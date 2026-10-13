# Environment and resource strategy

This document records where GroundedAI was developed and tested, what was downloaded, and
how to run it on a 16 GB RAM, CPU-only machine without paid services.

## Where development and testing happened

No remote or cloud workspace was available for this project, so all development and testing
ran **locally on the owner's laptop**:

| Resource | Value (measured with `scripts/check_environment.py` and system tools) |
|---|---|
| OS | Windows 11 Enterprise |
| CPU | 12th Gen Intel Core i7-1255U, 10 cores / 12 logical CPUs |
| RAM | 15.6 GB total; typically 2–3 GB free while other applications were open |
| GPU | Intel Iris Xe (integrated); no CUDA GPU, so everything runs on the CPU |
| Disk | 31.8 GB free on C: at the start of the work |
| Python / Node | Python 3.12.7, Node 24.19.0, npm 11.17.0 |
| Docker | Docker Desktop installed (engine 29.4.1); not running at the start. Used once on 2026-10-13 to verify the Compose stack (see below) |
| Ollama | Not installed at the start. On 2026-10-13 the portable Ollama 0.40.3 build and `qwen2.5:1.5b` were used for the real-LLM checks (see below) |
| Network | System-wide HTTP proxy variables set (see “Proxy” below) |

Laptop power matters for latency numbers: every report records the power source. The
published retrieval benchmark ran on mains power. Runs made on battery are labelled as such.

## How disk usage was controlled

* **No new model downloads.** Every model used for measurements was already in the local
  Hugging Face cache before this project started:
  `BAAI/bge-small-en-v1.5` (129 MB), `sentence-transformers/all-MiniLM-L6-v2` (175 MB),
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (88 MB) and `Qwen/Qwen2.5-0.5B-Instruct` (954 MB).
  The application never downloads models implicitly (`GROUNDEDAI_HF_OFFLINE=true` by default;
  `scripts/download_models.py` is the only path that downloads, and it asks first).
* **No second copy of PyTorch.** The optional ML environment (`.venv-ml`) was created with
  `--system-site-packages` so that it reuses the PyTorch 2.10 (CPU) already installed for the
  owner's other projects. It added sentence-transformers, transformers and their small
  dependencies only (237 MB in total).
* **Docker only when needed, and cleaned up.** Until 2026-10-12 the Dockerfile and Compose file
  were only validated with `docker compose config`, and the image was built in GitHub Actions.
  On 2026-10-13 the full stack was started once to verify it (pgvector image 631 MB, API image
  469 MB); afterwards the containers, volumes and both images were removed and Docker Desktop
  was stopped.
* **Ollama without installing it.** For the real-LLM checks the portable Ollama 0.40.3 zip
  (1.47 GB, SHA-256 checked against the release) was unpacked into a temporary folder and run
  bound to `127.0.0.1` only, with its model store in the same folder: no installer, service,
  autostart or PATH change. `qwen2.5:1.5b` (986 MB) was the only model pulled, after estimating
  its needs (about 1.5 GB of memory, 10–25 s per answer); measured: 1.3 GB and 10 s median.
* **PostgreSQL without Docker.** Postgres tests ran against a throwaway server from the
  `pgserver` wheel (12.8 MB, PostgreSQL 16.2 + pgvector 0.6.2), started and removed by
  `scripts/dev_postgres.py`.

Measured footprint of this project on disk:

| Item | Size |
|---|---|
| `.venv` (runtime + test tools) | 346 MB |
| `.venv-ml` (optional ML packages; reuses system PyTorch) | 237 MB |
| `frontend/node_modules` | 159 MB |
| Repository files incl. dataset, reports, screenshots, local data | ~46 MB |

The pip and npm download caches in the user profile also grew during installation (they were
already 1.9 GB and 0.6 GB in total, and the share added by this project was not separated).

### Cleanup

```bash
# project-local environments and build output (re-creatable)
rm -rf .venv .venv-ml frontend/node_modules frontend/dist data
# optional: shared download caches
pip cache purge
npm cache clean --force
```

Models in `~/.cache/huggingface` are shared with other projects; remove a single model with
`huggingface-cli delete-cache` (interactive) if it is no longer needed.

## Which components need models

| Component | Default (no model) | Optional (needs a model) |
|---|---|---|
| Embeddings | `hashing` – deterministic feature hashing, word overlap only | `sentence-transformers` (e.g. bge-small, ~130 MB) or `ollama` |
| Reranking | off | cross-encoder (~90 MB) |
| Generation | `extractive-demo` – sentence selection, **not an LLM** | `ollama` (e.g. `qwen2.5:1.5b`, ~1 GB) or `huggingface-local` (e.g. Qwen2.5-0.5B, ~1 GB) |

## Which tests run without models

All of them by default. The default suite (`pytest` in `backend/`) uses the hashing embedder,
the extractive demo generator and LangChain's fake chat models, and tests the Ollama
integration against a fake Ollama HTTP server. It needs no internet once dependencies are
installed. The PostgreSQL tests are skipped unless `GROUNDEDAI_TEST_POSTGRES_URL` is set. Real
model runs are separate commands (`groundedai eval ...` in the ML environment).

## Running on a 16 GB RAM CPU-only machine

* **Lightweight mode** (defaults): about 120 MB of memory for the API process with the
  sample corpus indexed (measured). Works on any machine that runs Python 3.12.
* **Real embeddings** (bge-small): about 630 MB for the API process (measured, PyTorch
  included).
* **Real local LLM**: Qwen2.5-0.5B through transformers used about 1.5–1.8 GB of working set
  (measured). With Ollama, `qwen2.5:1.5b` (Q4) needs roughly 1–2 GB; larger models need more and
  run slower on CPU.

See `docs/RESOURCE_PROFILE.md` for the full comparison.

## Optional cloud features and their limits

Nothing in the default workflow needs a cloud account. Optional pieces depend on external
services with their own limits:

* **GitHub Actions** (CI, Docker build, Postgres tests): free for public repositories within
  GitHub's usage limits.
* **GitHub Pages** (static demo): free for public repositories; serves only static files, so
  the online demo replays recorded answers and cannot run inference.
* No free backend host is assumed to keep a model running; most free tiers sleep or limit
  memory below what a local LLM needs.

## Proxy

The development machine sets `HTTP_PROXY`/`HTTPS_PROXY` without `NO_PROXY`. Python HTTP
clients then send even `localhost` requests to the proxy, which answered with HTTP 503. GroundedAI therefore ignores environment proxies for loopback model
URLs (`groundedai/core/net.py`), and maps HTTP 502/503/504 from Ollama to a
"model unavailable" error that mentions proxies. Remote model URLs still use the proxy.
