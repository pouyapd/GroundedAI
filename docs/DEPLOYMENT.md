# Deployment

GroundedAI runs well locally. Local deployment is the reliable mode; the online demo is a
static replay. See `docs/DEPLOYMENT_STATUS.md` for what has actually been deployed.

## 1. Local, no Docker (SQLite)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements/dev.txt
pip install --no-deps -e backend

groundedai ingest evaluation/datasets/groundedai-synthetic-v1/corpus   # optional sample data
groundedai serve                         # http://127.0.0.1:8000  (API + /docs)
```

UI during development (hot reload, proxies `/api` to port 8000):

```bash
cd frontend && npm ci && npm run dev     # http://localhost:5173
```

UI and API from one process:

```bash
cd frontend && npm ci && npm run build && cd ..
GROUNDEDAI_FRONTEND_DIST=frontend/dist groundedai serve
# PowerShell: $env:GROUNDEDAI_FRONTEND_DIST="frontend/dist"; groundedai serve
```

Data lives in `./data` (SQLite database and uploaded originals). Migrations run automatically
at startup; `groundedai migrate` runs them explicitly.

## 2. Docker Compose (PostgreSQL + pgvector)

```bash
cp .env.example .env        # set POSTGRES_PASSWORD; keep GROUNDEDAI_API_KEY empty for local use
docker compose up --build   # http://127.0.0.1:8000
docker compose logs -f api
```

* The `api` image contains the built UI. Its entrypoint runs `groundedai migrate`, then Uvicorn.
* Volumes: `pgdata` (database), `appdata` (uploaded originals), `hfcache` (models, if used).
* Health: Docker's `HEALTHCHECK` calls `/health`; `/ready` checks the database, migrations and
  the pgvector dimension.
* Image size: the default image has no PyTorch. `INSTALL_ML=true` in `.env` (or
  `--build-arg INSTALL_ML=true`) adds sentence-transformers with CPU-only PyTorch (about
  1 GB more).

Optional local LLM with Ollama in Compose:

```bash
docker compose --profile llm up --build
docker compose exec ollama ollama pull qwen2.5:1.5b      # explicit model download (~1 GB)
# in .env:
#   GROUNDEDAI_GENERATION_PROVIDER=ollama
#   GROUNDEDAI_OLLAMA_BASE_URL=http://ollama:11434
docker compose up -d api
```

Shut down and clean up:

```bash
docker compose down            # stop, keep data
docker compose down -v         # stop and delete volumes (database, uploads, models)
docker image rm groundedai-api # remove the built image
```

### Using real embeddings in Compose

Set in `.env`:

```bash
INSTALL_ML=true
GROUNDEDAI_EMBEDDING_PROVIDER=sentence-transformers
GROUNDEDAI_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
GROUNDEDAI_EMBEDDING_DIM=384
GROUNDEDAI_EMBEDDING_QUERY_PREFIX="Represent this sentence for searching relevant passages: "
```

Then download the model into the `hfcache` volume once (explicit, asks for confirmation):

```bash
docker compose run --rm --entrypoint python api scripts/download_models.py --embedding BAAI/bge-small-en-v1.5
```

Alternatively mount an existing local Hugging Face cache at `/models/huggingface`. The pgvector
column dimension is fixed by the first migration; changing the embedding dimension later needs a
fresh database or a new migration.

## 3. Production checklist

GroundedAI includes production-oriented features (migrations, health/readiness probes, structured
logs, request ids, metrics, rate limits, non-root container), but it has not been operated in
production. Before exposing it:

- [ ] `GROUNDEDAI_ENV=production` and a strong `GROUNDEDAI_API_KEY`
- [ ] TLS reverse proxy (Caddy, nginx, Traefik) in front; only the proxy is public
- [ ] `GROUNDEDAI_CORS_ORIGINS` set to the UI origin
- [ ] Rate limiting at the proxy if you run more than one worker or replica
- [ ] Backups of the PostgreSQL volume and the uploads volume
- [ ] Log shipping (stdout JSON) and a Prometheus scrape of `/api/v1/metrics/prometheus` (needs the API key)
- [ ] A model sized for the hardware; `GROUNDEDAI_MAX_CONCURRENT_GENERATIONS` tuned to the CPU/GPU
- [ ] Read `docs/SECURITY.md`, especially the limits of prompt-injection handling and the lack of per-user permissions

## 4. Static online demo (GitHub Pages)

The static demo is the React app built with `VITE_DEMO_MODE=true`. It has no backend: it
replays answers that were recorded from a real local run (`frontend/public/demo/answers.json`)
and shows the published benchmark reports. It is clearly labelled as a recording and never
performs inference.

Refresh the recorded data and build locally:

```bash
python scripts/export_demo_data.py         # records answers from a local run
cd frontend && npm run build:demo          # output in frontend/dist
npm run preview                            # check it at http://localhost:4173
```

Publish with GitHub Pages:

1. Push the repository to GitHub (public repository for free Pages).
2. Repository settings → Pages → Source: **GitHub Actions**.
3. The workflow `.github/workflows/pages.yml` builds and deploys on pushes to `main` that touch
   `frontend/`, or manually from the Actions tab.

The demo uses relative asset paths, so it works under `https://<user>.github.io/<repo>/`.

## 5. Backend hosting (optional)

No free backend host is assumed. If you host the API elsewhere, use the Docker image, an external
PostgreSQL with pgvector, the production checklist above, and a model that fits the host's memory.
Free tiers that sleep when idle or limit memory to 512 MB are not suitable for local LLM inference.
