<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img alt="GroundedAI: answers from your documents, with every citation checked" src="docs/assets/banner-light.svg" width="100%">
</picture>

<p align="center">
  <a href="https://github.com/pouyapd/GroundedAI/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/pouyapd/GroundedAI/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.12 and 3.13" src="https://img.shields.io/badge/python-3.12%20%7C%203.13-3776ab">
  <img alt="FastAPI, LangChain, React" src="https://img.shields.io/badge/FastAPI%20%C2%B7%20LangChain%20%C2%B7%20React-bf3989">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2ea043"></a>
  <img alt="Scope: local CPU, synthetic evaluation data" src="https://img.shields.io/badge/scope-local%20CPU%20%C2%B7%20synthetic%20eval%20data-6e7681">
</p>

<p align="center">
  <a href="#key-results">Key results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#screenshots">Screenshots</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#reproduce-the-evaluation">Reproduce</a> ·
  <a href="#what-is-verified-and-what-is-not">Verified / not verified</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a>
</p>

Retrieval-augmented generation (RAG) systems answer from documents, but their citations are
usually taken on trust, and their retrieval settings are chosen by habit. **GroundedAI** is a
complete RAG application that treats both as things to check. It ingests PDF and text documents,
retrieves passages with lexical, dense or hybrid search, and asks a language model to answer
only from those passages. The server then validates every citation against the passages the
model actually received, checks every statement of the answer against every retrieved passage
(and shows the exact supporting sentence, or says that none was found), reports retrieved
documents that disagree without choosing between them, flags planted instructions in retrieved
documents, and says so when the evidence is not enough. A built-in evaluation
framework measures retrieval and answer behaviour on a versioned dataset, with confidence
intervals, so configuration choices rest on numbers rather than assumptions.

In practice: upload a PDF, choose the document to ask, and get an answer whose every statement
comes with the quoted sentence and the page it is on, or a plain "not enough evidence" when the
document does not say.

Everything runs on a laptop CPU with no paid service: the default mode needs no model at all,
and local models (sentence-transformers, Ollama, transformers) are optional.

## Key results

- **The PDF journey is tested end to end.** A browser check drives the built UI against a real
  local server: damaged, text-less and unsupported files are refused with clear messages;
  a 3-page PDF is uploaded, and its answer quotes the right sentence and page (page 2), even
  though the short pages share one chunk; a question the selected document cannot answer gets
  "not enough evidence" instead of an answer from another document; switching and deleting
  documents leave no stale answers (24 of 24 checks, [script](scripts/e2e_browser.py)).
- **A real local model on that journey (Qwen2.5-0.5B-Instruct, transformers, CPU).** Of 5
  answerable questions it answered 1 correctly with its page and gave 1 malformed reply, which
  the claim check marked unverified; it wrongly refused the other 3. It refused both questions it
  should refuse ([report](evaluation/reports/journey/journey-qwen2.5-0.5b-20261018.json)). The
  small model is the weak link; the checks expose its failures rather than hide them.
- **Answer checks, version 2, measured once on a frozen test split.** On a 462-case synthetic
  test split, version 2 caught 69% of answers that miss the question (version 1: none) and every
  wrong citation. Disagreement recall rose from 0.38 to 0.47, and paraphrase acceptance fell from
  0.63 to 0.51. The labels were written and reviewed by models; the planned human-audit threshold
  was not met. Version 1 stays the default ([details](docs/RELIABILITY_V2.md)).

*Synthetic dataset written for this project (14 fictional documents, 56 questions, 43 with
evidence). Intel i7-1255U laptop, no GPU, mains power. Small sample: read the intervals.*

- **Hybrid retrieval did not beat BM25 here.** BM25 alone reached Recall@5 0.965 and MRR 0.934;
  hybrid Reciprocal Rank Fusion with bge-small reached the same Recall@5 (0.965) and MRR 0.915. No
  dense, hybrid or reranked configuration is distinguishable from BM25 (paired 95% bootstrap
  intervals all include zero), and BM25 is the fastest (3.6 ms median vs. 84 ms). At the
  application's default chunking (800/120 characters) BM25 reached Recall@5 1.000 and the shipped
  hybrid default 0.977. The questions reuse the documents' wording, which favours lexical search;
  the dataset limitation is stated in the report rather than hidden.
- **A valid citation is not a correct answer.** The deterministic extractive baseline has 100%
  citation validity by construction, yet it answered 5 of the 10 questions it should have refused,
  each with a valid citation (for "What is the capital of Australia?" it quoted a sentence about
  hotel limits in capital cities). This is why GroundedAI reports citation validity, sentence
  support and abstention behaviour separately.
- **A revised prompt stopped wrong refusals; citations are still the weak point.** With the
  original prompt, Qwen2.5-0.5B refused 29 answerable questions although retrieval had found the
  evidence. The `answer-v2` prompt, developed on a separate 25-question development set, was then
  frozen and run once on the evaluation set. With it, the configured model `qwen2.5:1.5b`
  (Ollama, CPU) answered all 43 answerable questions and found 94% of the key facts (v1: 84% and
  77%; both differences have paired 95% intervals above zero). It also refused fewer of the
  questions it should refuse (7 of 10, v1: 9 of 10), found neither of the two conflicts, and put
  a valid `[S#]` citation on only 13 of its 49 answers.
- **Its answers come from the passages; its citations do not.** Re-checking those recorded
  outputs statement by statement (no new model run) found 116 of its 118 statements in a
  retrieved passage, 47 of 49 answers fully matched, and 90% of key facts traced to the dataset's
  gold evidence; 9% of the statements it did cite were cited to a passage that does not contain
  them. The application
  now attaches the passage it found and says when the model's citation was wrong or missing. The
  check is lexical: its three wrong answers to unanswerable questions consist of true statements
  and pass it ([audit](docs/RELIABILITY_AUDIT.md)).
- **Disagreeing sources are shown, not resolved.** A new check found 12 of 16 disagreements in a
  held-out case set committed before the check was written (0 false alarms in 14 cases without
  one; the earlier figure check: 6 found, 3 false alarms). On the evaluation set it surfaces one of
  the two conflicts.

| Retrieval configuration (n = 43) | Recall@5 | MRR | nDCG@5 | p50 latency |
|---|:---:|:---:|:---:|:---:|
| BM25 (SQLite FTS5) | **0.965** [0.91, 1.00] | **0.934** [0.86, 0.99] | **0.931** [0.87, 0.98] | 3.6 ms |
| Dense, bge-small-en-v1.5 | 0.942 [0.87, 1.00] | 0.879 [0.80, 0.95] | 0.886 [0.80, 0.96] | 104 ms |
| Hybrid RRF (BM25 + bge-small) | **0.965** [0.91, 1.00] | 0.915 [0.84, 0.98] | 0.920 [0.85, 0.97] | 84 ms |
| Hybrid RRF + cross-encoder rerank | 0.942 [0.88, 0.99] | 0.920 [0.85, 0.98] | 0.917 [0.84, 0.98] | 810 ms |

| Answer evaluation (56 questions) | Correct behaviour | Answerable answered | Correct refusals | Key-fact recall | Citation format followed | p50 latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Extractive baseline (not an LLM) | 0.660 | 30 of 43 | 5 of 10 | 0.558 | 100% | < 0.1 s |
| Qwen2.5-0.5B, prompt v1 (transformers) | 0.340 † | 8 of 43 | 10 of 10 | 0.163 | 13% | 32 s |
| Qwen2.5-0.5B, prompt v2 | 0.528 | 21 of 43 | 7 of 10 | 0.302 | 8% | 38 s |
| qwen2.5:1.5b, prompt v1 (Ollama, Q4_K_M) | 0.830 | 36 of 43 | 9 of 10 | 0.767 | 30% | 11 s |
| **qwen2.5:1.5b, prompt v2** (default Ollama setup) | **0.906** | **43 of 43** | 7 of 10 | **0.942** | 27% | 10 s |

*Hybrid RRF retrieval with bge-small, top 5 passages, greedy decoding, Intel i7-1255U, no GPU,
mains power. "Correct behaviour" counts an expected answer as correct even if its content is
wrong; key-fact recall measures content. Neither conflict question was handled correctly by any
configuration. † The published 0.377, re-scored with the current refusal rule.*

| Recorded outputs re-checked (no model run) | Statements found in a passage | Answers with every statement found | Key facts traced to gold evidence | Model citations pointing to the right passage |
|---|:---:|:---:|:---:|:---:|
| qwen2.5:1.5b, prompt v2 | 98% (116 of 118) | 47 of 49 | 90% | 88% |
| qwen2.5:1.5b, prompt v1 | 86% | 30 of 37 | 86% | 90% |
| Qwen2.5-0.5B, prompt v2 | 72% | 15 of 25 | 57% | 79% |

Full tables, metric definitions and failure analysis: [docs/EVALUATION.md](docs/EVALUATION.md).

## How it works

```mermaid
flowchart LR
  UI["React UI"] --> API["FastAPI /api/v1"]
  CLI["CLI"] --> SVC
  API --> SVC["Services: ingestion · QA · evaluation · status"]
  SVC --> RET["Retrieval: BM25 + dense vectors → RRF → optional rerank"]
  RET --> DB[("SQLite FTS5 + vectors  or  PostgreSQL + pgvector")]
  SVC --> GEN["LangChain: prompt → chat model"]
  GEN --> LLM["extractive demo · Ollama · transformers"]
  SVC --> PRE["Before generation: disagreement check · injection flags → notes to the model"]
  SVC --> CHK["After generation: citation check · claim check against every passage · refusal detection"]
```

1. **Ingest.** Uploads are validated (type, size, magic bytes, encoding), parsed with page
   numbers, split into overlapping chunks (each records where its pages start), embedded and
   indexed. Identical files are detected by hash; deleting a document removes its chunks and
   stored file. A file that cannot be read (damaged, password-protected, no text) is deleted at
   once and the document keeps the reason; uploading it again retries.
2. **Retrieve.** Full-text search (BM25 on SQLite, `ts_rank_cd` on PostgreSQL) and vector search
   run in parallel and are fused with Reciprocal Rank Fusion. Every result carries its lexical
   rank, dense rank, fused and rerank scores, matched terms and timings.
3. **Generate.** A LangChain chain sends the policy and the numbered passages as separate
   messages, using a versioned prompt (`answer-v2` by default; `answer-v1` reproduces the earlier
   reports). Two deterministic checks run first. The disagreement check finds passages from
   different documents that give different values for what is asked (or a requirement and its
   denial), separates values for different dates or groups, and returns both quotes; the model
   gets a short note. Passages with instruction-like text are flagged. If nothing relevant is
   retrieved, no model is called and the answer says so.
4. **Check.** Each `[S#]` label must match a passage the model received. Every statement is then
   compared with every retrieved passage (figures with units, and most of its words): it is
   supported by its citation, cited to the wrong passage, found in a passage the model did not
   cite, partly found, or not found; the exact supporting sentence and its page are returned.
   Questions can be limited to chosen documents (`document_ids`). Instruction-like
   passage text never counts as evidence. Refusals are recognised even when a small model
   misspells them or adds a citation; a reply with only citation markers is reported as invalid.
5. **Measure.** The same pipeline runs offline over a versioned dataset and writes JSON, CSV and
   Markdown reports with bootstrap intervals and the hardware and power source.

If the configured model is unavailable, the API returns `503 model_unavailable` with setup
guidance; it never switches to a different model on its own. Design decisions:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Screenshots

Taken from the running application. The first two rows use the sample PDFs in `examples/`
(2026-10-18); the others use the sample corpus with bge-small embeddings.

| One document selected: the answer quotes its sentence and page (Qwen2.5-0.5B-Instruct, local) | The same question with the extractive demo generator (not an LLM) |
|---|---|
| ![Ask page with Search in set to the Harbourline Ferries Passenger Guide; a Qwen2.5-0.5B answer that a single adult ticket costs 14 euros, the badge Every statement found in the sources, and the supporting quote attributed to page 2](docs/screenshots/journey-answer-qwen-light.png) | ![The same answer from the extractive demo generator, with the quote on page 2 and the passage spanning pages 1 to 3](docs/screenshots/journey-answer-light.png) |
| **A question the selected document cannot answer** | **Documents: two ready PDFs, a damaged one and one without text, with the reasons** |
| ![Answer panel saying the searched documents do not contain enough evidence, for a library question asked of the ferry guide](docs/screenshots/journey-not-in-document-light.png) | ![Documents table with Ask, Chunks, Re-ingest and Delete buttons, the ferry guide's extracted text, and two failed uploads whose files were not kept](docs/screenshots/journey-documents-light.png) |
| **Claim check: each statement with the exact supporting sentence (extractive generator)** | **Disagreeing sources shown side by side, with values and dates (extractive generator)** |
| ![Answer with the verdict Every statement found in the sources, the claim check listing one statement as Supported by the cited passage with the quoted sentence from the travel policy, and the passage used as evidence](docs/screenshots/ask-claim-check-light.png) | ![Answer panel where the generator found no answer but a Different values for different dates panel shows the travel policy's 140 and 110 euro hotel limits next to the finance FAQ's 120 euro limit, updated May 2025](docs/screenshots/ask-disagreement-light.png) |
| **Evaluation: qwen2.5:1.5b with the answer-v2 prompt** | **Evaluation: retrieval benchmark** |
| ![Evaluation dashboard for the qwen2.5:1.5b answer-v2 report: 91% correct behaviour, 70% correct refusals, 94% key-fact recall, 27% citation format followed, and results by category](docs/screenshots/evaluation-ollama-light.png) | ![Evaluation dashboard with Recall@5, MRR and nDCG@5 per configuration, 95% intervals and latency](docs/screenshots/evaluation-light.png) |
| **Documents (dark theme)** | **Operations on a phone** |
| ![Documents table with status, pages and chunks](docs/screenshots/documents-dark.png) | ![Operations page at phone width](docs/screenshots/operations-mobile-dark.png) |

A static, read-only version of the UI can be built with `npm run build:demo`. Ten dataset
questions (five where the extractive generator behaves as expected, five known failures) were
recorded with every retrieval mode and passage count, so the Hybrid / Lexical / Dense and
passage controls replay the recording made with exactly that setting, and each answer shows the
dataset's expected result. The published reports are included; nothing is generated live. See
[docs/DEPLOYMENT_STATUS.md](docs/DEPLOYMENT_STATUS.md) for where it is hosted and
[docs/UI_CONTROL_AUDIT.md](docs/UI_CONTROL_AUDIT.md) for how every control was checked.

## Quick start

Requires Python 3.12+ and Node 22+ (developed with Python 3.12.7 and Node 24).

```bash
git clone https://github.com/pouyapd/GroundedAI.git && cd GroundedAI
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements/dev.txt
pip install --no-deps -e backend

groundedai ingest evaluation/datasets/groundedai-synthetic-v1/corpus    # sample documents
groundedai ask "During a boil water notice, how long should tap water be boiled?"
groundedai serve                          # API on http://127.0.0.1:8000, docs at /docs
```

UI in a second terminal: `cd frontend && npm ci && npm run dev`, then open http://localhost:5173.
Or serve the built UI from the API process:
`cd frontend && npm ci && npm run build && cd .. && GROUNDEDAI_FRONTEND_DIST=frontend/dist groundedai serve`.

**Try the PDF journey.** On **Documents**, upload `examples/harbourline-ferry-guide.pdf` and
`examples/orchard-library-rules.pdf`, then press **Ask** next to the ferry guide:

- "How much does a single adult ticket to Mirrow cost?" → 14 euros, quoted from page 2.
- "How many items can members borrow?" → not enough evidence in the ferry guide; choose the
  library rules under **Search in** and ask again → 12 items.

The default configuration uses the **hashing embedder** (word overlap, no model) and the
**extractive demo generator** (it copies sentences from the sources; it is not a language model).
The UI and every API response say which generator produced an answer.

**Local models (optional).** Install the ML extra (CPU-only PyTorch), download models explicitly,
and point GroundedAI at them:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements/ml.txt
pip install --no-deps -e backend
python scripts/download_models.py --preset recommended      # bge-small + reranker, asks first
ollama pull qwen2.5:1.5b                                    # https://ollama.com, about 1 GB
export GROUNDEDAI_GENERATION_PROVIDER=ollama GROUNDEDAI_EMBEDDING_PROVIDER=sentence-transformers
export GROUNDEDAI_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export GROUNDEDAI_EMBEDDING_QUERY_PREFIX="Represent this sentence for searching relevant passages: "
groundedai serve
```

Verified on the development laptop with Ollama 0.40.3 and `qwen2.5:1.5b` (Q4_K_M, digest
`65ec06548149`): about 1.3 GB of memory for the model runner and 10 s per answer (median) on an
i7-1255U without a GPU. Without Ollama, `GROUNDEDAI_GENERATION_PROVIDER=huggingface-local` runs
`Qwen/Qwen2.5-0.5B-Instruct` through transformers (download it first with
`python scripts/download_models.py --generation Qwen/Qwen2.5-0.5B-Instruct`); it is weaker (see
Key results). The prompt can be switched with `GROUNDEDAI_PROMPT_VERSION` (`answer-v2`, default,
or `answer-v1`).

Docker (PostgreSQL + pgvector + API/UI): `cp .env.example .env`, set `POSTGRES_PASSWORD`, then
`docker compose up --build` (verified on Windows with Docker Desktop: build, health checks,
ingestion, questions in all three retrieval modes, restart with data kept, and shutdown).
Details: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md),
[docs/MODELS.md](docs/MODELS.md), [docs/RESOURCE_PROFILE.md](docs/RESOURCE_PROFILE.md).

## API

```bash
curl -s http://127.0.0.1:8000/api/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms for Aldervale suppliers?", "include_diagnostics": true}'
```

The response contains the answer with `[S#]` markers, `sources` (title, page, excerpt, scores,
injection flags, which claims each passage supports), `claims` (status and exact quote with
character offsets per statement), `verification` (verdict and counts), `conflicts` (disagreeing
passages with values and dates), `sentences` (the strict `[S#]` check), `citation_check`,
`warnings`, `provider` (including `is_llm`), `timings` and `usage`. Other endpoints: document upload,
listing, chunks, re-ingest and delete; retrieval-only `/retrieve`; evaluation reports and runs;
JSON and Prometheus metrics; health and readiness. Reference with a recorded example response:
[docs/API.md](docs/API.md).

## Reproduce the evaluation

```bash
groundedai eval retrieval --config evaluation/configs/retrieval_offline.json   # no models, ~5 s
HF_HUB_OFFLINE=1 groundedai eval retrieval --config evaluation/configs/retrieval_full.json
groundedai eval generation --provider extractive-demo
groundedai eval generation --provider ollama --model qwen2.5:1.5b --checkpoint data/q15.jsonl  # resumable
python scripts/compare_generation_reports.py BASE.json OTHER.json   # paired bootstrap intervals
python scripts/revalidate_recorded_answers.py REPORT.json   # re-check recorded outputs, no model run
python scripts/evaluate_conflict_cases.py                   # held-out disagreement cases
```

The dataset files are hash-checked before every run, and each report records the dataset
fingerprint, configuration, model revisions, package versions, hardware and power source.
Retrieval metrics are deterministic (tested). Exact commands for the published reports:
[docs/EVALUATION.md](docs/EVALUATION.md#reproducing).

## Testing

```bash
cd backend && ruff check groundedai tests && mypy && pytest      # offline: no models, keys or network
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres   # real PostgreSQL, no Docker
python scripts/e2e_browser.py               # the PDF journey in Chromium against a real server
python scripts/journey_check.py --out r.json   # the same journey through the API, configured model
```

`e2e_browser.py` needs `playwright` and its Chromium build, which are not project dependencies;
CI does not run it.

Unit, integration, API, security and end-to-end tests run on SQLite with deterministic providers;
in the test suite the Ollama client talks to a fake Ollama HTTP server (real Ollama inference was
checked separately, see below); PostgreSQL + pgvector tests run against a real server. Counts and
the latest results: [docs/TESTING.md](docs/TESTING.md).

## What is verified and what is not

| Verified | Not verified |
|---|---|
| 419 backend tests (offline, SQLite) and 70 front-end tests; ruff and strict mypy clean (Windows and Linux targets) (2026-10-18) | CI on the local commits after `100f124` (not pushed); CI last passed, 7 of 7 jobs, on `100f124` |
| the PDF journey in Chromium against a real local server, 24 checks ([script](scripts/e2e_browser.py), 2026-10-18) | the browser check in CI (it needs playwright, which is not a project dependency) |
| a real local model (Qwen2.5-0.5B-Instruct, transformers, CPU) on that journey through the HTTP API and the UI (2026-10-18) | the 56-question evaluation with the checks of 2026-10-14 and later: only the 7-question journey was run with a model; the Ollama files were deleted |
| PostgreSQL 16 + pgvector: migrations, all retrieval modes, cascade delete, evidence pages (7 tests, bundled local server, 2026-10-18) | retrieval and answer quality on real or larger document collections |
| real Ollama inference (0.40.3, qwen2.5:1.5b) through the API, the UI and a 56-question evaluation (2026-10-13) | |
| the claim check on the recorded outputs of three model runs, with the rebuilt passages verified against every recorded result | that a "verified" statement is correct: the check is lexical; human or model judging was not done |
| the disagreement check on 30 held-out cases written before it | larger models; any LLM-as-judge or human evaluation |
| restart: documents, chunks and answers unchanged after restarting on the same data (test) | |
| the full Docker Compose stack on Windows: build, health, ingestion, questions, restart, shutdown (2026-10-13) | the Docker image with the 2026-10-18 changes (Docker Desktop was not running, and a build downloads packages); the Ollama service inside Compose (`--profile llm`) |
| retrieval benchmark rerun reproduces every published metric and interval exactly | a statistically supported advantage of hybrid retrieval over BM25 (none was found) |
| every UI control, live and in the static demo, at desktop and phone width ([audit](docs/UI_CONTROL_AUDIT.md)) | the private claude.ai preview as rendered for a signed-in viewer (checked through its files and a local sandboxed copy) |

## Technology

| Layer | Technology |
|---|---|
| API and services | Python 3.12, FastAPI, Pydantic, Uvicorn |
| Orchestration | LangChain (prompts, LCEL chain, text splitter, Ollama and Hugging Face integrations) |
| Storage and search | SQLAlchemy 2, Alembic; SQLite FTS5 or PostgreSQL 16 with pgvector (HNSW) |
| Models (optional) | sentence-transformers (bge-small, MiniLM), cross-encoder reranker, Ollama, transformers |
| UI | React 19, TypeScript, Vite; hand-written SVG charts |
| Quality | pytest, ruff, mypy (strict), Vitest, ESLint, pip-audit, npm audit, GitHub Actions |
| Deployment | Docker (multi-stage, non-root), Docker Compose, static demo build |

## Repository layout

```
backend/groundedai/   api · core · ingestion · retrieval · generation · services · evaluation
                      persistence (models, migrations, search backends) · observability · security
backend/tests/        unit · integration · api · e2e
examples/             two small sample PDFs with known content (scripts/make_sample_pdfs.py)
frontend/             React UI (src/) and static-demo data (public/demo/)
evaluation/           datasets/ · configs/ · reports/published/ · scripts/
scripts/              check_environment · download_models · dev_postgres · export_demo_data · bench_api
docs/                 architecture · API · evaluation · security · deployment · status documents
```

## Limitations

- The evaluation dataset is small, synthetic and lexically easy; it cannot show where dense
  retrieval helps, and its numbers do not transfer to other domains.
- The claim check is lexical: a "found" statement can still be wrong or beside the question, and
  a correct paraphrase can be marked "partly found".
- The local models tested here are small (0.5B and 1.5B). qwen2.5:1.5b usually gives the right
  fact but rarely cites it in the required form, and it answers some questions it should refuse.
- The disagreement check is pattern-based: it misses synonyms and conflicts that share one value,
  and it does not decide which source is right ([known issues](docs/RELIABILITY_AUDIT.md#known-issues)).
- Prompt injection is reduced (separate messages, neutralised tags and labels, pattern flags,
  a note to the model), not eliminated. See [docs/SECURITY.md](docs/SECURITY.md).
- No OCR, tables or images; English-focused tokenisation. A quote's page is the page on which it
  starts. The sentence splitter joins a PDF heading line to the sentence after it ("Tickets A
  single adult ticket…"); the quote is still the document's exact text.
- Single-tenant: one API key, no per-user document permissions. Rate limits and in-process
  metrics are per process.

More: [docs/RELIABILITY_AUDIT.md](docs/RELIABILITY_AUDIT.md) ·
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) ·
[docs/ENVIRONMENT_AND_RESOURCE_STRATEGY.md](docs/ENVIRONMENT_AND_RESOURCE_STRATEGY.md) ·
[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

## Licence

MIT, see [LICENSE](LICENSE). Third-party packages and optional models keep their own licences:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
