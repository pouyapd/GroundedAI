# Testing

## Layers

| Layer | Location | What it covers |
|---|---|---|
| Unit | `backend/tests/unit` | text utilities, upload validation, PDF/text parsing, chunking, embeddings, fusion, citations and support checks, context assembly, injection heuristics, extractive generator, evaluation metrics, configuration, rate limiter, metrics registry, dataset integrity, CLI |
| Integration | `backend/tests/integration` | Alembic migrations (upgrade, downgrade, schema matches models), ingestion lifecycle (failure, retry, re-ingest, delete without orphans, crash recovery), SQLite search and hybrid retrieval, reranking, QA service failure modes (timeout, busy, provider errors, invalid citations, hallucinated numbers), LangChain retriever adapter, Ollama integration against a fake Ollama HTTP server, evaluation runners on the bundled dataset |
| PostgreSQL | `backend/tests/integration/test_postgres.py` | migrations with pgvector/tsvector/HNSW, all retrieval modes, hostile queries, cascade delete (runs when `GROUNDEDAI_TEST_POSTGRES_URL` is set) |
| API | `backend/tests/api` | every endpoint, status codes, error shape, validation, size limits, API key, rate limiting, CORS, security headers, request ids, OpenAPI paths, metrics, evaluation runs |
| End to end | `backend/tests/e2e` | upload PDF → ask → cited answer with page → abstention → delete → metrics, through HTTP only |
| Front end | `frontend/src/test` | citation rendering, answer view states, abstention, LLM vs demo labels, offline backend, model unavailable, question flow, empty documents, evaluation report rendering (invalid outputs, latency, category names), static demo replay and expected-result comparison, upload validation, formatting |
| Build/config | CI | lint, type checks, front-end build (live and demo), Docker image build and container smoke test, Compose validation, dependency audits |

Mocks are used only at the model boundary (LangChain fake chat models, a slow/failing model, a
fake Ollama server, a deterministic reranker). Retrieval, persistence, citation checks and the
API run for real against SQLite (and PostgreSQL in the Postgres job).

## Commands

```bash
cd backend
ruff check groundedai tests && ruff format --check groundedai tests
ruff check --config pyproject.toml ../scripts ../tools ../evaluation/scripts
mypy                                    # strict mode
pytest                                  # default suite: offline, no models, no API keys
pytest --cov                            # with coverage

# PostgreSQL tests without Docker (bundled Postgres from the pgserver wheel)
pip install pgserver
cd .. && python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres

cd frontend
npm run lint && npm run typecheck && npm test && npm run build
```

## Results recorded on 2026-10-12 (development laptop, Windows 11, Python 3.12.7, Node 24.19.0)

| Check | Result |
|---|---|
| `pytest --cov` (default suite, run from `backend/` as in CI) | 219 passed, 6 skipped (the 6 PostgreSQL tests, no database URL set); line coverage 91% |
| `pytest -m postgres` via `scripts/dev_postgres.py` (PostgreSQL 16.2, pgvector 0.6.2) | 6 passed |
| `ruff check`, `ruff format --check` (`groundedai`, `tests`, `scripts`, `tools`, `evaluation/scripts`) | clean |
| `mypy` (strict, 75 files) | no issues |
| `vitest` | 25 passed |
| `eslint`, `tsc -b` | clean |
| `npm run build`, `npm run build:demo` | succeeded |
| `pip-audit` (prod, dev and ML locks) | no known vulnerabilities (the CPU-only torch wheel is skipped: not on PyPI) |
| `npm audit` | 0 vulnerabilities |
| `docker compose config` (with and without the `llm` profile) | valid |
| Docker image build | **not run locally** (no image downloads on this machine); runs in CI |
| UI in a browser (live app and static demo) | 390 px and 1280 px wide: no horizontal scrolling on any tab or recorded answer; chart text at its designed size (viewBox width equals rendered width) |

The GitHub Actions results are recorded in `docs/DEPLOYMENT_STATUS.md`.

## Regression tests for bugs found during development

| Bug | Test |
|---|---|
| Citations written after a full stop (`Fact. [S1] Next.`) were credited to the next sentence | `test_citation_after_full_stop_belongs_to_previous_sentence` |
| Misspelled or paraphrased abstentions from a small LLM (`[Insuffcient Evidence: ...]`) were scored as ungrounded answers | `test_tolerant_abstention_detection_for_small_models` |
| With `HTTP_PROXY` set, requests to a local Ollama went to the proxy and failed with HTTP 503 | `test_loopback_urls_bypass_environment_proxies`, `test_unreachable_server_is_reported` |
| A module named `types.py` shadowed the standard library when run from its directory | module renamed to `column_types.py` |
| Python's salted `hash()` would make hashing embeddings differ between processes | `test_hashing_embeddings_are_deterministic_across_instances` |
| Plain-text documents showed their file name instead of their first-line title | `test_text_title_only_from_short_heading_like_first_line` |
| Stray space before punctuation after removing citation markers | `test_strip_citations` |
| An interrupted LLM evaluation lost every completed answer (results were written only at the end) | `test_generation_eval_resumes_from_checkpoint` |
| A small LLM's citation-only replies (`[S1][S2]`, no text) were scored as answers, inflating behaviour and coverage | `test_citation_markers_without_text_are_invalid_output` |
| "Insufficient evidence" after leading labels (`[S1][S4]\nInsufficient Evidence: ...`) was scored as an answer | `test_abstention_after_leading_labels_is_recognised` |
| A checkpoint scored with older rules could be resumed after a scoring change | `SCORING_VERSION` in the checkpoint header (mismatch is refused) |
| Charts scaled their text with the container (unreadable small multiples, oversized latency chart) | checked in the browser; charts now render at natural size |
| On phones the charts were scaled down to fit, so 12 px labels showed at about 6.7 px | checked in the browser at 390 px; charts measure their container (`ResizeObserver`) and lay out for that width |
| A visually hidden table header escaped its scroll container and widened the page on phones | checked in the browser at 390 px; `.table-wrap` is positioned |
| The "not an LLM" badge under an answer did not wrap and widened the page on phones by 6 px | checked in the browser at 390 px for all ten recorded answers |
| The static demo showed wrong recorded answers without saying they were wrong | `compareWithExpected` tests; the demo shows the dataset's expected result next to each recording |
| `ruff` sorted imports differently from the repository root and from `backend/` | `known-first-party = ["groundedai"]`; CI lints `scripts/` and `tools/` with the project rules |
