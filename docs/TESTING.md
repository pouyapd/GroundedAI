# Testing

## Layers

| Layer | Location | What it covers |
|---|---|---|
| Unit | `backend/tests/unit` | text utilities, upload validation, PDF/text parsing, chunking, embeddings, fusion, citations and support checks, context assembly, injection heuristics, extractive generator, evaluation metrics, configuration, rate limiter, metrics registry, dataset integrity, CLI |
| Integration | `backend/tests/integration` | Alembic migrations (upgrade, downgrade, schema matches models), ingestion lifecycle (failure, retry, re-ingest, delete without orphans, crash recovery), SQLite search and hybrid retrieval, reranking, QA service failure modes (timeout, busy, provider errors, invalid citations, hallucinated numbers, citation-marker-only replies), what reaches the model (documents only as data, conflict and injection notes, answer-v1 unchanged), LangChain retriever adapter, Ollama integration against a fake Ollama HTTP server (including model digest and server version), evaluation runners on the bundled dataset |
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
mypy --platform linux                   # on Windows: also check what CI (Linux) sees
pytest                                  # default suite: offline, no models, no API keys
pytest --cov                            # with coverage

# PostgreSQL tests without Docker (bundled Postgres from the pgserver wheel)
pip install pgserver
cd .. && python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres

cd frontend
npm run lint && npm run typecheck && npm test && npm run build
```

## Results recorded on 2026-10-13 (development laptop, Windows 11, Python 3.12.7, Node 24.19.0)

| Check | Result |
|---|---|
| `pytest --cov` (default suite, run from `backend/` as in CI) | 266 passed, 6 skipped (the 6 PostgreSQL tests, no database URL set); line coverage 91% |
| — of which `tests/unit` / `tests/integration` / `tests/api` / `tests/e2e` | 162 / 62 (+6 skipped) / 41 / 1 passed |
| `pytest -m postgres` via `scripts/dev_postgres.py` (PostgreSQL 16.2, pgvector 0.6.2) | 6 passed |
| `ruff check`, `ruff format --check` (`groundedai`, `tests`, `scripts`, `tools`, `evaluation/scripts`) | clean |
| `mypy` (strict, 76 files), also with `--platform linux` | no issues |
| `vitest` | 53 passed (6 files) |
| `eslint`, `tsc -b` | clean |
| `npm run build`, `npm run build:demo` | succeeded |
| `docker compose config` (with and without the `llm` profile) | valid |
| Docker Compose stack on Windows (Docker Desktop, engine 29.4.1, Compose 5.1.3) | built (about 16 min), both services healthy; 3 documents ingested (one a 4-page PDF); questions in lexical, dense and hybrid mode reached the API with the right mode; a page-cited, grounded answer; the conflict check fired; data and metrics kept after an API restart; `down -v` removed containers and volumes |
| Real Ollama inference (0.40.3, qwen2.5:1.5b) | through the API and the browser UI, and the 56-question evaluation (0 errors, 0 timeouts) |
| Dataset fingerprint and reproducibility | fingerprint `fa56ebd64e3b1eee` on Windows and in CI; the retrieval benchmark rerun and the extractive answer rerun match the published reports exactly |
| UI in a browser (live app and static demo) | every control checked at 390 px and 1280 px; no horizontal scrolling ([UI_CONTROL_AUDIT.md](UI_CONTROL_AUDIT.md)) |
| `pip-audit` (prod, dev and ML locks), `npm audit` | run on 2026-10-12: no known vulnerabilities (CPU-only torch not auditable); no dependency changed since |
| GitHub Actions | last run on `5cd6d3f` (2026-10-12): all 7 jobs passed. The commits after it are local and not pushed, so CI has **not** run on them |

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
| The dataset fingerprint included raw line endings, so a manifest written on Windows (CRLF) gave a different fingerprint from a clone (LF) | `test_fingerprint_does_not_depend_on_line_endings`, `test_fingerprint_changes_when_a_question_changes`; report and export writers now write LF |
| Static demo: Hybrid/Lexical/Dense and the passage selector were disabled without a visible state; the diagnostics box had no effect | `interactions.test.tsx`, `demo.test.ts`, `demo-data.test.ts` |
| Navigation depended on assigning `location.hash` before switching the view | `still switches pages when the hosting frame blocks history changes` |
| Switching tabs lost the Ask page's answer and settings | `keeps the answer and settings on the Ask page after visiting another tab` |
| A refusal that carried a citation ("The sources do not provide … [S5]") counted as an answer | `test_refusals_that_carry_a_citation_are_refusals` (and the hedged-answer counter-cases) |
| A passage or title could pose as another source with `[S3]` | `test_v2_passages_cannot_pose_as_another_source`, `test_v2_titles_cannot_open_prompt_blocks` |
| The tag neutraliser missed `<sources>` | `test_sources_wrapper_tag_is_neutralised` |
| `ruff` sorted imports differently from the repository root and from `backend/` | `known-first-party = ["groundedai"]`; CI lints `scripts/` and `tools/` with the project rules |
| Strict mypy passed on Windows but failed in CI on Linux: the Windows registry lookup (`winreg`) is untyped there | the lookup sits behind a literal `sys.platform == "win32"` check; `mypy --platform` win32, linux and darwin are clean |
