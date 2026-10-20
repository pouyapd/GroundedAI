# Testing

## Layers

| Layer | Location | What it covers |
|---|---|---|
| Unit | `backend/tests/unit` | figures with units and number words, the claim check (citation forms, labels before text, wrong and missing sources, figure-only answers, instruction text), the disagreement check (conflicts, different dates and groups, agreement, incomplete evidence, a requirement and its denial), text utilities, upload validation, PDF/text parsing, chunking, embeddings, fusion, citations and support checks, context assembly, injection heuristics, extractive generator, evaluation metrics, configuration, rate limiter, metrics registry, dataset integrity, CLI |
| Integration | `backend/tests/integration` | Alembic migrations (upgrade, downgrade, schema matches models), ingestion lifecycle (failure, retry, re-ingest, delete without orphans, crash recovery), SQLite search and hybrid retrieval, reranking, QA service failure modes (timeout, busy, provider errors, invalid citations, hallucinated numbers, citation-marker-only replies), what reaches the model (documents only as data, conflict and injection notes, answer-v1 unchanged), LangChain retriever adapter, Ollama integration against a fake Ollama HTTP server (including model digest and server version), evaluation runners on the bundled dataset |
| PostgreSQL | `backend/tests/integration/test_postgres.py` | migrations with pgvector/tsvector/HNSW, all retrieval modes, hostile queries, cascade delete (runs when `GROUNDEDAI_TEST_POSTGRES_URL` is set) |
| API | `backend/tests/api` | every endpoint, status codes, error shape, validation, size limits, API key, rate limiting, CORS, security headers, request ids, OpenAPI paths, metrics, evaluation runs |
| End to end | `backend/tests/e2e` | upload PDF → ask → cited answer with page → abstention → delete → metrics, through HTTP only; the sample-PDF journey (extracted text, a quote on page 2 of a one-chunk 3-page PDF, a question the selected document cannot answer, isolation between two documents, deletion); three application starts on one data directory (documents, chunks, answers, claims and disagreements unchanged; re-ingest and delete kept) |
| Browser | `scripts/e2e_browser.py` (not in CI) | the same journey in Chromium against a real local server and the built UI: damaged, text-less and unsupported files, upload, Ask from the documents table, quote and page, citation highlight, not-in-document question, switching documents, repeated questions, deletion, console errors (24 checks) |
| Front end | `frontend/src/test` | claim check and disagreement panel (quotes, statuses, withheld answer, no repeated warnings), scoring-version-4 tiles and re-checked reports, the hashing-embedder note, citation rendering, answer view states, abstention, LLM vs demo labels, offline backend, model unavailable, question flow, empty documents, evaluation report rendering (invalid outputs, latency, category names), static demo replay and expected-result comparison, upload validation, formatting |
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

# browser journey against a real server (needs: pip install playwright && playwright install chromium)
cd .. && python scripts/e2e_browser.py --screenshots /tmp/shots
```

## Online demo and release (2026-10-19, evening; mains power)

| Check | Kind | Result |
|---|---|---|
| `pytest` (default suite) | unit, integration, API, e2e | 487 passed, 7 skipped (PostgreSQL tests, run separately) |
| `tests/unit/test_browser_runtime.py` | the demo's runtime under CPython: extractive answer with quote and page, the two-step page-model flow, an invented figure not found, refusal, changed passages refused, status | 10 passed |
| Pyodide rehearsal in Node (scratch script, not committed) | the real FastAPI app in Pyodide 314.0.7 with only the pinned wheels: start-up, PDF uploads, a 1.5 MB upload through the disk-spooled path, extractive and page-model answers | worked; ready after 11–12 s |
| `vitest` | front end, including `BrowserApi` (two-step flow, timings, errors, reset) | 77 passed; three repeated runs without unhandled errors |
| `scripts/e2e_demo.py` | the demo build in Chromium and in Chrome, served under `/GroundedAI/` | 16 of 16 (start-up 25–47 s with a cold cache; about 26 MB downloaded) |
| Heading fix (`split_sentences`) | reliability dev split (v1, v2), 30 held-out conflict cases, replays of recorded qwen2.5:1.5b answers (dev, 56-question; v1, v2) | every measure unchanged |
| CI on pull request #2 | GitHub Actions | first run: backend lint (script line lengths) and a front-end test race failed; both fixed; run 39316142259 on `c95d612`: 7 of 7 jobs passed, including the Docker image build and smoke test |
| First Pages build (run 39317852549) | GitHub Actions | failed: the page never rendered on the Linux runner. Cause: `frontend/.env.browser` was excluded by `.gitignore`, so CI built the ordinary UI. Fixed in PR #3, which also adds the demo check to CI |
| CI on `main` after PR #3 | GitHub Actions | run 39392655791 on `ef1dd43`: 8 of 8 jobs, including the demo in a browser (16 of 16) |
| Pages workflow after PR #3 | GitHub Actions | run 39392655930: built and checked (16 of 16), deployed, then the deployed URL checked from GitHub's runner (16 of 16, start-up 13 s) |

The in-page model (WebLLM, Qwen2.5-1.5B-Instruct) needs WebGPU. In Playwright's own Chromium on
Windows the adapter was found but the device failed (`dxil.dll` missing; Chrome ships it); the
demo then showed "The language model could not be loaded … The extractive generator still
works", as intended. The model check therefore uses the installed Chrome (`--channel chrome`).

## Release preparation checks (2026-10-19, later; commit `83c5668` plus this round's changes; mains power)

| Check | Kind | Result |
|---|---|---|
| `pytest` (default suite) | unit, integration, API, e2e; scripted fake models | 475 passed, 7 skipped (PostgreSQL tests, run separately below) |
| `scripts/dev_postgres.py -- pytest -m postgres` | PostgreSQL 16 + pgvector, bundled local server | 7 passed |
| `ruff`, `ruff format --check` (backend and scripts), `mypy` strict (Windows and `--platform linux`) | static | clean |
| `eslint`, `tsc -b`, `vitest`, `npm run build:demo`, `npm run build` | front end | clean; 70 passed; both builds succeed |
| Workflow files | YAML parse (no `actionlint` available) | both parse; `contents: read`; Pages jobs only add `pages`/`id-token` write |
| `scripts/reliability_suite.py score --split dev` (v1, v2) | checks, model-written labels | unchanged: v1 A 37, B 31, C 21, D 35; v2 A 47, B 34, C 41, D 47 |
| `scripts/evaluate_conflict_cases.py` | 30 held-out conflict cases, aggregate only | 12 of 16 found, 0 of 14 false alarms (unchanged) |
| `groundedai eval generation` on the development set | **real model** (qwen2.5:1.5b) | 16 of 19 answerable correct, 2 false refusals, 5 of 6 refusals; outputs identical to the earlier run ([EVALUATION.md](EVALUATION.md#2026-10-19-release-preparation-each-behaviour-separately)) |
| `scripts/journey_check.py` with Ollama | **real model**, API, PDF upload | 2 correct and verified, 3 correct but not fully verified, 2 correct refusals; 2–7 s per answer |
| `scripts/e2e_browser.py` | browser, real server | 24 of 24 with the default generator; 24 of 24 with qwen2.5:1.5b on the first run; no test folder left behind |
| `review.html` in a browser (two browser profiles) | tooling, placeholder labels | restored-label notice, *Clear all labels* and the download work; `score` reads the files |
| Docker image and Compose | – | **not run**: Docker Desktop is not running |
| CI | – | **not run** on any commit after `100f124`: nothing pushed |

Fixed in this round, each with a test: `claim_review.py score` scored only the first labeller's
file (now: only statements all labellers agree on, disputed ones listed, optional adjudication
file); the review page could show one labeller's stored labels to the next labeller in the
same browser; `e2e_browser.py` silently left its test folder (uploaded files and database) in
the temporary directory on Windows, because the server's database is still locked for a moment
after the server is stopped.

## Release-candidate checks (2026-10-19, commit `cca6670` plus the e2e timeout change; laptop on battery)

| Check | Kind | Result |
|---|---|---|
| `pytest` (default suite) | unit, integration, API, e2e; scripted fake models | 467 passed, 7 skipped (PostgreSQL tests, run separately below) |
| `scripts/dev_postgres.py -- pytest -m postgres` | PostgreSQL 16 + pgvector, bundled local server | 7 passed |
| `ruff`, `ruff format --check` (backend and scripts), `mypy` strict (Windows and `--platform linux`) | static | clean |
| `eslint`, `tsc -b`, `vitest`, `npm run build`, `npm run build:demo` | front end | clean; 70 passed; both builds succeed |
| Workflow files | YAML parse only (no `actionlint` available) | both parse; jobs unchanged |
| `scripts/reliability_suite.py score --split dev` (v1, v2) | checks, model-written labels | unchanged by the release fixes: v1 A 37, B 31, C 21, D 35; v2 A 47, B 34, C 41, D 47 |
| `scripts/journey_check.py` with Ollama | **real model** (qwen2.5:1.5b), API | 2 correct and verified, 3 correct but not fully verified, 2 correct refusals; 2–6 s per answer |
| `scripts/e2e_browser.py` | browser, real server | 24 of 24 with the default generator; 24 of 24 with qwen2.5:1.5b (see note) |
| `scripts/claim_review.py collect` | **real model**, new review set | 30 questions, 56 statements exported; labels not collected |
| Docker image and Compose | – | **not run**: Docker Desktop is not running (CI builds and smoke-tests the image) |
| CI | – | **not run** on these commits: nothing pushed |

Note on the browser check with Ollama: the first two runs failed, both while the model was
loading on battery power. One timed out after 20 s waiting for the first answer (the script now
allows 300 s for an answer), and one timed out at an upload step that does not use the model.
The third run, with the model loaded, passed 24 of 24. Running `npm run build:demo` after
`npm run build` replaces the UI the check serves; build the normal UI before running it.

## Results recorded after the claims-v2 revision (2026-10-19, mains power, commit `7082970`)

Changes: claims-v2 judges claims on supporting sentences, reads equivalent wordings, names,
ordinals and unresolved labels, and warns when an answer does not state a conflict;
disagreement-v2 reads requirement wording and equivalent words (both opt-in; defaults unchanged).

| Check | Kind | Result |
|---|---|---|
| `pytest` (default suite) | unit, integration, API, e2e; scripted fake models where a model is needed | 464 passed, 7 skipped (PostgreSQL); 33 new tests, invented examples |
| `ruff`, `ruff format --check`, `mypy` (strict, also `--platform linux`), scripts lint | static | clean |
| `vitest`, `eslint`, `tsc -b`, `npm run build` | front end (unchanged) | 70 passed; clean; built |
| `scripts/reliability_suite.py validate`; dev split with v1 and v2 | dataset, checks (labels written by models) | 0 errors; v1 identical in every case; v2 in [RELIABILITY_V2.md](RELIABILITY_V2.md#revision-of-2026-10-19-development-split-only) |
| Development set (25, claims-v2) and journey (7, both versions) | **real model**: qwen2.5:1.5b, Ollama 0.14.2 | outputs identical to the earlier runs; results in [EVALUATION.md](EVALUATION.md#2026-10-19-later-the-revised-claims-v2-on-real-model-answers) |
| 56-question set, once | recorded real-model outputs re-checked with both versions (no model call) | same section |
| `scripts/e2e_browser.py` | browser, real server | 24 of 24 with the default checks, with claims-v2, and with claims-v2 and qwen2.5:1.5b (the script now accepts the claims-v2 label "Every statement found") |
| Docker | – | not run: Docker Desktop is not running |

## Results recorded with qwen2.5:1.5b (2026-10-19, mains power)

Changes: month-first dates are read as dates; evidence quotes ignore document-title words.

| Check | Kind | Result |
|---|---|---|
| `pytest` (default suite) | unit, integration, API, e2e; scripted fake models where a model is needed | 431 passed, 7 skipped (PostgreSQL) |
| `ruff`, `ruff format --check`, `mypy` (strict, also `--platform linux`) | static | clean |
| `vitest`, `eslint`, `tsc -b`, `npm run build` | front end | 70 passed; clean; built |
| `scripts/reliability_suite.py validate`; dev split scored with v1 and v2 before and after each fix | dataset, checks | 0 errors; results unchanged in every category |
| Development set (25), journey (7), 56-question set (once) | **real model**: qwen2.5:1.5b, Ollama 0.14.2 | no errors or timeouts; results in [EVALUATION.md](EVALUATION.md#2026-10-19-qwen2515b-through-ollama) |
| `scripts/e2e_browser.py` | browser, real server | 24 of 24 with the default generator and 24 of 24 with qwen2.5:1.5b |
| Docker | – | not run: Docker Desktop is not running |

## Results recorded after the answer-quality diagnosis (2026-10-18, on battery power)

Change: claims-v2 reports replies made only of bracketed fragments as invalid output.

| Check | Result |
|---|---|
| `pytest` (default suite) | 428 passed, 7 skipped (PostgreSQL) |
| `ruff check`, `ruff format --check`, `mypy` (strict, also `--platform linux`) | clean |
| Real model: dev set (25 questions) with claims-v1 and claims-v2, journey with claims-v2 | ran; Qwen2.5-0.5B-Instruct through transformers, offline; results in [EVALUATION.md](EVALUATION.md#2026-10-18-the-pdf-journey-with-a-real-local-model) |
| Front end | not rerun: no front-end change |

## Results recorded on 2026-10-18 (same laptop, on battery power)

Changes: document selection on the Ask page, the page of each evidence quote (migration 0002),
discarding unreadable uploads, the sample-PDF journey tests.

| Check | Result |
|---|---|
| `pytest --cov` (default suite, run from `backend/`) | 419 passed, 7 skipped (the PostgreSQL tests, no database URL set); line coverage 92% |
| — of which `tests/unit` / `tests/integration` / `tests/api` / `tests/e2e` | 289 / 80 (+7 skipped) / 47 / 3 passed |
| `pytest -m postgres` via `scripts/dev_postgres.py` | 7 passed |
| `ruff check`, `ruff format --check` (`groundedai`, `tests`, `scripts`, `tools`, `evaluation/scripts`) | clean |
| `mypy` (strict, 83 files), also with `--platform linux` | no issues |
| `vitest` | 70 passed (7 files) |
| `eslint`, `tsc -b`, `npm run build`, `npm run build:demo` | clean, both builds succeeded |
| `scripts/e2e_browser.py` (Chromium 1208, default providers) | 24 of 24 checks passed |
| `scripts/journey_check.py` with Qwen2.5-0.5B-Instruct (transformers, CPU, offline) | ran without errors; results in [EVALUATION.md](EVALUATION.md#2026-10-18-the-pdf-journey-with-a-real-local-model) |
| Migration 0002 on a database with documents from before | upgraded at start-up; old chunks answer with `page: null` for multi-page quotes until re-ingested, then with the page (checked by hand and in `test_migrations.py`) |
| `scripts/reliability_suite.py validate` | 660 cases, 0 errors, 0 warnings (frozen test split unchanged; it was not scored again) |
| `docker compose config` | valid |
| Not run | the Docker image build and Compose stack (Docker Desktop not running; a build downloads packages); GitHub Actions (nothing pushed); dependency audits (need the network) |

## Results recorded after the PR review fixes (2026-10-14, same laptop)

Fixes: the conflict explanation for a requirement against its denial, and the recording of every
setting that can change an answer evaluation ([RELIABILITY_AUDIT.md](RELIABILITY_AUDIT.md),
"Fixes after the PR review").

| Check | Result |
|---|---|
| `pytest --cov` (default suite, run from `backend/`) | 352 passed, 6 skipped (the PostgreSQL tests, no database URL set); line coverage 93% |
| — of which `tests/unit` / `tests/integration` / `tests/api` / `tests/e2e` | 233 / 73 (+6 skipped) / 44 / 2 passed |
| `pytest -m postgres` via `scripts/dev_postgres.py` (PostgreSQL 16.2, pgvector 0.6.2) | 6 passed |
| `ruff check`, `ruff format --check` (`groundedai`, `tests`, `scripts`, `tools`, `evaluation/scripts`) | clean |
| `mypy` (strict, 79 files), also with `--platform linux` | no issues |
| `vitest` | 62 passed (7 files) |
| `eslint`, `tsc -b`, `npm run build`, `npm run build:demo` | clean, both builds succeeded |
| New regression tests run against the code before the fix | the conflict-text tests (backend and UI) failed with the reported text ("S1 says ,", "Value: "), as intended |
| Disagreement check, old and new code on the 30 held-out inputs | identical pairs and kinds; one explanation changed (the broken one) |
| `scripts/revalidate_recorded_answers.py` on the qwen2.5:1.5b answer-v2 run of 2026-10-13 | rebuild verified; every per-question row and the summary identical to the published replay; the new report records the pipeline settings (defaults) and no `GROUNDEDAI_*` variables |
| Not rerun for these fixes | browser check (the change is the text of the disagreement panel, covered by tests); Docker Compose; real language-model inference (no model available) |

## Results recorded on 2026-10-14 (development laptop, Windows 11, Python 3.12.7, Node 24.19.0)

| Check | Result |
|---|---|
| `pytest --cov` (default suite, run from `backend/`) | 343 passed, 6 skipped (the PostgreSQL tests, no database URL set); line coverage 92% |
| — of which `tests/unit` / `tests/integration` / `tests/api` / `tests/e2e` | 227 / 70 (+6 skipped) / 44 / 2 passed |
| `pytest -m postgres` via `scripts/dev_postgres.py` (PostgreSQL 16.2, pgvector 0.6.2) | 6 passed |
| `ruff check`, `ruff format --check` (`groundedai`, `tests`, `scripts`, `tools`, `evaluation/scripts`) | clean |
| `mypy` (strict, 79 files), also with `--platform linux` | no issues |
| `vitest` | 61 passed (7 files) |
| `eslint`, `tsc -b`, `npm run build`, `npm run build:demo` | clean, both builds succeeded |
| Live application in a browser (extractive generator, bge-small) | claim check and disagreement panel at 1100 px and 390 px, light and dark; no horizontal scrolling |
| Real language-model inference | **not run**: the portable Ollama build and its model were deleted on 2026-10-13. The claim and disagreement checks were measured on recorded outputs instead ([RELIABILITY_AUDIT.md](RELIABILITY_AUDIT.md)) |
| Docker Compose | **not rerun** (verified on 2026-10-13; the Dockerfile and Compose file did not change) |
| GitHub Actions | at the time: last run on `c3cfbf1` (2026-10-13, run 38414818764). The 11 commits were pushed later on 2026-10-14 and CI run 38612773081 on `45b12a2` passed all 7 jobs |

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
| GitHub Actions | at the time: last run on `5cd6d3f` (2026-10-12). The commits were pushed on 2026-10-13 and CI run 38414818764 on `c3cfbf1` passed all 7 jobs |

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
| `[Source 1]`, `(S1)` and bracketed titles were not read as citations, so correct answers showed as "not grounded" | `test_variant_reference_forms_are_understood_and_counted_as_variants` |
| Labels written before the text (`[S1]: …`) were credited to the previous sentence | `test_labels_written_before_the_text_apply_to_that_text`, `test_an_answer_that_opens_with_a_label_uses_labels_before_text` |
| "SX-5" and the "1" of `[Source 1]` counted as invented figures | `test_identifiers_are_not_figures`, `test_codes_are_checked_as_words_not_as_figures` |
| "€0.30" was read as the time 00:30; "20:00." at a sentence end was not read as a time | `test_decimal_money_is_not_a_time_but_a_closing_time_is` |
| A Markdown heading after a sentence merged two sentences, so a verbatim quote looked partly supported | `test_a_markdown_heading_after_a_sentence_starts_a_new_claim` |
| Words pooled from two documents "supported" a claim (one organisation's name, another's fact) | `test_words_are_never_pooled_across_passages` |
| A figure-only answer ("€260") matched the wrong sentence of a passage | `test_figure_only_answer_is_read_against_the_sentence_that_matches_the_question` |
| Planted instruction text counted as evidence for a claim | `test_instruction_text_in_a_passage_is_not_evidence` |
| Years, editions and unrelated figures made sentences "conflict" | `test_years_in_headings_are_not_values`, `test_only_values_of_the_asked_kind_are_compared`, `test_statements_that_do_not_match_the_question_are_ignored` |
| The UI repeated a disagreement's explanation as a warning | `does not repeat a disagreement's explanation in the warning list` |
