# Testing

## Layers

| Layer | Location | What it covers |
|---|---|---|
| Unit | `backend/tests/unit` | figures with units and number words, the claim check (citation forms, labels before text, wrong and missing sources, figure-only answers, instruction text), the disagreement check (conflicts, different dates and groups, agreement, incomplete evidence, a requirement and its denial), text utilities, upload validation, PDF/text parsing, chunking, embeddings, fusion, citations and support checks, context assembly, injection heuristics, extractive generator, evaluation metrics, configuration, rate limiter, metrics registry, dataset integrity, CLI |
| Integration | `backend/tests/integration` | Alembic migrations (upgrade, downgrade, schema matches models), ingestion lifecycle (failure, retry, re-ingest, delete without orphans, crash recovery), SQLite search and hybrid retrieval, reranking, QA service failure modes (timeout, busy, provider errors, invalid citations, hallucinated numbers, citation-marker-only replies), what reaches the model (documents only as data, conflict and injection notes, answer-v1 unchanged), LangChain retriever adapter, Ollama integration against a fake Ollama HTTP server (including model digest and server version), evaluation runners on the bundled dataset |
| PostgreSQL | `backend/tests/integration/test_postgres.py` | migrations with pgvector/tsvector/HNSW, all retrieval modes, hostile queries, cascade delete (runs when `GROUNDEDAI_TEST_POSTGRES_URL` is set) |
| API | `backend/tests/api` | every endpoint, status codes, error shape, validation, size limits, API key, rate limiting, CORS, security headers, request ids, OpenAPI paths, metrics, evaluation runs |
| End to end | `backend/tests/e2e` | upload PDF → ask → cited answer with page → abstention → delete → metrics, through HTTP only; three application starts on one data directory (documents, chunks, answers, claims and disagreements unchanged; re-ingest and delete kept) |
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
```

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
