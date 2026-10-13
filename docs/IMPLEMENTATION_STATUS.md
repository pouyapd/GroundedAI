# Implementation status

Last updated: 2026-10-13 (end of the third work session). This file is the hand-over point for
the next session: what is done, what was verified, and what is left.

## Definition of done

| Item | Status |
|---|---|
| Backend implemented and importable | Done |
| Frontend implemented and buildable | Done (`npm run build`, `npm run build:demo`) |
| Ingestion works on supported fixtures | Done (TXT, MD, PDF with pages; failure, retry, delete, crash recovery tested) |
| Retrieval works on the test dataset | Done (SQLite and PostgreSQL) |
| Hybrid retrieval has a meaningful baseline comparison | Done: 7 configurations, bootstrap intervals, paired comparisons. Result: hybrid did **not** beat BM25 on this dataset |
| Grounded answers expose validated source references | Done |
| Insufficient evidence handled explicitly | Done (no-retrieval abstention, model abstention detection, UI states) |
| Fake provider supports deterministic offline testing | Done (extractive demo model + LangChain fake chat models + fake Ollama server) |
| Real local LLM integration optional and documented | Done: Ollama (verified with qwen2.5:1.5b on 2026-10-13) and transformers (Qwen2.5-0.5B) |
| Evaluation metrics implemented and tested | Done |
| Benchmark outputs reproducible | Done (deterministic metrics test; dataset hash check; configs versioned) |
| Database schema and migrations documented | Done (`docs/ARCHITECTURE.md`) |
| API validation and error handling | Done |
| Security controls tested where feasible | Done (`docs/SECURITY.md` lists tests) |
| Structured logs and operational metrics | Done |
| Unit and integration tests executed | Done (results below) |
| Frontend build executed | Done |
| Docker configuration validated | `docker compose config` passes; the full stack was run on Windows on 2026-10-13, and the image is built and smoke-tested in CI (see `docs/DEPLOYMENT_STATUS.md`) |
| CI workflow configured | Done; results of the first runs in `docs/DEPLOYMENT_STATUS.md` |
| README and technical documentation complete | Done |
| Model weights and secrets excluded from Git | Done (`.gitignore`, secret scan before committing) |
| Zero-mandatory-cost local workflow documented | Done |
| Online demo status documented | Done: private preview of the static demo; GitHub Pages waits for a public repository (`docs/DEPLOYMENT_STATUS.md`) |
| No fake numbers | All numbers come from report files or recorded command output |
| Remaining limitations listed | Done (README, `docs/EVALUATION.md`, `docs/SECURITY.md`) |

## Verified results (2026-10-13, Windows 11, Python 3.12.7, Node 24.19.0, mains power)

| Check | Result |
|---|---|
| Backend tests (`pytest --cov`, offline) | 266 passed, 6 skipped (PostgreSQL tests without a database URL); coverage 91%. Unit 162, integration 62, API 41, end-to-end 1 |
| PostgreSQL 16.2 + pgvector 0.6.2 (`scripts/dev_postgres.py`) | 6 passed |
| `ruff`, `ruff format`, `mypy --strict` (Windows and Linux targets) | clean (76 source files type-checked) |
| Front end: Vitest, ESLint, `tsc -b`, `build`, `build:demo` | 53 passed; clean; both builds succeed |
| Every UI control, live and in the static demo | see `docs/UI_CONTROL_AUDIT.md` |
| Real Ollama inference (0.40.3, qwen2.5:1.5b) | API, browser UI and 56-question evaluation; 0 errors, 0 timeouts |
| Docker Compose stack on Windows | built, healthy, ingestion, questions in all modes, restart, shutdown |
| Retrieval benchmark rerun (2026-10-13) | identical to the 2026-10-11 report in every metric and interval; hybrid did not beat BM25 |
| Answer evaluation, qwen2.5:1.5b, answer-v1 → answer-v2 (56 questions) | correct behaviour 0.830 → 0.906 (interval includes 0); answerable answered 36 → 43 of 43 and key-fact recall 0.767 → 0.942 (intervals exclude 0); correct refusals 9 → 7 of 10; conflicts 0 of 2; citation format followed 30% → 27%; p50 10 s |
| Answer evaluation, Qwen2.5-0.5B, answer-v1 → answer-v2 | correct behaviour 0.340 (published 0.377, re-scored) → 0.528 (interval excludes 0); 8 citation-marker-only replies |
| GitHub Actions | passed on `5cd6d3f` (2026-10-12); the 2026-10-13 commits are local and not pushed, so CI has not run on them |

Full tables: `docs/EVALUATION.md` and `docs/TESTING.md`.

## Remaining work

1. **Push the local commits** so CI runs on them (needs the owner's approval), then decide on
   merging pull request #1.
2. **Citations from small models.** qwen2.5:1.5b gets most facts right but cites them in the
   `[S#]` form in only about a quarter of its answers. Options to measure, each reported
   separately: mapping unambiguous variants such as `[Source S1]` to labels; a larger local
   model (estimate memory and latency first); constrained output.
3. **Conflicting sources.** No configuration handled either conflict question, and the
   figure-conflict check missed both. A better check needs more than word overlap.
4. **Dataset v2** with low-overlap paraphrases and more conflict and unanswerable questions, so
   differences between configurations can be measured with useful precision.
5. **Visibility.** The repository is private; making it public enables the CI badge for visitors
   and GitHub Pages for the static demo (`docs/DEPLOYMENT_STATUS.md`).
6. Optional: an NLI- or LLM-based support check, reported separately from the deterministic one.

## Next commands

```bash
# re-run every local check
cd backend && ruff check groundedai tests && ruff format --check groundedai tests && mypy && pytest
cd .. && python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres
cd frontend && npm run lint && npm run typecheck && npm test && npm run build && npm run build:demo

# refresh the static demo data after changing behaviour
python scripts/export_demo_data.py
```
