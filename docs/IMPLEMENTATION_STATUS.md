# Implementation status

Last updated: 2026-10-12 (end of the second work session). This file is the hand-over point for
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
| Real local LLM integration optional and documented | Done: Ollama (tested against a fake server only) and transformers (measured with Qwen2.5-0.5B) |
| Evaluation metrics implemented and tested | Done |
| Benchmark outputs reproducible | Done (deterministic metrics test; dataset hash check; configs versioned) |
| Database schema and migrations documented | Done (`docs/ARCHITECTURE.md`) |
| API validation and error handling | Done |
| Security controls tested where feasible | Done (`docs/SECURITY.md` lists tests) |
| Structured logs and operational metrics | Done |
| Unit and integration tests executed | Done (results below) |
| Frontend build executed | Done |
| Docker configuration validated | `docker compose config` passes locally; the image is built and smoke-tested only in CI (see `docs/DEPLOYMENT_STATUS.md`) |
| CI workflow configured | Done; results of the first runs in `docs/DEPLOYMENT_STATUS.md` |
| README and technical documentation complete | Done |
| Model weights and secrets excluded from Git | Done (`.gitignore`, secret scan before committing) |
| Zero-mandatory-cost local workflow documented | Done |
| Online demo status documented | Done: private preview of the static demo; GitHub Pages waits for a public repository (`docs/DEPLOYMENT_STATUS.md`) |
| No fake numbers | All numbers come from report files or recorded command output |
| Remaining limitations listed | Done (README, `docs/EVALUATION.md`, `docs/SECURITY.md`) |

## Verified results (2026-10-12, Windows 11, Python 3.12.7, Node 24.19.0, mains power)

| Check | Result |
|---|---|
| Backend tests (`pytest --cov`, offline) | 219 passed, 6 skipped (PostgreSQL tests without a database URL); coverage 91% |
| PostgreSQL 16.2 + pgvector 0.6.2 (`scripts/dev_postgres.py`) | 6 passed |
| `ruff`, `ruff format`, `mypy --strict` | clean (75 source files type-checked) |
| Front end: Vitest, ESLint, `tsc -b`, `build`, `build:demo` | 25 passed; clean; both builds succeed |
| `pip-audit` (prod, dev, ML locks), `npm audit` | no known vulnerabilities (CPU torch wheel not auditable: not on PyPI) |
| `docker compose config` | valid |
| Browser checks (live app and static demo, 390 px and 1280 px) | no horizontal scrolling; chart text at its designed size |
| Retrieval benchmark (published report, 2026-10-11) | 7 configurations; hybrid RRF did not beat BM25 (paired 95% intervals include 0) |
| Answer evaluation, extractive baseline (2026-10-11) | behaviour accuracy 0.660; evidence coverage 0.663 |
| Answer evaluation, Qwen2.5-0.5B-Instruct on CPU (2026-10-12, 56 questions) | behaviour accuracy 0.377; 10 answered, 42 abstained, 4 invalid; p50 latency 32.3 s |

Full tables: `docs/EVALUATION.md` and `docs/TESTING.md`.

## Remaining work

1. **Decide on visibility.** The repository is private. Making it public enables the CI badge for
   visitors and GitHub Pages for the static demo (`docs/DEPLOYMENT_STATUS.md`).
2. **Enable GitHub Pages** once the repository is public.
3. **Evaluate an Ollama model** (`qwen2.5:1.5b`) with
   `groundedai eval generation --provider ollama --model qwen2.5:1.5b` once Ollama is installed,
   and publish the report.
4. **Dataset v2** with low-overlap paraphrases to test where dense retrieval helps.
5. Optional: an NLI- or LLM-based support check, reported separately from the deterministic one.

## Next commands

```bash
# re-run every local check
cd backend && ruff check groundedai tests && ruff format --check groundedai tests && mypy && pytest
cd .. && python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres
cd frontend && npm run lint && npm run typecheck && npm test && npm run build && npm run build:demo

# refresh the static demo data after changing behaviour
python scripts/export_demo_data.py
```
