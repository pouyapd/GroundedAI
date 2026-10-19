# Implementation status

Last updated: 2026-10-18. This file is the hand-over point for the next session: what is done,
what was verified, and what is left.

## 2026-10-18: the PDF journey as a product

Local branch `feature/reliability-v1` (not pushed; pull request #1 is unchanged at `100f124`).

- Ask page: choose the document to search (`document_ids`); asking is disabled until a document
  is ready; an **Ask** button on each ready document; the answer says what was searched.
- Evidence quotes and conflict statements carry the page on which the quote starts, also inside
  chunks that span pages (migration 0002 adds `chunks.page_breaks`).
- Unreadable uploads are deleted at once; re-uploading retries; documents report
  `source_stored`, and re-ingest is disabled in the UI when the file is gone.
- Reproducible journey: sample PDFs in `examples/`, an API end-to-end test, a browser check
  (`scripts/e2e_browser.py`, 24 checks) and `scripts/journey_check.py` for a real model.
- A real local model ran on the journey (Qwen2.5-0.5B-Instruct via transformers, from the local
  cache, offline): it works end to end, but the model refused 3 of 5 answerable questions
  ([EVALUATION.md](EVALUATION.md#2026-10-18-the-pdf-journey-with-a-real-local-model)).
- Results of every check: [TESTING.md](TESTING.md#results-recorded-on-2026-10-18-same-laptop-on-battery-power).
- Earlier on 2026-10-18: answer checks version 2, evaluated on a frozen test split
  ([RELIABILITY_V2.md](RELIABILITY_V2.md)).

2026-10-19: Ollama 0.14.2 installed and `qwen2.5:1.5b` pulled; the development set, the
journey, the browser check and, once, the 56-question evaluation ran with it; two check fixes
(month-first dates, quotes) ([EVALUATION.md](EVALUATION.md#2026-10-19-qwen25-15b-through-ollama)).
Later the same day: claims-v2 and disagreement-v2 revised (support from sentences, equivalent
wordings, requirement conflicts, an unstated-conflict warning); measured on the development
split and real-model answers only, defaults unchanged
([RELIABILITY_V2.md](RELIABILITY_V2.md#revision-of-2026-10-19-development-split-only)).
Release candidate (2026-10-19): README rewritten, release checklist and changelog, security
review, a claim-review set with real model answers awaiting human labels, and two fixes found
on it (cross-organisation conflict reports; bracketed answers hidden as invalid output)
([RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)).

Still open: push and CI (owner's approval), a Docker image build with these changes, conflict
questions (no model run states a conflict), and a public demo (needs a public repository or a
host; see [DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md)).

## Definition of done

| Item | Status |
|---|---|
| Backend implemented and importable | Done |
| Frontend implemented and buildable | Done (`npm run build`, `npm run build:demo`) |
| Ingestion works on supported fixtures | Done (TXT, MD, PDF with pages; failure, retry, delete, crash recovery tested) |
| Retrieval works on the test dataset | Done (SQLite and PostgreSQL) |
| Hybrid retrieval has a meaningful baseline comparison | Done: 7 configurations, bootstrap intervals, paired comparisons. Result: hybrid did **not** beat BM25 on this dataset |
| Grounded answers expose validated source references | Done; since 2026-10-14 every statement is checked against every retrieved passage, with the exact supporting quote (lexical check) |
| Conflicting sources shown without resolving them | Done (2026-10-14): disagreement check with conflict / different-dates classification; 12 of 16 held-out disagreements found, 0 false alarms in 14 |
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

## Verified results (2026-10-14)

| Check | Result |
|---|---|
| Backend tests (`pytest --cov`, offline) | 343 passed, 6 skipped; coverage 92%. Unit 227, integration 70, API 44, end-to-end 2 |
| PostgreSQL 16.2 + pgvector 0.6.2 | 6 passed |
| `ruff`, `ruff format`, `mypy --strict` (Windows and Linux targets) | clean (79 source files) |
| Front end: Vitest, ESLint, `tsc -b`, `build`, `build:demo` | 61 passed; clean; both builds succeed |
| Live UI in a browser | claim check and disagreement panel at desktop and phone width, light and dark |
| Recorded outputs re-checked (3 model runs, no model run) | rebuild verified against every recorded result; qwen2.5:1.5b answer-v2: 116 of 118 statements found in a passage, 47 of 49 answers fully matched |
| Held-out disagreement cases (30) | 12 of 16 found, kind right in all 12, 0 of 14 false alarms |
| Retrieval at the application's defaults | BM25 Recall@5 1.000, MRR 0.985; no configuration better |
| Real language-model inference | not run (Ollama files deleted on 2026-10-13) |
| GitHub Actions | passed on `45b12a2` (run 38612773081, all 7 jobs) after the 2026-10-14 commits were pushed |

Details: [RELIABILITY_AUDIT.md](RELIABILITY_AUDIT.md).

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
| GitHub Actions | passed on `5cd6d3f` (2026-10-12); the 2026-10-13 commits were pushed later that day and passed on `c3cfbf1` |

Full tables: `docs/EVALUATION.md` and `docs/TESTING.md`.

## Remaining work

1. **Push the local commits** of 2026-10-14 so CI runs on them (needs the owner's approval), then
   decide on merging pull request #1.
2. **A model run with the new checks.** The disagreement notes changed the model input for four
   evaluation questions; only a new run shows whether a model then reports both sides. Needs a
   local model again (the Ollama files were deleted).
3. **Answers that are true but beside the question** (q041 price, q042 person, q044 code) pass
   the claim check. A check for "the requested item is missing" needs new held-out questions
   before it is designed.
4. **Known weaknesses of the checks** (stemming of "changed"/"change", synonyms, shared range
   ends, paraphrases marked partial): fix only together with new held-out cases, so the fixes are
   not tuned to results already seen ([RELIABILITY_AUDIT.md](RELIABILITY_AUDIT.md#known-issues)).
5. **Dataset v2** with low-overlap paraphrases and more conflict and unanswerable questions, so
   differences between retrieval configurations can be measured.
6. **Visibility** (owner's decision): see "Before making the repository public" in
   `docs/SECURITY.md`; the static demo preview has not been updated with the 2026-10-14 demo.

## Next commands

```bash
# re-run every local check
cd backend && ruff check groundedai tests && ruff format --check groundedai tests && mypy && pytest
cd .. && python scripts/dev_postgres.py -- python -m pytest backend/tests -m postgres
cd frontend && npm run lint && npm run typecheck && npm test && npm run build && npm run build:demo

# refresh the static demo data after changing behaviour
python scripts/export_demo_data.py

# re-check recorded model outputs and score the held-out disagreement cases (no model needed)
python scripts/revalidate_recorded_answers.py evaluation/reports/published/generation-ollama-groundedai-synthetic-v1-20261013T113618Z.json
python scripts/evaluate_conflict_cases.py
```
