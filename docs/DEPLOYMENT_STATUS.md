# Deployment status

Last updated: 2026-10-20.

The first Pages build (run 39317852549) failed: `frontend/.env.browser`, which selects the
browser build, was excluded by `.gitignore`, so CI built the ordinary UI. It is now tracked,
and CI runs the demo's browser check on every pull request.

| Target | Status | Evidence |
|---|---|---|
| Online demo (GitHub Pages) | **Live: https://pouyapd.github.io/GroundedAI/** | GroundedAI runs in the visitor's browser (Pyodide), optional WebGPU model ([ONLINE_DEMO.md](ONLINE_DEMO.md)). Pages run 39392655930 on `ef1dd43` (2026-10-20): built, checked in Chromium (16 of 16), deployed, and the deployed URL checked from GitHub's runner (16 of 16, start-up 13 s) |
| Local, SQLite (`groundedai serve`) | **Working** | API and built UI from one process on `127.0.0.1`; browser journey `scripts/e2e_browser.py` 24 of 24 with the extractive generator and with qwen2.5:1.5b through Ollama (2026-10-19) |
| Local, PostgreSQL + pgvector | **Working (tests)** | `pytest -m postgres` passed against PostgreSQL 16 + pgvector started by `scripts/dev_postgres.py` (no Docker) |
| Docker image | **Built and smoke-tested in CI** | Every pull request: image built, container started, `/ready`, `/api/v1/system/status` and the UI answered; last local run 2026-10-13 |
| Docker Compose | **Config validated in CI; last run locally 2026-10-13** | 2026-10-13: `docker compose up -d --build`, both services healthy, documents ingested, a page-cited answer, data kept after a restart |
| Local LLM through Ollama | **Working (2026-10-19)** | Ollama 0.14.2 with `qwen2.5:1.5b`: API, browser journey, development set and the 56-question evaluation |
| GitHub repository | **Public since 2026-10-19** | [`pouyapd/GroundedAI`](https://github.com/pouyapd/GroundedAI); PR #1 and #2 merged into `main` with merge commits (no history rewritten), then #3 (demo fix) |
| GitHub Actions (CI) | **Passing** | 8 jobs, including the demo in a browser: run 39392408635 on the last pull request before the merge into `main` |
| Backend hosting | **Not deployed** | No server is needed for the demo; a hosted backend would need a paid or account-based host ([DEPLOYMENT.md](DEPLOYMENT.md)) |

### CI results

Pull request [#1](https://github.com/pouyapd/GroundedAI/pull/1), CI run 38302045492 on commit
`0cf0182` (2026-10-12, `ubuntu-24.04` runners): **all 7 jobs passed**.

| Job | Result |
|---|---|
| Backend (Python 3.12) | ruff and strict mypy clean; 221 passed, 6 skipped; coverage 91% |
| Backend (Python 3.13) | ruff and strict mypy clean; 221 passed, 6 skipped |
| Backend on PostgreSQL + pgvector (`pgvector/pgvector:0.8.7-pg16-bookworm`) | 6 passed |
| Offline retrieval benchmark (smoke) | completed; dataset fingerprint `fa56ebd64e3b1eee`, the same as on Windows |
| Frontend | ESLint and `tsc -b` clean; 25 passed; `build` and `build:demo` succeed |
| Dependency audit | pip-audit (prod and ML locks): no known vulnerabilities; `npm audit --omit=dev`: 0 vulnerabilities |
| Docker image and Compose config | image built, container started, `/ready`, `/api/v1/system/status` and the UI answered; Compose config valid |

The first run (on commit `490de73`) failed in both backend jobs: strict mypy on Linux rejected
the Windows registry lookup used to name the CPU. The next commit fixed it (see
`docs/TESTING.md`); the other five jobs had already passed. The last pushed commit, `5cd6d3f`
(run 38302429733), also passed all 7 jobs.

The commits of 2026-10-13 (UI control fixes, answer-v2, scoring version 3, new reports) were
pushed on 2026-10-13; run 38414818764 on `c3cfbf1` passed all 7 jobs: backend on Python 3.12 and
3.13 (266 passed, 6 skipped, coverage 91%), PostgreSQL + pgvector (6 passed), offline benchmark
(fingerprint `fa56ebd64e3b1eee`), front end (53 tests, both builds), dependency audit (no known
vulnerabilities), Docker image and Compose config.

The 11 commits of 2026-10-14 (claim check, disagreement check, scoring version 4, re-checked
reports, UI) were pushed on 2026-10-14; run 38612773081 on `45b12a2` passed all 7 jobs: backend
on Python 3.12 and 3.13 (343 passed, 6 skipped, coverage 92%), PostgreSQL + pgvector (6
passed), offline benchmark (fingerprint `fa56ebd64e3b1eee`), front end (61 tests, both builds),
dependency audit (no known vulnerabilities), Docker image and Compose config.
