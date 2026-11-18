# Deployment status

Last updated: 2026-11-18.

The first Pages build (run 39317852549) failed: `frontend/.env.browser`, which selects the
browser build, was excluded by `.gitignore`, so CI built the ordinary UI. It is now tracked,
and CI runs the demo's browser check on every pull request.

| Target | Status | Evidence |
|---|---|---|
| Online demo (GitHub Pages) | **Live: https://pouyapd.github.io/GroundedAI/** | GroundedAI runs in the visitor's browser (Pyodide), optional WebGPU model ([ONLINE_DEMO.md](ONLINE_DEMO.md)). Pages run 43761256978 on `323fde5` (2026-11-15, reliability round 3: version-aware disagreement, relevance, added details, the optional NLI meaning check, re-indexing): built, checked, deployed, verified; on the live URL `e2e_demo.py` 16 of 16, `--llm` 19 of 19, round-3 cases 9 of 9, and the meaning check measured ([RELIABILITY_V3.md](RELIABILITY_V3.md#deployment-check)). Earlier: Pages run 43465011957 on `3628f16` (2026-11-13, the current version): built, checked (16 of 16), deployed, deployed URL checked (16 of 16); then on the live URL `e2e_demo.py --llm` 19 of 19, the 22 live-test questions with answers identical to the run on `b3583ab` and all 64 quotes verbatim on the page shown, and a blocked model download handled with the new retry message. Before that, Pages run 43461690251 on `b3583ab`: built, checked in Chromium (16 of 16), deployed, and the deployed URL checked from GitHub's runner (16 of 16). The served Python package is byte-identical to `main`; on the live URL `e2e_demo.py` passed 16 of 16, and 19 of 19 with the in-page model ([RESEARCH_EVALUATION.md](RESEARCH_EVALUATION.md#live-demo-after-deployment-2026-11-13)). First deployed: run 39392655930 on `ef1dd43` (2026-10-20) |
| Local, SQLite (`groundedai serve`) | **Working** | API and built UI from one process on `127.0.0.1`; browser journey `scripts/e2e_browser.py` 24 of 24 with the extractive generator and with qwen2.5:1.5b through Ollama (2026-10-19) |
| Local, PostgreSQL + pgvector | **Working (tests)** | `pytest -m postgres` passed against PostgreSQL 16 + pgvector started by `scripts/dev_postgres.py` (no Docker) |
| Docker image | **Built and smoke-tested in CI** | Every pull request: image built, container started, `/ready`, `/api/v1/system/status` and the UI answered; last local run 2026-10-13 |
| Docker Compose | **Config validated in CI; last run locally 2026-10-13** | 2026-10-13: `docker compose up -d --build`, both services healthy, documents ingested, a page-cited answer, data kept after a restart |
| Local LLM through Ollama | **Working (2026-10-19)** | Ollama 0.14.2 with `qwen2.5:1.5b`: API, browser journey, development set and the 56-question evaluation |
| GitHub repository | **Public since 2026-10-19** | [`pouyapd/GroundedAI`](https://github.com/pouyapd/GroundedAI); PRs #1 to #15 merged into `main` with merge commits (no history rewritten) |
| GitHub Actions (CI) | **Passing** | 8 jobs, including the demo in a browser and the dependency audit: run 43761266292 on `main` at `0eb3eb5` (2026-11-15) |
| Cloud API (Render free web service + Neon free PostgreSQL) | **Live: https://groundedai-api-jyza.onrender.com** ([API docs](https://groundedai-api-jyza.onrender.com/docs)) | Deployed 2026-11-18 from `main` (`aabb946`) with the unchanged Dockerfile, region Frankfurt for both. The entrypoint migrated the Neon database (`/ready`: `postgresql`, migrations `current=0002 head=0002`, `vector(384)`, pgvector 0.8.7 on PostgreSQL 18.7). `scripts/check_cloud_deployment.py` on the live URL: 19 of 19 (no key and a wrong key refused with 401, a non-PDF named .pdf 415, a 5 MB+ upload 413, both sample PDFs uploaded and indexed, "14 euros" quoted from page 2, the wrong document refused, "12 items"); the same checks passed again after a redeploy, with the documents still in Neon. Before deploying, the 8 PostgreSQL tests passed against a temporary Neon database (deleted afterwards). Logs checked: no errors, no connection string, key or question text. Free-tier limits: [DEPLOYMENT.md section 5](DEPLOYMENT.md#5-cloud-api-render-free-tier--neon-postgresql) |

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
