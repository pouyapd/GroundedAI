# Deployment status

Last updated: 2026-10-13.

| Target | Status | Evidence |
|---|---|---|
| Local, SQLite (`groundedai serve`) | **Working** | API and built UI served from one process on `127.0.0.1`; sample corpus ingested with the CLI; questions answered; all four UI tabs checked in a browser at desktop and phone width, light and dark (screenshots in `docs/screenshots/`) |
| Local, PostgreSQL + pgvector | **Working (tests)** | `pytest -m postgres` passed against PostgreSQL 16.2 + pgvector 0.6.2 started by `scripts/dev_postgres.py` (no Docker) |
| Docker image | **Built and run locally and in CI** | 2026-10-13 on the development machine (Docker Desktop on Windows) as part of the Compose check below; in CI on every push (results below) |
| Docker Compose | **Working (verified locally)** | 2026-10-13: `docker compose up -d --build` with a throwaway password; both services healthy; 3 documents ingested (one a PDF); questions in lexical, dense and hybrid mode; a page-cited answer; data kept after an API restart; `down -v` and the images removed afterwards. The optional `ollama` service (`--profile llm`) was not started; Ollama ran natively instead |
| GitHub repository | **Private** | [`pouyapd/GroundedAI`](https://github.com/pouyapd/GroundedAI); branch `feature/groundedai-v0.1`, pull request #1 into `main` (open, not merged). The commits of 2026-10-13 are local and **not pushed** |
| GitHub Actions (CI) | **Passing on `5cd6d3f`; not run on later commits** | All 7 jobs passed on 2026-10-12 (results below). CI runs when the local commits are pushed |
| Static demo build | **Built and checked locally** | `npm run build:demo`; every control checked at 390 px and 1280 px, also inside a sandboxed iframe that blocks history changes ([UI_CONTROL_AUDIT.md](UI_CONTROL_AUDIT.md)) |
| Local LLM through Ollama | **Working (verified locally)** | Ollama 0.40.3 (portable build, bound to 127.0.0.1) with `qwen2.5:1.5b` (digest `65ec06548149`): answers through the API and the UI, and the 56-question evaluation |
| Private online preview | **Published, private** | The static demo as a private claude.ai artifact (details below) |
| GitHub Pages static demo | **Not deployed** | `.github/workflows/pages.yml` is ready but skips its build while the repository is private (Pages on private repositories needs a paid plan) |
| Backend hosting | **Not deployed** | No free host is assumed for running models; see `docs/DEPLOYMENT.md` |

## Private online preview

The static demo build (`npm run build:demo`: 120 recorded answers (10 questions × 3 retrieval
modes × 4 passage counts), the documents list and the eight published reports; no backend and no
live inference) is published as a private artifact on claude.ai. Version 1: 2026-10-12.
Version 2: 2026-10-13, with the repaired controls and the new reports.

* **Access:** only the owner's signed-in claude.ai account. An anonymous visit was checked on
  2026-10-12 and showed "Sign in to view this page". It can be shared with named people, or by
  link, from the page's Share menu; it was not made public.
* **Checked through the service (version 2):** the page and its 16 supporting files (JavaScript,
  CSS and 14 JSON data files) are listed with the expected sizes, the two old bundles were
  removed, and `demo/reports/index.json` read back through the artifact URL lists all eight
  reports with their prompt versions.
* **Not checked:** the signed-in rendering itself, because the browser used for testing has no
  claude.ai session. The same build was checked in a local browser and inside a local sandboxed
  iframe that blocks history changes (the suspected cause of the navigation problem reported in
  the preview; see `docs/UI_CONTROL_AUDIT.md`).

The artifact URL is not listed in this public-facing file; it is in the owner's claude.ai gallery
under the name "GroundedAI".

## GitHub

The repository was created private: the code has not been reviewed for public release by its
owner, and making it public is the owner's decision. CI runs on pushes to `main` and on pull
requests. To make the repository public and deploy the static demo:

```bash
gh repo edit pouyapd/GroundedAI --visibility public --accept-visibility-change-consequences
```

Then Settings → Pages → Source: **GitHub Actions**, and run "Static demo (GitHub Pages)" from the
Actions tab.

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

**Not yet run:** the commits made on 2026-10-13 (UI control fixes, answer-v2, scoring version 3,
new reports) are on the local branch only. They pass the same checks locally (`docs/TESTING.md`),
but CI has not seen them; pushing the branch would run it.
