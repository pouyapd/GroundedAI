# Deployment status

Last updated: 2026-10-12.

| Target | Status | Evidence |
|---|---|---|
| Local, SQLite (`groundedai serve`) | **Working** | API and built UI served from one process on `127.0.0.1`; sample corpus ingested with the CLI; questions answered; all four UI tabs checked in a browser at desktop and phone width, light and dark (screenshots in `docs/screenshots/`) |
| Local, PostgreSQL + pgvector | **Working (tests)** | `pytest -m postgres` passed against PostgreSQL 16.2 + pgvector 0.6.2 started by `scripts/dev_postgres.py` (no Docker) |
| Docker image | **Built in CI only** | Not built on the development machine (no image downloads there). The CI job `docker` builds it and smoke-tests `/ready`, `/api/v1/system/status` and the UI; result below |
| Docker Compose | **Configuration validated** | `docker compose --env-file .env.example config` passes (services `db`, `api`, optional `ollama`); not started |
| GitHub repository | **Private** | [`pouyapd/GroundedAI`](https://github.com/pouyapd/GroundedAI); the work is on branch `feature/groundedai-v0.1` with a pull request into `main` |
| GitHub Actions (CI) | **First run pending** at the time of this commit | Result below |
| Static demo build | **Built and checked locally** | `npm run build:demo` succeeds; at 390 px and 1280 px it replays the recorded answers with their expected results and shows the published reports, with no horizontal scrolling |
| Private online preview | **Published, private** | The static demo as a private claude.ai artifact (details below) |
| GitHub Pages static demo | **Not deployed** | `.github/workflows/pages.yml` is ready but skips its build while the repository is private (Pages on private repositories needs a paid plan) |
| Backend hosting | **Not deployed** | No free host is assumed for running models; see `docs/DEPLOYMENT.md` |

## Private online preview

The static demo build (`npm run build:demo`: recorded answers, documents list and the three
published reports; no backend and no live inference) is published as a private artifact on
claude.ai, version 1, 2026-10-12.

* **Access:** only the owner's signed-in claude.ai account. An anonymous visit was checked and
  shows "Sign in to view this page". It can be shared with named people, or by link, from the
  page's Share menu; it was not made public.
* **Checked through the service:** the page and its 11 supporting files (JavaScript, CSS and nine
  JSON data files) are listed with the expected sizes, and `demo/manifest.json` read back through
  the artifact URL matches the export of 2026-10-12 19:08 UTC.
* **Not checked:** the signed-in rendering itself, because the browser used for testing has no
  claude.ai session. The same files were checked in a local browser with `vite preview`.

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

Pending at the time of this commit.
