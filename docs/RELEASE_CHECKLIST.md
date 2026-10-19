# Release checklist (0.1.0)

Preparation and publication are separate steps. Everything under "Prepared" was done locally;
everything under "Owner decisions" and "Publish" needs the repository owner.

## Prepared (local branch `feature/reliability-v1`)

- [x] Product: PDF upload → extraction → indexing → retrieval → answer → quote and page →
      claim statuses; refusals, invalid output, conflicts and warnings shown in the UI.
- [x] Backend tests, lint, format and strict type checks pass (Windows; mypy also for Linux).
- [x] Front-end lint, type check, tests and production build pass.
- [x] Browser journey (24 checks) passes with the extractive generator and with qwen2.5:1.5b.
- [x] Real local-model smoke test (Ollama, qwen2.5:1.5b) from PDF upload to verified answer.
- [x] README for first-time users: setup, Ollama, configuration, checks, results, limitations,
      privacy and security; troubleshooting, contributing guide, changelog.
- [x] `.env.example` with placeholders only; `.gitignore` excludes data, models, caches,
      virtual environments, secrets and the private reliability split.
- [x] Public-release audit of tracked files and full history (see `docs/SECURITY.md`).
- [x] Claim-review set and tool for human labels (`evaluation/reviews/claims-review-v1`).

## Owner decisions before publication

1. **Author email in history.** Every commit carries the author's personal email address, and
   it becomes public with the repository. Keeping it needs no action; changing it means
   rewriting history (all branches, PR #1) before the first public push, which was not done.
2. **Release branch.** `main` holds only the initial commit. PR #1
   (`feature/groundedai-v0.1`, at `100f124`) is open; this branch adds commits on top of it.
   Choose: merge PR #1 and open a second PR from `feature/reliability-v1`, or open one PR from
   `feature/reliability-v1`. Nothing has been pushed.
3. **CI on the release commits.** CI last ran (7 of 7 green) on `100f124`. Push the branch and
   let CI run before merging; the Docker job is the only check of the current image.
4. **Default checks.** `claims-v1` stays the default. Switching to `claims-v2` should wait for
   human labels on `groundedai-claims-review-v1` (two labellers), then
   `scripts/claim_review.py score`.
5. **GitHub Actions artifacts** from earlier runs (offline benchmark report, Docker build
   records with event data) become visible with the repository; delete them first if unwanted.

## Publish (after the decisions above)

1. Push the branch, open or update the pull request, wait for CI to pass.
2. Merge into `main` (no force-push, no history rewrite unless decided in item 1).
3. Change the repository visibility to public.
4. Tag `v0.1.0` on the merge commit and publish a GitHub release with the CHANGELOG text.
5. Optional: enable GitHub Pages (Settings → Pages → GitHub Actions) for the static demo, and
   set the social preview image (`docs/assets/social-preview.png`).
6. Check the public page: README images render, badges resolve, links work.

## Not verified in this release candidate

- The Docker Compose stack locally (Docker Desktop was not running; last full run 2026-10-13).
- CI on any commit after `100f124`.
- Human-labelled accuracy of either claim-check version.
