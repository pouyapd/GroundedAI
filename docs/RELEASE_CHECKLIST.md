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
- [x] Claim-review set and tool for human labels (`evaluation/reviews/claims-review-v1`),
      with a guide for two independent labellers; the scoring path was tested with placeholder
      labels only.
- [x] Final checks on mains power (2026-10-19): development set with qwen2.5:1.5b, each
      behaviour reported separately (`docs/EVALUATION.md`); browser check with qwen2.5:1.5b
      passed on the first run (`docs/TESTING.md`).

## Decisions taken for the release (2026-10-19)

1. **Author email in history.** Kept. Every commit carries the owner's personal address
   (`p***@gmail.com`); the same address is already public in the owner's other public
   repositories, so publishing adds no new exposure. No history was rewritten.
2. **Release branch.** PR #1 (`feature/groundedai-v0.1`) is merged first, then PR #2
   (`feature/reliability-v1`, which contains it) after CI passes. Merge commits, no
   force-push.
3. **CI.** Run on the release branch before merging (7 of 7 on `c95d612`).
4. **Default checks.** `claims-v1` and `disagreement-v1` stay the defaults.
5. **Old GitHub Actions artifacts** (offline benchmark report, Docker build records with the
   pull-request text, the public account login and masked headers) are kept; they contain no
   secrets or private data.
6. **GitHub Pages** serves the live demo, built by the Pages workflow from `main`.

## Publish

1. Push the branch, open the pull request, wait for CI to pass.
2. Merge PR #1, then the release pull request, into `main`.
3. Change the repository visibility to public (Pages on a free plan needs it).
4. Enable Pages with GitHub Actions as the source; run the Pages workflow; it checks the built
   site and the deployed URL in a browser.
5. Tag `v0.1.0` on the merge commit and publish a GitHub release with the CHANGELOG text and the
   demo URL.
6. Check the public page: README images render, badges resolve, links work, the demo opens.

## Not verified in this release

- The Docker Compose stack locally since 2026-10-13 (CI builds and smoke-tests the image and
  validates the Compose file).
- Human-labelled accuracy of either claim-check version.
- Any model larger than qwen2.5:1.5b; no evaluation of the demo's in-browser model build.
