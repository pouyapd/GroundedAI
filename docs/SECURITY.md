# Security

## Reporting a problem

Please open a private security advisory on the repository (GitHub → Security → Advisories)
rather than a public issue.

## Deployment rules

* **Default is local-only.** `groundedai serve` binds to `127.0.0.1`, and Docker Compose
  publishes ports on `127.0.0.1` only. In development mode no API key is required.
* **Before exposing the service to other machines:** set `GROUNDEDAI_ENV=production` and a
  random `GROUNDEDAI_API_KEY` of at least 24 characters (the app refuses to start in production
  without one), restrict `GROUNDEDAI_CORS_ORIGINS` to your UI origin (wildcards are refused in
  production), and put a TLS-terminating reverse proxy in front. Do not publish the API without
  authentication: uploads and questions use CPU, disk and model time.
* **Secrets** come only from environment variables or an untracked `.env` file.
  `.env.example` contains placeholders; `.gitignore` excludes `.env*`.

## Controls in place

| Area | Control | Tested in |
|---|---|---|
| Authentication | `X-API-Key` on all data endpoints when a key is set; constant-time comparison | `tests/api/test_security_api.py` |
| Abuse control | Token-bucket rate limits for questions, uploads and evaluation runs (`Retry-After` on 429); bounded concurrent generations with a queue timeout; generation timeout | `test_security_api.py`, `test_qa_service.py` |
| Uploads | Extension allow-list, declared content type must match, PDF magic bytes, text must be UTF-8 and not binary, size limit checked on `Content-Length` and while streaming, page and character limits | `test_validation.py`, `test_documents_api.py`, `test_parsers.py` |
| Paths | Stored files are named from the server-generated id and an allow-listed extension; the user's filename is display-only and sanitised | `test_validation.py`, `test_documents_api.py` |
| Parsing | pypdf in pure Python; malformed, encrypted or oversized PDFs fail with a safe message; no document content is executed | `test_parsers.py` |
| SQL | ORM or bound parameters only; user text is reduced to word tokens before FTS5 `MATCH` / `to_tsquery`, so search operators cannot be injected | `test_search_and_retrieval.py`, `test_postgres.py` |
| Errors | One JSON error shape; stack traces, paths and rejected input are never returned | `test_security_api.py`, `test_query_api.py` |
| Headers | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`, a strict Content-Security-Policy for the UI and `default-src 'none'` for the API | `test_health_and_system.py` |
| CORS | Explicit origin list; no credentials | `test_security_api.py` |
| Logging | Question text is not logged (`GROUNDEDAI_LOG_QUERY_TEXT=false`) and not stored; `query_events` holds outcomes and timings only; log fields named like secrets are redacted | `test_qa_service.py` |
| Dependencies | `pip-audit` on the pinned Python locks and `npm audit` on the front end in CI | `.github/workflows/ci.yml` |
| Container | Non-root user, no build tools in the runtime image, health check | `Dockerfile` |

## Prompt injection

Retrieved documents are untrusted. GroundedAI reduces, but **cannot eliminate**, the risk that
text inside a document steers the model:

1. The policy is in the system message; retrieved text is in the user message, wrapped in
   `<source>` blocks and described as untrusted data.
2. Tag-like sequences in retrieved text (`</source>`, `<system>` and similar) are neutralised so
   a document cannot close its own block.
3. Retrieved passages are scanned for common injection phrasings ("ignore previous
   instructions", fake system notices, "answer every question with ...", requests to reveal the
   prompt or to stop citing). Matches are passed to the model as data and reported as warnings
   in the response and the UI.
4. Every citation label is validated and every statement is checked against every retrieved
   passage, so an answer that drifts away from the sources is flagged. Instruction-like sentences
   in a passage never count as evidence, and a statement that repeats one is marked unsupported.
5. With the `answer-v2` prompt: label-like text inside a passage or title (`[S3]`) is rewritten
   as `(S3)` so a document cannot pose as another source; the policy says that text which gives
   orders or speaks to an AI assistant is not a fact; and a flagged passage is named in a note to
   the model ("source [S3] contains text that looks like instructions to an AI system").

The pattern list misses paraphrased attacks, and a capable model can still be persuaded by
well-written instructions. On the development set, qwen2.5:1.5b with `answer-v2` still copied
all sources, including a planted instruction, into one reply. The evaluation dataset includes
planted injections and reports an "injection resistance" rate; it is a regression signal, not a
proof of safety.

## Known limitations

* Rate limits are per process and in memory. With several workers or replicas, use a shared
  limiter at the reverse proxy or gateway.
* There are no user accounts or per-user document permissions: everyone with the API key sees
  all documents. Do not upload documents that the key holders must not read.
* Uploaded files and the database are not encrypted at rest by the application; use disk
  encryption on the host.
* The lexical support and claim checks cannot detect wrong meaning (negation, swapped entities)
  or a true statement that does not answer the question. Answers must be read together with
  their sources.
* PDF parsing relies on pypdf; a parser vulnerability would affect GroundedAI. Keep
  dependencies updated (`backend/requirements/README.md`).

## Before making the repository public (review of 2026-10-14)

* Git history (all 24 commits on 2026-10-14, all branches): no secrets, keys, tokens, local paths
  or deleted files; the largest file is the static demo's recordings (1.15 MB). Every commit
  carries the author's personal email address,
  which becomes public with the repository; rewriting history to change it is the owner's decision.
* Tracked files: the author's name and GitHub handle appear as authorship (LICENSE, `pyproject.toml`,
  README, banner images); `.env.example` holds placeholders only; test fixtures and datasets
  contain deliberately planted prompt-injection text, which is test data.
* GitHub Actions: no repository secrets are configured or used. Run logs mask the workflow token
  and contain no personal email addresses or local paths. Artifacts (retention 90 days): the offline
  benchmark report and Docker build records; the build records contain the GitHub event data
  (pull-request text, public account login and id) and masked request headers. They become
  visible with the repository.
* Dependencies: runtime Python packages are permissive (MIT, BSD, Apache-2.0, MPL-2.0, PSF) except
  psycopg (LGPL-3.0, used unmodified as a library; listed in `THIRD_PARTY_NOTICES.md`); the front
  end ships React, React DOM and Scheduler (MIT). `pgserver`, used only by the developer
  PostgreSQL script, has no licence metadata.

## Release-candidate review (2026-10-19)

Scope: every tracked file on `feature/reliability-v1`, the new and changed files of the release
candidate, and the content of every commit on all branches.

* Secrets: no API keys, tokens, private keys or passwords other than the documented placeholders
  (`change-me...` in `.env.example`, `ci-only-password` for the CI database service).
* Local paths and machine details: no absolute local paths or user names in tracked files or in
  any commit's content. Reports record the CPU model, RAM, operating system and power source on
  purpose (they explain the timings); they contain no host names or paths.
* Personal data: the author's name and GitHub handle appear as authorship; every commit carries
  the author's personal email address and the author's time zone offset (unchanged from the
  2026-10-14 review: owner's decision). Rechecked later on 2026-10-19: the same address is
  already public in commits of the owner's public repositories, and the GitHub profile shows no
  public email, so publishing this repository adds no new exposure of the address
  (`docs/RELEASE_CHECKLIST.md`, item 1).
* Private evaluation data: the reliability suite's test split, its documents, the writer notes,
  private audit files and sealed reports are git-ignored and were never committed (checked
  across all history). The committed manifest lists test documents by id only, with file hashes;
  `REVISIONS.jsonl` holds test case ids and hashes only; the committed test reports are
  aggregates without per-case rows. The development split, the synthetic evaluation set, the
  claim-review set and their reports are public by design (fictional organisations).
* Binary files: screenshots of the application with synthetic data, two generated sample PDFs,
  one synthetic corpus PDF, and the social-preview image. No databases, model weights, caches or
  uploaded user documents are tracked.
* Workflows: no repository secrets are used; jobs have read-only permissions except the Pages
  deployment, which runs only for a public repository.

## Online demo (GitHub Pages, 2026-10-19)

* **No server, no credentials.** The demo is static files; the application runs in the
  visitor's browser. No API key, token or secret is built into the site, and the Pages
  workflow uses only the job token GitHub issues for deployment.
* **Visitor data stays in the tab.** Uploaded files and questions are processed in a Web Worker
  and kept in memory; nothing is sent to this project. A reload or *Reset demo* deletes them.
* **Third parties the browser contacts:** jsDelivr (the pinned Pyodide release and its
  packages), and, only if the visitor chooses the language model, Hugging Face (model weights)
  and GitHub (the model's WebGPU library, through the MLC project's URLs). Their own privacy
  terms apply to those requests.
* **Pinned code.** The wheels served with the site are checked against SHA-256 hashes in
  `frontend/browser-runtime.lock.json` when the site is built; Pyodide is pinned to one
  release. The CDN files themselves are not hash-checked in the browser.
* **Headers.** GitHub Pages does not allow custom response headers, so the demo has no Content
  Security Policy. It renders documents as text, never as HTML.
