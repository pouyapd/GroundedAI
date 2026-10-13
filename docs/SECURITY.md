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
4. Every citation label is validated and every sentence is checked against its cited text, so an
   answer that drifts away from the sources is flagged.
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
* The lexical support check cannot detect wrong meaning (negation, swapped entities). Answers
  must be read together with their sources.
* PDF parsing relies on pypdf; a parser vulnerability would affect GroundedAI. Keep
  dependencies updated (`backend/requirements/README.md`).
