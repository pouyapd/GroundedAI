# HTTP API

Base path: `/api/v1`. Interactive documentation (OpenAPI/Swagger) is served at `/docs` and the
schema at `/openapi.json` when the server runs.

## Conventions

* JSON in and out (uploads use `multipart/form-data`).
* Every response has an `X-Request-ID` header. Send your own `X-Request-ID` (letters, digits,
  `.`, `_`, `-`, up to 64 characters) to correlate client and server logs.
* Errors always have the same shape and never contain stack traces or file paths:

  ```json
  {"error": {"code": "model_unavailable", "message": "Cannot reach Ollama at http://localhost:11434. ...", "details": {}, "request_id": "4f40b9fc..."}}
  ```

* When `GROUNDEDAI_API_KEY` is set, all `/api/v1` endpoints except `health`, `ready` and
  `system/status` need the header `X-API-Key: <key>`; otherwise they return `401 unauthorized`.
* `POST /query`, `POST /retrieve`, uploads/re-ingestion and evaluation runs are rate-limited per
  API key (or per client address). Exceeding the limit returns `429 rate_limited` with a
  `Retry-After` header.

## Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health` (also `/api/v1/health`) | Liveness | 200 |
| GET | `/ready` (also `/api/v1/ready`) | Database, migrations, embedding provider (and vector dimension on PostgreSQL) | 200, or 503 with failing checks |
| GET | `/api/v1/system/status` | Configured providers, availability, demo-mode flag, limits | 200 |
| POST | `/api/v1/documents` | Upload `.pdf`, `.txt` or `.md` (field `file`, optional `title`) | 202 (processing continues in the background) |
| GET | `/api/v1/documents?offset&limit&status` | List documents, newest first | 200 |
| GET | `/api/v1/documents/{id}` | Metadata and ingestion status | 200 |
| GET | `/api/v1/documents/{id}/chunks?offset&limit` | Indexed chunks with page ranges | 200 |
| POST | `/api/v1/documents/{id}/reingest` | Process again (after a failure or configuration change); 409 with `details.reason = "source_missing"` when the file is no longer stored | 202 |
| DELETE | `/api/v1/documents/{id}` | Delete document, chunks and stored file | 204 |
| POST | `/api/v1/query` | Grounded answer with sources, citation check and timings | 200 |
| POST | `/api/v1/retrieve` | Retrieval only, with component scores and timings | 200 |
| GET | `/api/v1/evaluations/reports` | Published benchmark reports | 200 |
| GET | `/api/v1/evaluations/reports/{id}` | One full report | 200 |
| POST | `/api/v1/evaluations/runs` | Evaluate this instance's retrieval settings on the bundled dataset (isolated index) | 202 |
| GET | `/api/v1/evaluations/runs`, `/runs/{id}` | Run history and results | 200 |
| GET | `/api/v1/metrics?window_hours=24` | Operational metrics for the dashboard | 200 |
| GET | `/api/v1/metrics/prometheus` | Process metrics, Prometheus text format | 200 |

### Status codes used

| Code | `error.code` values |
|---|---|
| 401 | `unauthorized` |
| 404 | `not_found` |
| 409 | `duplicate_document` (details include the existing `document_id`; a document whose processing failed is replaced instead), `conflict` |
| 413 | `file_too_large`, `payload_too_large` |
| 415 | `unsupported_file_type` |
| 422 | `invalid_input` (validation errors list `loc`, `msg`, `type` but never echo the input), `empty_document`, `document_parse_failed` |
| 429 | `rate_limited` |
| 502 | `generation_failed` |
| 503 | `model_unavailable`, `server_busy` |
| 504 | `model_timeout` |

## Examples

Upload and check status:

```bash
curl -s -F "file=@evaluation/datasets/groundedai-synthetic-v1/corpus/brindlewick-water-faq.txt" \
     http://127.0.0.1:8000/api/v1/documents
# {"id": "doc_bcee98dbd52e4be372ae", "status": "pending", ...}

curl -s http://127.0.0.1:8000/api/v1/documents/doc_bcee98dbd52e4be372ae
# {"id": "doc_bcee98dbd52e4be372ae", "status": "ready", "chunk_count": 7, ...}
```

Stored files: the original file is kept while its document exists (re-ingestion needs it) and is
deleted with the document. A file that cannot be read at all (damaged, password-protected, no
extractable text) is deleted as soon as that is known; the document stays `failed` with the
reason, `source_stored` is `false`, and uploading the file again retries it. A file over a
settings limit (pages, characters) is kept, since a configuration change can fix that.

Ask a question (add `"document_ids": ["doc_…"]` to search only those documents):

```bash
curl -s http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "During a boil water notice, how long should tap water be boiled?", "include_diagnostics": true}'
```

Response (recorded from a local run with bge-small embeddings and the extractive demo
generator; long fields shortened):

```json
{
  "status": "answered",
  "answer": "During a boil water notice, bring tap water to a rolling boil for 1 minute before drinking it or using it for cooking [S1]. What should I do during a boil water notice [S1]?",
  "grounding": "grounded",
  "sources": [
    {
      "label": "S1",
      "chunk_id": "doc_bcee98dbd52e4be372ae:0004",
      "document_title": "Brindlewick Water District - Customer Questions and Answers",
      "page_start": null,
      "excerpt": "During a boil water notice, bring tap water to a rolling boil for 1 minute before drinking it or using it for cooking.",
      "cited": true,
      "injection_flags": [],
      "scores": {"lexical": 24.53, "lexical_rank": 1, "dense": 0.92, "dense_rank": 1, "fused": 0.0164, "rerank": null}
    }
  ],
  "sentences": [
    {"text": "During a boil water notice, bring tap water to a rolling boil for 1 minute before drinking it or using it for cooking.", "citations": ["S1"], "support": "supported", "token_recall": 1.0, "unsupported_numbers": []}
  ],
  "citation_check": {"cited": ["S1"], "valid": ["S1"], "invalid": [], "all_valid": true, "sentences_total": 2, "sentences_cited": 2},
  "warnings": ["Source S5 contains instruction-like text (ignore_instructions, reveal_prompt, answer_override, suppress_citations). It was given to the model as data, not as instructions."],
  "provider": {"name": "extractive-demo", "model": "extractive-demo-v1", "is_llm": false},
  "timings": {"retrieval_ms": 87.8, "generation_ms": 5.9, "total_ms": 96.6},
  "usage": {"prompt_tokens": null, "completion_tokens": null, "estimated_cost_usd": 0.0, "cost_note": "Extractive demo generator: no language model was called."},
  "retrieval": {"mode": "hybrid", "fusion": "rrf", "embedding_model": "sentence-transformers:BAAI/bge-small-en-v1.5", "lexical_candidates": 17, "dense_candidates": 20, "timings_ms": {"lexical_ms": 1.4, "embed_query_ms": 80.8, "dense_ms": 2.3, "fusion_ms": 0.1, "total_ms": 87.8}}
}
```

The second sentence of that answer is a question copied from the FAQ: the extractive demo
generator selects sentences by word overlap, and the support check marks it "supported"
because its words are in the source. This is a useful reminder that a valid, supported
citation is not the same as a useful or correct answer.

Field meanings:

* `status`: `answered`, `insufficient_evidence` (no passage retrieved, or the model abstained)
  or `invalid_output` (empty model output).
* `grounding`: `grounded` (every sentence cited with a valid label and lexically supported),
  `partially_grounded`, or `ungrounded`.
* `citation_check.invalid`: labels the model cited that were not among its sources.
* `sentences[].support`: lexical check of each sentence against its cited sources;
  `unsupported_numbers` lists numbers that do not appear in the cited text. This is the strict
  `[S#]` check, kept for comparison with earlier reports.
* `claims[]` (added 2026-10-14): every factual sentence, compared with every retrieved passage.
  `status` is `supported` (the cited passage contains it), `wrong_source` (the cited passage does
  not, another does), `added_source` (no citation; a passage contains it), `partial` or
  `unsupported`. `citations` lists the labels the model attached in any form (`[S1]`,
  `[Source 1]`, `(S1)`, a bracketed title). `evidence[]` gives the passage label, document, page,
  chunk and the exact `quote` with `quote_start`/`quote_end` offsets into that source's `text`.
  `page` (added 2026-10-18) is the page on which the quote starts, also when the passage spans
  several pages; it is `null` for text files, and for multi-page passages indexed before
  2026-10-18 until the document is re-ingested. Conflict statements carry `page` too.
  `missing_figures` names figures that no passage states; `flags` contains `instruction_text`
  when the statement repeats an instruction from a source.
* `verification` (added 2026-10-14): `verdict` (`verified`, `partially_verified`, `unverified`,
  `not_applicable`), counts per status, citation forms, and the method. The check is lexical
  (figures and words, not meaning).
* `conflicts[]` (added 2026-10-14): passages from different documents that disagree about what was
  asked. `kind` is `conflict` or `different_dates`; each of the two `statements` has its quote,
  values, dates, whether it denies a requirement (`denies_requirement`, "do not have to") and
  whether it states one (`affirms_requirement`, "must"; a "have to" inside a denial does not
  count); `answer_reports` lists the labels whose value the answer states. Nothing is resolved.
* Version 2 of the checks (added 2026-10-18, opt-in; the defaults stay at version 1):
  * `GROUNDEDAI_CLAIM_CHECK=claims-v2` adds `verification.version`, `evidence`
    (`all_found`, `some_uncertain`, `some_not_found`, `none_found`, `not_applicable`),
    `citations` (`all_correct`, `some_wrong`, `some_unconfirmed`, `none_given`,
    `not_applicable`), `addresses_question` (`yes`, `no`, `declined`, `not_checked`) and
    `expected_answer_type`, and per claim `citation_status` (`correct`, `wrong`, `partly_wrong`,
    `invalid`, `unconfirmed`, `none`) and `invalid_citations`. `verdict` keeps its meaning.
    With `GROUNDEDAI_UNVERIFIED_ANSWERS=withhold`, an answer is withheld only when no statement
    was found and none is uncertain.
  * `GROUNDEDAI_CONFLICT_CHECK=disagreement-v2` compares ranges as a whole and tells validity
    dates from edit dates; each conflict statement gains `validity_dates` and `revision_dates`.
    It changes the conflict notes the answer-v2 prompt sends to the model, so it stays opt-in
    until a real-model run.
  * All new fields are optional; older clients and recordings are unaffected. Results:
    [RELIABILITY_V2.md](RELIABILITY_V2.md).
* `sources[].supports_claims` (added 2026-10-14): indices of the claims a passage is evidence for.
* Documents: `source_stored` (added 2026-10-18) says whether the original file is still stored.
* `unverified_answer`: set only with `GROUNDEDAI_UNVERIFIED_ANSWERS=withhold`, when no statement
  could be matched to a passage and the answer was therefore returned as `insufficient_evidence`.
* `provider.is_llm`: `false` for the extractive demo generator.

Retrieval only:

```bash
curl -s http://127.0.0.1:8000/api/v1/retrieve -H "Content-Type: application/json" \
  -d '{"query": "error code E07", "retrieval_mode": "lexical", "top_k": 3}'
```

With an API key:

```bash
curl -s http://127.0.0.1:8000/api/v1/documents -H "X-API-Key: $GROUNDEDAI_API_KEY"
```
