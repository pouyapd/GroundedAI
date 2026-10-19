# Changelog

## 0.1.0 — 2026-10-19

The first public version. Everything below runs locally on a CPU, and the online demo runs in
the visitor's browser; no hosted inference service is part of the release.

### Online demo

- **Live:** https://pouyapd.github.io/GroundedAI/ (GitHub Pages, static files only).
- The unchanged FastAPI application runs in the browser (Pyodide in a Web Worker): uploads,
  PDF extraction, indexing, retrieval, the prompt and all checks. Two synthetic sample PDFs
  and example questions; files uploaded by visitors stay in the tab.
- Optional language model in the page: Qwen2.5-1.5B-Instruct (MLC 4-bit build) on WebGPU
  through WebLLM, asked in two steps so its reply goes through the same claim check.
- Pinned runtime: wheels with SHA-256 in `frontend/browser-runtime.lock.json`;
  `scripts/build_browser_runtime.py` builds it, `scripts/e2e_demo.py` checks the demo in a
  browser (also the deployed URL, from the Pages workflow). See `docs/ONLINE_DEMO.md`.
- Interface: demo panel with start-up progress, generator choice and reset; sample questions;
  refined styling (header, cards, answer, evidence quotes); header static on phones.

### Application

- PDF, text and Markdown upload with validation (type, size, magic bytes, encoding), page-aware
  chunking, and deletion of unreadable files with the reason kept; re-upload retries.
- Hybrid retrieval (BM25 + embeddings, Reciprocal Rank Fusion, optional cross-encoder rerank) on
  SQLite FTS5 or PostgreSQL + pgvector; questions can be limited to selected documents.
- Generators: a deterministic extractive demo (default), Ollama (tested with qwen2.5:1.5b) and
  transformers (tested with Qwen2.5-0.5B-Instruct). A missing model gives `503
  model_unavailable` with setup guidance, never a silent fallback.
- After generation: citation-label validation, a claim check that gives every statement a
  status with the exact quote and page, refusal recognition, and `invalid_output` for replies
  without answer text.
- Before generation: a disagreement check that shows conflicting sources with both quotes
  (values for different dates or groups kept apart), and flags for instruction-like passages.
- React UI: Documents, Ask (document selector, claim check, disagreement panel, evidence with
  pages), Evaluation dashboards and Operations status; light and dark themes; phone layout.
- FastAPI with OpenAPI docs, Alembic migrations, health and readiness probes, Prometheus metrics,
  rate limits, optional API key; Docker image and Compose file.

### Checks, two versions

- `claims-v1` (default) and `claims-v2` (opt-in, `GROUNDEDAI_CLAIM_CHECK=claims-v2`): claims-v2
  judges each statement on the few sentences that carry it, reads a short list of equivalent
  wordings, compares negation sentence by sentence, reads ordinals and names, reports
  unresolvable citation labels, gives separate evidence / citation / relevance verdicts, and
  warns when an answer does not mention a conflict.
- `disagreement-v1` (default) and `disagreement-v2` (opt-in): ranges compared as a whole,
  validity dates kept apart from edit dates, requirement wording, equivalent words.

### Evaluation

- Retrieval and answer benchmarks on a 56-question synthetic set, a 25-question development set,
  and a 660-case reliability suite for the checks (frozen test split, model-written labels).
- Recorded runs with qwen2.5:1.5b (Ollama) and Qwen2.5-0.5B (transformers), with hardware,
  model revision and power source in every report.
- `groundedai-claims-review-v1` and `scripts/claim_review.py`: real model answers on a new
  question set, exported statement by statement, with a scorer that compares labels with both
  claim-check versions.

### Fixed in the release candidate

- A section heading on its own line is no longer part of the next sentence, so quotes and
  extractive answers no longer start with it ("Tickets A single adult ticket ..."). No
  evaluation measure changed.
- Conflict reports no longer pair statements from a document that never mentions the
  organisation the question names (both check versions).
- claims-v2 no longer hides a whole answer that a model wrapped in brackets.
- `scripts/claim_review.py score` scores only statements on which every label file agrees (it
  used the first file alone) and lists disputed statements; the review page announces labels
  restored from the browser and can clear them.
- `scripts/e2e_browser.py` no longer leaves its test folder (uploaded files and database)
  behind on Windows.
- `scripts/answer_behaviours.py` splits answer-evaluation reports into separate behaviours.

### Known limitations

See the README's Limitations section: lexical checks, small local models, synthetic and small
evaluation data, no OCR, single-tenant.
