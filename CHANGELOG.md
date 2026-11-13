# Changelog

## Unreleased

### Real-document evaluation and experiments (November 2026)

Full write-up: `docs/RESEARCH_EVALUATION.md`.

- **Changed defaults:** `GROUNDEDAI_TEXT_CLEANUP=v2` and `GROUNDEDAI_EXTRACTIVE_MATCHING=prefix`
  (Experiment A), after checks on every other document and question set
  (`docs/RESEARCH_EVALUATION.md`, "Making Experiment A the default"). Documents indexed earlier
  keep their text until they are re-ingested; `v1` and `exact` restore the old behaviour.
  Reports recorded before these settings existed are replayed with `v1` and `exact`.
- **Changed:** text clean-up v2 removes a line that is only a number only when it follows the
  page order, and counts only valid lower-case roman numerals as page numbers.
- **Security:** pypdf 6.21.4 (from 6.20.0) and certifi 2026.10.23, for published advisories;
  also in the online demo's runtime.
- **Security:** filelock 4.2.10 and setuptools 84.2.1 in the development and ML locks.
- **Online demo:** when the model cannot be loaded, the message now says how to retry (choose
  the model again) and keeps its sentences apart; found by blocking the download on the live
  site. The live site was re-tested after deployment (`docs/RESEARCH_EVALUATION.md`, "Live demo
  after deployment").

- **Fixed:** facts written before an `INSUFFICIENT_EVIDENCE` marker were treated as part of a
  refusal and never checked; they are now checked and shown as a partial answer, with a warning.
- **Fixed:** the chunk viewer showed only the first 50 chunks of a document; it now pages
  ("Show more chunks").
- **Added (opt-in at first):** `GROUNDEDAI_TEXT_CLEANUP=v2` removes repeated page headers and footers,
  joins words hyphenated at line ends and expands ligatures in PDF text;
  `GROUNDEDAI_EXTRACTIVE_MATCHING=prefix` lets the extractive generator match other forms of the
  question's words.
- **Added:** real-PDF datasets on NIST publications (`groundedai-nist-v1`, 53 test questions;
  `groundedai-nist-val-v1`, validation; `groundedai-claims-nist-v1`, 108 hand-written claims),
  `scripts/fetch_nist_corpus.py` (PDFs are downloaded and hash-checked, not committed), and an
  end-to-end evaluation (`groundedai.evaluation.rag_eval`, `scripts/rag_eval.py`): retrieval,
  correctness, abstention, citations, quotes, pages, conflicts and time, reported separately.
- **Added:** `experiments/` (not used by the application): embedding fine-tuning, a claim
  verifier (rule-based, pretrained NLI, fine-tuned), an answer-level check, ONNX export and int8
  quantization, and the data and code for a LoRA run that was not completed.
- **Measured:** reports in `evaluation/reports/research-2026-11/`. The clean-up and matching
  settings raised correct extractive answers on the test set from .600 to .725; fine-tuned
  embeddings and the fine-tuned verifier showed no reliable gain; a pretrained NLI model
  verified hand-written claims much better than the lexical check (macro-F1 .723 against .409).

### Earlier

- `scripts/record_demo_preview.py`: an animated preview recorded from the live demo (real
  screenshots of a real answer); `scripts/make_social_preview.py`: the repository's social
  preview image from two real fragments of a live answer.
- `scripts/e2e_demo.py` accepts refusals in the model step and takes card screenshots without
  the sticky header.
- The in-browser model was checked end to end on the live site (`docs/ONLINE_DEMO.md`).
- Launch visuals (`docs/images/linkedin-launch*.png`) built around crops of a real answer card
  from the live demo: `scripts/capture_launch_fragments.py`, `scripts/make_linkedin_visual.py`.

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
