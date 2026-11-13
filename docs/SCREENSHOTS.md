# Screenshots

## Online demo (2026-10-19)

Taken by `scripts/e2e_demo.py --screenshots docs/images` from the demo build running in Chrome
(GroundedAI in the browser, extractive generator, the two sample PDFs from `examples/`).

| Start page: generator choice, question form, sample questions | An answer: status, quoted sentence and page |
|---|---|
| ![The live demo's start page with the extractive generator selected, the Qwen2.5-1.5B-Instruct option, the question form and example questions about the two sample PDFs](images/demo-start.png) | ![Answer that a single adult ticket costs 14 euros, marked Supported by the cited passage, with the quoted sentence from page 2](images/demo-answer-card.png) |
| **A question the selected PDF cannot answer** | **Phone width** |
| ![Answer panel saying the searched documents do not contain enough evidence, for a library question asked of the ferry guide](images/demo-not-enough-evidence.png) | ![The demo at 390 px width with an answer about bicycle prices quoted from page 3](images/demo-mobile.png) |

Full-page version of the answer: [images/demo-answer.png](images/demo-answer.png).

| The in-browser model (Qwen2.5-1.5B-Instruct, WebGPU) on the live site | Animated preview (`scripts/record_demo_preview.py`) |
|---|---|
| ![Answer from the in-browser model: 14 euros supported by the cited passage on page 2; an invented citation S2 struck through and flagged as invalid; Children under 5 travel free found on page 2 without a citation](images/demo-llm-answer-card.png) | ![Recording of the live demo in the dark theme: a document is selected, the question is typed and sent, and the answer appears with its claim check, quote and page](images/demo-preview.gif) |

## Full application

All taken from the running application on the development laptop. The journey screenshots use
the sample PDFs in `examples/`; the others use the sample corpus in
`evaluation/datasets/groundedai-synthetic-v1` with bge-small embeddings. Dates are when each was
taken; the UI has not changed in ways that affect them since.

| One document selected, qwen2.5:1.5b through Ollama (2026-10-19) | The same question with the extractive demo generator, not an LLM (2026-10-18) |
|---|---|
| ![Ask page with Search in set to the Harbourline Ferries Passenger Guide; the qwen2.5:1.5b answer that a single adult ticket costs 14 euros, its claim check, and the supporting quote attributed to page 2](screenshots/journey-answer-qwen15-light.png) | ![The same answer from the extractive demo generator, with the quote on page 2 and the passage spanning pages 1 to 3](screenshots/journey-answer-light.png) |
| **A question the selected document cannot answer** | **Documents: two ready PDFs, a damaged one and one without text, with the reasons** |
| ![Answer panel saying the searched documents do not contain enough evidence, for a library question asked of the ferry guide](screenshots/journey-not-in-document-light.png) | ![Documents table with Ask, Chunks, Re-ingest and Delete buttons, the ferry guide's extracted text, and two failed uploads whose files were not kept](screenshots/journey-documents-light.png) |
| **Claim check: each statement with the exact supporting sentence (extractive generator)** | **Disagreeing sources shown side by side, with values and dates (extractive generator)** |
| ![Answer with the verdict Every statement found in the sources, the claim check listing one statement as Supported by the cited passage with the quoted sentence from the travel policy](screenshots/ask-claim-check-light.png) | ![A Different values for different dates panel showing the travel policy's 140 and 110 euro hotel limits next to the finance FAQ's 120 euro limit, updated May 2025](screenshots/ask-disagreement-light.png) |
| **Evaluation: qwen2.5:1.5b with the answer-v2 prompt (2026-10-13 report)** | **Evaluation: retrieval benchmark** |
| ![Evaluation dashboard for the qwen2.5:1.5b answer-v2 report with correct behaviour, refusals, key-fact recall, citation format and results by category](screenshots/evaluation-ollama-light.png) | ![Evaluation dashboard with Recall@5, MRR and nDCG@5 per configuration, 95% intervals and latency](screenshots/evaluation-light.png) |
| **Documents (dark theme)** | **Operations on a phone** |
| ![Documents table with status, pages and chunks](screenshots/documents-dark.png) | ![Operations page at phone width](screenshots/operations-mobile-dark.png) |

Qwen2.5-0.5B-Instruct (transformers) on the same journey (2026-10-18):
![Ask page with a Qwen2.5-0.5B answer that a single adult ticket costs 14 euros, quoted from page 2](screenshots/journey-answer-qwen-light.png)

A static, read-only version of the UI can be built with `npm run build:demo`: ten dataset
questions recorded with every retrieval mode and passage count, replayed without a backend
([docs/DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md), [docs/UI_CONTROL_AUDIT.md](UI_CONTROL_AUDIT.md)).

## Launch visuals

`docs/images/linkedin-launch.png` and `linkedin-launch-minimal.png` (1200 × 675; `@2x` versions
2400 × 1350) are layouts around crops of one real answer card, captured from the live demo on
2026-10-20 in the dark theme: Qwen2.5-1.5B-Instruct in the browser answered the ferry-guide
question, cited a source (S2) it was never given, and the check flagged it. The crops are not
edited; the gaps between them are marked, and the numbered rings only point at parts of the
interface. Regenerate with `scripts/capture_launch_fragments.py` (needs a Chrome profile with the
cached model) and `scripts/make_linkedin_visual.py`.

The launch post's other images are plain crops of the same screenshots, with nothing drawn on
them: `linkedin-post-2-invalid-citation.png` (the answer, the model and the warning about S2),
`linkedin-post-3-verified-evidence.png` (the claim check with both quotes and pages) and
`linkedin-post-4-interface.png` (the whole Ask page at desktop width with the model's answer).

![LinkedIn launch visual: the GroundedAI name and tagline on the left with three numbered points, and on the right three crops of a real answer card with numbered rings on the invalid citation S2, the status Supported by the cited passage, and the quote from page 2](images/linkedin-launch.png)

