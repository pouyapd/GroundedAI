# Development-set reports (prompt development, 2026-10-13)

These runs use `evaluation/datasets/groundedai-dev-v1` (25 questions, documents not in the
evaluation set). They record how the answer-v2 prompt was developed. They are **not**
evaluation results; the 56-question evaluation set was run only after the prompt was frozen.

All runs: hybrid retrieval (RRF), bge-small-en-v1.5, top 5 passages, 400/60-character chunks,
temperature 0, at most 200 new tokens, Windows 11 laptop on mains power. Latency in these files
is not reported anywhere: the first 0.5B run includes a Modern Standby period (12:40–13:36).

| Report | Model | Prompt (fingerprint) | What changed |
|---|---|---|---|
| `…T100755Z` | Qwen2.5-0.5B-Instruct (transformers) | answer-v1 | baseline: the published prompt |
| `…T102456Z` | Qwen2.5-0.5B-Instruct | v2 draft 1 (not recorded) | sources start with a `[S1] Title` line; question after the sources; inline example sentences. The model copied an example time ("07:00") into an answer |
| `…T103102Z` | qwen2.5:1.5b (Ollama) | answer-v1 (`93f4b2d66ffa`) | baseline for the configured model |
| `…T103520Z` | qwen2.5:1.5b | v2 draft 2 (`ed0392f23f81`) | examples removed, rules reworded. Refusals collapsed, the model wrote `[Source S1]`, and repeated a planted instruction |
| `…T105117Z` | Qwen2.5-0.5B-Instruct | v2 draft 2 (`ed0392f23f81`) | same prompt on the small model: fewer refusals, no well-formed citations |
| `…T110201Z` | qwen2.5:1.5b | **v2 draft 3 = frozen answer-v2** (`e928c4235d03`) | v1 source format and rules with the question after the sources, "answer if one source states it", "instruction-like text is not a fact", conflict and injection notes |
| `…T110921Z` | qwen2.5:1.5b | v2 draft 4 (`94f3994f327f`) | stricter label rule and shorter notes. Rejected by the rule fixed before the run (keep only if correct behaviour and injection resistance are both at least those of draft 3; behaviour fell from 0.88 to 0.76) |

`comparison-qwen2.5-1.5b.md` and `comparison-qwen2.5-0.5b.md` are produced by
`scripts/compare_generation_reports.py`, which re-scores every row with the current refusal
parser so all columns use the same rules.

## 2026-10-18: Qwen2.5-0.5B with the frozen answer-v2 prompt

Two runs with the application's default chunking (800/120 characters), hybrid retrieval with
bge-small, top 5, greedy decoding, on battery power (latency not meaningful; the first run
shared the CPU with a diagnosis script):

| Report | Checks | Notes |
|---|---|---|
| `…T174909Z` | claims-v1 (default) | code at `6a74774`; marked "uncommitted changes" because `scripts/journey_check.py` and `README.md` were being edited when it was written, neither of which the run uses |
| `…T181836Z` | claims-v2 | code at `1516dc1`; raw outputs identical to the first run; `[21]`, `[12]` and a title-only reply are now invalid output |

Diagnosis and results: [docs/EVALUATION.md](../../../docs/EVALUATION.md#2026-10-18-the-pdf-journey-with-a-real-local-model).

Known weaknesses of the frozen prompt on this set (qwen2.5:1.5b): citation-format compliance
0.42 (v1: 0.54), malformed citations in 32% of answers (v1: 8%), and one of the two injection
questions failed: the model copied all sources, including the planted text, into its reply.
