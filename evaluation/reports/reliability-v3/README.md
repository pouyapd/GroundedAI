# Reports of reliability round 3 (November 2026)

Measured outputs behind [docs/RELIABILITY_V3.md](../../../docs/RELIABILITY_V3.md). Do not edit
them. All runs were made on one laptop (Intel Core i7-1255U, 16 GB RAM, no GPU) on mains power.

| Folder | Content | Made with |
|---|---|---|
| `versions/` | the disagreement checks v1, v2 and v3 on `groundedai-versions-v1` (dev and validation, including the intermediate dev runs and the first and final validation runs of v3), false alarms on the NIST validation documents (`valfp-*`) and on 242 questions about the training documents (`trainfp-*`; the question set is in `train-fp-check/`, the PDFs are `experiments/finetune/corpus`) | `scripts/eval_disagreement.py` |
| `heldout-conflicts/` | the 30 held-out cases of `groundedai-conflicts-v1`, scored with v1, v2 and v3 | `scripts/evaluate_conflict_cases.py --check` |
| `relevance/` | extractive generator, old defaults vs new defaults, on the validation sets and on `groundedai-dev-v1` | `scripts/rag_eval.py`, `groundedai eval generation` |
| `protected-test/` | the one run on the protected sets: `groundedai-nist-v1` with old and new defaults (extractive) and new defaults (qwen2.5:1.5b), the paired comparisons, the version-conflict questions, the 108 test claims (lexical rules, NLI, both), the answer-level NLI check of both test runs, and `groundedai-synthetic-v1` | `scripts/rag_eval.py`, `scripts/compare_rag_reports.py`, `scripts/eval_disagreement.py`, `experiments/verifier/run_verifiers.py`, `experiments/verifier/answer_level.py` |
| `nli/` | claim-level NLI on validation sets (int8 ONNX vs PyTorch), the lexical rules on the same sets, and the answer-level choice of windows on validation answers (input: `research-2026-11/regression/validation-qwen15-text-cleanup-v2.json`) | `run_verifiers.py`, `answer_level.py` |
| `nli/browser/` | the meaning check in Chromium (local build): three runs with nothing else running (`local-run*.json`) and three earlier runs made while other programs used the CPU, two of them with settings that were then dropped (`local-busy-cpu-*.json`) | `scripts/bench_meaning_check.py` |
| `reindex/` | the three NIST test PDFs indexed with clean-up v1, re-indexed with v2 and compared with a fresh v2 ingestion | `groundedai ingest`, `groundedai reindex`, a comparison of the SQLite chunk tables |
| `lora-smoke/` | the bounded LoRA smoke test (8 examples) and its repeat (3 examples); no model was trained | `experiments/finetune/train_lora.py --max-steps N --smoke` |

The protected-set reports were each produced once, after the new defaults had been committed
(`08ff4cc`). Reports that were used to choose settings are the ones in `versions/`,
`relevance/` and `nli/`.
