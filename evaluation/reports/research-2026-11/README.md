# Reports of the November 2026 research round

Measured outputs behind [docs/RESEARCH_EVALUATION.md](../../../docs/RESEARCH_EVALUATION.md).
Do not edit them. Each end-to-end report stores its settings (`settings_overrides`), the code
commit, package versions, hardware and power source, and one row per question.

| Folder | Content | Made with |
|---|---|---|
| `test/` | the 53-question test set (`groundedai-nist-v1`): baseline and Experiment A with the extractive generator and with qwen2.5:1.5b; Experiment B (base and fine-tuned bge-small, hybrid and dense) | `scripts/rag_eval.py` |
| `validation/` | the validation set (`groundedai-nist-val-v1`): ablations a00-a17 (a09 was not run), Experiment B b01-b04, and the embedding training log | `scripts/rag_eval.py`, `experiments/finetune/train_embeddings.py` |
| `verifier/` | claim verifiers on the 108 hand-written test claims and on the rule-made sets, with the prediction for every claim; the verifier training log | `experiments/verifier/run_verifiers.py`, `train_verifier.py` |
| `answer-level/` | lexical check and NLI check on finished answers: validation (whole-passage and sentence-window premises) and test (sentence windows) | `experiments/verifier/answer_level.py` |
| `deploy/` | ONNX export and int8 quantization of the verifier: size, speed, agreement with PyTorch | `experiments/deploy/export_verifier_onnx.py` |
| `regression/` | checks before Experiment A became the default: what text clean-up v2 removes from each local PDF; the synthetic and development sets and the offline retrieval benchmark with the old and new settings; qwen2.5:1.5b on the validation set with and without v2 | `scripts/compare_text_cleanup.py`, `groundedai eval generation`, `groundedai eval retrieval`, `scripts/rag_eval.py` |
| `live/` | the live demo test of 2026-11-09: the questions and expected answers written before the test, and the log of what the page showed | a Playwright script (not in the repository) |

Notes:

* The test reports were each produced by one run with settings fixed beforehand. The validation
  reports are the ones used to choose settings.
* Times in `validation/` and `verifier/` were measured while other jobs used the CPU. The
  times in `deploy/` and `answer-level/test-*` were measured with nothing else running.
* There is no report for Experiment C (LoRA): it was not completed.
