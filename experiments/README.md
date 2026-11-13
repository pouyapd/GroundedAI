# Experiments: fine-tuning and verification (optional, not used by the application)

Research code for four questions, kept apart from the application so that nothing here can
change how GroundedAI answers. Results and their interpretation are in
[docs/RESEARCH_EVALUATION.md](../docs/RESEARCH_EVALUATION.md).

| Question | Code | Trained model used by the app? |
|---|---|---|
| Do domain-adapted embeddings retrieve better on unseen documents? | `finetune/generate_queries.py`, `finetune/train_embeddings.py` | no |
| Does LoRA fine-tuning make a small answer model more grounded (format, citations, abstention)? | `finetune/make_sft_data.py`, `finetune/train_lora.py` | no |
| Can an NLI model, pretrained or fine-tuned, verify claims better than the lexical check? | `verifier/*.py` | no |
| Can the verifier run small and fast enough to ship (ONNX, int8)? | `deploy/export_verifier_onnx.py` | no |

Status: the embedding and verifier experiments were run and measured. **The LoRA run was not
completed** (it was stopped before the first epoch ended, and saved nothing); its data, code and
commands are here so it can be run later. The verifier scripts also include
`verifier/answer_level.py`, which checks the statements of finished answers against the
retrieved passages.

## Requirements

A separate environment with PyTorch (CPU), `transformers`, `sentence-transformers`, `peft`,
`accelerate`, `onnxruntime` and `onnx` (versions used: torch 2.10, transformers 5.19,
sentence-transformers 6.1, peft 0.21.2, accelerate 1.10.1, onnxruntime 1.23.2, onnx 1.19.1),
plus the backend package. None of them is a dependency of the application.
Query generation also needs a local Ollama with `qwen2.5:1.5b`. Everything runs on a CPU; no
paid service or GPU is needed.

```bash
python scripts/fetch_nist_corpus.py        # all NIST PDFs (test, validation, training)
```

## Data (all generated from the training or validation documents only)

- `finetune/generate_queries.py`: a local model writes one question and a verbatim answer
  phrase per sampled passage; pairs whose answer is not copied from the passage are dropped.
- `verifier/make_verifier_data.py`: four-label claim/evidence pairs made by rule (FactCC-style
  perturbations), from the training documents (train), the validation documents (validation)
  and the fictional synthetic documents (an out-of-domain check).
- `finetune/make_sft_data.py`: the application's exact prompts for each generated question,
  with a grounded target reply (the sentence that holds the answer, cited) or the abstention.

Generated data, downloaded PDFs and trained models are written to `finetune/data/`,
`verifier/data/`, `finetune/runs/` and `deploy/runs/` and are not committed (`.gitignore`),
except the two question files named below.

## Reproducing

The commands below are the ones used for the reported runs (Windows, Git Bash, `.venv-ml` for
the experiments, `.venv` for query generation). Trained models go to `finetune/runs/` and
generated data to `finetune/data/` and `verifier/data/`; these are not committed, except the two
model-written question files (`finetune/data/train_queries.jsonl`, `val_queries.jsonl`), which a
new run of a local model might not reproduce exactly. Everything else is rebuilt the same way
from them and the PDFs (fixed seeds).

```bash
python scripts/fetch_nist_corpus.py                     # PDFs, checked by SHA-256

# Query generation (local Ollama, qwen2.5:1.5b). The committed files were made by runs that
# stopped and were resumed with other --per-doc values (train: 150, then 70; validation: 70,
# then 60). The script skips passages already done, so one command does not remake them
# exactly; that is why they are committed.
python experiments/finetune/generate_queries.py --split train --per-doc 70 --out data/train_queries.jsonl
python experiments/finetune/generate_queries.py --split validation --per-doc 60 --out data/val_queries.jsonl

# Experiment B: embeddings (from experiments/finetune)
python train_embeddings.py --base BAAI/bge-small-en-v1.5 --train data/train_queries.jsonl --val data/val_queries.jsonl --out runs/bge-small-ft --epochs 3 --threads 5

# Experiment C: LoRA (from experiments/finetune)
python make_sft_data.py --queries data/train_queries.jsonl --corpus corpus.json --out data/sft_train.jsonl
python make_sft_data.py --queries data/val_queries.jsonl --corpus ../../evaluation/datasets/groundedai-nist-val-v1/manifest.json --out data/sft_val.jsonl
python balance_sft.py --input data/sft_train.jsonl --out data/sft_train_balanced.jsonl
python train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --train data/sft_train_balanced.jsonl --val data/sft_val.jsonl --out runs/qwen05-lora --threads 8

# Experiment D: claim verifier (data from the repository root, the rest from experiments/verifier)
for s in train validation synthetic; do
  python experiments/verifier/make_verifier_data.py --split $s --per-doc $([ $s = train ] && echo 160 || echo 120) --out experiments/verifier/data/$s.jsonl
done
python train_verifier.py --base cross-encoder/nli-deberta-v3-xsmall --train data/train.jsonl --val data/validation.jsonl --out ../finetune/runs/verifier-xsmall --epochs 3 --threads 6
python run_verifiers.py --out verifier-zero-shot.json --method rule:claims-v1 --method rule:claims-v2 --method nli:cross-encoder/nli-deberta-v3-xsmall --method nli:cross-encoder/nli-deberta-v3-base
python run_verifiers.py --out verifier-finetuned.json --method ft:../finetune/runs/verifier-xsmall/epoch-2

# Answer-level verification of finished rag_eval reports (from the repository root)
python experiments/verifier/answer_level.py --report <rag_eval report.json> --dataset evaluation/datasets/groundedai-nist-v1 --premise window --out answer-level.json

# Deployment: ONNX export, int8 quantization, size and speed (from the repository root)
python experiments/deploy/export_verifier_onnx.py --model cross-encoder/nli-deberta-v3-xsmall --mapping nli --out experiments/deploy/runs/nli-xsmall --report onnx-report.json
```

End-to-end runs use `scripts/rag_eval.py` (see its docstring); the exact `--set` values of each
reported run are stored in the report itself (`settings_overrides`).
