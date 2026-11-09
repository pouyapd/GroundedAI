# Experiments: fine-tuning and verification (optional, not used by the application)

Research code for three questions, kept apart from the application so that nothing here can
change how GroundedAI answers. Results and their interpretation are in
[docs/RESEARCH_EVALUATION.md](../docs/RESEARCH_EVALUATION.md).

| Question | Code | Trained model used by the app? |
|---|---|---|
| Do domain-adapted embeddings retrieve better on unseen documents? | `finetune/generate_queries.py`, `finetune/train_embeddings.py` | no |
| Does LoRA fine-tuning make a small answer model more grounded (format, citations, abstention)? | `finetune/make_sft_data.py`, `finetune/train_lora.py` | no |
| Can an NLI model, pretrained or fine-tuned, verify claims better than the lexical check? | `verifier/*.py` | no |

## Requirements

A separate environment with PyTorch (CPU), `transformers`, `sentence-transformers`, `peft` and
`accelerate` (versions used: torch 2.10, transformers 5.19, sentence-transformers 6.1, peft 0.21.2,
accelerate 1.10.1), plus the backend package. None of them is a dependency of the application.
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
`verifier/data/` and `finetune/runs/` and are not committed (`.gitignore`).

## Reproducing

```bash
cd experiments
python finetune/generate_queries.py --split train --per-doc 70 --out data/train_queries.jsonl
python finetune/generate_queries.py --split validation --per-doc 60 --out data/val_queries.jsonl
python finetune/train_embeddings.py --train data/train_queries.jsonl --val data/val_queries.jsonl --out runs/bge-small-ft

python verifier/make_verifier_data.py --split train --per-doc 160 --out verifier/data/train.jsonl
python verifier/make_verifier_data.py --split validation --per-doc 120 --out verifier/data/validation.jsonl
python verifier/make_verifier_data.py --split synthetic --per-doc 120 --out verifier/data/synthetic.jsonl
python verifier/train_verifier.py --train verifier/data/train.jsonl --val verifier/data/validation.jsonl --out finetune/runs/verifier-xsmall
python verifier/run_verifiers.py --out verifier-report.json --method rule:claims-v1 --method rule:claims-v2 \
    --method nli:cross-encoder/nli-deberta-v3-xsmall --method ft:finetune/runs/verifier-xsmall/epoch-2

python finetune/make_sft_data.py --queries data/train_queries.jsonl --corpus corpus.json --out data/sft_train.jsonl
python finetune/train_lora.py --train data/sft_train.jsonl --out runs/qwen05-lora
python ../scripts/rag_eval.py --dataset ../evaluation/datasets/groundedai-nist-v1 --generator hf \
    --hf-model finetune/runs/qwen05-lora/merged --set text_cleanup=v2 --out lora-test.json
```
