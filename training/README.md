# LoRA fine-tuning of the answer model

Full description, data, settings and results: [docs/FINETUNING.md](../docs/FINETUNING.md).
Run it on a free Colab GPU with
[notebooks/finetuning/GroundedAI_LoRA_Colab.ipynb](https://colab.research.google.com/github/pouyapd/GroundedAI/blob/main/notebooks/finetuning/GroundedAI_LoRA_Colab.ipynb).

| File | Purpose |
|---|---|
| `prepare_dataset.py` | builds `data/` through GroundedAI's own pipeline (needs the NIST training and validation PDFs: `python scripts/fetch_nist_corpus.py`) |
| `finetune_lora.py` | LoRA training; `--report-hardware`, `--smoke`, resumable checkpoints |
| `evaluate_finetuned.py` | base or base + adapter through GroundedAI's evaluation runner |
| `requirements-colab.txt` | pinned training libraries (PyTorch comes with Colab) |
| `data/train.jsonl`, `data/validation.jsonl` | 713 / 269 examples: exact prompt messages and target replies |
| `data/synthetic-eval/` | held-out fictional organisations as a GroundedAI evaluation dataset |
| `data/synthetic-documents/` | every fictional document, by split |
| `data/dataset_report.json` | counts, dropped examples, hashes, seed |

Model weights, checkpoints and adapters are not committed (`training/runs/` is ignored).
