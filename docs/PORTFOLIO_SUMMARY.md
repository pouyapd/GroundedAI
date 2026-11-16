# GroundedAI: portfolio summary

A short, factual summary of the project for CVs, applications and interviews. Every figure links
to the document or report it comes from.

## Project overview

GroundedAI answers questions about PDF documents and shows the evidence for each statement: the
exact quote, its page and source label, and a list of the checks that ran and what they found.
It is a research and portfolio prototype, built and evaluated by one author.

- **Application:** FastAPI backend, LangChain prompt and model interface, React 19 + TypeScript
  frontend, SQLite FTS5 or PostgreSQL + pgvector, Alembic migrations, Docker, GitHub Actions CI.
- **Live demo:** <https://pouyapd.github.io/GroundedAI/>. The same Python backend runs in the
  browser tab (Pyodide, WebAssembly). It has an optional Qwen2.5-1.5B-Instruct model on the
  visitor's GPU (WebLLM) and an optional NLI meaning check (DeBERTa-v3-xsmall, int8 ONNX).
- **Local-first:** no paid API is needed. Generators: a deterministic extractive generator, a local
  model through Ollama, or Transformers.

## Engineering contributions

- **Retrieval and indexing:** PDF text extraction with page tracking and clean-up of running
  headers and footers; hybrid retrieval (BM25 + vectors, Reciprocal Rank Fusion); safe bulk
  re-indexing after a settings change (one document at a time, old index kept on failure).
- **Answer checks:** citation-label validation; a lexical claim check that matches every
  statement to an exact quote and page; a disagreement check across documents and editions; a
  question-relevance check; an optional NLI meaning check in the browser. Each check is shown
  separately, never merged into one score.
- **Evaluation:** separate development, validation and protected test sets (NIST publications,
  questions written by hand before any run). Reports record commit, settings and hardware.
  Changes are compared with paired bootstrap intervals, and the test sets are run once per
  configuration ([docs/RELIABILITY_V3.md](RELIABILITY_V3.md),
  [docs/RESEARCH_EVALUATION.md](RESEARCH_EVALUATION.md)).
- **LoRA fine-tuning pipeline** (below): data generation through the application's own pipeline,
  training, evaluation and a restart-safe Colab notebook.

## LoRA fine-tuning: implementation and GPU training

Full write-up: [docs/FINETUNING.md](FINETUNING.md). Raw reports:
[evaluation/reports/finetuning-2026-11/run1/](../evaluation/reports/finetuning-2026-11/run1/).

- **Goal:** teach the small answer model the application's output format: answer only from the
  passages, cite each sentence with `[S#]`, and reply `INSUFFICIENT_EVIDENCE` when the passages
  do not answer. Correctness had to be preserved.
- **Data** (`training/prepare_dataset.py`): 713 training and 269 validation examples. Each was
  made by sending the question through the real QA service and recording the exact prompt the
  model would see. The sources are fictional organisations generated from templates (validation
  organisations never seen in training) and five NIST training publications that are in no
  test set. Files are deterministic and pinned by SHA-256.
- **Training** (`training/finetune_lora.py`, PEFT): Qwen2.5-1.5B-Instruct with LoRA rank 16,
  alpha 32, dropout 0.05 on all attention and MLP projections (18,464,768 trainable parameters,
  1.18%). fp16 base weights with fp32 LoRA weights, batch 1 with 8-step gradient accumulation,
  gradient checkpointing, learning rate 1e-4, 2 epochs, seed 13. The sequence length is chosen
  automatically on the GPU (2048 fitted).
- **GPU run:** free Google Colab NVIDIA Tesla T4 (14.6 GB): 180 optimizer steps in 40.3
  minutes, 13.15 GB peak GPU memory, validation loss .150 → .064. An eight-stage smoke test
  passed on the GPU before training: loading, LoRA creation, forward and backward pass, three
  optimizer steps, validation loss, saving, reloading, generation.
- **Reliability engineering:** the Colab notebook runs with *Run all* after any restart. No cell
  depends on another cell's variables. Training resumes from the newest complete checkpoint, and
  finished runs and reports are never repeated. An incompatible preinstalled torchao is removed
  without a manual restart, and outputs are written atomically. 26 automated tests cover the
  notebook and its recovery logic.
- **Merging:** the merged model (adapter folded into the fp16 weights) gave the same replies as
  the adapter on 20 of 20 validation questions.

## Evaluation design

- Base model and adapter run through the **same** evaluation code, index, retrieval, prompt,
  checks and greedy decoding; only the weights differ.
- **Order fixed in advance:** validation sets first, then the protected NIST test set once with
  the final adapter. The checkpoint was chosen by validation loss only.
- **Separate measures,** never one combined score: correct answers (every key fact present),
  false refusals, correct refusals, output-format validity, citation precision and recall,
  lexical support of statements, and retrieval Recall@5 / MRR@10.
- **Paired bootstrap 95% intervals** over questions (5,000 resamples) for every difference.

## Main findings

| | Base | With the adapter |
|---|---|---|
| Protected NIST test set: correct answers (40 answerable) | 34 | 23 (paired −.275, CI −.425 to −.150) |
| — false refusals | 1 | 8 |
| — refused, of 6 that should be refused | 5 | 6 |
| — replies in the application's format | 27 of 53 | 53 of 53 |
| — answers with no valid source label | 24 of 46 | 0 of 35 |
| NIST validation: correct answers | 14 of 15 | 6 of 15 |
| NIST relevance validation: correct answers | 5 of 7 | 1 of 7 |
| Held-out fictional documents: correct answers | 151 of 166 | 147 of 166 (not distinguishable) |
| — refused, of 62 that should be refused | 15 | 57 |
| Retrieval Recall@5 / MRR@10 | same | same |

- **Positive:** the adapter learned the format and abstention. Every reply was in the
  application's format, every answer had a valid source label, and refusals of questions that
  should be refused rose sharply.
- **Negative:** the adapter refused too often and copied single sentences that did not answer the
  question, so correctness on real NIST documents fell clearly on validation and on the test set.
- **Decision:** the adapter is not used. The base model stays the production default because the
  adapter did not keep answer correctness.
- **Likely causes (hypotheses, not proven):** 31% of the NIST training examples are refusals that
  look almost like answer examples; answer targets were kept only when the answer appeared word
  for word in one retrieved sentence; targets are always one copied sentence; and the validation
  loss, mostly on templated fictional text, could not show the problem.

## Limitations

- One training run, one seed, one base model; no hyperparameter search.
- Small evaluation sets (40 answerable test questions; one question is 2.5 points), one
  publisher, labelled by one person. "Correct" means that every key fact is present as a string.
- The NIST training questions were written by a small model.
- The protected test set has now been used for this adapter. A next adapter must be chosen on
  validation data and compared on a new test set.

## CV bullets

- Implemented and evaluated LoRA fine-tuning of Qwen2.5-1.5B-Instruct on an NVIDIA T4 (free
  Colab tier). Built a reproducible PEFT training pipeline with fp16 training, checkpoint
  resumption, an eight-stage GPU smoke test and 26 automated tests.
- Compared the base and adapted models through the same RAG pipeline on evidence-grounded QA,
  citation compliance and abstention, with paired bootstrap intervals. The adapter raised
  citation-format compliance from 27/53 to 53/53 replies but lowered correct answers on a
  protected real-document test set from 34/40 to 23/40. I reported the negative result and kept
  the base model in production.
- Built GroundedAI, an open-source RAG application (FastAPI, LangChain, React) that matches every
  answer statement to an exact quote and page in the source PDFs and runs entirely in the browser
  (Pyodide, WebLLM).

## Interview explanation (about two minutes)

> GroundedAI answers questions from PDFs and checks each statement against the passages it
> cites. The small answer model, Qwen2.5-1.5B, usually found the right fact but often ignored the
> required citation format. So I tried LoRA fine-tuning on the application's own prompt format.
> I generated about 700 training examples by sending questions through the real pipeline and
> recording the exact prompts. The targets were either a cited sentence from the retrieved
> passages or the application's refusal. I trained rank-16 LoRA on a free Colab T4 in fp16 with
> gradient checkpointing: 180 steps in 40 minutes. To make the free GPU usable, the notebook is
> restart-safe and resumable and runs a GPU smoke test first.
>
> For evaluation, the base model and the adapter went through identical retrieval, prompt and
> decoding. I ran validation sets first and the protected test set once, with paired bootstrap
> intervals. The format problem was solved: 53 of 53 replies were valid, against 27, and on
> held-out synthetic documents the questions that should be refused were refused 57 times out of
> 62, against 15. But correct answers on the real
> NIST test set fell from 34 to 23 of 40, and validation showed the same drop. Most new refusals
> had the evidence in the passages. My reading is that the training data taught refusal too
> strongly: many refusal examples looked almost like answer examples, and every answer was a
> single copied sentence. The validation loss could not show this. So I kept the base model in
> production and documented the result. The next run would change the data and choose the
> checkpoint on validation correctness, not loss.
