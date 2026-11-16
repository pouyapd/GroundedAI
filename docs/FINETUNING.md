# LoRA fine-tuning of the answer model

**Status: prepared and tested on a CPU; the GPU training run has not been done yet.** Nothing in
this document claims that fine-tuning helps. The results section is filled in only from a real
run.

## What and why

GroundedAI's language-model path sends the retrieved passages and the question to a small model
(Qwen2.5-1.5B-Instruct in the browser demo). The model must answer only from the passages, cite
each sentence with `[S#]`, and reply `INSUFFICIENT_EVIDENCE: …` when the passages do not answer.
On the protected test set, qwen2.5:1.5b (through Ollama) gave the right fact often (.775
correct) but rarely used the citation format: 28 of 45 answers in round 2 and 35 of 48 in round 3
had no valid source label. The NLI check caught 5 of its 9 wrong answers. The experiment asks whether LoRA fine-tuning on
GroundedAI's own prompt format improves citation, abstention and support without losing
correctness.

## How to run it (Google Colab, free tier)

1. Open the notebook in Colab:
   <https://colab.research.google.com/github/pouyapd/GroundedAI/blob/main/notebooks/finetuning/GroundedAI_LoRA_Colab.ipynb>
2. *Runtime → Change runtime type → T4 GPU* (the free GPU). Do not buy Colab Pro or compute
   units; if no GPU is offered, try later.
3. *Runtime → Run all*. Allow Google Drive access when asked (checkpoints and results go to
   `MyDrive/groundedai-lora/run1/`).
4. If the runtime disconnects, open the notebook again and *Run all* with the same `RUN_NAME`:
   training continues from the last checkpoint and finished evaluations are not repeated.
5. At the end, download `groundedai-lora-results.zip` (reports, logs, adapter) from the Files
   panel, or take the folder from Drive.

The first cell prints the GPU, its memory, CUDA and PyTorch versions, RAM and disk, and stops the
notebook when there is no GPU.

## Data (`training/prepare_dataset.py` → `training/data/`)

Every example is made with GroundedAI's own pipeline: the question goes through the real QA
service (text clean-up v2, hybrid retrieval, top 5, the answer-v2 prompt), the exact messages the
model would receive are recorded, and the target is built from the passages actually retrieved.
An answer target is a sentence copied from a retrieved passage of the right document with that
passage's label; every other target is the application's abstention format. Nothing is
model-written except the NIST questions (see below).

| Source | Content | Train | Validation |
|---|---|---|---|
| fictional documents (`synthetic-documents/`, generated from templates, seed 20261116) | 8 kinds of organisation; each document states a random subset of facts with random values; one organisation per split also has two editions with one changed value | 30 organisations, 36 documents (6 with a second edition) | 16 held-out organisations, 24 documents, including two kinds (clinic, museum) never seen in training |
| NIST PDFs | training: five publications that are in no validation or test set (`experiments/finetune/corpus.json`); validation: the two NIST validation PDFs | model-written questions (qwen2.5:1.5b via Ollama, `experiments/finetune/generate_queries.py`) | same |

Kinds of example (counts from `training/data/dataset_report.json`):

| Kind | Target | Train | Validation |
|---|---|---|---|
| answer (one stated fact) | the sentence, cited | 246 | 124 |
| multi (two stated facts) | both sentences, each cited | 30 | 16 |
| conflict (two editions differ) | "The sources disagree." and both sentences, each cited | 6 | 8 |
| unanswerable (fact not in the document; other documents may state it) | abstention | 45 | 20 |
| insufficient (a related fact is stated, the asked one is not) | abstention | 51 | 26 |
| irrelevant (only another organisation's document in scope) | abstention | 30 | 16 |
| partial (two facts, one stated) | the stated one, cited, then the abstention for the other | 25 | 10 |
| detail bait (a stated fact plus a detail no document states) | the fact, cited, then the abstention for the detail | 30 | 16 |
| NIST answer | the sentence, cited | 173 | 24 |
| NIST question asked in another document | abstention | 77 | 9 |
| **total** | | **713** | **269** |

Quality control:

* targets are built only from retrieved passages, and an example is dropped when the passage with
  the answer was not retrieved (71 NIST and 2 fictional training examples, 5 NIST validation);
* questions name the organisation, and all training documents share one index, so retrieved
  passages include other organisations' documents with other values (to be ignored);
* fictional names were checked against every dataset, document and test in the repository (three
  colliding names were replaced);
* the split is by organisation (and by kind for clinic and museum), so validation documents were
  never seen in training;
* the files' SHA-256 hashes are in `dataset_report.json` and the notebook checks them;
* the protected test sets (`groundedai-nist-v1`, `groundedai-claims-nist-v1`) are not read.

Token lengths (Qwen2.5 tokenizer, prompt and target): median 1,119, 99th percentile 1,604,
longest 2,010 in training; median 1,177, longest 1,648 in validation.

Limitations: the fictional documents are short and regular, and the targets copy one sentence,
so the model is taught a format more than a skill; the NIST questions were written by a small
model and some are odd; the abstention share (about 30% of training examples, more with partial
answers) is a choice, not a measured optimum.

## Training (`training/finetune_lora.py`)

| Setting | Value |
|---|---|
| base model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA | rank 16, alpha 32, dropout 0.05; q, k, v, o, gate, up and down projections |
| precision | fp16 base weights, fp32 LoRA weights, fp16 autocast with loss scaling (no bf16, no 4-bit) |
| memory | batch 1, gradients accumulated over 8 examples, gradient checkpointing |
| sequence length | `auto`: the longest example is tried at 2048, 1536 and 1024 tokens on the GPU; the largest that fits is used, longer examples are left out and counted |
| optimiser | AdamW, learning rate 1e-4, linear schedule, 5% warm-up, gradient clipping 1.0 |
| length of training | 2 epochs; validation loss every 20 steps; the checkpoint with the lowest validation loss is kept |
| seed | 13 |
| smoke test | 3 steps at 512 tokens with prompts cut, before the real run |

Saved: `adapter/` (LoRA weights and configuration), `tokenizer/`, `training_config.json`,
`environment.json` (library versions and the GroundedAI commit), `hardware.json`,
`loss_history.json`, `data_report.json`, `result.json`, and checkpoints (resumable).

## Evaluation (`training/evaluate_finetuned.py`)

The base and the fine-tuned model are run through GroundedAI's evaluation runner with identical
index, retrieval, prompt, checks and greedy decoding (at most 384 new tokens); only the weights
differ. Measures: correct answers (every key fact), false refusals, abstention precision and
recall, citation validity, statements found by the lexical claim check, answers with added
details (counted with the optional rule, which does not change any status), coverage of asked
words, valid output format, and results by kind of question. Paired differences come from
`scripts/compare_rag_reports.py` (bootstrap, 5,000 resamples).

Order: validation sets first (held-out fictional organisations, `groundedai-nist-val-v1`,
`groundedai-nist-val-relevance-v1`); the protected test set `groundedai-nist-v1` once at the end,
after the adapter is final. Model selection uses only the validation loss.

## Tests done so far (CPU, no GPU)

| Check | Result |
|---|---|
| `prepare_dataset.py` | 713 / 269 examples, 236 held-out evaluation questions; the evaluation set loads as a GroundedAI dataset; a second build gave identical files (same SHA-256) |
| `finetune_lora.py --smoke` with Qwen2.5-0.5B on the CPU (the 1.5B model is not downloaded on this machine) | 3 optimizer steps, validation loss recorded, adapter and tokenizer saved (570 s) |
| the same command again | resumed from the checkpoint (`"resumed": true`) |
| `evaluate_finetuned.py` with the 0.5B model on 3 held-out questions: base, base + adapter, merged adapter | all three ran through the pipeline with every measure; the adapter loaded with PEFT; the merged model gave the same reply as the adapter on 3 of 3 |
| `finetune_lora.py --report-hardware` | prints the report; on this laptop: no GPU |

The GPU path (fp16, the automatic sequence length, the 1.5B model) has not run anywhere yet;
the first notebook run will be its first test.

These show that the code runs. They are not training results.

## Results

Not available yet: the GPU run has not been done.

## Browser deployment (WebLLM)

Not done. The demo runs an MLC 4-bit build of Qwen2.5-1.5B. The fine-tuned model has the same
architecture, so the merged weights could in principle be converted with MLC LLM
(`mlc_llm convert_weight … --quantization q4f16_1`, `mlc_llm gen_config …`) and loaded by WebLLM
with the existing Qwen2.5-1.5B WebGPU library. This needs the MLC LLM package, a host for the
converted weights (for example a Hugging Face model repository, which needs an account) and a
change to the demo's model list. Until that has been done and tested in the deployed demo, the
current model stays the only one there.
