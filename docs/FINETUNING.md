# LoRA fine-tuning of the answer model

**Status: completed (run1, 2026-11-16, free Google Colab Tesla T4). The adapter is not used by
the application: it made the answers' format reliable and refused every question it should
refuse, but it answered far fewer answerable NIST questions correctly (protected test set: 34 →
23 of 40).** The application's default model is unchanged. [Results](#results-run1-2026-11-16)
are taken from the run's own JSON reports, committed unchanged in
[`evaluation/reports/finetuning-2026-11/run1/`](../evaluation/reports/finetuning-2026-11/run1/)
with a SHA-256 manifest; `backend/tests/unit/test_finetuning_results.py` checks the figures
quoted here against them.

**In brief**

| | Base Qwen2.5-1.5B-Instruct | With the LoRA adapter |
|---|---|---|
| Protected NIST test set: correct answers (40 answerable) | **34** (.850) | **23** (.575); paired −.275, 95% CI −.425 to −.150 |
| — false refusals (40 answerable) | 1 | 8 |
| — refused when it should (6) | 5 | 6 |
| — replies in the application's format, every label valid (53) | 27 | 53 |
| — answers with no valid source label | 24 of 46 answers | 0 of 35 answers |
| NIST validation: correct (15 answerable) | 14 | 6 |
| NIST relevance validation: correct (7 answerable) | 5 | 1 |
| Held-out fictional organisations: correct (166 answerable) | 151 (.910) | 147 (.886); not distinguishable |
| — refused when it should (62) | 15 | 57 |
| Retrieval (Recall@5, MRR@10), every set | same | same (identical index and passages) |

Fine-tuning taught the output format and abstention, but it also taught the model to abstain or
to copy one sentence where the real NIST questions needed an answer. The likely causes are in the
training data (see [Why correctness fell](#why-correctness-fell-hypotheses)); they are
hypotheses, not proven.

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
2. *Runtime → Change runtime type →* a free GPU (for example T4) *→ Save*. Do not buy Colab Pro
   or compute units; if no GPU is offered, try later.
3. *Runtime → Run all*. Nothing has to be edited, and no Google Drive permission is asked for.
4. When the last cell has finished, download `/content/groundedai-lora-results.zip`. The download
   starts by itself in most browsers; otherwise use the *Files* panel (left), find the file under
   `/content`, and choose *Download* in its menu.

What the steps do:

| Step | What happens | Stops the notebook when |
|---|---|---|
| 1. Settings and code | the settings (written out in the cell), then a clone of the repository into `/content/GroundedAI` with `subprocess` and explicit arguments (an existing clone is updated to the latest `main`; an unusable folder is moved aside); the commit is printed; the settings are saved to `/content/groundedai-lora/settings.json` | cloning fails or a needed file is missing |
| 2. Libraries and compatibility check | stops at once without a GPU; installs the pinned Transformers, PEFT and Accelerate with PyTorch held at its installed version; removes torchao when it is older than PEFT accepts (Colab had torchao 0.10.0; GroundedAI does not use torchao); installs GroundedAI; then, in a fresh process, creates LoRA layers on a tiny model and runs a forward and backward pass | any install fails, PyTorch would change, or the check fails |
| 3. Training data | checks the training files against their SHA-256 hashes | a file differs |
| 4. Smoke test | eight stages with PASS or FAIL: model loading, LoRA creation, forward and backward pass, three optimizer steps, validation loss, saving the adapter, loading it on a fresh base model, generating with it | any stage fails (the full error is shown) |
| 5. Training | prints the plan (GPU, data, sequence length, estimated optimizer steps, folder, fresh or resumed), then trains | the smoke test has not passed |
| 6. Validation | base and fine-tuned model on the three validation sets, with paired comparisons | |
| 7. Protected test set | base and fine-tuned model once | |
| 8. Merge check and archive | merged model against the adapter; the results archive | |

**Restarts.** No cell uses a variable from another cell: step 1 writes the settings to a file,
and every later cell imports `training/colab_pipeline.py` again and reads them. Heavy work runs
in separate Python processes, so removing torchao does not need a kernel restart. After a kernel
restart, *Run all* again: step 1 updates the clone, step 2 checks the libraries again, training
continues from the newest *complete* checkpoint (an incomplete one, for example cut off while
saving, is renamed `incomplete-checkpoint-…` and skipped), a finished training run is not
repeated, and finished evaluation reports are not run again (reports are written to a
`.partial` file and renamed when complete). A run folder keeps the training settings it started
with; other settings need a new `run_name`.

**Storage.** By default everything is under `/content/groundedai-lora/run1/` on the runtime's
disk. It survives a kernel restart but **not the end of the runtime** (a disconnect, an idle
timeout, *Disconnect and delete runtime*); download the archive before closing. With
`"use_drive": True` in step 1 the run folder is in `MyDrive/groundedai-lora/` instead; if Drive
cannot be mounted (for example "credential propagation was unsuccessful"), the notebook says so
and continues with local storage.

The notebook is generated by `training/build_notebook.py`; `backend/tests/unit/test_finetuning_notebook.py`
checks that it is up to date, that no cell has shell lines or unfilled placeholders, that no cell
uses a variable it does not define, that the settings are the experiment's, and the recovery
logic.

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

* targets are built only from retrieved passages, and an example is dropped when no retrieved
  sentence contains the answer (69 NIST and 2 fictional training examples, 5 NIST validation; 5
  more NIST "other document" examples were dropped because the other document also contained
  the answer);
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
model and some are odd; the abstention share (203 of 713 training targets, 28.5%, are pure
abstentions; 258, 36.2%, contain the abstention marker, counting partial and detail-bait
answers) is a choice, not a measured optimum. The run's results suggest that these choices
matter (see [Why correctness fell](#why-correctness-fell-hypotheses)).

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

## Tests before the GPU run (CPU)

**Problems found in Colab and how they were fixed (2026-11-16).** A first Colab session on a
Tesla T4 (Python 3.13.16, PyTorch 2.14.1+cu130, PEFT 0.21.2, torchao 0.10.0) showed three
failures:

| Failure | Cause | Fix |
|---|---|---|
| `fatal: repository '{REPO}' does not exist` after a runtime restart, then missing files | the clone was a shell line (`!git clone {REPO} …`) that needed a variable from an earlier cell; after a restart the placeholder stayed literal | step 1 defines the URL, branch and paths itself and clones with `subprocess` and explicit arguments; no cell has shell lines or uses another cell's variables (tested) |
| `ImportError: Found an incompatible version of torchao … 0.10.0` when the LoRA layers were created | Colab has torchao 0.10.0 installed; PEFT 0.21 refuses torchao older than 0.16 whenever it is installed, even though GroundedAI does not use it | step 2 removes a torchao older than 0.16 before anything imports PEFT, checks it is gone, and proves LoRA creation works in a fresh process; training runs in its own process, so no restart is needed |
| `MessageError: credential propagation was unsuccessful` from the Drive mount | the mount ran by default | Drive is off by default; when switched on and it fails, the notebook continues with local storage |

**Reproduced and tested on this laptop** (CPU only; a Python 3.13.5 environment with the same
PyTorch 2.14.1 (CPU build), Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.10.1 and torchao
0.10.0 as Colab):

| Check | Result |
|---|---|
| LoRA creation with torchao 0.10.0 installed | the same `ImportError` as in Colab |
| `training/check_environment.py` before the fix | `"ok": false` with that error |
| the whole notebook, *Run all* from a clean kernel (`nbclient`), test mode: Qwen2.5-0.5B on the CPU, 2 training steps, 2 questions per evaluation | all 8 steps ran. Step 1: the real repository was cloned. Step 2: torchao removed, PyTorch unchanged (2.14.1), GroundedAI installed, LoRA creation and a forward and backward pass passed. Step 3: hashes verified. Step 4: all 8 smoke stages PASS. Step 5: trained, validation loss 1.099 → 0.990. Steps 6–8: see the next row (the first run stopped in step 6 on a bug, now fixed: a comparison with no question of one kind divided by zero) |
| the whole notebook again in a new kernel, after deleting the finished adapter and `result.json` and adding a damaged `checkpoint-3` | all 8 steps finished (15.6 min). The clone was updated in place; the damaged checkpoint was renamed `incomplete-checkpoint-3-…`; training resumed from `checkpoint-2`; the validation and test evaluations and comparisons ran; the merge check ran; the archive was written (checkpoints left out) |
| step 1 alone with a non-git folder in place of the clone, then again | the folder was moved aside and the repository cloned; the second time the clone was updated in place |
| `backend/tests/unit/test_finetuning_notebook.py` (26 tests, also in CI) | the notebook matches `training/build_notebook.py`; no shell lines or unfilled placeholders (the old notebook had 11 shell lines); no cell uses a variable it does not define; the settings are the experiment's; every needed file exists; data hashes; settings saved and re-read, missing step 1, unknown settings, Drive fallback, a run keeping its settings, torchao decisions, newest complete checkpoint, damaged trainer state, finished-run detection, smoke result required for the same model, partial reports, archive contents |
| `prepare_dataset.py` twice | identical files (same SHA-256) |

When training resumes from the very last checkpoint of a run, the Trainer (Transformers 5.19)
runs one more step; its learning rate is 0 at that point of the linear schedule, so the weights
and the validation loss do not change (0.99026 before and after in the test above).

These CPU tests showed that the code and the notebook work; they are not training results. The
GPU path (fp16 training, the automatic sequence length, Qwen2.5-1.5B itself) was first exercised
in run1 below, which ran the notebook from the merged `main` (commit `c077c48`) without changes.

## Results (run1, 2026-11-16)

All figures come from the files in
[`evaluation/reports/finetuning-2026-11/run1/`](../evaluation/reports/finetuning-2026-11/run1/),
copied unchanged from the Colab run folder. `MANIFEST.json` lists the SHA-256 of every file of
that folder, including the files that are not in Git (adapter weights, tokenizer). One file was
not in the folder received: `reports/nist-val-base.json`, the base model's per-question report on
`groundedai-nist-val-v1`. Its aggregate figures survive in `reports/nist-val-comparison.json`,
which the notebook computed from it in Colab, so the NIST validation comparison below is complete
but the base model's individual answers on that set cannot be inspected.

### Hardware and environment

| | |
|---|---|
| GPU | NVIDIA Tesla T4, 14.6 GB (compute capability 7.5), free Colab tier |
| peak GPU memory during training | 13.15 GB |
| host | 2 vCPUs (Intel Xeon 2.00 GHz), 12.7 GB RAM |
| software | Python 3.13.16, PyTorch 2.14.1+cu130 (CUDA 13.0), Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.10.1; torchao removed by step 2 (`environment_check.json`: `"torchao": null`, LoRA creation and a forward and backward pass passed) |
| code | GroundedAI `c077c4813142ffcf722ba7ddf166d39fd20e1aac` (no uncommitted changes, recorded in every report) |

### Smoke test (on the T4, before training)

All eight stages passed with Qwen2.5-1.5B-Instruct in fp16 on the GPU (`smoke/smoke_result.json`):

| Stage | Result |
|---|---|
| 1. model loading | PASS, 35.8 s |
| 2. LoRA adapter creation | PASS, 18,464,768 trainable parameters |
| 3. forward and backward pass | PASS, loss 0.794 |
| 4. three optimizer steps | PASS, training loss 1.953 |
| 5. validation loss | PASS, 1.941 |
| 6. adapter saving | PASS |
| 7. adapter loading on a fresh base model | PASS, 392 LoRA tensors, largest difference 0 |
| 8. generation with the adapter | PASS |

### Training

| | |
|---|---|
| base model | Qwen/Qwen2.5-1.5B-Instruct (1,562,179,072 parameters) |
| LoRA | rank 16, alpha 32, dropout 0.05 on q, k, v, o, gate, up and down projections; 18,464,768 trainable parameters (1.18%) |
| precision | fp16 base weights, fp32 LoRA weights, fp16 autocast with loss scaling |
| batch | 1, gradients accumulated over 8 examples, gradient checkpointing |
| optimiser | AdamW, learning rate 1e-4, linear schedule with 5% warm-up |
| sequence length | 2048 (the first size tried; the longest example has 2,010 tokens, so no example was left out) |
| data | 713 training and 269 validation examples (hashes checked by the notebook) |
| steps | 180 optimizer steps = 2 complete epochs (90 per epoch); fresh run, not resumed |
| time | 2,415.5 s (40.3 min) in the Trainer, including the nine validation-loss evaluations (about 92 s each); 2,524.3 s (42.1 min) for the whole training script, including model loading and saving |
| seed | 13 |

Validation loss (269 examples, every 20 steps; `loss_history.json`):

| step | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 |
|---|---|---|---|---|---|---|---|---|---|
| validation loss | .150 | .105 | .091 | .078 | .071 | .080 | .068 | .066 | **.064** |

The training loss fell from 0.818 (first 5 steps) to about 0.02 at the end (mean over the run
0.085). The lowest validation loss was at the last step, so `checkpoint-180` became the adapter.
Model selection used only this validation loss, as fixed before the run.

Saved: `adapter/adapter_model.safetensors` (73,911,112 bytes, SHA-256
`7af02566c8625b206cfd453300fc50925effe5ac59cbcbfc53cad17d4cf80a21`) with `adapter_config.json`
(PEFT 0.21.2, `r=16`, `lora_alpha=32`), the tokenizer, and the records listed above. The weights
are not in Git and are not published anywhere; anyone can reproduce them with the notebook
(same data hashes, seed and library versions; GPU kernels can still make the weights differ
slightly between runs).

### Evaluation setup

The base model and the adapter were evaluated with the same code
(`training/evaluate_finetuned.py`), index, retrieval (hybrid BM25 + hashing embedder, top 5),
prompt (`answer-v2`), checks (`claims-v1`, `disagreement-v3`) and decoding (greedy, at most 384
new tokens, fp16 on the T4). Only the weights differ. "Correct" means that the answer contains
every key fact (`key_fact_recall` = 1); a refusal is never correct. Differences are paired
bootstrap 95% intervals over questions (5,000 resamples, seed 13); "distinguishable" means the
interval excludes zero.

Order: the three validation sets, then the protected test set `groundedai-nist-v1` once with the
final adapter, then the merge check. No setting was changed after any of these runs, and the
protected test results are not used to choose anything.

These base-model figures were measured with Transformers in fp16. Earlier figures for
qwen2.5:1.5b in the README (for example .775 correct on the test set) come from Ollama's 4-bit
build on a laptop CPU, a different setup; they are not compared with these.

### Protected NIST test set (53 questions, run once)

| | Base | LoRA | Paired difference |
|---|---|---|---|
| correct answers (40 answerable) | 34 (.850) | 23 (.575) | −.275, CI −.425 to −.150 (11 lost, 0 gained) |
| false refusals (40 answerable) | 1 | 8 | +.175, CI +.050 to +.325 |
| refused, of 6 that should be refused | 5 | 6 | +.167, CI 0 to +.500 (not distinguishable) |
| all refusals (precision of refusing) | 7 (.714 should be refused) | 18 (.333 should be refused) | |
| ambiguous questions (3): refused | 1 | 2 | |
| conflicting-version questions (4): correct | 0 | 0 | |
| replies in the application's format (53) | 27 | 53 | |
| answers given | 46 | 35 | |
| answers with no valid source label | 24 of 46 | 0 of 35 | |
| statements found in the passages by the lexical check | 99 of 109 (.908) | 38 of 39 (.974) | |
| quotes verbatim and on the right page | all | all | |
| retrieval Recall@5 / MRR@10 (44 questions with evidence) | .864 / .792 | .864 / .792 | 0 / 0 |
| median time per question (T4) | 2.18 s | 2.20 s | |
| generated tokens, all 53 questions | 3,534 | 1,387 | |

By kind of question: factual 32 → 22 correct of 36 (refused 1 → 7), multi-chunk 2 → 1 of 4.

"Replies in the application's format" (`format_valid`) counts every reply: an answer whose
sentence-level citations all name a provided passage and which has at least one, or a refusal
in the `INSUFFICIENT_EVIDENCE` format. The base model often wrote citations the application does
not read, such as "According to Source S1" or "[Source S1]".

**Citation precision and recall are not comparable between the two columns as plain numbers.**
They are averaged over different answers:

| | Base | LoRA |
|---|---|---|
| citation precision (cited passages that hold the evidence) | .414 over **21** answers (those with at least one valid label and known evidence) | .794 over **34** answers |
| citation recall (evidence passages that were cited) | .302 over **43** answers (answers with no valid label count as 0) | .750 over **34** answers |

The adapter's figures are higher partly because the base model's 24 unlabelled answers count as
zero recall, and partly because the adapter answered 11 fewer questions, refusing some that the
base model answered with poor citations. They show that the adapter's labels are usable, not
that its answers are better.

### Validation sets

| | Base | LoRA | Paired difference |
|---|---|---|---|
| **NIST validation** (`groundedai-nist-val-v1`, 18 questions): correct of 15 | 14 | 6 | −.533, CI −.800 to −.200 (9 lost, 1 gained) |
| — false refusals of 15 | 0 | 5 | +.333, CI +.133 to +.600 |
| — refused, of 3 that should be | 2 | 3 | not distinguishable |
| **NIST relevance validation** (`groundedai-nist-val-relevance-v1`, 14 questions): correct of 7 | 5 | 1 | −.571, CI −.857 to −.143 (4 lost) |
| — false refusals of 7 | 0 | 5 | +.714, CI +.286 to +1.000 |
| — refused, of 7 that should be | 5 | 7 | not distinguishable |
| — replies in the application's format (14) | 7 | 14 | |
| **Held-out fictional organisations** (`synthetic-eval`, 236 questions): correct of 166 | 151 (.910) | 147 (.886) | −.024, CI −.090 to +.042 (14 gained, 18 lost; not distinguishable) |
| — false refusals of 166 | 9 | 19 | +.060, CI 0 to +.121 (not distinguishable) |
| — refused, of 62 that should be | 15 | 57 | +.677, CI +.565 to +.790 |
| — replies in the application's format (236) | 94 | 236 | |
| — answers with no valid source label | 127 of 212 answers | 0 of 158 answers | |
| Retrieval Recall@5 / MRR@10 | identical in every set | | 0 |

The 62 fictional questions that should be refused are of three kinds: unanswerable (base 2,
LoRA 15 of 20 refused), insufficient (5 → 26 of 26) and irrelevant (8 → 16 of 16). Other kinds:
detail bait 8 → 15 of 16 correct, partial 7 → 9 of 10, multi 15 → 12 of 16, single answers
121 → 111 of 124, conflicts 3 → 0 of 8.

### Merge check

The adapter merged into the base weights (`merge_and_unload`, fp16) gave the same reply as the
unmerged adapter on 20 of 20 fictional validation questions (`reports/merge-check.json`). A
merged model could therefore replace the adapter in a deployment without changing answers on
that sample.

### Conclusion

LoRA fine-tuning on GroundedAI's prompt format made the format and abstention reliable: every
reply followed the application's format, no answer lacked a valid source label, and the
fictional questions that should be refused were refused in 57 of 62 cases instead of 15. It did
not preserve correctness on real documents: on the NIST validation sets and on the protected
test set the adapter answered many fewer questions correctly, mostly by refusing questions whose
evidence had been retrieved or by copying a sentence that does not answer the question. On the
held-out fictional organisations, correctness did not change measurably. Because the application
needs correct answers first, **the adapter is not used**, and the base model stays the default.

### Why correctness fell (hypotheses)

These are explanations consistent with the data, not tested causes; testing them needs new
training runs judged on validation data (the protected test set has now been used for this
adapter and must not guide the next one).

1. **The adapter refuses when the evidence was there.** Retrieval is identical between the two
   models, and the passages with the evidence were retrieved for most of the new refusals: 6 of
   the 8 test false refusals, 4 of the 5 in NIST validation, and 2 of the 5 in NIST relevance
   validation had Recall@5 = 1. The refusals come from the model, not from retrieval.
2. **The NIST training data teaches a weak cue for refusing.** 77 of the 250 NIST training
   examples (31%) are refusals of a NIST question asked over passages from *another* NIST
   publication. These passages use the same vocabulary as the right ones, so what separates
   "answer" from "refuse" is subtle. 52 question texts occur with both targets and 23 only as a
   refusal. The model may have learned that NIST-style passages without an exact match mean
   "refuse".
3. **Answer targets were kept only when the answer was a verbatim substring of one retrieved
   sentence.** 69 of the 242 NIST training questions (28.5%) were dropped because no single
   retrieved sentence contained the reference answer word for word. The model never saw an
   answer target built from paraphrased evidence, from several sentences, or from a list, which
   is what many test and validation questions need (for example "the four functions of the AI
   RMF Core"). It may treat such evidence as insufficient.
4. **Targets are single copied sentences.** 170 of the 173 NIST answer targets have one
   citation, and the fictional answers are short (median 60 characters). The adapter's test
   answers are about half as long as the base model's (median 122 against 260 characters; 1,387
   against 3,534 generated tokens). 5 of the 11 lost test answers are one copied sentence that
   misses the key fact. For t20 (which function most users start with) the adapter replied with
   the fragment "However users integrate the functions, the process [S1]."; for t22 (whose
   definition of AI actors the framework uses; key fact "OECD") it copied a sentence about one
   group of AI actors.
5. **The validation loss could not show the problem.** 236 of the 269 validation examples are
   fictional and templated, and the loss falls smoothly (.150 → .064). Choosing the checkpoint by
   loss therefore rewarded the format, not correct answers on real documents. The NIST validation
   sets showed the drop in correctness before the test set was run.

What would test these hypotheses: fewer and harder-to-confuse refusal examples; answer targets
that need paraphrase, several sentences or lists (kept by a check on key facts instead of a
verbatim match); checkpoints chosen on NIST validation correctness and false refusals instead of
loss; then a new, previously unused NIST test set for the final comparison.

## Browser deployment (WebLLM)

Not done, and not planned for this adapter, because it lowered correctness. The demo keeps its
MLC 4-bit build of the base Qwen2.5-1.5B-Instruct; the server keeps its generators unchanged.
A future adapter with the same architecture could be merged (the merge check above passed),
converted with MLC LLM (`mlc_llm convert_weight … --quantization q4f16_1`, `mlc_llm gen_config …`)
and loaded by WebLLM with the existing Qwen2.5-1.5B WebGPU library. That would need the
MLC LLM package, a host for the converted weights and a change to the demo's model list, and it
should only follow an adapter that keeps correctness on validation data.
