# Real-document evaluation, pipeline ablations and fine-tuning experiments

This document covers the November 2026 round of work on GroundedAI: an audit and live test of the
public demo, an evaluation on real PDFs, ablations of the RAG pipeline, and fine-tuning
experiments: embeddings and a claim verifier were trained and measured; a LoRA-tuned answer model
was prepared but not trained to the end, and has no results. Every number below was
measured on one laptop CPU (Intel Core i7-1255U, 16 GB RAM, mains power) and is reproducible
with the commands given. Where a measurement was not made, the table says so.

**Defaults.** "Defaults" and "baseline" in the tables below mean the settings before this round
(text clean-up v1, exact matching). Experiment A's two settings became the defaults afterwards,
after the checks in [Making Experiment A the default](#making-experiment-a-the-default-2026-11-13).

Contents: [Protocol](#protocol) · [Live demo test](#live-demo-test) ·
[Baseline on real PDFs](#baseline-on-real-pdfs) · [Pipeline ablations](#pipeline-ablations) ·
[Fine-tuning](#fine-tuning) · [Configurations compared](#configurations-compared) ·
[Deployment](#deployment) ·
[Making Experiment A the default](#making-experiment-a-the-default-2026-11-13) ·
[Live demo after deployment](#live-demo-after-deployment-2026-11-13) · [Limitations](#limitations) · [Reproducing](#reproducing)

## Protocol

**Documents.** Ten NIST publications (U.S. Government works), split by document:

| Split | Documents | Used for |
|---|---|---|
| test | SP 800-63B (2017), SP 800-63B-4 (2025), AI RMF 1.0 (AI 100-1) | reported results only |
| validation | AI 600-1 (Generative AI Profile), NISTIR 8312 (explainable AI) | choosing settings, early stopping |
| training | CSF 2.0, SP 800-61r3, SP 800-88r1, SP 800-207, SP 800-218 | fine-tuning data only |

No document is in two splits. The 2017 and 2025 versions of SP 800-63B are both in the test split,
because their differing requirements are the conflict cases.

**Questions and labels written before any run.** The 53 test questions
([groundedai-nist-v1](../evaluation/datasets/groundedai-nist-v1/README.md)), the 18 validation
questions and the 108 test claims
([groundedai-claims-nist-v1](../evaluation/datasets/groundedai-claims-nist-v1/README.md)) were
written by hand from the PDF text before any retrieval, model or verifier was run on those
documents. Training data are generated from the training documents only (model-written questions,
rule-made claims), and are never scored as results.

**Fixed protocol.** Settings are chosen on the validation split; the test split is run once per
reported configuration, with the configuration fixed beforehand. No model was trained on, and no
setting was chosen from, the test documents, questions or claims. Two places where test data were
looked at before a choice are stated where they occur: the conflict check was diagnosed on test
questions (and for that reason left unchanged), and one diagnostic run of the answer-level
verifier was made on test answers before its premise setting was chosen on validation data. Random seeds are fixed
(sampling 20261109, training 13); generation is greedy (temperature 0). Each report records the
settings, code commit, package versions, hardware and power source.

**Separate measures.** Retrieval, answer correctness, abstention, citations, GroundedAI's own
claim check, quote and page accuracy, conflict reporting and latency are reported separately.
No single combined score is computed. With 40 answerable test questions, a difference of one
question is 2.5 points; 95% bootstrap intervals (5,000 resamples, paired where two runs are
compared) are given where they matter.

**What the claim check's numbers mean.** "Found" is GroundedAI's lexical check grading the
answers; it is the system under test, not an independent judge of faithfulness. Answer
correctness is a separate check (every key fact present, as a string).

## Live demo test

The public demo (https://pouyapd.github.io/GroundedAI/, the code on `main` at 6e944ef) was tested
in Chromium driven by Playwright on 2026-11-09. Three real PDFs were uploaded: SP 800-63B (2017,
80 pages), AI RMF 1.0 (48 pages) and SP 800-63B-4 (2025, 129 pages). 22 questions and their
expected answers were written down before the first question was asked (direct, numeric,
terminology, paraphrase, date, missing, multi-passage, ambiguous, entity, version conflict,
document switching, all documents). The extractive generator answered all 22; the WebLLM model
(Qwen2.5-1.5B-Instruct, q4f16, WebGPU on Intel Iris Xe) answered a subset of 6, plus the sample
ferry question twice.

| Check | Result |
|---|---|
| Page load to ready | 17 s (Pyodide and packages from the CDN) |
| Upload and indexing | SP 800-63B 8.6 s (367 chunks), AI RMF 2.5 s (172), SP 800-63B-4 8.6 s (468) |
| WebLLM model load | 35 s (model files already in the browser cache) |
| Answer time | extractive about 0.2 s; WebLLM 11-50 s |
| Damaged PDF, scanned PDF (no text), empty file | refused, with a reason |
| File over 10 MB, `.docx` | upload button disabled, reason shown |
| Question too short or too long | button disabled |
| Question of punctuation only | "Not enough evidence" |
| Page refresh | uploaded documents are gone (by design: data stays in the tab); re-upload works |
| Browser console errors | none |

**The S2 fabricated citation.** The sample ferry question was asked twice with WebLLM. Both times
the model wrote "S2 Children under 5 travel free." while only one passage (S1) had been given to
it. Both times the page showed "The model cited sources that were not provided: S2. Those
citations are invalid." So an invalid label is detected and shown. Two things are still weak: the
answer banner said "Every statement found in the sources" at the same time, and a valid label is
not proof that the cited passage supports the sentence (the claim check tests whether a statement
is in a retrieved passage, which is a weaker test). The claim-verifier experiment below measures
how well a model can make that second test.

**Answer quality** (key facts checked as strings against the answers written beforehand):

| Generator | Answer questions with all key facts | Should refuse: refused | Ambiguous question | Version conflicts flagged |
|---|---:|---:|---|---:|
| Extractive | 6 of 17 | 1 of 2 (L16 answered with unrelated text) | answered, did not ask | 0 of 2 |
| WebLLM Qwen2.5-1.5B | 2 of 3 (L01 wrong) | 2 of 2 | not asked | 0 of 1 |

Problems found, and what was done:

| Problem | Seen in | Status |
|---|---|---|
| Running page headers and footers become "answers" | L07 | fixed: `TEXT_CLEANUP=v2` removes repeated edge lines (opt-in at first, now the default) |
| Words split at line ends ("trustwor- thy") and ligatures | L12, L13 | fixed: `TEXT_CLEANUP=v2` (now the default) |
| Extractive generator refuses when the question uses another word form ("minutes" / "reauthenticated") | L03 | fixed: `EXTRACTIVE_MATCHING=prefix` (now the default) |
| Facts written before an `INSUFFICIENT_EVIDENCE` marker were never checked | crafted model replies | fixed: they are now checked and shown as a partial answer |
| The chunk viewer showed only the first 50 chunks | 367-chunk PDF | fixed: "Show more chunks" |
| A wrong answer can be shown as "Every statement found" | L01 (WebLLM), L12, L16 | documented: the check tests that sentences are in the sources, not that they answer the question |
| Version conflicts (2017 vs 2025) are not reported | L19, L20 | documented with the root cause (see [Limitations](#limitations)); not tuned, because the conflict questions are test data |

The two fixes that change answers were added as opt-in settings and evaluated as Experiment A;
they became the defaults after the checks in
[Making Experiment A the default](#making-experiment-a-the-default-2026-11-13). The public demo
gets them when that change is merged and deployed.

## Baseline on real PDFs

The test set has 53 questions on the three test PDFs (1,007 chunks): 40 answerable, 6 that must be
refused, 3 ambiguous and 4 version conflicts; 44 questions have marked evidence pages. Settings
are the application defaults (hybrid retrieval with the hashing embedder, chunks of 800/120
characters, top 5 passages). The answer models are the extractive generator and `qwen2.5:1.5b`
through Ollama (CPU, greedy).

| Measure | Extractive | qwen2.5:1.5b |
|---|---:|---:|
| Recall@1 / @5 / @10 (evidence pages, n = 44) | .625 / .807 / .932 | same retrieval |
| Precision@5 | .186 | same |
| MRR@10 / nDCG@10 | .753 / .782 | same |
| Correct answers (all key facts), of 40 | .600 (95% CI .45-.75) | .750 (.60-.875) |
| Completeness (share of key facts found) | .631 | .775 |
| False refusals, of 40 | 4 | 1 |
| Abstention precision / recall (6 to refuse) | .286 / .333 | .750 / 1.000 |
| Citation precision / recall (evidence pages) | .444 / .769 | .481 / .279 |
| Answers with no valid source label | 0 of 46 | 28 of 45 |
| Statements found by the claim check | .961 | .928 |
| Quotes verbatim / on the right page | 1.000 / 1.000 | 1.000 / 1.000 |
| Version conflicts reported, of 4 | 0 | 0 |
| Median time per question | 0.12 s | 22 s |
| Index build / peak memory of the app | 51 s / 151 MB | 42 s / 64 MB (Ollama separate) |

What the baseline shows:

* **Retrieval is the first limit for the extractive generator.** Of its 16 wrong answers, 8 have
  no evidence page in the top 5 and 8 have the evidence but pick the wrong sentences.
* **Qwen answers more questions correctly but cites badly.** 28 of its 45 answers had no valid
  `[S#]` label (it numbers its own sentences, or writes no label), so its citation recall is low.
  On one question (t12) it invented a date ("March 2006"); the claim check marked that sentence as
  not found. On another (t07) the answer was wrong but all 6 statements were "found": every
  sentence was copied from a retrieved passage that did not answer the question.
* **No version conflict was reported** by either generator (see [Limitations](#limitations)).

## Pipeline ablations

One change at a time, on the validation split (15 questions with evidence on AI 600-1 and NISTIR
8312, 466 chunks), extractive generator. The test split was not used for these choices.

| Run | R@5 | MRR@10 | nDCG@10 | Correct | Median time | Note |
|---|---:|---:|---:|---:|---:|---|
| a00 defaults | .733 | .633 | .688 | .467 | 0.09 s | |
| a01 `TEXT_CLEANUP=v2` | .867 | .717 | .755 | .533 | 0.12 s | headers, hyphenation, ligatures |
| a02 `EXTRACTIVE_MATCHING=prefix` | .733 | .633 | .688 | .533 | 0.12 s | |
| **a03 v2 + prefix** | **.867** | **.717** | **.755** | **.667** | 0.06 s | citation recall .857 |
| a04 v2, lexical only | .867 | .647 | .701 | .533 | 0.05 s | |
| a05 v2, dense only (hashing) | .733 | .518 | .585 | .467 | 0.02 s | |
| a06 v2, bge-small hybrid | .867 | .704 | .760 | .533 | 0.33 s | |
| a07 v2, bge-small dense | .867 | .697 | .754 | .533 | 0.32 s | |
| a08 v2, all-MiniLM-L6 hybrid | .800 | .606 | .670 | .533 | 0.20 s | |
| a10 v2, chunks 400/60 | .667 | .522 | .559 | .533 | 0.07 s | 857 chunks |
| a11 v2, chunks 1200/180 | .867 | .789 | .809 | .533 | 0.09 s | 309 chunks |
| a12 v2, rerank ms-marco-MiniLM | 1.000 | .743 | .806 | .533 | 3.3 s | |
| a13 v2, rerank bge-reranker-base | 1.000 | .730 | .796 | .600 | 13.8 s | |
| a14 / a15 / a16 v2, bge hybrid with alpha .3 / alpha .7 / RRF k 20 | .867 | .703 / .724 / .701 | .773 / .774 / .757 | .533 | | no clear change |
| a17 v2, bge hybrid + ms-marco rerank | 1.000 | .743 | .806 | .533 | 2.5 s | |

A bge-base run (a09) was planned but not run: the local copy of the model was incomplete, and the
run was dropped rather than downloading it again.

With 15 questions, one question is 6.7 points, so only large differences mean much. Times were
measured while other jobs used the CPU, so they show order of size only. The choice made before
the test run was **Experiment A = `TEXT_CLEANUP=v2` + `EXTRACTIVE_MATCHING=prefix`**, keeping the
hashing embedder and 800/120 chunks. It gave the largest gain in correct answers, costs nothing in
time or size, and needs no model download, so it also works in the browser demo. Rerankers raised
Recall@5 to 1.0 but took 25-140 times longer per question and did not give more correct answers
than a03; larger chunks raised MRR but not correctness.

### Experiment A on the test set

Experiment A (`TEXT_CLEANUP=v2`, `EXTRACTIVE_MATCHING=prefix`), fixed on the validation split, was
run once on the test set. Paired differences are against the baseline with the same generator.

| Measure | Extractive baseline | Extractive A | Change (95% CI) | qwen2.5:1.5b baseline | qwen2.5:1.5b A | Change (95% CI) |
|---|---:|---:|---|---:|---:|---|
| Recall@5 | .807 | .864 | +.057 (.000 to +.136) | .807 | .864 | same retrieval |
| MRR@10 | .753 | .792 | +.038 (-.003 to +.094) | .753 | .792 | |
| nDCG@10 | .782 | .790 | | | | |
| Correct, of 40 | .600 | .725 | +.125 (+.025 to +.250); 5 gained, 0 lost | .750 | .775 | +.025 (-.100 to +.150); 4 gained, 3 lost |
| False refusals | 4 | 3 | | 1 | 1 | |
| Abstention precision / recall | .286 / .333 | .250 / .167 | one more wrong answer (t38) | .750 / 1.000 | .833 / .833 | |
| Citation precision / recall | .444 / .769 | .451 / .829 | | .481 / .279 | .419 / .256 | |
| Quotes verbatim / right page | 1.000 / 1.000 | 1.000 / 1.000 | | 1.000 / 1.000 | 1.000 / 1.000 | |
| Conflicts reported, of 4 | 0 | 0 | | 0 | 0 | |
| Chunks | 1,007 | 927 | -8% (repeated headers removed) | | | |
| Median time per question | 0.12 s | 0.10 s | | 22 s | 22 s | |

For the extractive generator the gain in correct answers is clear (the interval does not include
zero); the gain in retrieval is small and its interval touches zero. For Qwen the change is within
noise. One question that should be refused (t38, "What minimum accuracy does the AI RMF require for
hiring models?") was answered after the change with a sentence about AI RMF hiring profiles:
with prefix matching, "require" also matches "requirements", so the passage shared enough words
with the question.

## Fine-tuning

Three experiments, each kept outside the application (`experiments/`). Training data come only
from the five training PDFs; settings and early stopping use the validation PDFs; the test PDFs
are used only for the reported numbers. B and D were trained and measured; C was prepared but
not completed.

### Experiment B: fine-tuned embeddings

**Data.** A local model (`qwen2.5:1.5b` through Ollama, greedy) wrote one question and a short
answer for passages sampled from the five training PDFs; a pair was kept only when its answer is
copied word for word from the passage. 242 training pairs were kept (pairs without a verbatim answer,
and passages where the local model timed out, were dropped) and 29 validation pairs from the
validation PDFs. The question files are committed
(`experiments/finetune/data/*_queries.jsonl`), because a local model's output cannot be relied on
to be identical in another run.

**Training.** `BAAI/bge-small-en-v1.5` (33M parameters, 127 MiB of weights), contrastive loss with in-batch
negatives plus one hard negative per pair (the passage of the same document that best matches the question
by BM25 word overlap, other than the answer passage), batch 16, learning rate 2e-5, 3 epochs, seed 13; best epoch (2) chosen by
validation MRR. Training took 65 minutes on the CPU. On the 29 validation questions (each searched
among the 29 validation passages only, an easy task) MRR@10 went from .865 to .907.

**Results in the full pipeline** (Experiment A settings, extractive generator; the same settings
with the base model are the comparison):

| Split | Retrieval | Model | R@1 | R@5 | MRR@10 | nDCG@10 | Correct |
|---|---|---|---:|---:|---:|---:|---:|
| validation (15) | hybrid | hashing (Exp A) | .600 | .867 | .717 | .755 | .667 |
| validation | hybrid | bge-small | .600 | .867 | .704 | .760 | .667 |
| validation | hybrid | bge-small fine-tuned | .600 | .867 | .693 | .750 | .600 |
| validation | dense | bge-small | .600 | .867 | .697 | .754 | .600 |
| validation | dense | bge-small fine-tuned | .600 | .733 | .683 | .741 | .600 |
| test (44 / 40) | hybrid | hashing (Exp A) | .670 | .864 | .792 | .790 | .725 |
| test | hybrid | bge-small | .648 | .898 | .791 | .803 | .725 |
| test | hybrid | bge-small fine-tuned | .648 | .886 | .788 | .797 | .725 |
| test | dense | bge-small | .375 | .727 | .591 | .633 | .600 |
| test | dense | bge-small fine-tuned | .398 | .784 | .610 | .657 | .650 |

Paired differences on the test set, fine-tuned minus base: dense Recall@5 +.057 (95% CI -.045 to
+.170), MRR +.020 (-.035 to +.088), correct +.050 (-.075 to +.175); hybrid Recall@5 -.011 (-.080
to +.057), correct .000.

What this shows: fine-tuning on 242 synthetic pairs **did not give a measurable gain**. Dense
retrieval moved in the right direction on the test set but within noise (and the other way on the
validation set); in hybrid mode, where BM25 already finds most of the evidence, nothing changed.
On these documents a dense model of any kind added little over hybrid retrieval with the hashing
embedder, which needs no model at all. The likely reasons are the small amount of training data
and that NIST text is already well covered by bge-small's general training.

### Experiment C: LoRA fine-tuning of the answer model (not completed)

**Status: not run to completion. There are no results for this experiment.** The data and the
training code are complete. Training was started once and stopped after about 25 minutes, before
the first epoch ended, because the only available machine (a laptop CPU, no GPU) was needed for
other work. The script saves the adapter only at the end, so nothing was kept from that run, its
speed was not recorded, and no number from it is reported.

**What was planned.** Teach a small instruct model GroundedAI's answer format: answer only from
the passages, cite each sentence with `[S#]`, and reply with the `INSUFFICIENT_EVIDENCE` marker
when the passages do not answer the question. The baseline showed that this is where the small
model is weak (28 of its 45 answers had no valid source label).

**Data (made, in `experiments/finetune/data/`, rebuilt by the commands below).**

| File | Examples | Content |
|---|---:|---|
| `sft_train.jsonl` | 327 (173 answers, 154 abstentions) | the application's exact prompt for each training question; target = the passage sentence that holds the answer, cited, or the abstention marker (the same question asked against another document) |
| `sft_train_balanced.jsonl` | 259 (173 answers, 86 abstentions) | all answers and a half-sized random sample of the abstentions (seed 20261109), so the model is not taught to refuse too often |
| `sft_val.jsonl` | 43 (24 answers, 19 abstentions) | the same, from the validation PDFs |

**Configuration (fixed before the run).** Base model `Qwen/Qwen2.5-0.5B-Instruct`, fp32, CPU;
LoRA rank 16, alpha 32, dropout 0.05 on the attention and MLP projections (`q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); base weights frozen; loss on the reply
tokens only; AdamW, learning rate 2e-4 with linear warm-up (5%) and decay; one example per step,
gradients accumulated over 8 examples; gradient clipping 1.0; 1 epoch; maximum 1,792 tokens per
example; seed 13. The adapter is merged into a copy of the base model, which GroundedAI loads
like any local model.

**Requirements.** Python with torch 2.10 (CPU), transformers 5.19, peft 0.21.2, accelerate
1.10.1; about 1 GB of disk for the base model and 2 GB for the merged copy; the stopped run used
about 4.3 GB of RAM. The time for one epoch on this CPU was not measured; each of the 259 steps
is a forward and backward pass over a full prompt of five passages, so plan for hours. The script
runs on the CPU as written; a GPU run needs the model and tensors moved to the device, which was
not done or tested. QLoRA (4-bit base weights through bitsandbytes) was not an option here
because it needs a CUDA GPU.

**Commands** (from `experiments/finetune`, after `python scripts/fetch_nist_corpus.py`):

```bash
python make_sft_data.py --queries data/train_queries.jsonl --corpus corpus.json --out data/sft_train.jsonl
python make_sft_data.py --queries data/val_queries.jsonl --corpus ../../evaluation/datasets/groundedai-nist-val-v1/manifest.json --out data/sft_val.jsonl
python balance_sft.py --input data/sft_train.jsonl --out data/sft_train_balanced.jsonl
python train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --train data/sft_train_balanced.jsonl --val data/sft_val.jsonl --out runs/qwen05-lora --threads 8
```

**Evaluation plan (fixed now, to be run once).** The comparison is the same base model without
the adapter, with the same settings, first on the validation split and then once on the test
split (from the repository root):

```bash
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-v1 --generator hf --hf-model Qwen/Qwen2.5-0.5B-Instruct --set text_cleanup=v2 --label expC-base-qwen05 --out expC-base.json
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-v1 --generator hf --hf-model experiments/finetune/runs/qwen05-lora/merged --set text_cleanup=v2 --label expC-lora-qwen05 --out expC-lora.json
```

Measures to compare, each with a paired bootstrap interval: correct answers, false refusals,
abstention precision and recall, answers without a valid source label, citation precision and
recall, statements found by the claim check, and time per answer. The expected risk, from the
verifier experiment, is that the model learns the copy-one-sentence style of the synthetic
targets and gets worse on questions that need two passages.

### Experiment D: claim verifier

**Task.** Given one claim and the evidence passage it cites, say `supported`, `partial` (part of
the claim is stated, part is not), `contradicted`, or `nee` (not enough evidence). This is the
test GroundedAI's lexical claim check cannot make: it checks that the words and numbers of a
claim appear in a passage, not what the passage means.

**Data.**

| Set | Pairs | Made how | Documents |
|---|---:|---|---|
| test | 108 (supported 31, partial 22, contradicted 34, nee 21) | written by hand, before any verifier was run | test PDFs |
| training | 2,743 | by rule (FactCC-style edits: changed numbers, added or removed negation, weaker modal verb, swapped term; added clause; sentence from elsewhere) | training PDFs |
| validation | 900 | the same rules | validation PDFs |
| synthetic (out of domain) | 153 | the same rules | the fictional ferry and library documents |

**Methods.**

* Rule-based: GroundedAI's own claim check (`claims-v1`, the default, and `claims-v2`), with a
  fixed mapping of its statuses to the four labels.
* Pretrained NLI, zero-shot: `cross-encoder/nli-deberta-v3-xsmall` (about 70M parameters with the
  embeddings) and `-base` (about 184M). Entailment -> supported, contradiction -> contradicted; neutral -> partial if the claim
  has several clauses and one of them alone is entailed (and none contradicted), else nee. This
  rule was fixed before any run.
* Fine-tuned: the xsmall model with a new 4-label head, trained on the 2,743 rule-made pairs
  (3 epochs, AdamW 3e-5, batch 16, seed 13; epoch 2 chosen by validation macro-F1).

**Results on the 108 hand-written test claims.**

| Method | Macro-F1 | Accuracy | F1 supported / partial / contradicted / nee | Supported: false positive rate | Supported: false negative rate | Size |
|---|---:|---:|---|---:|---:|---:|
| Rule `claims-v1` (default) | .409 | .463 | .787 / .392 / .000 / .457 | .078 | .226 | none |
| Rule `claims-v2` | .386 | .426 | .702 / .392 / .000 / .451 | .078 | .355 | none |
| NLI xsmall, zero-shot | .723 | .750 | .853 / .581 / .853 / .607 | .052 | .161 | 270 MB |
| NLI base, zero-shot | .720 | .741 | .862 / .581 / .879 / .557 | .026 | .194 | 704 MB |
| NLI xsmall, fine-tuned | .740 | .732 | .745 / .864 / .696 / .654 | .013 | .387 | 270 MB |

"False positive rate" is the share of claims that are not fully supported but are labelled
`supported`: the error that lets a wrong answer through. "False negative rate" is the share of
supported claims that are not labelled `supported`.

Confusion matrices on the test claims (rows: true label; columns: predicted, in the order
supported / partial / contradicted / nee):

| True label | NLI xsmall zero-shot | NLI xsmall fine-tuned |
|---|---|---|
| supported (31) | 26 / 0 / 1 / 4 | 19 / 1 / 9 / 2 |
| partial (22) | 1 / 9 / 0 / 12 | 0 / 19 / 0 / 3 |
| contradicted (34) | 3 / 0 / 29 / 2 | 1 / 0 / 24 / 9 |
| nee (21) | 0 / 0 / 4 / 17 | 0 / 2 / 2 / 17 |

Out of domain and on the rule-made validation pairs (macro-F1): rule `claims-v1` .495 / .524
(validation / synthetic), `claims-v2` .603 / .695, NLI xsmall .814 / .813, NLI base .797 / .825,
fine-tuned .994 / .993.

What this shows:

* **The rule-based check cannot tell a contradiction from a missing fact** (F1 0 for
  `contradicted`): contradicting claims come out as not found (all 15 with a changed number) or
  partly found, and 5 of 34 as found (a reversed meaning, a weaker modal verb, a swapped name, an
  added "not").
* **A pretrained NLI model is much better with no training** (macro-F1 .72 vs .41), and the small
  model is as good as the base model at 38% of the size and about a fifth of the time.
* **Fine-tuning on rule-made pairs did not give a reliable gain.** Macro-F1 rose by .016 (paired
  bootstrap 95% CI -.081 to +.120); accuracy fell by .018. The fine-tuned model learned the
  `partial` label well (F1 .58 -> .86) and almost never calls a wrong claim supported (false
  positive rate .013), but it labels 9 of 31 correct, reworded claims as `contradicted`: the
  rule-made `supported` claims are near-copies of the evidence, so a reworded claim looked like an
  edited one. Its .994 on the rule-made validation set shows only that it learned the rules; that
  set could not reveal the problem. A hand-written validation set, and training claims that
  paraphrase, are the next step (see [Limitations](#limitations)).
* Time per claim, measured on a busy CPU, was 0.7 ms for the rules, about 57 ms for NLI xsmall
  and 282 ms for NLI base (the fine-tuned run, 110 ms, ran beside another job; it has the same
  architecture as xsmall). Measurements on an idle CPU are in [Deployment](#deployment).

**The verifier on whole answers (answer level).** The table above uses short evidence (one to
three sentences). In the application the evidence is five retrieved passages of up to 800
characters, so the pretrained NLI model (xsmall, zero-shot) was also run on the finished test
answers (`experiments/verifier/answer_level.py`; no answer was generated again, and the rebuilt
passages gave the same lexical counts as the reports for every answer). An answer "passes" when
all its statements pass; a wrong answer is one without all key facts, or any answer to a
question that should be refused. Ambiguous and conflict questions are left out.

Choosing the premise, on the validation split only:

| Premise given to the NLI model | Wrong answers flagged (a00 / a03) | Correct answers flagged (a00 / a03) |
|---|---:|---:|
| each whole passage | 7 of 8 / 4 of 5 | 4 of 7 / 7 of 10 |
| windows of one or two sentences | 2 of 8 / 0 of 5 | 0 of 7 / 0 of 10 |

With whole passages the model flags most answers, right or wrong: it fails to find entailment
for a sentence copied word for word from a long passage. (The first whole-passage run was made on
the test baseline before this was understood: 13 of 16 wrong and 21 of 24 correct answers
flagged. The choice between the two premises was then made on validation data.) Sentence windows
were chosen, and run once on the four test reports:

| Test report | Answers scored | Wrong | Lexical check: wrong flagged | NLI: wrong flagged | Lexical: correct flagged | NLI: correct flagged |
|---|---:|---:|---:|---:|---:|---:|
| Extractive, baseline | 40 | 16 | 1 | 2 | 2 of 24 | 2 of 24 |
| Extractive, Experiment A | 42 | 13 | 2 | 2 | 2 of 29 | 3 of 29 |
| qwen2.5:1.5b, baseline | 39 | 9 | 4 | 7 | 2 of 30 | 6 of 30 |
| qwen2.5:1.5b, Experiment A | 40 | 9 | 1 | 5 | 5 of 31 | 5 of 31 |

* **For the extractive generator a verifier does not help.** Its wrong answers are real
  sentences from the documents that do not answer the question; they are supported, so no
  support check can flag them. This is a relevance problem, not a faithfulness problem.
* **For the language model the NLI check flags more wrong answers** than the lexical check (7 and
  5 of 9, against 4 and 1 of 9), at the price of more false alarms in one of the two runs (6
  against 2 of 30). With 9 wrong answers per run these counts are small; they show a direction,
  not a rate.
* **Cost.** Checking every statement against every sentence window of five passages took about
  10 s per answer with PyTorch on this CPU (390 to 420 s for 39 to 42 answers, measured while the
  machine was otherwise idle). That is too slow for interactive use as it is; the int8 ONNX model
  in [Deployment](#deployment) and checking fewer windows are the obvious next steps, not yet
  measured at answer level.
* The fine-tuned verifier was not run at answer level: on the claim test it had no reliable
  advantage and it rejects reworded statements.

## Configurations compared

All rows are test-set measurements already reported above, put side by side. Changes are against
the baseline with the same generator; intervals are paired bootstrap 95% intervals.

| Configuration | What changes | Recall@5 / MRR@10 | Correct, extractive | Correct, qwen2.5:1.5b | Extra download | Time per question |
|---|---|---|---|---|---|---|
| Baseline | defaults | .807 / .753 | .600 | .750 | none | 0.12 s / 22 s |
| A: text clean-up + prefix matching | `TEXT_CLEANUP=v2`, `EXTRACTIVE_MATCHING=prefix` | .864 / .792 (+.057, .000 to +.136 / +.038, -.003 to +.094) | .725 (+.125, +.025 to +.250; relative +21%) | .775 (+.025, -.100 to +.150) | none | 0.10 s / 22 s |
| B: A + fine-tuned bge-small, hybrid | embedding model | .886 / .788 (against A: +.023, -.045 to +.091 / -.004, -.099 to +.091) | .725 (against A: .000, -.075 to +.075) | not run | 127 MiB | 0.11 s |
| B: A + fine-tuned bge-small, dense only | embedding model, no BM25 | .784 / .610 | .650 | not run | 127 MiB | 0.13 s |
| C: LoRA answer model | | not completed | not completed | not completed | | |
| D: NLI verifier (zero-shot) instead of the lexical check | checking only; answers unchanged | unchanged | unchanged | unchanged | 270 MB (83 MB as int8 ONNX) | 19 to 59 ms per claim with short evidence; about 10 s per answer at answer level (PyTorch) |

What D changes is the check, measured in two ways: on the 108 hand-written claims (macro-F1 .723
against .409 for the lexical check; fine-tuned .740, not reliably different), and on whole
answers (table above).

**Combined configuration.** Chosen from the validation results, not from the test table:
Experiment A's settings (the only change with a clear gain), the hashing embedder (no dense
model gave more on validation, fine-tuned or not), and the pretrained NLI model with
sentence-window premises as an optional second check for language-model answers. No separate run
was needed or made for it: its answers are the "A" rows above (correct .725 extractive, .775
qwen2.5:1.5b), and its check is the "qwen2.5:1.5b, Experiment A" row of the answer-level table
(5 of 9 wrong answers flagged against 1 of 9; 5 of 31 correct answers flagged by both). The
fine-tuned embedder and the fine-tuned verifier are not part of it, because neither showed a
reliable gain. The NLI check is not wired into the application; that step is future work.

**Failures that no configuration fixed:** version conflicts (0 of 4 reported everywhere); wrong
extractive answers shown as "Every statement found" (11 of 13 with Experiment A); the small
model's missing source labels (28 and 33 answers without a valid label, baseline and A); and one
more wrong answer to a question that should be refused with prefix matching (t38).

## Deployment

**What was measured.** The claim verifier was exported to ONNX (opset 17) and quantized to int8
(onnxruntime dynamic quantization of the weights) with
`experiments/deploy/export_verifier_onnx.py`, then scored on the 108 test claims. Method: one
claim at a time (batch 1), 4 threads, after a warm-up of 8 pairs; the time per claim includes the
extra clause checks the 4-label rule makes for the NLI model; median and 90th percentile over
the 108 claims. Machine: Intel Core i7-1255U (12 logical CPUs), 15.6 GB RAM, Windows 11, mains
power, no other job running (system CPU load measured just before timing: 10.5% and 8.8%);
torch 2.10, onnxruntime 1.23.2.

| Model | Version | File | Load | Median per claim | 90th percentile | Macro-F1 | Same label as PyTorch |
|---|---|---:|---:|---:|---:|---:|---:|
| NLI xsmall, zero-shot | PyTorch fp32 | 270 MB | 0.7 s | 59 ms | 136 ms | .723 | |
| | ONNX fp32 | 271 MB | 2.2 s | 32 ms | 66 ms | .723 | 108 of 108 |
| | ONNX int8 | 83 MB | 1.8 s | 19 ms | 42 ms | .687 | 104 of 108 |
| NLI xsmall, fine-tuned | PyTorch fp32 | 270 MB | 0.9 s | 60 ms | 137 ms | .740 | |
| | ONNX fp32 | 271 MB | 2.2 s | 30 ms | 46 ms | .740 | 108 of 108 |
| | ONNX int8 | 83 MB | 1.8 s | 17 ms | 25 ms | .775 | 99 of 108 |

ONNX fp32 gives the same labels as PyTorch at about half the time. int8 is 31% of the size and
about a third of the PyTorch time; it changes 4 and 9 of the 108 labels, and the macro-F1 moves
in opposite directions for the two models (-.036 and +.035), which is noise at this sample size,
not a gain or a loss. Memory use of the ONNX sessions was not measured.

**Not measured.** The verifier was not run in a browser (onnxruntime-web, WASM or WebGPU), the
embedder and the answer model were not exported, and nothing in this round was deployed. The
figures above are for a desktop CPU with native onnxruntime.

**What this means for the application.**

* The public demo is unchanged. It still starts with the extractive generator and the hashing
  embedder, with no model download; the WebLLM model stays optional.
* Experiment A needs no model and no download, so it is the one change that could go into the
  demo as it is: two settings. They are now the defaults (see
  [Making Experiment A the default](#making-experiment-a-the-default-2026-11-13)); the public demo
  gets them when that change is merged and deployed.
* An 83 MB int8 verifier is small enough to consider as an optional download for a local
  installation. For the browser demo it would be a second optional download next to the 880 MB
  WebLLM model; whether it runs fast enough there has to be measured first. Plan: load
  `model.int8.onnx` with onnxruntime-web in the existing Web Worker, check only statements the
  lexical check marks "found", measure load and per-statement time in Chromium on the same
  laptop, and keep the lexical check as the default.
* The fine-tuned embedder (127 MiB) is not worth shipping: no measured gain.

## Making Experiment A the default (2026-11-13)

Before `TEXT_CLEANUP=v2` and `EXTRACTIVE_MATCHING=prefix` became the defaults, they were checked
on every other document and question set available, and the clean-up rules were reviewed.
Reports: [evaluation/reports/research-2026-11/regression](../evaluation/reports/research-2026-11/regression/).

**What v2 removes, document by document** (`scripts/compare_text_cleanup.py --datasets`, all 15
local PDFs: the 10 NIST PDFs, the synthetic manual, the two example PDFs and the demo's
byte-identical copies of them):

* The synthetic manual, both example PDFs and the demo's sample PDFs are **unchanged** (byte for
  byte): they are too short for the header rule and have no ligatures or broken words.
* In the 10 NIST PDFs every page keeps its number, no page gains a word that was not on it, and
  no page becomes empty. The removed lines are running headers ("NIST SP 800-63B DIGITAL IDENTITY
  GUIDELINES:"), footers ("This publication is available free of charge from: ..."), dates in
  the header, page numbers and underscore rules, with one exception: in SP 800-61r3 a table's
  column header line ("CSF Element CSF Element Description Priority Recommendations,
  Considerations, Notes"), repeated at the top of 23 pages, is removed too. The table rows stay.
* Per NIST document, up to 402 ligatures and 5 to 235 line-break hyphens were repaired.

**Two rules were made stricter** after this review (before any default changed): a line that
is only a number is now removed only when it follows the page order (number minus page position
the same on at least half of the pages), so a table value at a page edge is kept; and only valid
lower-case roman numerals count as page numbers, so words such as "mix" or a lone "C" are kept.
On all 15 PDFs the output is identical before and after this change, so every result above still
holds. Updating pypdf from 6.20.0 to 6.21.4 (security advisories) also left the text of all 15
PDFs identical.

**Other question sets** (extractive generator; old defaults against v2 + prefix):

| Set | Questions | Answers changed | Effect |
|---|---:|---:|---|
| Synthetic evaluation set (`groundedai-synthetic-v1`) | 56 | 1 | q011 gained one more correct sentence ("Without registration the warranty is 5 years"); still verified, same key facts |
| Development set (`groundedai-dev-v1`) | 25 | 0 | identical |
| Offline retrieval benchmark (synthetic, 43 questions with evidence) | 3 configurations | - | identical metrics (only the text clean-up applies to retrieval) |

**Language models.** Prefix matching is used only by the extractive generator, so for Qwen
(Ollama, Hugging Face or WebLLM in the browser) only the text clean-up can matter, and on the
synthetic and development sets their input is identical. On the NIST documents:

| qwen2.5:1.5b, Ollama | Old defaults | v2 | Paired change (95% CI) |
|---|---:|---:|---|
| Test set (40 answerable): correct | .750 | .775 | +.025 (-.100 to +.150); 4 gained, 3 lost |
| Validation set (15 answerable): correct | .933 | 1.000 | +.067 (.000 to +.200); 1 gained, 0 lost |
| Validation: should refuse, refused | 2 of 3 | 2 of 3 | same questions |
| Validation: Recall@5 / MRR@10 | .733 / .633 | .867 / .717 | +.133 (.000 to +.333) / +.084 (+.017 to +.170) |
| Validation: quotes verbatim / right page | 1.000 / 1.000 | 1.000 / 1.000 | |
| Statements found by the claim check | test .928, validation .812 | test .934, validation .647 | |

The one measure that fell is the share of statements found on validation (.812 to .647, 32 and 34
statements). It comes from three answers in which the model wrote different text after seeing
the cleaner passages: one listed four principles with a definition for each (3 of 5 statements
not found), one gave the publication month in its own words ("was published on September 2021",
correct but not matched), and one copied a URL. On the larger test set the share did not fall.
This is a change in the model's wording, which the lexical check scores sentence by sentence; it
is not a change in what the documents say.

**Regression tests added** (`tests/unit/test_pdf_cleanup.py`,
`tests/integration/test_text_cleanup_pipeline.py`, `tests/unit/test_browser_runtime.py`): numbers
off the page order are kept; words made of roman-numeral letters are kept; every fact of a
multi-page PDF with a running header keeps its page through ingestion with v2; answers cite and
quote the right page with v2 and both matching modes; prefix matching gives identical answers
where exact matching already answers; an unanswerable question is still refused; the defaults are
v2 and prefix, and the online demo uses them. Reports recorded before the two settings existed
are replayed with v1 and exact (`--settings-from`).

**Trade-offs of the new defaults.**

* Gains: on the NIST test set, correct extractive answers .600 to .725 (+.125, CI +.025 to
  +.250), Recall@5 +.057; chunks without running headers; ligature and hyphen repairs.
* Losses: one question that should be refused (t38) was answered with prefix matching; a
  repeated table column header is dropped in one document; a PDF of four or more pages whose
  only text is repeated header lines would be refused as having no extractable text.
* Unchanged: page numbers, quotes (all verbatim and on the right page in every run), citations,
  the claim check, refusals on the synthetic and development sets.
* Documents indexed before the change keep their v1 text until they are re-ingested;
  `GROUNDEDAI_TEXT_CLEANUP=v1` and `GROUNDEDAI_EXTRACTIVE_MATCHING=exact` restore the old
  behaviour.

## Live demo after deployment (2026-11-13)

After the merges (`main` at `b3583ab`), GitHub Pages run 43461690251 built, checked and
deployed the demo, then checked the deployed URL from GitHub's runner (16 of 16). The deployed
version was confirmed independently: the Python package served by the live site is
byte-identical to `backend/groundedai` on `main`, its pypdf wheel is 6.21.4, and the page's
script bundle has the same hash as a local build of `main`.

The same 22 questions as in the [first live test](#live-demo-test), written before that test,
were asked again on the public URL in Chrome (Intel Iris Xe, WebGPU). Reports:
[evaluation/reports/research-2026-11/live](../evaluation/reports/research-2026-11/live/).

| Check | 2026-11-09 (old defaults) | 2026-11-13 (v2 + prefix) |
|---|---|---|
| Upload and indexing, SP 800-63B / AI RMF / SP 800-63B-4 | 8.6 / 2.5 / 8.6 s; 367 / 172 / 468 chunks | 4.5 / 2.5 / 4.5 s; 306 / 173 / 448 chunks |
| Extractive: answer questions with all key facts | 6 of 17 | 9 of 17 (L03, L07, L17 gained; none lost) |
| Extractive: should refuse, refused | 1 of 2 (L16 answered) | 1 of 2 (L16 still answered) |
| Extractive: version conflicts flagged | 0 of 2 | 0 of 2 (L20 now names neither value) |
| Quotes shown: verbatim in the PDF and on the page shown | 60 of 60 (against the text as then extracted; 7 were running headers, which v2 removes) | 64 of 64 |
| WebLLM (Qwen2.5-1.5B, cached): load, answer questions with all key facts | 35 s; 2 of 3 | 10 s; 2 of 3 (L01 still wrong) |
| WebLLM: should refuse, refused | 2 of 2 | 2 of 2 |
| The invented `S2` citation (asked twice) | flagged both times | flagged both times |
| Damaged, scanned and empty files; too large; wrong type | refused / blocked with a reason | the same |
| Page refresh, then upload again and ask | works (uploads cleared by design) | works |
| Chunk viewer on a 306-chunk PDF | first 50 only | 100 shown, "Show more chunks" |
| Browser console errors | none | none |

The quote check takes each quote and page label shown on the page and looks for the quote in the
original PDF's text of that page. "Key facts" are checked as strings against the answers written
beforehand, as in the first test.

**The repository's own browser checks on the live URL:** `scripts/e2e_demo.py --url ...`: 16 of 16;
with the model (`--llm`, Chrome with WebGPU): 19 of 19, including loading with download progress,
an answer checked against the PDF, and a paraphrased question.

**Model download failure on the live site.** With a fresh browser profile and the model files
blocked, the page said the model could not be loaded, kept the extractive generator selected,
and answered with it; after unblocking, choosing the model again restarted the download (9%
after 8 s). The message read "...could not be loaded: TypeError: Failed to fetch The extractive
generator still works." (no sentence break, no hint that a retry works); it now says that
choosing the model again continues the download. After that fix was deployed (`3628f16`), the
whole live test was repeated: the 28 answers were identical, all 64 quotes verbatim on the page
shown, `e2e_demo.py --llm` 19 of 19, and the blocked download showed the new message.

**Crafted model replies through the deployed code.** The in-page model's reply goes through
`BrowserDemo.answer`; the same code (loaded from the live site's package) was given crafted
replies to the ferry question, where only one passage (S1) is provided:

| Reply | Outcome |
|---|---|
| correct, cited `[S1]` | supported; quote from page 2 |
| correct sentence plus "Children under 5 travel free [S2]" | `S2` reported as a citation to a source that was not provided; the statement itself is found in S1 and shown as found there |
| correct, cited `[S9]` | `S9` reported as invalid; statement found in S1 |
| "costs 20 euros" (the PDF says 14) | not found; answer marked unverified |
| "Children under 5 must pay the full adult fare" (the PDF: they travel free) | partly found; answer marked unverified (not labelled a contradiction) |
| correct, no citation | found, source added, with a warning that the model cited nothing |
| a correct sentence, then `INSUFFICIENT_EVIDENCE` for the rest | the sentence is checked; a partial-answer warning |
| "14 euros and includes a free hot meal on board" | **found (supported)** with the default `claims-v1`: 73% of its words are in the passage. `claims-v2` (opt-in) marks it partly found. |

The last row is a known limit of the lexical check: an added detail that shares enough words
with the passage passes. The NLI verifier experiment (Experiment D) measures this gap; the NLI
model is not part of the application, so nothing on the live site is checked for meaning.

## Limitations

* **Small test sets.** 53 questions (40 answerable) and 108 claims, on three documents of one
  publisher and one domain (security and AI-risk guidance). One answerable question is 2.5
  points; most differences between configurations are inside their 95% intervals, and only the
  intervals that exclude zero are described as gains.
* **One annotator.** Questions, key facts, evidence pages and claim labels were written by one
  person, before the runs, with no second annotator and so no agreement figure. Answer
  correctness is a string check of key facts, which can miss a correct answer in other words
  and count an answer that contains the facts but also wrong statements.
* **Synthetic training data.** All fine-tuning data are made from the training PDFs by a local
  1.5B model (questions) or by rules (claims). The verifier result shows the main risk: a model
  can learn the rules of the generator rather than the task, and a validation set made by the same
  rules cannot show it.
* **Version conflicts are not detected.** None of the 4 conflict questions (2017 vs 2025 versions
  of SP 800-63B) was reported as a conflict by the application, with any configuration. The cause
  is in the disagreement check, which compares two passages only when they share at least two of
  the question's content words: "AAL2" is dropped as a capitalised name, and "reauthenticated" and
  "reauthentication" are different words to it. Even "30 minutes" against "45 minutes" in two
  retrieved passages was not paired. The conflict questions exist only in the test split, so the
  check was not changed in this round (changing it while looking at those 4 questions would make
  them useless as a test). A fix needs its own validation conflict cases.
* **An added detail can pass the default claim check.** On the live site, "14 euros and includes a
  free hot meal on board" was reported as found (73% of its words are in the passage);
  `claims-v2` (opt-in) marks it partly found.
* **The claim check is lexical.** "Every statement found" means each statement's words and
  figures are in a retrieved passage. It does not mean the statement answers the question, or
  that the passage says the same thing. On the hand-written claims it reported all 15 claims
  with a changed number as not found, but it can never say "contradicted", and 5 of the 34
  contradicting claims (a reversed meaning, a weaker modal verb, a swapped name, an added "not")
  were reported as found. The verifier experiments measure the size of this gap; the lexical
  check is still what the application shows.
* **CPU only, shared CPU.** Every number comes from one laptop CPU on mains power. Several jobs
  ran at the same time during this round, so the times in the ablation and verifier tables show
  order of size only; [Deployment](#deployment) has times measured with nothing else running.
  No GPU, cloud service or paid API was used.
* **Documents indexed before the default changed keep their v1 text** until they are re-ingested
  (tested: they keep answering; re-ingesting applies v2).
* **Experiment C has no results.** The LoRA fine-tuning of the answer model was prepared but not
  trained to the end (see above); nothing in this document says whether it helps.
* **Not tried.** A bge-base embedder (the local copy was incomplete), larger answer models than
  1.5B on the CPU, QLoRA (4-bit training needs a CUDA GPU), and a second fine-tuning round for the
  verifier with paraphrased `supported` claims and a hand-written validation set.

## Reproducing

```bash
python scripts/fetch_nist_corpus.py     # the ten NIST PDFs, checked by SHA-256 (not committed)

# Baseline and Experiment A on the test set (extractive; add --generator ollama for qwen2.5:1.5b)
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-v1 --generator extractive --set text_cleanup=v1 --set extractive_matching=exact --out baseline.json
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-v1 --generator extractive --set text_cleanup=v2 --set extractive_matching=prefix --out expA.json
```

Every report in [evaluation/reports/research-2026-11](../evaluation/reports/research-2026-11/README.md)
stores its own settings (`settings_overrides`), the code commit, package versions, hardware and
power source; repeating a run means passing those settings with `--set`. The fine-tuning, verifier
and ONNX commands are in [experiments/README.md](../experiments/README.md). Retrieval and the
extractive generator are deterministic; language-model answers on another machine or Ollama
version can differ.
