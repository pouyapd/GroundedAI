# Reliability round 3 (November 2026)

This round worked on seven known weaknesses. Each section below gives the root cause, what was
built, how it was tested and evaluated, the results, and what is still wrong. Every number comes
from a saved report in [`evaluation/reports/reliability-v3/`](../evaluation/reports/reliability-v3/).

| Issue | Status |
|---|---|
| A. Disagreement between two versions of a document (NIST SP 800-63B 2017 / 2025) | Partially resolved |
| B. Statements that add details the sources do not state | Partially resolved |
| C. Meaning check (NLI) in the browser | Resolved and tested (deployed; deployed timings measured on battery) |
| D. Whether the extractive answer addresses the question | Partially resolved |
| E. Test question t38 answered instead of refused | Resolved and tested |
| F. LoRA fine-tuning of the answer model | Blocked by hardware or technical constraints |
| G. Documents indexed with an older pipeline | Resolved and tested |

"Resolved and tested" means the specific failure is fixed and covered by tests; it does not mean
the general problem is solved. C is deployed and was checked on the live site; its mains-power
timings come from the local build of the same code.

## How this round was run

* **New validation data first.** Before any check was changed, two datasets were written:
  [`groundedai-versions-v1`](../evaluation/datasets/groundedai-versions-v1/README.md) (four
  fictional documents, each in two editions; 52 questions and 40 claims, split by document
  family into *dev* and *validation*) and
  [`groundedai-nist-val-relevance-v1`](../evaluation/datasets/groundedai-nist-val-relevance-v1/README.md)
  (14 relevance and abstention questions on the two NIST validation PDFs). Neither uses a test
  document.
* **Protected test sets.** `groundedai-nist-v1` (53 questions on three NIST PDFs, including the
  four version-conflict questions t44-t47) and `groundedai-claims-nist-v1` (108 hand-written
  claims) were not opened while building. They were run once, after the new defaults were fixed
  and committed (commit `08ff4cc`), comparing the old defaults with the new ones at the same
  commit. Where a test result showed a weakness, the cause is reported here and was **not**
  fixed, because a fix chosen on the test set would make the test number meaningless.
* **Development history that matters.** One change to the version check (matching word forms
  when two editions' statements are aligned) was made after the first validation score had been
  seen. The first validation score is therefore the clean one; the later one is reported next to
  it and marked.
* **Statistics.** Where the same questions are answered twice, differences come with a 95%
  paired percentile bootstrap interval over questions (5,000 resamples, seed 13;
  `scripts/compare_rag_reports.py`). Small sets (7, 16 or 24 cases) get counts, not intervals.
  No significance claims are made.
* **Machine.** One laptop: Intel Core i7-1255U, 16 GB RAM, no GPU, Windows 11, on mains power.

## A. Version-aware disagreement

**Problem.** The test set has four questions (t44-t47) whose answer changed between NIST SP
800-63B (2017) and SP 800-63B-4 (2025). The earlier checks reported none of them.

**Root cause** (traced on the test questions, then reproduced on new fictional editions):

1. The checks had no idea of editions. Two documents that are versions of each other were
   compared like two unrelated sources, and when retrieval returned a passage from only one
   edition, there was nothing to compare.
2. Changes of requirement ("SHOULD" in one edition, "SHALL NOT" in the other) are not values,
   and the checks compared only values.
3. Asked words were too strict: an acronym such as "AAL2" was dropped as a name, and word forms
   ("reauthenticated" / "reauthentication") did not match.
4. Terminology changed between editions ("memorized secret" in 2017, "password" in 2025), so
   statements about the same thing share few words.

**Implementation** (`conflict_check=disagreement-v3`, the new default;
`backend/groundedai/generation/versions.py`, `disagreement_v3.py`):

* Editions are found from near-identical titles (at least 60% of the title words shared, after
  removing edition labels and years), with the edition label and date read from the title or the
  opening text ("Revision B, March 2024"). The AI RMF and its Generative AI Profile are, correctly,
  not treated as editions of each other.
* When two editions are in scope and only one was retrieved, the two best passages of the other
  edition are added (replacing the lowest-ranked passages of documents that have several).
* Statements of two editions are aligned when they share at least 30% of their word forms, and
  compared value by value (numbers with units, both directions) and by normative stance (RFC 2119
  words). Section numbers ("4 Visits") are not values. Statements about another level named in
  the question (AL1 when AL2 is asked) are left out.
* The result is a new kind, `different_versions`: both passages, both editions with their dates,
  pages and values. GroundedAI does not say which version applies. Pairs that are not editions are
  checked exactly as before (`disagreement-v1`).

**Tests.** `backend/tests/unit/test_reliability_v3.py` (values, requirements, units, the AL1
context, ranges, section numbers, names, the delegation to v1) and
`backend/tests/integration/test_versions_pipeline.py` (editions and dates read from PDFs, the
missing edition added, the response fields).

**Results** (found = the expected pair was reported with the expected values in the quotes):

| Set | disagreement-v1 | disagreement-v2 | disagreement-v3 |
|---|---|---|---|
| versions-v1 dev: found (of 11) / false alarms (of 17) | 5 / 1 | 6 / 1 (and 1 wrong report) | 10 / 0 |
| versions-v1 validation: found (of 7) / false alarms (of 17) | 1 / 0 | 1 / 1 | 4 / 0 first run; 6 / 0 after the word-form change (see above) |
| held-out cases `groundedai-conflicts-v1`: found (of 16) / kind right / false alarms (of 14) | 12 / 12 / 0 | 13 / 11 / 0 | 12 / 10 / 0 |
| NIST training documents: false alarms (of 242 questions) | 18 | 23 | 18 |
| NIST validation documents: false alarms (of 18) | 0 | – | 0 |
| **Protected test** `groundedai-nist-v1`: found (of 4) / false alarms (of 49) | 0 / 0 | – | **0 / 0** |

On the held-out cases v3 finds the same 12 as v1, but labels 2 of them `different_versions`
(their titles look like editions of one document), so "kind right" drops from 12 to 10.

**Why the test questions are still missed** (diagnosis only; nothing was changed for them):

| Question | Cause |
|---|---|
| t44 | the 2025 passage is not retrieved, and the editions use different terms ("memorized secret" / "password") |
| t45 | the two editions' sentences are worded too differently to be aligned |
| t46 | the 2025 sentence with "1 hour" is not among the passages added for that edition |
| t47 | aligned similarity 0.27, under the 0.30 threshold (terminology change, and "e.g." splits into the tokens "e" and "g") |

**Limitations.** The check works on fictional editions written for it and finds 0 of 4 real
version changes in the test set, so it is **partially resolved**. Aligning statements by shared
words cannot follow a change of terminology; that needs either a semantic similarity model or a
terminology map, and new validation editions to choose it on. Two editions are recognised only
from their titles.

## B. Added details (fabricated detail inside a matching statement)

**Problem.** A statement such as "A single ticket costs 14 euros and includes a free hot meal on
board" was shown as a lexical match when the passage only says the ticket costs 14 euros.

**Root cause.** The lexical check (`claims-v1`) accepts a statement when enough of its words and
all its figures are found. Words that appear in no passage do not change the result as long as
the share of found words stays above the threshold.

**Implementation (lexical).** `claim_detail_check=v1` (new default): a matching statement that
contains two or more consecutive content words found in no retrieved passage becomes `partial`,
with the flag `added_detail` and the words listed ("hot meal"). The UI shows them under the
statement and in "Checks that ran on this answer".

**Implementation (meaning).** A second, separate check with an entailment (NLI) model; see C.
It is reported per statement as *entailed*, *contradicted* or *not entailed*, next to the lexical
result and never merged into it. NLI is a model's judgement, not proof that a statement is true.

**Tests.** `test_reliability_v3.py` (added details make a matching claim partial; one unmatched
word is not an added detail; the check can be switched off), `meaning.test.tsx` (the meaning
check per statement and in the summary, never changing the lexical result).

**Results, lexical rule** (claim-level macro-F1 over supported / partial / contradicted / not
enough evidence; "false supported" = share of claims that are not supported but were marked
supported; "false not-supported" = share of supported claims marked otherwise):

| Set (claims) | claims-v1 | claims-v1 + added details (new default) |
|---|---|---|
| versions-v1 dev (24) | .397, false supported .294 | .470, false supported .118 |
| versions-v1 validation (16) | .454 | .454 (no change) |
| rule-made validation pairs (900) | .495, false supported .242 | .535, false supported .158 |
| synthetic, out of domain (153) | .524 | .532 |
| **Protected test** (108) | **.409**, false supported .078, false not-supported .226 | **.388**, false supported .078, false not-supported .323 |

On the hand-written test claims the rule **did not help**: it caught no additional unsupported
claim and marked 3 more of the 31 supported claims as partial (their paraphrases use words the
passage does not). This is reported as a negative result. The default was fixed before the test
run and is kept, because its error is to flag too much rather than too little; it can be switched
off with `GROUNDEDAI_CLAIM_DETAIL_CHECK=off`.

NLI results on the same claims: see C (macro-F1 .716, false supported .039).

## C. Meaning check (NLI) in the browser

**Problem.** The online demo had only the lexical check: a statement that uses the passage's words
but changes its meaning, or a correct paraphrase in other words, was judged on words alone.

**Implementation** (`frontend/src/nli/`, `frontend/src/browser/nliWorker.ts`, `nliModel.ts`):

* Model: `cross-encoder/nli-deberta-v3-xsmall` (Apache-2.0), zero-shot, as converted to ONNX and
  quantized to int8 by the transformers.js project (`Xenova/nli-deberta-v3-xsmall`, pinned to
  revision `2a4f614a`). It runs with ONNX Runtime Web 1.31.1 on WebAssembly, one thread (several
  threads need cross-origin isolation, which GitHub Pages does not provide), in its own Web
  Worker. **WebGPU is not used for this check**; it was not tested.
* It is off by default. Switching it on downloads the model files once from Hugging Face (kept in
  the browser's Cache Storage) and the runtime file from the demo site, with a progress bar.
  Switching it off stops the worker and frees its memory. If loading or a check fails, the answer
  keeps its lexical checks and says that the meaning check did not run; both generators work
  without it.
* Each statement is compared with the six sentence windows (one or two consecutive sentences) of
  the retrieved passages that share the most words with it, the SummaC zero-shot idea (Laban et
  al., 2022). A statement is *entailed* if a window entails it, else *contradicted* if a window
  contradicts it, else *not entailed*. The result is shown per statement ("Meaning check (NLI):
  entailed by S1 — quote") and as its own row in "Checks that ran on this answer";
  `checks_run` gets `meaning_nli` only when it ran. The lexical result is never changed.
* Questions and documents stay in the tab; the only network traffic is the model download.

**Tests.** `frontend/src/test/meaning.test.tsx` (10 tests: windows, the decision rule, outcomes per
statement, refusals left alone, the summary row, download progress, retry after a failed
download, stopping the worker). `scripts/bench_meaning_check.py` switches the check on in a real
Chromium and checks that the per-statement line and the summary row appear.

**Choice of six windows** (validation answers of qwen2.5:1.5b on `groundedai-nist-val-v1`,
16 answers of which 1 is wrong; chosen on validation data only):

| Check | wrong answers caught | correct answers flagged (of 15) | pairs per answer |
|---|---|---|---|
| lexical | 1 / 1 | 6 | – |
| NLI, every window (PyTorch or int8: same result) | 1 / 1 | 4 | 99.6 |
| NLI, 6 nearest windows (chosen) | 1 / 1 | 5 | 12.8 |

**Browser measurements** (local build, headless Chromium 145, the laptop above on mains power,
three runs with nothing else running; two earlier runs made while other programs used the CPU
were about twice as slow, 186-218 ms per pair, and are kept in the reports):

| Measure | Result |
|---|---|
| bytes received on first use | 104.2 MB: 87.3 MB model, 2.0 MB tokenizer (compressed in transit), 14.8 MB runtime file (served uncompressed by the local test server) |
| first load (download and start) | 3.7-4.0 s on this connection |
| load from the browser cache, new browser | 1.1-1.3 s; no model bytes downloaded again |
| time per premise/statement pair | median 95-99 ms (Python onnxruntime, same file, one thread: 30-33 ms) |
| one ferry question with the check on (6 pairs) | 0.86-0.90 s for the whole answer, 0.28-0.33 s of it for the check |
| memory of the page's renderer process (RSS) | about 408 MB before; 1,024 MB loaded; 909 MB after classifying; 515 MB after switching the check off |
| agreement with Python onnxruntime, 160 validation pairs | same top label on 159; median probability difference 0.0004; largest 0.21 |

The one disagreement is "Meters are read twice a year." against "Meters are read every 6 months."
(browser: entailment .41, contradiction .36; Python: contradiction .46, entailment .34), a case
where both are unsure; the int8 kernels of the two runtimes differ. Two settings were tried and
dropped: turning off ONNX Runtime's memory arena (same memory, slightly slower), and handing the
runtime file to ONNX Runtime as a buffer (the browser then cannot keep the compiled code: 2.4 s
instead of 1.6 s from the cache). These numbers are from the local build; the deployed demo is
checked in [Deployment check](#deployment-check) (functionally the same; timings there were taken on battery).

**Results on the protected sets** (run once; the int8 file the browser uses):

Claims (108 hand-written test claims; macro-F1 over four labels; "false supported" = share of the
77 claims that are not supported but were marked supported; "false not-supported" = share of the
31 supported claims marked otherwise):

| Check | macro-F1 | accuracy | false supported | false not-supported |
|---|---|---|---|---|
| lexical, claims-v1 + added details (default) | .388 | .435 | .078 | .323 |
| NLI, int8 ONNX | .716 | .741 | .039 | .129 |
| both: supported only if both agree, contradicted if either says so (rule fixed before the run) | .627 | .648 | .026 | .355 |

The full PyTorch model scored .723 on the same claims in the previous round.

Whole answers (an answer passes a check when all its statements pass; it is wrong when a key fact
is missing or it answers a question that should be refused):

| Answers | Check | wrong answers caught | correct answers flagged |
|---|---|---|---|
| qwen2.5:1.5b, new defaults (39 scored, 9 wrong) | lexical | 1 / 9 | 4 / 30 |
| | NLI, 6 windows | 5 / 9 | 6 / 30 |
| | both | 5 / 9 | 8 / 30 |
| extractive, new defaults (32 scored, 7 wrong) | lexical | 0 / 7 | 2 / 25 |
| | NLI, 6 windows | 0 / 7 | 2 / 25 |
| | both | 0 / 7 | 2 / 25 |

For the language model, the NLI check catches four more wrong answers than the lexical check, at
the cost of two more false alarms. For the extractive generator neither check catches anything:
its wrong answers copy passage sentences faithfully, so they are entailed; they are wrong because
they do not answer the question, which is a relevance problem (D), not a faithfulness one. Two
Qwen answers were left out because the rebuilt passages differed from the recorded ones, and 7
answers per run had no statements to score.

**Limitations.** NLI is a model's judgement, not proof: on the test claims it still marked 3 of
77 unsupported claims as supported and missed 4 of 31 supported ones. The demo shows it as a
second opinion next to the lexical check. About 0.6 GB of extra memory while it is loaded is a
real cost on small devices.

## D. Does the extractive answer address the question?

**Problem.** The extractive generator picked the passage sentence with the most words in common
with the question, even when the question asked for something no passage mentions. Answers to
unanswerable questions were shown with valid citations and matching statements.

**Root cause.** Sentence selection used every question word, including the document title's words
("AI RMF") and words such as "require", which reach the 0.5 overlap needed without the attribute
actually asked about ("hiring", "minimum accuracy").

**Implementation** (`extractive_relevance=v2`, new default; `generation/relevance.py`):

* Asked words leave out title words, the word after "how", generic question verbs and numbers.
* An asked word that appears in no retrieved passage is *unmatched*. The extractive generator then
  refuses and says which words are missing ("The sources do not mention "hiring", which the
  question asks about"). Sentences are scored without title words.
* Every answer, from either generator, reports `relevance` (asked words, unmatched words, how
  many asked words the answer covers) and a warning when words are unmatched. This is a lexical
  check: a passage can mention every asked word and still be about something else.

**Tests.** `test_reliability_v3.py` (relevance words, refusal, the "Error codes" heading
regression), `test_versions_pipeline.py` (refusal names the missing word).

**Results, validation sets** (extractive generator, old vs new defaults):

| Set | Measure | Old | New |
|---|---|---|---|
| versions-v1 + nist-val + nist-val-relevance (46 answerable, 20 to refuse) | refused, of 20 | 8 | 20 |
| | false refusals, of 46 | 1 | 7 |
| | correct answers, of 46 | 32 | 32 |
| groundedai-dev-v1 (25) | false-refusal rate | .53 | .32 |
| | abstention recall | .83 | 1.00 |
| | key-fact recall | .37 | .63 |
| | behaviour accuracy | .56 | .76 |
| groundedai-synthetic-v1 (56, run once with the test sets) | abstention recall / precision | .50 / .28 | .90 / .38 |
| | incorrect refusals | .30 | .35 |
| | behaviour accuracy | .66 | .70 |

**Results, protected test** `groundedai-nist-v1`, reported separately by question type:

| Question type | Measure | Extractive, old | Extractive, new | Qwen2.5-1.5B (Ollama), new |
|---|---|---|---|---|
| answerable (36 factual + 4 multi-passage) | correct | 29 / 40 (.725) | 25 / 40 (.625) | 31 / 40 (.775) |
| | false refusals | 3 | 10 | 0 |
| unanswerable (5) | refused | 1 | 3 | 4 |
| out of scope (1) | refused | 0 | 1 | 1 |
| ambiguous (3) | answered (no gold answer) | 3 | 3 | 3 |

Paired differences, extractive generator (new − old): correct answers −.100 (95% interval −.225
to 0; t14 gained; t03, t12, t26, t49, t50 lost), false refusals +.175 (+.075 to +.300), refusals of
the 6 questions to refuse +.50 (+.17 to +.83: t36, t38 and t40 are now refused). Abstention
precision rose from .25 to .29. Retrieval is unchanged (recall@5 .864, MRR@10 .792). For Qwen,
compared with the Experiment A run of 2026-11-13
(`evaluation/reports/research-2026-11/test/expA-ollama-qwen15.json`), correct answers did not
change (.775, no question gained or lost), false refusals went from 1 to 0, abstention precision
from .83 to 1.00, and the share of statements with a lexical match fell from .934 to .890.

**Cause of the new false refusals on the test** (diagnosis only): words that name the kind of
value asked for ("maximum", "month", "year", "longest", "kind", "long") count as asked words, and
a passage that gives the value ("8 characters") often does not contain them. Fixing this was not
done, because the cause was found on the test set; it needs new validation questions.

**Status: partially resolved.** Refusals of unanswerable questions improved on every set, but the
extractive generator now refuses too many answerable test questions.

## E. Test question t38

**Problem.** t38 ("What minimum accuracy does the AI RMF require for hiring models?") must be
refused: the AI RMF sets no such number. After Experiment A (2026-11-13) the extractive generator
answered it with a sentence about requirements in general.

**Root cause.** With text clean-up v2 and prefix matching, the title words ("AI", "RMF") and
"require"/"requirements" reached the sentence-overlap threshold without "hiring" or "accuracy".
It was reproduced on new validation questions of the same form
(`groundedai-nist-val-relevance-v1`: "What minimum accuracy does the Generative AI Profile require
for chatbots?") before anything was changed.

**Fix.** The relevance gate of D: "hiring" appears in no retrieved passage, so the question is
refused and the missing word is named.

**Result.** On the protected test run t38 is refused (and so are t36 and t40). Results by
question type are in the table in D; the cost is the higher false-refusal count reported there.

## F. LoRA fine-tuning of the answer model

**Status: blocked by hardware. No model was trained and no result is claimed.**

The machine has no GPU. A bounded smoke test was run instead, to check that the training script
works and to measure what a full run would cost (`--max-steps 8 --smoke`, a new option that saves
only a log; `experiments/finetune/train_lora.py`):

| Measure | Result |
|---|---|
| base model | Qwen/Qwen2.5-0.5B-Instruct, fp32, CPU, 8 threads |
| trainable parameters | 8,798,208 of 502,830,976 (LoRA rank 16 on the attention and MLP projections) |
| training examples | 257 of 259 (2 are longer than 1,792 tokens) |
| time per example (forward and backward) | 41.7-63.9 s, mean 52.0 s, for 948-1,458 tokens |
| peak memory | 9.4 GB of 15.6 GB |
| one epoch, extrapolated | about 3.7 hours of training, plus about half an hour of validation loss |
| reproducibility | a second run gave the same first three losses to four decimals (1.8717, 0.3765, 0.4854) |

The eight examples include one optimizer update, made at learning rate 0 (the first warm-up
step), so the validation loss could not change (0.9964 on two validation examples, before and
after). The smoke test shows that training runs, not that it learns. A full run was not started:
it would occupy the only machine for about four hours, and the comparison planned in
`docs/RESEARCH_EVALUATION.md` (Experiment C) adds two evaluation runs. On a CUDA GPU the epoch
should take minutes, but the script would first need the model and tensors moved to the device,
which has not been done or tested.

## G. Documents indexed with an older pipeline

**Problem.** Since text clean-up v2 became the default (2026-11-13), documents indexed before keep
their v1 chunks (running headers, hyphenated line breaks). Nothing showed which documents were
affected, and the only remedy was to re-ingest documents one by one.

**Root cause** (found while building the fix): re-ingestion starts a document over. If it fails,
the document becomes `failed` and loses its chunks. Worse, when the current settings find no text
in a file, re-ingestion treated the file as unreadable and **deleted the stored original**, even
though it had been indexed with the earlier settings. Clean-up v2 can do this: on pages whose
lines all sit at the page edges and differ only in their numbers, every line is removed as a
running header (shown by a test).

**Implementation.**

* Every ready document is compared with the current settings: its chunking record (chunk size,
  overlap, splitter, chunker version, text clean-up; a record without `text_cleanup` means v1)
  and its embedding model. `GET /api/v1/documents` lists the differences per document;
  `GET /api/v1/documents/reindex` gives counts, what differs, and how many documents lack their
  original file.
* Re-indexing (`POST /api/v1/documents/reindex`, the Documents page, or `groundedai reindex`)
  rebuilds outdated documents one at a time. A document stays `ready` and searchable with its
  previous chunks while the new ones are built; they replace the old ones in one transaction,
  under a row lock, and only if nothing else wrote the document meanwhile (otherwise the result is
  discarded). Document and chunk ids are kept. Any failure (parsing, empty text, embedding, the
  write, a missing file) keeps the previous index and the stored file.
* No job state has to survive a restart: what is outdated is read from the database, so starting
  again continues with what is left. A running job is never started twice in one process.
* Re-ingesting a document that was indexed before no longer deletes its stored file when the
  current settings find no text in it.
* The Documents page shows a notice with what differs, a "Re-index N documents" button, progress
  ("Re-indexing: 2 of 5 documents done · Now: …"), and after the run what failed and why. Each
  outdated row says "Indexed with older settings: text clean-up v1 → v2".

**Tests** (`backend/tests/integration/test_reindex.py`, 16 tests; a PostgreSQL test; a CLI test;
`frontend/src/test/reindex.test.tsx`, 4 tests):

| Case | Checked |
|---|---|
| detection | v1 records without the key, changed chunk size, other embedding model, no record |
| re-index result | equal to a fresh ingestion of the same file (ids, text, pages, page breaks, hashes, embeddings) |
| no duplicates or orphans | chunks = distinct chunk ids = full-text rows; the removed header text is no longer found |
| repeated run | the second run does nothing; `updated_at` unchanged |
| partial failure | embedding failure, empty text, failed write (rolled back), missing file: previous index kept |
| concurrent change | re-ingest or delete while a document is being re-indexed: nothing written, no orphans |
| mixed states | ready-current, ready-outdated, failed, pending, missing file: only the outdated ready ones with a file are touched |
| stop and restart | a job stopped in the middle; a new process on the same data continues with what is left |
| API, CLI, UI | status and job over HTTP, `groundedai reindex`, the notice, progress and the row note |
| PostgreSQL | the same, with the row lock, in CI and with a local server |

**On real PDFs** (`evaluation/reports/reliability-v3/reindex/`): the three NIST test PDFs were
indexed with clean-up v1 and re-indexed with `groundedai reindex` (13 s; 172 → 173, 468 → 448 and
367 → 306 chunks). The result was identical to a fresh v2 ingestion of the same files: 927 chunks
with the same ids, text hashes, pages, page breaks and embeddings, and 927 full-text rows in
both. A second run did nothing.

**Limitations.** Only recorded settings are compared; a change to the PDF parser that keeps them
the same is not detected. Progress is kept in memory: after a restart the page shows the
remaining outdated documents, not the history of the interrupted run. With several server
processes, two jobs could run at once; the row lock and the change check keep the data correct,
but work may be done twice.

## Deployment check

PRs #12 to #15 were merged into `main` on 2026-11-15 (merge commits; `main` at `0eb3eb5`). CI run
43761266292 on `main` passed all 8 jobs:
- backend on Python 3.12 and 3.13: 555 passed, 8 skipped each (the PostgreSQL tests, which run
  in their own job);
- PostgreSQL + pgvector: 8 passed;
- front end: lint, type check, 102 tests, both builds;
- online demo in a browser: 16 of 16;
- offline benchmark;
- dependency audits: no known vulnerabilities;
- Docker image and Compose file.

Pages run 43761256978 built, checked, deployed and verified the demo. On the live URL
(`https://pouyapd.github.io/GroundedAI/`, evidence in
`evaluation/reports/reliability-v3/live/` and `nli/browser/deployed-*`):

| Check | Result |
|---|---|
| `scripts/e2e_demo.py --url …` | 16 of 16 |
| `scripts/e2e_demo.py --url … --llm --channel chrome` (Qwen2.5-1.5B on WebGPU, Intel Iris Xe) | 19 of 19 |
| two fictional editions uploaded into the tab, "How quickly are reported leaks attended?" | reported as "Two versions of the same document differ" |
| a question about borrowing, asked about the ferry guide | refused, naming the words the passages do not mention |
| "Checks that ran on this answer" | every row present; meaning shown as not checked while the check is off |
| meaning check switched on, ferry question | "entailed by S1" per statement and in the summary row ([screenshot](images/demo-checks-meaning.png)) |
| meaning check: bytes received on first use | 93.2 MB: 87.3 MB model, 2.0 MB tokenizer and 3.9 MB runtime file, both compressed in transit by their servers |
| meaning check: agreement with Python onnxruntime, 160 validation pairs | same top label on 159 (as locally) |
| meaning check: memory of the page's renderer | 401-418 MB before; 897-1,013 MB loaded; about 850 MB after classifying; 450-458 MB after switching off |

**The deployed timings were measured on battery power** (the laptop was unplugged at that
point) and are not comparable with the local mains-power runs: 212-250 ms per pair, first load
4.9-6.7 s, load from the cache 2.5-4.2 s, 0.65-1.24 s for the 6 pairs of the ferry question;
Python onnxruntime on the same file was also 4 to 5 times slower than on mains power. Once, the
page did not finish starting within five minutes; a second load worked. The mains-power numbers
in C come from the local build of the same code.

## Reports and reproduction

The reports are in [`evaluation/reports/reliability-v3/`](../evaluation/reports/reliability-v3/README.md).
From the repository root (the NIST PDFs are downloaded and hash-checked first with
`python scripts/fetch_nist_corpus.py`; the fictional editions are rebuilt and checked with
`python scripts/build_versions_corpus.py --check`):

```bash
# A: version conflicts (dev / validation splits are selected by tag), held-out cases
python scripts/eval_disagreement.py --dataset evaluation/datasets/groundedai-versions-v1 --split dev --set conflict_check=disagreement-v3 --out dev-v3.json
python scripts/evaluate_conflict_cases.py --check disagreement-v3 --out heldout
# D/E: old vs new defaults on the protected test set, and the paired comparison
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-v1 --generator extractive --set extractive_relevance=v1 --set claim_detail_check=off --set conflict_check=disagreement-v1 --out old.json
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-v1 --generator extractive --out new.json
python scripts/compare_rag_reports.py old.json new.json
# B/C: claims (from experiments/verifier) and whole answers
python run_verifiers.py --sets test --method rule:claims-v1+detail --method onnx:../deploy/runs/xenova-nli-xsmall --method "both:rule:claims-v1+detail|onnx:../deploy/runs/xenova-nli-xsmall" --out claims.json
python experiments/verifier/answer_level.py --report new.json --dataset evaluation/datasets/groundedai-nist-v1 --premise window --max-windows 6 --onnx experiments/deploy/runs/xenova-nli-xsmall --out answer-level.json
# C: the browser (build the demo first, see docs/ONLINE_DEMO.md)
python scripts/bench_meaning_check.py --out bench.json
# F: the smoke test
python experiments/finetune/train_lora.py --train data/sft_train_balanced.jsonl --val data/sft_val.jsonl --out runs/smoke --max-steps 8 --val-limit 2 --smoke
# G: re-indexing
groundedai reindex --dry-run
```

The int8 model directory `experiments/deploy/runs/xenova-nli-xsmall` holds the four files of
`Xenova/nli-deberta-v3-xsmall` at revision `2a4f614a701367a02d51389039afc998faeda637`
(`config.json`, `tokenizer.json`, `tokenizer_config.json`, `onnx/model_quantized.onnx`), the
same files the browser downloads.

## Limitations of this round

* **Small, partly self-written data.** The fictional editions, the relevance questions and the
  hand-written claims were written by the same person who wrote the checks. Counts of 7, 16 or 24
  cases support only rough comparisons.
* **The real version changes are still missed** (A: 0 of 4 on the test set).
* **The relevance gate is lexical and refuses too often** on the test set (D: 10 false refusals
  of 40 with the extractive generator). The cause is known but was found on the test set, so it
  was not fixed.
* **The added-detail rule did not transfer** to the hand-written test claims (B).
* **The meaning check is a second opinion, not a verdict.** It helps most with a language model's
  answers; it cannot tell that a faithful quotation does not answer the question.
* **Single machine, CPU only.** Times are from one laptop; the browser numbers depend on the
  visitor's device. WebGPU was not used or tested for the meaning check.
* **LoRA (F) has no result.**
* **No independent judge.** Correctness is "every key fact present, as a string"; it can miss
  correct answers in other words and accept answers that contain the facts with extra wrong ones.
