# Reliability audit: citations, grounding, conflicts and retrieval

Dates: 2026-10-13 (audit) and 2026-10-14 (changes and measurements). No language model was run
for any number in this document: the portable Ollama build and its model had been deleted, and
the results below either re-check outputs recorded on 2026-10-13 or use the deterministic
extractive baseline. What still needs a model run is listed at the end.

## How the audit was done

The 2026-10-13 runs stored each model's raw output. For every question, the exact passages the
model saw (S1–S5) were rebuilt from the dataset with the run's settings (400/60-character chunks,
bge-small embeddings, hybrid retrieval, top 5, 6,000-character context). The rebuild reproduced
the recorded citations, refusals, grounding and evidence coverage of all 56 questions of every
run, so the checks below were applied to exactly what the model had received.

## Findings (root causes)

### A. Citations and grounding (`backend/groundedai/generation/citations.py`)

1. Only `[S#]` counted as a citation. qwen2.5:1.5b used it in 13 of its 49 answers; 24 had no
   reference and 12 wrote `[Source 1]` or `[Source S1]`. Those answers were shown as "Not
   grounded" even when they repeated a passage word for word (q001).
2. Uncited sentences were never compared with any passage. 93 of 107 uncited sentences in the
   qwen2.5:1.5b answer-v2 run were supported word for word by a retrieved passage.
3. The support check had false negatives: the "1" of `[Source 1]` and the "5" of "SX-5" counted as
   invented figures (q001, q007, q012, q044); attribution phrases and document-title words lowered
   the overlap (q052).
4. Labels written before the text (`[S1]: … [S2]: …`) were credited to the previous sentence
   (q056).
5. A citation to the wrong passage and an invented claim both appeared as "unsupported".
6. Evidence coverage counted only cited passages, so correct uncited answers scored 0.

### B. Answers to questions that should be refused

The three wrong answers of the qwen2.5:1.5b answer-v2 run to unanswerable questions (q041 price,
q042 person, q044 access code) consist of true statements from the passages that do not answer
the question. A claim check cannot catch them: the claims are supported. Not addressed in this
round (no development data of this kind; see "Needs a model run").

### C. Conflicts (`backend/groundedai/generation/conflicts.py`)

On the rebuilt contexts the figure check found 1 of the 4 conflicts in the evaluation and
development sets and raised false alarms on 8 of the 73 other questions. Causes: only digits
counted ("three times" vs "twice" and "no fixed schedule" vs "every 90 days" were invisible);
"nightly" did not match "night"; any two shared words (including names like "Kestrel Valley
Rail") made sentences comparable; years and dates ("2024 edition") counted as values; units were
not compared (€300 stipend vs 28 days of leave); different dates were not distinguished from
contradictions; and the only output was a warning and a prompt note.

### D. Retrieval

BM25 alone reached Recall@5 0.965 and MRR 0.934 on the frozen benchmark; hybrid RRF the same
Recall@5 and MRR 0.915. 41 of 43 questions were in the top 5 for both, so the benchmark cannot
separate the methods. The published numbers used 400/60-character chunks and 30 candidates; the
application ships with 800/120 and 20, which had never been measured. The default "dense"
retriever is the hashing embedder (word overlap, not meaning), but the UI called it embedding
similarity search.

### E. Static demo

The demo replays recordings of the extractive generator (labelled). Any change to the response
had to be re-recorded, or the demo would show an older response format.

## What changed

| Area | Change | Where |
|---|---|---|
| Claim check | Every factual sentence is checked against every retrieved passage: all figures (with units, number words, times, dates; codes like SX-5 are not figures) and 70% of the content words must be in one passage, document titles excluded. Status per claim: supported, wrong_source, added_source, partial, unsupported. Each claim carries the exact quote with character offsets, document, page and chunk. Variant citation forms are read and counted separately; labels before text are handled; instruction-like passage text is never evidence. | `generation/claims.py`, `core/quantities.py` |
| Uncertainty | The response has a verdict (verified, partially_verified, unverified) and plain-language warnings. With `GROUNDEDAI_UNVERIFIED_ANSWERS=withhold` an answer with no matched statement is returned as insufficient evidence, with the model's text kept in `unverified_answer` (default: label only, see below). | `services/qa_service.py` |
| Disagreements | New check `disagreement-v1`: compares values of the kind the question asks for, only in sentences about the asked property, from different documents; classifies agreement, different context (group, plan, day, class, place), different dates and conflict, including a requirement and its denial. Both quotes are returned with values and dates; the answer is never "resolved". The published figure check remains selectable (`GROUNDEDAI_CONFLICT_CHECK=figures-v1`). | `generation/disagreement.py` |
| Evaluation | Scoring version 4 (claim match, cited-claim precision, key facts traced to the gold evidence, disagreements surfaced and false alarms); runs record the check versions and the embedding query prefix; a replay tool re-checks recorded outputs; a held-out case file for disagreements. | `evaluation/generation_eval.py`, `scripts/revalidate_recorded_answers.py`, `scripts/evaluate_conflict_cases.py` |
| UI | Verdict badge, claim list with quotes, disagreement panel, the strict check kept for comparison, a note when dense retrieval uses the hashing embedder, scoring-version-4 tiles. Static demo re-recorded. | `frontend/src` |
| Tests | Unit tests for quantities, claims and disagreements; API tests for the new fields; a test that restarts the application three times on one data directory. | `backend/tests` |

The default for unverified answers was chosen on development data: withholding them would have
turned 16 wrong answers into refusals but also 3 correct ones across the seven development runs,
and the only one it would have changed in the frozen-prompt qwen2.5:1.5b run was correct.

## Results

### Disagreement check on held-out cases

`evaluation/datasets/groundedai-conflicts-v1`: 30 cases committed (f0008b8) before the check was
written, run once (`evaluation/reports/conflicts/conflict-cases-20261014T073843Z.md`).

| Check | Disagreements found (of 16) | Kind also right | False alarms (of 14) |
|---|---|---|---|
| disagreement-v1 (new) | 12 (8 of 10 conflicts, 4 of 6 different dates) | 12 | 0 |
| figures-v1 (published) | 6 | – | 3 |

The four misses: synonyms kept one statement below the topic threshold ("bag" / "baggage",
"ask" / "request"); two visiting-hour ranges share the end time 20:00, and any shared value
counts as agreement; one FAQ sentence matches too few of the question's words. The same author
wrote the cases and the check, so this is not an independent test.

On the 56-question evaluation set (hybrid, top 5) the new check reports the hotel-limit conflict
q052 (as values for different dates, because the FAQ passage says "updated May 2025"), misses the
password conflict q053 (the "every 90 days" passage was not retrieved, and the stemmer does not
join "changed" and "change"), and raises one false alarm, on the ambiguous q050 ("What is the
limit?"). The figure check had found neither conflict and raised two false alarms. Both
evaluation-set conflicts had been inspected in this audit, so this is not a blind result.

### Recorded model outputs re-checked (no model run)

Replays of the 2026-10-13 runs through the current checks
(`evaluation/reports/published/revalidated-*`). The strict `[S#]` rate is unchanged; the
claim-level numbers are new.

| Run (56 questions) | Answers | Strict: every sentence `[S#]`-cited | Statements found in a passage | Model citations pointing to the right passage | Answers with every statement found | Key facts traced to gold evidence |
|---|---|---|---|---|---|---|
| qwen2.5:1.5b, answer-v2 | 49 | 14% | 98% (116 of 118) | 88% | 96% (47) | 90% |
| qwen2.5:1.5b, answer-v1 | 37 | 11% | 86% | 90% | 81% (30) | 86% |
| Qwen2.5-0.5B, answer-v2 | 25 | 20% | 72% | 79% | 60% (15) | 57% |

What the numbers do and do not say:

- The 1.5B model's answers are mostly drawn from the passages; its problem is the citation
  format, not invented content. The check now attaches the evidence it did not cite.
- "Found in a passage" is lexical. The three wrong answers to unanswerable questions (q041, q042,
  q044) are "verified": their statements are true, they just do not answer the question.
- False negatives exist: "The minimum password length … is 14 characters long" (q021) is only
  "partial" because the passage says "at least 14 characters".
- Four questions in each answer-v2 run (q002, q006, q050, q052) would get a different note from
  the new disagreement check, so a new model run could answer them differently.

### Extractive baseline, each retrieval mode (current pipeline)

| Retrieval (56 questions) | Correct behaviour | Evidence coverage | Key facts | Conflict questions surfaced | False disagreement alarms |
|---|---|---|---|---|---|
| Lexical (BM25) | 0.660 | 0.663 | 0.558 | 0 of 2 | 1 of 54 |
| Dense (bge-small) | 0.660 | 0.663 | 0.581 | 1 of 2 | 1 of 54 |
| Hybrid (RRF) | 0.660 | 0.663 | 0.558 | 1 of 2 | 1 of 54 |

Every quoted claim of the baseline is matched (claim match 100%), as it must be. Lexical
retrieval misses q052's second document in its top 5, so the disagreement cannot be shown.

### Retrieval at the application's default settings

`evaluation/configs/retrieval_app_defaults.json` (800/120-character chunks, 41 chunks, 20
candidates), same 43 questions.

| Configuration | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| BM25 | 1.000 | 0.985 | 0.990 |
| Hashing vectors (default embedder) | 0.942 | 0.915 | 0.910 |
| Hybrid RRF with hashing (the shipped default) | 0.977 | 0.950 | 0.956 |
| bge-small | 0.954 | 0.884 | 0.892 |
| Hybrid RRF with bge-small | 1.000 | 0.959 | 0.969 |

No configuration beats BM25; hybrid with bge-small has a slightly lower nDCG@5 (−0.021, interval
excludes 0). No retrieval change was made: none improved a measured metric. The default hybrid
mode was kept because the benchmark questions reuse the documents' wording and cannot show where
dense retrieval helps; whether to default to BM25 is left open.

## Evaluation integrity

- The claim check and the disagreement check were developed on the development set's recorded
  outputs and retrieval contexts, and on new unit tests. Their thresholds are the earlier ones
  (70% / 40% word overlap); no threshold was tuned on the evaluation set.
- The held-out disagreement cases were committed before the check existed and were run once;
  their file is hash-checked. After the PR review (below), their inputs were used once more only
  to confirm that a wording fix left the check's output unchanged: the old and new code were
  compared with each other on all 30 cases, not scored against the expected results.
- The evaluation set's failures were inspected during the audit; results on its conflict
  questions are therefore not blind.
- Every new report records the commit (all clean), the check versions and the explicit settings;
  no earlier report was changed. The reports of 2026-10-14 do **not** record the pipeline settings
  that the shell environment could change (`candidate_k`, `rrf_k`, `hybrid_alpha`,
  `rerank_candidates`, `unverified_answers`, `max_question_chars`). The commands logged for those
  runs set none of them and the machine's user and system environment defines none, but the
  reports themselves cannot show it. Reports written after the fix below record them.

## Fixes after the PR review (2026-10-14)

- **Conflict text.** For a passage that requires something ("must") against one that says it is
  not required, the explanation read "S1 says , S2 says there is no such requirement" and the UI
  showed an empty "Value:". It now reads "S1 says it is required" and the UI shows "Value:
  required" / "Value: not required"; the API marks such a statement with `affirms_requirement`.
  Which passages are paired and how they are classified did not change (old and new code compared
  on the 30 held-out cases: identical pairs and kinds; one explanation changed, from the broken
  text to the fixed one). None of the published reports or demo recordings contained the broken
  text.
- **Settings recorded with each evaluation.** `GROUNDEDAI_*` environment variables could change
  what an answer evaluation did without appearing in its report. Every answer report now records
  the effective value of every setting that can change results, and the names of the variables
  that were set; `groundedai eval generation --settings-from REPORT.json` runs again with exactly
  those values. Defaults and results without such variables are unchanged.

## Known issues

- Lexical checks: paraphrases can be marked partial ("minimum length" / "at least"); true
  statements that do not answer the question are verified; negation and wrong subjects are only
  caught when they change the overlap.
- Disagreement check: misses synonyms and conflicts with a shared range end; the light stemmer
  does not join "changed" and "change"; one-sided dates are reported as "different dates".
- Changing the stemmer or any threshold now would be tuning after seeing the held-out and
  evaluation results; it needs new held-out cases first.
- The server sends `index.html` without `Cache-Control`, so a browser can show an old UI after an
  upgrade until it reloads (seen once during the browser check).

## Needs a model run (not possible on 2026-10-14)

- qwen2.5:1.5b (or another model) with the new disagreement notes, to see whether it then reports
  both sides; the recorded outputs answer the old notes.
- A check for answers that state true facts but not the requested item (price, person, code),
  built on new held-out questions first.
- Prompt work on the `[S#]` format (now less critical, since evidence is attached either way).
