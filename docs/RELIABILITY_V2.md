# Answer checks, version 2: evaluation on groundedai-reliability-v1

Version 2 of GroundedAI's deterministic answer checks (`claims-v2`, `disagreement-v2`) compared
with version 1 on the suite `groundedai-reliability-v1`
([README](../evaluation/datasets/groundedai-reliability-v1/README.md)). No language model was
run: every case has fixed passages and a fixed statement or answer.

## What changed in version 2

| Area | Version 2 |
|---|---|
| Separate results | `evidence`, `citations` and `addresses_question` verdicts next to the unchanged `verdict`; a citation status per statement |
| Word forms | `claim_stem_v2` (the Porter inflection steps, written in-house): change/changed, apply/applied, cancel/cancelled; plan/plane stay apart. `light_stem` is unchanged and pinned by a test |
| Limit words | "at least 14" must not be matched with "at most 14" ("minimum" = "at least", "must not exceed" = "at most") |
| Negation | a statement whose negation differs from its best evidence sentence is only partly found (kept because it helped on the dev split) |
| Ranges | compared as a whole: "08:00–20:00" and "08:00–18:00" differ; a value at one end agrees |
| Dates | validity dates ("2025 season", "valid from …") are kept apart from edit dates ("last updated …"); `different_dates` needs differing validity dates on both sides |
| Malformed replies (2026-10-18) | a reply made only of bracketed fragments (`[21]`, document titles) is `invalid_output`, not an answer; bracketed refusals stay refusals |
| Caching | `index.html` and demo data are revalidated; hashed assets are cached for a year; demo data requests carry the export time |

Version 1 stays the default and selectable, and its behaviour is unchanged: its predictions on
all 198 development cases are identical before and after the change.

## Evaluation protocol actually used

The suite's original protocol froze the test split only after a blind human audit of 50 test
labels reached 45 agreements. That did not happen, and the auditor is not available for more
labelling. The protocol used instead:

1. **Human evidence kept as it is.** Round 1 agreed on 39 of 50 (A 14/14, B 9/9, C 9/12, D 7/15),
   below the threshold; see [`audit/ROUND1.md`](../evaluation/datasets/groundedai-reliability-v1/audit/ROUND1.md).
   An independent adjudication found no proposed label wrong under the specification; the
   disagreements came from unclear audit-page rules (6), slips (4) and one unanswered case. The
   11 unconfirmed cases were replaced. Round 2 was prepared but **not performed**.
2. **Replacements reviewed.** A separate agent checked the 11 replacements against the
   specification and the source passages: 11 valid, 0 invalid, 0 ambiguous.
3. **Blind model review of the whole test split.** Agents running a second large language
   model (a smaller model of the same family as the one that wrote the cases) labelled all 462 test cases from anonymised files without
   ids, subtypes, labels or rationales. Their transcripts show they opened only their own
   file and the rules. They agreed with the writer's labels on **462 of 462**.
4. **Frozen before any scoring** (commit `babca1b`): test split SHA-256
   `7b0c8840cfd223579937803b51ab8233565dedac2cba6bbdce03860a5e166913`, pinned in
   `reliability.py`; the manifest records the label status.
5. **Two views, one run each per version.** The full test split (462 cases, model-written labels)
   and the **human-confirmed subset** (the 39 round-1 cases whose blind human label agreed; id
   list SHA-256 `8d4206288174adcdecda52113c807c5a0c99799a796facf2f676f0c5fa0ab6a0`). Version 1
   was scored once at the freeze commit (`56151b1` records it), and version 2 once with the final
   code (`4d1e361`). All tuning used the development split; the test split was never inspected
   case by case.

**What this does and does not establish.** The test labels are model-written. A second model
reproducing all of them shows the labels are consistent with the written rules, but models of
one family can share blind spots, and agreement between them is not human verification. The
pre-registered human threshold was **not met**; the only human-verified test labels are the 39
cases of the subset, too few for precise rates.

## Results on the frozen test split

Measures with k/n and 95% Wilson intervals. "Dev" shows the same measure on the development
split (tuning data, v1 → v2).

| Measure | Version 1 (baseline) | Version 2 (final) | Dev v1 → v2 |
|---|---|---|---|
| A: unsupported recall | 37/63 = 0.59 (0.46–0.70) | 44/63 = 0.70 (0.58–0.80) | 0.70 → 0.85 |
| A: unsupported precision | 37/55 = 0.67 (0.54–0.78) | 44/66 = 0.67 (0.55–0.77) | 0.68 → 0.74 |
| A: supported false flag rate | 18/63 = 0.29 (0.19–0.41) | 22/63 = 0.35 (0.24–0.47) | 0.33 → 0.30 |
| A: paraphrase acceptance | 22/35 = 0.63 (0.46–0.77) | 18/35 = 0.51 (0.36–0.67) | 0.47 → 0.53 |
| B: correct citation precision | 41/48 = 0.85 (0.73–0.93) | 39/39 = 1.00 (0.91–1.00) | 0.84 → 1.00 |
| B: correct citation recall | 41/42 = 0.98 (0.88–1.00) | 39/42 = 0.93 (0.81–0.98) | 0.89 → 0.89 |
| B: wrong citation recall | 35/42 = 0.83 (0.69–0.92) | 42/42 = 1.00 (0.92–1.00) | 0.83 → 1.00 |
| C: off target recall | 0/42 = 0.00 (0.00–0.08) | 29/42 = 0.69 (0.54–0.81) | 0.00 → 0.78 |
| C: false warning rate | 0/49 = 0.00 (0.00–0.07) | 3/49 = 0.06 (0.02–0.17) | 0.00 → 0.10 |
| C: refusal recognition | 2/21 = 0.10 (0.03–0.29) | 9/21 = 0.43 (0.24–0.63) | 0.00 → 0.89 |
| D: disagreement recall | 29/77 = 0.38 (0.28–0.49) | 36/77 = 0.47 (0.36–0.58) | 0.27 → 0.45 |
| D: disagreement precision | 29/35 = 0.83 (0.67–0.92) | 36/42 = 0.86 (0.72–0.93) | 0.90 → 0.94 |
| D: false alarm rate | 6/63 = 0.10 (0.04–0.19) | 6/63 = 0.10 (0.04–0.19) | 0.04 → 0.04 |
| D: kind accuracy | 23/29 = 0.79 (0.62–0.90) | 36/36 = 1.00 (0.90–1.00) | 0.89 → 1.00 |

Cases handled correctly, with the paired exact McNemar test on the same cases:

| Category | v1 | v2 | Fixed / broke | p | Human-confirmed subset, v1 → v2 |
|---|---|---|---|---|---|
| A | 82/126 | 85/126 | 7 / 4 | 0.55 | 11/14 → 9/14 |
| B | 76/84 | 81/84 | 7 / 2 | 0.18 | 8/9 → 9/9 |
| C | 51/112 | 84/112 | 36 / 3 | < 0.0001 | 8/9 → 8/9 |
| D | 86/140 | 93/140 | 7 / 0 | 0.016 | 5/7 → 5/7 |

Reports: `evaluation/reports/reliability/test-v1-20261018T075535Z` and
`test-v2-20261018T134933Z` (`.json`, `.md`), development reports in `…/reliability/dev/`, and
the run log `test-runs.jsonl`. Per-case outcomes are sealed in a private, git-ignored folder.

## Reading the results

- **Clear gains:** C (from no answer check at all to 69% of off-target answers caught, with 6%
  false warnings), B (every wrong citation caught; the "one of two citations wrong" case is now
  seen) and D kind accuracy (validity vs edit dates). D recall rose with ranges compared as a
  whole (7 cases fixed, none broken).
- **No gain or worse on A:** unsupported recall rose (0.59 → 0.70), but paraphrase acceptance
  fell (0.63 → 0.51) and false flags rose. On the dev split both moved the other way, so the
  limit-word and negation rules generalise less well than dev suggested.
- **Refusal recognition generalised poorly:** 0.89 on dev, 0.43 on test. The recogniser's
  phrases fit the development answers better than the test ones.
- **Targets:** of the 14 measures, version 2 reaches the 80% target (or the 20% ceiling) on 7; it
  is *established* (interval bound past the target) on 6. A recall, A precision, A false flags,
  A paraphrase, C off-target recall, C refusal and D recall stay below target.
- **Not fixed by design (out of this round's scope; partly addressed by the revision of
  2026-10-19 below):** synonyms ("fee" / "charge"), missing
  denial phrases ("no booking is needed"), statements dropped by the topic filter, an
  incidental shared value counting as agreement, "€3 million" and ordinals ("10th") not read as
  numbers, and wrong-subject statements.

## Revision of 2026-10-19 (development split only)

Version 2 was revised after the qwen2.5:1.5b runs of 2026-10-19 showed three kinds of error:
correct paraphrases not confirmed ("do not have to pay" for "travel free"), claims "supported"
by words scattered over a passage ("bring" in one sentence, "dog" in another), and conflicts
between documents that the check could not see. The changes (commit `7082970`) are opt-in, like
all of version 2; version 1 and every default are unchanged.

| Area | Revised version 2 |
|---|---|
| Support | A claim is compared with the few sentences of one passage that carry it: the best sentence, plus at most two more, each adding at least two of the claim's words (or a missing figure and one word). The quotes are those sentences, word for word. An FAQ question is evidence only together with its answer |
| Wording | A short list of equivalent phrases and words (`generation/equivalents.py`): "do not have to pay" / "at no cost" / "free of charge"; fee / charge / cost / price; annual / yearly; and a few others |
| Negation | Compared part by part for every quoted sentence, after the rewrite; a condition ("if you have not paid") is not a negation; an FAQ question answered "No" is |
| Names and figures | A name in the claim ("the Minnow class") must be in the evidence or its document's title; ordinals ("the 10th") are figures; a figure from the question is exempt only inside a condition; a date without a year matches when the evidence gives the year |
| Labels | A bracketed reference that names no passage ("[Source A]", a title that is not one of the documents) is an invalid citation, and its words are not part of the claim |
| Conflicts (`disagreement-v2`) | Equivalent words; "no X is needed", "not compulsory", "compulsory"; FAQ questions are not statements unless answered "No"; the question's "need" / "must" / "compulsory" is not an asked word; a statement may have two fewer asked words than the best one (one before), and a requirement statement needs two |
| Answers | With claims-v2, a warning when the passages conflict and the answer does not say so |

**Development split, version 2 before → after** (198 cases; the labels were written by models,
see "Evaluation protocol actually used"):

| Measure | Before (`b80a13a`) | After (`7082970`) |
|---|---|---|
| A. unsupported statements caught (recall) | 23 / 27 | 25 / 27 |
| A. unsupported statements accepted (false negatives) | 4 / 27 | 2 / 27 |
| A. supported statements flagged (false flags) | 8 / 27 | 5 / 27 |
| A. paraphrases accepted | 8 / 15 | 10 / 15 |
| A. all cases right | 42 / 54 | 47 / 54 |
| B. all cases right | 34 / 36 | 34 / 36 |
| C. all cases right | 41 / 48 | 41 / 48 |
| D. disagreements found (recall) | 15 / 33 | 22 / 33 |
| D. reports that are real disagreements (precision) | 15 / 16 | 22 / 24 |
| D. false alarms | 1 / 27 | 2 / 27 |
| D. kind right | 15 / 15 | 22 / 22 |
| D. all cases right | 41 / 60 | 47 / 60 |

Version 1 gives the same result in every case before and after.

**What this does not show.** The changes were designed against the development split and the
journey, and three choices were made by measuring on it: the "two added words" rule for further
sentences, the conflict check's looser topic limit, and which equivalent words to list. The
development numbers are therefore optimistic. The frozen test split was **not** rescored: the
2026-10-18 test results above describe version 2 as it was then (code `b5fe83b`), and the
previous round showed that development gains on statement support did not carry over to the
test split. A fair test of this revision needs a new, versioned test set.

Still wrong on the development split: two wrong-subject statements are accepted (a person named
in the other clause of the same sentence; a price attached to the wrong pontoons in a sentence
that gives both prices), five paraphrases are not confirmed ("split the fee into three
payments" for "pay in three instalments", "billed" for "pay"), two false alarms pair values for
different groups whose sentences do not name the group (two berth sizes; two courses), and 11 of
33 disagreements are missed.

## Limitations

- Model-written labels; the human-audit threshold was not met; same-model-family review.
- Synthetic cases chosen around known failure modes: rates are not real-world rates.
- Lexical, pattern-based checks: good scores do not show that they understand meaning.
- Category E (end-to-end answers from a real model) is not part of this suite; no model was run.
- `disagreement-v2` stays opt-in until a real-model run checks its effect on the prompt notes.
- If these results call for more tuning, the next round needs a new, versioned test set.
