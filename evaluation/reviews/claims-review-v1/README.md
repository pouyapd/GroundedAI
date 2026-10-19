# Claim review v1: real model answers for human labelling

**Status: no human labels yet.** Nothing in this folder measures accuracy until people have
labelled the statements. Do not quote verifier "accuracy" from it.

## What is here

| File | Content |
|---|---|
| `review.html` | open it in a browser: each question, the model's answer, its statements and the passages the model received; a label per statement and the supporting passages; *Download labels* saves `labels-<initials>.json`. It never shows what the checks said |
| `answers.json` | the same content as data, including the model's raw output |
| `checks.json` | what `claims-v1` and `claims-v2` said about each statement, the conflicts reported and the warnings. **Do not open it before labelling** |

Collected on 2026-10-19 with `python scripts/claim_review.py collect --out
evaluation/reviews/claims-review-v1` at commit `cca6670` (clean tree): qwen2.5:1.5b through
Ollama 0.14.2, the application's default retrieval (hashing embedder, hybrid, top 5,
800/120-character chunks), prompt answer-v2, disagreement-v1. The laptop was on battery; no
timing is reported from this run.

Dataset: [`groundedai-claims-review-v1`](../../datasets/groundedai-claims-review-v1) (four
invented documents about two fictional organisations, 30 questions: 24 to answer, 4 to refuse,
2 with conflicting sources). 56 statements to label from 25 answers; 2 replies were
bracket-only (invalid output) and 3 questions were refused (one of them answerable).

## How to label

Labellers follow [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md): two people, independently, each in
their own browser profile; open `review.html`, label every statement from the passages shown,
tick the supporting passages, *Download labels*. Then:

```bash
python scripts/claim_review.py score --run evaluation/reviews/claims-review-v1 \
  --labels labels-a.json labels-b.json [--adjudication labels-agreed.json]
```

`score` reports for each check version: false acceptance (accepted, people say not supported),
false rejection (not accepted, people say supported) and whether the quoted evidence is among
the passages people ticked. With two label files it scores only the statements both labellers
agree on, lists the disputed ones, and reports how often the two agree; an adjudication file
settles disputed statements only. Conflict detection is compared with the dataset's own
`expected_behavior`, written by the dataset author, not by labellers.

`python scripts/claim_review.py page --run evaluation/reviews/claims-review-v1` rebuilds
`review.html` from `answers.json` without a model (done on 2026-10-19 to add the notice about
labels restored from the browser and the *Clear all labels* button; the answers are unchanged).

## How independent this is, honestly

- The documents and questions were written after the checks were frozen at `7082970`, by the
  same assistant that wrote the checks. The answers and statements come from the model.
- A first collection on the same questions found two bugs, fixed before this collection
  (`92a46f5`): conflict reports that paired one organisation's prices with another
  organisation's, and a full answer wrapped in brackets hidden as invalid output. For these two
  error types the set is no longer blind.
- While preparing the release the author read the answers and noticed one wrong statement that
  both versions accept: "Rides lasting longer than 45 minutes cost an additional £1 per minute"
  (the terms say £1 per 15 minutes; the check does not compare the unit a rate is per). It was
  left unfixed so the set keeps measuring it.
- Nothing else was tuned on this set.
