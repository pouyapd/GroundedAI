# Label audit, round 1 (2026-10-16)

Counts only. The per-case records (the page, the sample, the answers, the disagreements and
their adjudication) are private and never committed. They stay on the owner's machine: in a
local staging folder until the freeze, then in `audit/private/`.

## What was audited

- Draft test split: SHA-256 `e6d184a1bb8ccdc9592e8ce32a2d902e7c899d04b8f303949c04b75b830980b7`
  (462 cases; the draft is archived privately as it was).
- Sample: 50 test cases, stratified as A 14, B 9, C 12, D 15, drawn with seed 20261015 by
  `scripts/reliability_suite.py audit-prepare` (commit `495041c`).
  - Page SHA-256: `48535397a68886efb2c839a34dbd49a566f001ceaa0f465e48aaa2300163d963`.
  - Sample file SHA-256: `3f8b575925a630c7991e39755a05744b8ee9b4fc202a69407e5dac357a7d4933`.
- The page showed each case's question, passages and statement or answer, with a short list of
  rules per group. It showed no proposed label, subtype, case id or system output.

## The answers

- File: `blind_answers.json`, saved by the page on 2026-10-16 (2,823 bytes).
  - SHA-256: `db2dbf382573a6dc8948717deeb2a3f03e31f299382d6cb3d45441055014ba1c`.
  - The original is kept unchanged; a copy was processed.
- 50 entries, keyed 1–50; no duplicate keys, no unknown keys, no invalid labels, no notes.
- One entry had no label (audit number 43, group D). Under the protocol, it counts as not
  confirming the proposed label.

## Agreement with the proposed labels

| Group | Agree | Cases |
|---|---|---|
| A (statement support) | 14 | 14 |
| B (citations) | 9 | 9 |
| C (does the answer address the question) | 9 | 12 |
| D (disagreement between passages) | 7 | 15 |
| **All** | **39** | **50** |

- Agreement: 39 of 50, 78% (95% Wilson interval 0.65–0.87).
- Among the 49 labelled cases, 39 agree (80%).
- The protocol requires at least 45 of 50, so the threshold was **not met**.

This measures the labels, not the checks. No version of the checks has been scored on the test
split.

## Why the labels disagreed

A separate agent adjudicated the 11 disagreements. It takes no part in building the checks.
For each case, it first labelled the case itself under `SPEC.md` and under the page's short
rules, and only then read both labels and the writer's rationale. Its per-case record is
private.

| Cause | C | D |
|---|---|---|
| The page's short rules were unclear on the deciding point; the case is clear under `SPEC.md` | 3 | 3 |
| Slip: the rules were explicit for the case | 0 | 4 |
| No label chosen (the case is clear under both rule sets) | 0 | 1 |
| The proposed label is wrong under `SPEC.md` | 0 | 0 |
| `SPEC.md` itself does not settle the case | 0 | 0 |

Rule-level findings:

- **C.** The off-target option, "the passages contain the answer", is easily read as "the
  passages contain what the answer says". That is also true of an answer to a question the
  passages cannot answer. Neither the page nor the spec said that the same kind of fact for
  another plan, period, group or item is not the asked item.
- **D, ranges.** In the spec's subtype table, two ranges that share one end and differ at the
  other are a conflict. Rule 8 and the page mentioned only "a value equal to one end of a range"
  (no disagreement).
- **D, dates.** Rule 8 and the page described the date rules as if dates alone decided the
  label. Only the subtype table said the values must differ first: the same value in documents
  with different dates is no disagreement.
- **Extent.** These points decide every case of three subtypes: `unanswerable_answered` (21
  test, 9 dev), `range_shared_endpoint` (14 test, 6 dev) and `dated_agreement` (7 test, 3 dev).
  The adjudicator found the proposed label correct in all of them, so no case was relabelled.

The adjudicator is of the same model family as the writer, so its agreement with the writer is
weaker evidence than a human's would be.

## Decisions

- **The rules are clarified.** Rules 7a and 8a, under "Clarifications after the first label
  audit" in `SPEC.md`, restate the subtype tables and change no label. The page's short rules
  and option texts were rewritten to match.
- **The 11 unconfirmed cases are replaced.** That is 3 in C and 8 in D, including the
  unlabelled one. The protocol replaces a case whose proposed label was not confirmed, whatever
  the adjudication found. Keeping a case on the strength of another model's judgment would treat
  the writer's label as correct without the auditor's confirmation.
  - A new writer agent (same model family) wrote cases of the same subtypes, ids and domains
    from the clarified specification. All of its own checks passed.
    - File SHA-256: `4a7b1fd033d6ef4a411e871bfda916c71137be7ea7cdf87c10ab3c31b475c5d0`.
    - Its 14 decision notes are kept privately with the writer's other notes.
    - It also ran read-only helper scripts on the files it was allowed to read, beyond its
      instruction to run programs only to write, check and hash its own files. It reports
      opening nothing else.
  - `replace-cases` applied them and logged 11 entries in `REVISIONS.jsonl` (id, old and new
    hash, reason; no subtype or label).
  - Only those 11 lines changed (3 in `test/C.jsonl`, 8 in `test/D.jsonl`). Validation:
    0 errors, 0 warnings, exact counts.
  - Hashes after the replacements (draft 2):

    | Split | SHA-256 |
    |---|---|
    | test | `7b0c8840cfd223579937803b51ab8233565dedac2cba6bbdce03860a5e166913` |
    | dev (unchanged) | `176aae44a5fcf614d26c4619b04b1a2fdead463a65ccf3cb78f448e4aab81b9a` |
    | combined | `f953f796e72588ed778f3a35fcc2d1c0218f39a2a81ed861c172b191172751f6` |
- **Round 2.**
  - It audits a fresh random sample of 50 test cases with the same stratification (seed
    20261016), excluding the 50 round-1 cases, plus the 11 replacements shuffled in: 61 cases.
  - The random sample alone is held to the 45-of-50 threshold; each replacement must agree.
  - Page SHA-256: `de2b532286b13e06ebf0ff3a7fdc52089c8d95fbb38a5c0f4fa7c7eb7a2ffa45`.
  - Sample file SHA-256: `50b996e00a6b961169ef531189aa2bb569042578ac80e56899bc27803906cadc`.
  - A check of the page found no case id, label, subtype or rationale text in any case.
- **The test split is frozen only after round 2 meets the threshold.**
- **The auditor's round-1 answers are kept unchanged.**

## Tool fixes found during this round

- The reveal page showed the unlabelled case as if "No disagreement" had been chosen. A missing
  label was stored as `none`, which is that label's value in group D (fixed in `5729738`).
- The blind page reported unanswered cases only as a count. It now names and marks them when
  saving (`89b5037`).
- `replace-cases` first logged each replaced case's subtype, which implies its label. The log
  now leaves it out, and the 11 entries were rewritten without it (`cde29a1`).
