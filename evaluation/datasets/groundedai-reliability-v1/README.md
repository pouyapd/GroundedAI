# groundedai-reliability-v1

An evaluation suite for GroundedAI's answer checks, written to be independent of them. It
measures four tasks on fixed passages and fixed answers, so no language model is needed:

| Category | Task | Positive class |
|---|---|---|
| A | Is a statement supported by the passages? | unsupported statement |
| B | Are the citation labels in a statement right? | wrong citation |
| C | Does an answer address the question? | answer that does not |
| D | Do passages from different documents disagree? | real disagreement |

660 cases: 198 development and 462 held-out test cases, with exact counts per subtype in
[`SPEC.md`](SPEC.md). End-to-end behaviour with a real model (category E) is not part of this
version.

## How the suite is made

1. **Specification first.** `SPEC.md` (task, domains, formats, annotation rules, counts) was
   committed before any case existed, together with the validator and the scorer
   (`backend/groundedai/evaluation/reliability.py`, `scripts/reliability_suite.py`).
2. **Separate generator.** A separate agent writes the documents and cases from `SPEC.md` alone, in
   its own folder, without access to this repository (source, tests, reports or rules). It
   receives no description of how the checks work. This relies on its instructions; it is not
   technically enforced. The generator is the same model family as the author of the checks, so
   shared blind spots are possible.
3. **New material only.** New invented organisations in twelve new domains; development and test
   use different domains and share no document. None of the 14 documents or 56 questions of
   `groundedai-synthetic-v1`, the development set or the 30 conflict cases is reused.
   Previously inspected examples (such as q053) are regression tests in `backend/tests`, never
   evaluation data.
4. **Validation.** `python scripts/reliability_suite.py validate` checks JSON, ids, labels,
   passage references, that every passage is a verbatim excerpt of its document, split
   separation, duplicate questions, exact counts and hashes. It never prints case text.
5. **Blind label audit.** 50 test cases (A 14, B 9, C 12, D 15; seed 20261015) are labelled by
   Pouya on a page that shows no proposed labels, subtypes, ids or system output. Proposed labels
   are revealed only for disagreements. The protocol: at least 45 of 50 must agree, otherwise
   the rules are clarified and a fresh sample is audited; a case whose proposed label is not
   confirmed is replaced (same subtype) and the replacement is audited blind. In a later round
   the replacements are shuffled into the fresh sample but scored apart: the threshold applies
   to the random sample, and each replacement must agree. Counts go to `audit/`; per-case files
   stay in `audit/private/`, which is not committed.
   - Round 1 (2026-10-16): 39 of 50 agreed, below the threshold. The disagreements came from the
     page's short rules, not from wrong labels, so the rules were clarified (`SPEC.md`, rules 7a
     and 8a) and the 11 unconfirmed cases replaced. See [`audit/ROUND1.md`](audit/ROUND1.md).
     Round 2 (61 cases: a fresh sample of 50 and the 11 replacements) is ready for the auditor.
6. **Freeze.** The test split's hash is recorded in `manifest.json` and pinned in
   `reliability.py` (`FROZEN_TEST_SHA256`); the scorer refuses any other content.
7. **Amended protocol (2026-10-18).** The auditor was not available for round 2, so the
   45-of-50 human threshold was **not met**. The test split (SHA-256 `7b0c8840…`) was frozen on
   other evidence, recorded under `label_status` in `manifest.json`: the 11 replacements were
   reviewed (11 valid), and a blind review of all 462 test cases by a second model agreed with
   every label. Labels are model-written, not human-verified. Each version of the checks is
   reported on the full split and on the 39-case human-confirmed subset; see
   [`docs/RELIABILITY_V2.md`](../../../docs/RELIABILITY_V2.md).

## Rules for using the test split

- The coordinating (implementing) agent does not open held-out cases. The scorer prints counts,
  rates and intervals only; per-case outcomes are written as `{id: {category, correct}}` to
  `evaluation/reports/reliability/sealed/` for the paired comparison.
- Held-out material is never committed: `test/`, `documents/test/`, the writer's `notes/`, the
  per-case audit files in `audit/private/` and the sealed outcomes are listed in `.gitignore` and
  exist only on the owner's machine. The committed `manifest.json` and `FROZEN_TEST_SHA256`
  record their exact content, so a copy can be checked, but they cannot be rebuilt from the
  repository: keep a private backup. The committed manifest lists paths, hashes and counts
  only, without the test documents' titles. Committed audit records and reports hold counts
  only, and `REVISIONS.jsonl` names replaced cases without their subtypes.
- The test split is scored **once per version of the checks**: once with the code at freeze time
  (baseline) and once with the final, frozen implementation. Every run is logged in
  `evaluation/reports/reliability/test-runs.jsonl`; a second run of the same version is refused
  unless a reason is given, which is logged.
- All debugging and tuning uses the development split.
- If the final results call for more tuning, the next round needs a new, versioned test set.

## How each task is scored

Each case's passages are labelled and assembled exactly as the application does
(`assemble_context`), then one version of the checks runs.

| Task | Prediction (version 1 = the checks at freeze time) | Prediction (version 2) |
|---|---|---|
| A | flagged unless every statement is found in a passage (statuses supported, wrong source, found but not cited) | flagged unless the evidence verdict is "all found" |
| B | judged correct when no cited label is invalid and every cited statement is supported by its cited passage | judged correct when the citation verdict is "all correct" |
| C | version 1 has no check of whether an answer addresses the question, so it never flags; a refusal is recognised by the answer parser | flagged when the answer check says the answer does not give the asked item, or the answer is a refusal |
| D | the disagreement check's reports; detected when a report names the gold pair | the same, with version 2 of the disagreement check |

"Partly found" counts as flagged in A (it is not confirmed).

## Measures and targets

Every measure is reported with k, n and a 95% Wilson interval. There is no combined score.
*Met* means the point estimate reaches the target; *established* means the interval's lower
bound (upper bound, for rates that should be low) does too. 80% is the target to investigate and
90% the stretch; neither is a promised result.

| Measure | Definition | Test n | Target (stretch) |
|---|---|---|---|
| A unsupported recall | flagged ÷ unsupported | 63 | ≥ 0.80 (0.90) |
| A unsupported precision | unsupported ÷ flagged | varies | ≥ 0.80 (0.90) |
| A false-flag rate | supported flagged ÷ supported | 63 | ≤ 0.20 (0.10) |
| A paraphrase acceptance | accepted ÷ supported paraphrases | 35 | ≥ 0.80 (0.90) |
| B correct-citation precision | truly correct ÷ judged correct | varies | ≥ 0.80 (0.90) |
| B correct-citation recall | judged correct ÷ truly correct | 42 | ≥ 0.80 (0.90) |
| B wrong-citation recall | judged wrong ÷ truly wrong | 42 | ≥ 0.80 (0.90) |
| C off-target recall | flagged ÷ (off target + unanswerable answered) | 42 | ≥ 0.80 (0.90) |
| C false-warning rate | correct answers flagged ÷ correct answers | 49 | ≤ 0.20 (0.10) |
| C refusal recognition | recognised ÷ correct refusals | 21 | ≥ 0.80 (0.90) |
| D disagreement recall | gold pair reported ÷ real disagreements | 77 | ≥ 0.80 (0.90) |
| D disagreement precision | cases with the gold pair reported ÷ cases with any report | varies | ≥ 0.80 (0.90) |
| D false-alarm rate | cases with a report ÷ cases without a disagreement | 63 | ≤ 0.20 (0.10) |
| D conflict-type accuracy | right kind ÷ detected disagreements | ≤ 77 | ≥ 0.80 (0.90) |

F1, false-negative rates and per-subtype rates are reported as well; subtypes have 7–35 test
cases each and are descriptive only. The baseline and final runs are compared on the same cases
with an exact McNemar test (cases fixed and broken, per category).

## Limitations

- Synthetic and written by one generator; label errors are estimated by the audit.
- Subtypes were chosen around known failure modes, so rates over the suite are not real-world
  rates. Synonym conflicts are included although this round does not address them.
- Lexical and pattern-based checks: good results do not show that they understand meaning, and
  results here do not prove generalisation to other documents.
- No language model was run.
