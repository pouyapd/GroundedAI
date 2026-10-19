# groundedai-conflicts-v1 (held-out cases for the disagreement check)

30 small cases for testing how GroundedAI handles sources that give different values. Each case
is a question and 2–3 short passages from different documents, as the model would receive them
after retrieval. No retrieval or language model is involved: the cases test the deterministic
check on its own.

| Class | Cases | What the passages contain | Expected result |
|---|---|---|---|
| `conflict` | 10 | Different values for the same thing, same scope, no dates that separate them (digits, number words, different units, times of day, a requirement and its negation) | reported as a **conflict** between the two passages in `expected_pair` |
| `temporal` | 6 | Different values that apply to different dates (effective dates, editions, "until", "since", dated price lists) | reported as **values for different dates** between the passages in `expected_pair` |
| `context` | 6 | Different values for different groups or situations (students vs adults, free vs Pro plan, weekdays vs Saturdays) | **nothing reported** |
| `agreement` | 4 | The same value written differently ("08:00" and "8 am", "two" and "2", "€500" and "500 euros") | **nothing reported** |
| `incomplete` | 4 | One passage gives a value, the other mentions the topic without one | **nothing reported** |

## How it was used

* The file was written and committed **before** the check it tests
  (`backend/groundedai/generation/disagreement.py`) existed. The git history shows the order.
  Development used the unit tests and the development set `groundedai-dev-v1` instead.
* The check is run on these cases once, by `scripts/evaluate_conflict_cases.py`, and the result
  is reported whatever it is. If the check changes afterwards, the new result is reported next to
  the old one; the cases are not edited to fit the check.
* `manifest.json` records the SHA-256 of `cases.jsonl`; the script refuses to run if it differs.

## Limits

* The same author wrote the cases and the check, so the cases are not independent of the
  author's ideas about what a conflict is. Some cases are deliberately hard (unit conversion,
  "twice a year" vs "every 3 months") and are expected to fail.
* 30 cases cannot give precise rates; they show which kinds of disagreement the check handles.

Written 2026-10-13. Licence: MIT, like the rest of the repository.
