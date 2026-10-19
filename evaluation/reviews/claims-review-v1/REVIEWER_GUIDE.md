# Labelling guide: claim review v1

For two people who label the statements in `review.html` independently. It takes about one
hour for 56 statements. No knowledge of GroundedAI is needed.

## Before you start

- Work alone. Do not discuss statements with the other labeller until both files are saved.
- Do not open `checks.json`, `answers.json`, the dataset's `questions.jsonl` or the
  documentation of the checks. They contain what the software decided, or what the dataset
  author expected.
- Use your own browser or browser profile. The page keeps your progress in the browser; if it
  says that labels were restored and they are not yours, press *Clear all labels*.
- Type your initials (letters only) at the top. They are used only in the file name.

## For every statement

Read the statement and the passages under *Passages the model received*. Decide from those
passages only, not from what you know or what seems likely.

| Label | Use it when |
|---|---|
| **Supported** | the passages state it; other words are fine ("no charge" = "free") |
| **Partly supported** | one part is stated and another part is not, or is stated differently |
| **Not supported** | the passages do not state it, or they say something else |
| **Unclear** | you cannot decide, for example because the passages are ambiguous |

Then tick every passage (S1, S2 …) that states the statement, or the supported part of it.
Leave the boxes empty for *Not supported*.

Rules for common cases:

- **Numbers, prices, dates and times** must match exactly, including the unit and what they are
  per ("€3 per hour" does not support "€3 per day").
- **Names** of organisations, places and products must match. A fact about one organisation
  does not support the same claim about another.
- **Conditions matter.** "Members over 65 ride free on Sundays" does not support "members ride
  free on Sundays".
- **Two passages that disagree.** If the statement repeats one of them, it is *Supported* by
  that passage; tick only that one.
- **Judge the statement, not the answer.** Do not mark down a statement because the answer is
  unhelpful, too long, or has no citation.
- Use the comment box for anything you were unsure about. Comments are not scored.

## When you have finished

1. Press *Download labels*. The file is `labels-<initials>.json`.
2. Give the file to the person running the review. Keep a copy.
3. After both files exist, the two labellers may compare the statements listed under
   `disagreements` in the score output and agree on a label. Write the agreed labels in the
   same format to `labels-agreed.json` (only the disputed statements are read from it).

## Scoring (for the person running the review)

```bash
python scripts/claim_review.py score --run evaluation/reviews/claims-review-v1 \
  --labels labels-a.json labels-b.json            # agreed statements only
python scripts/claim_review.py score --run evaluation/reviews/claims-review-v1 \
  --labels labels-a.json labels-b.json --adjudication labels-agreed.json
```

Report both runs, the agreement rate, the number of disputed and *unclear* statements, and
that the documents and questions were written by the same assistant that wrote the checks (see
`README.md`). With 56 statements, a single statement moves a rate by about two percentage
points or more.

Label files contain only labels, passage names, initials and comments. Decide before
committing them whether the initials and comments may become public.
