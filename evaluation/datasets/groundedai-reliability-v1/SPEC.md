# groundedai-reliability-v1: case-generation specification

This is the complete instruction given to the separate agent that writes the cases. It says
nothing about how GroundedAI's checks work, on purpose. The coordinating agent wrote it before
any case existed; the scoring rules are in `README.md` and in
`backend/groundedai/evaluation/reliability.py`, which the generator never sees.

---

## Task

You write evaluation cases for a system that answers questions from short document passages and
cites them as `[S1]`, `[S2]`. A separate checking component will be scored on your cases. You are
not told how it works and must not try to find out.

## Limits

- Work only inside the output folder you are given. Do not open, list or search any other folder,
  repository, source code, test file or report. Do not use the internet.
- Run programs only to write, split, validate and hash your own files in the output folder.
- Invent all organisations, people, products and places. Never use real companies or real people.
- Never use these names, or names close to them: Aldervale, Solvane, Brindlewick, Orion, Tessera,
  Kestrel Valley, Fenwick, Halden, Quillon, Amberly, Ashby, Brightwing, Brookfield, Calder,
  Cobalt Print, Corvid, Elmstead, Fairlight, Greyfield, Harlow, Kelso, Larkspur, Lumen,
  Marrow Lane, Nimbus, Orbis, Ostend, Pine Hollow, Port Ellis, Quarry Hill, Redfern, Saltmarsh,
  St Ansel, Tamsin, Tolland, Vellum, Westbrook, Wren House.
- Your final message lists only files, counts, hashes and the number of notes you wrote. Never
  quote a case, passage or label in it.

## Domains

| Split | Domains |
|---|---|
| dev | municipal waste collection; language school; marina berths; childcare nursery |
| test | university exam rules; veterinary clinic; car-sharing service; dental practice; ferry operator; laboratory safety; pension fund member guide; theatre ticketing |

Development cases use only development domains; test cases use only test domains. Nothing
(document, passage, organisation, person) is shared between the splits.

## Documents and passages (categories A, B and C)

- Write two documents per domain (8 dev, 16 test), for example a policy or handbook and an FAQ or
  notice from the same invented organisation. Each is 600–1,200 words of plain text; the first
  line is the title.
- Include many concrete facts: amounts (several currencies and formats), percentages, durations,
  deadlines, times of day, dates, counts, codes and reference numbers, roles and named people,
  limits and thresholds, conditions and exceptions, ranges ("09:00–17:30", "3 to 5 working days").
- Some documents say when they apply ("Valid from 1 March 2026"), some say when they were last
  edited ("Last updated: 12 May 2025"); most say neither. The two documents of a domain agree
  with each other.
- Save each as `documents/<split>/<document_id>.md`, with ids such as `dev-waste-1`,
  `test-ferry-2`.
- Split every document into passages and write them to `documents/<split>/passages.jsonl`, one
  object per line: `{"passage_id": "P-T-001", "document_id": "test-ferry-2", "text": "..."}`. A
  passage is a verbatim excerpt of its document, 300–800 characters, starting and ending at a
  sentence boundary; consecutive passages do not overlap. Passage ids are `P-D-###` (dev) and
  `P-T-###` (test). The title line is not part of any passage.

A case in A, B or C shows 3–5 passages, labelled `S1`, `S2`, … in the order listed. Include
related passages that do not contain the answer, and vary the position of the key passage.

## Categories

### A: statement support

A case has a question, passages and a statement of 1–2 sentences about the question's topic,
cited the way an AI assistant might cite it. Gold: `supported` and `supporting_labels`.

| Subtype | Meaning | Gold `supported` |
|---|---|---|
| `verbatim` | close to the passage's own wording | true |
| `paraphrase` | the same meaning in other words, word forms or sentence structure | true |
| `number_form` | the same value written differently (unit, number word, currency format, 12/24-hour time) | true |
| `wrong_value` | one value changed: number, unit, date, or a limit's direction ("at least" → "at most") | false |
| `wrong_subject` | a true fact attached to the wrong group, item or person | false |
| `negation` | the passage's meaning reversed | false |
| `absent` | plausible, but stated in no passage | false |

When the statement is supported, cite the supporting passage(s) and list them in
`supporting_labels`. When it is not supported, cite the passage it seems to come from and leave
`supporting_labels` empty.

### B: citations

Every statement is fully supported by the passages; only the citation labels vary. Gold:
`citation_correct` and `supporting_labels` (the passages that do support it). Also give
`cited_labels`: the labels written in the statement.

| Subtype | Meaning | Gold `citation_correct` |
|---|---|---|
| `correct_single` | one label, the passage that states it | true |
| `correct_two` | two parts, each stated in a different passage; both cited | true |
| `wrong_related` | cites a passage on the same topic that does not state it | false |
| `wrong_unrelated` | cites an unrelated passage | false |
| `one_of_two_wrong` | two labels: one passage states it, the other does not | false |
| `nonexistent_label` | cites a label that is not among the passages (e.g. `[S7]` with 4 passages) | false |

### C: does the answer address the question?

A case has a question, passages and a full answer (1–4 sentences with citations, or a refusal).
Gold: `answer_kind`, `answerable` and `question_type`.

| Subtype | Meaning | `answer_kind` | `answerable` |
|---|---|---|---|
| `correct_value` | states the asked item; the question asks for a value or a person | `correct_answer` | true |
| `correct_other` | answers a yes/no, how or why question correctly | `correct_answer` | true |
| `off_target` | the passages contain the asked item, but the answer states only other true facts | `off_target` | true |
| `unanswerable_answered` | the passages do not contain the asked item; the answer states related true facts as if answering | `unanswerable_answered` | false |
| `correct_refusal` | the passages do not contain the asked item; the answer clearly says the documents do not give it | `correct_refusal` | false |

`question_type` is one of `amount`, `duration`, `date`, `time`, `count`, `person_or_role`,
`code`, `place` (for `correct_value`, and allowed for the other subtypes) or `yes_no`,
`procedure`, `reason` (for `correct_other`, and allowed for the other subtypes). In
`off_target` and `unanswerable_answered` cases every statement must be true according to the
passages.

### D: conflicts

Each case defines 2–4 short invented documents (100–800 characters each) on a domain of its
split and shows passages from at least two of them. Put dates in titles or text the way real
documents do. Gold: `disagreement`, `kind` and `pair` (the labels of the two disagreeing
passages, or `null`).

| Subtype | Meaning | `disagreement` | `kind` |
|---|---|---|---|
| `plain_conflict` | different values for the same thing, same group and period; nothing explains it | true | `conflict` |
| `synonym_conflict` | as above, but the two documents name the thing with different words ("fee" / "charge") | true | `conflict` |
| `range_shared_endpoint` | two ranges with one endpoint equal and the other different | true | `conflict` |
| `requirement_denial` | one says something is required, the other says it is not | true | `conflict` |
| `revision_date_only` | different values; the documents only say when they were last edited, or one gives no date | true | `conflict` |
| `validity_dates` | different values, and the documents say different periods apply ("from 1 March 2026", "2025 price list") | true | `different_dates` |
| `agreement` | the same value written differently | false | `none` |
| `different_groups` | different values for clearly different groups, plans, days, places or classes | false | `none` |
| `incomplete` | only one document gives a value; the other mentions the topic without one, or another property | false | `none` |
| `range_consistent` | a range and a single value equal to one of its ends, or the same range written two ways | false | `none` |
| `dated_agreement` | dated documents that give the same value | false | `none` |

## Annotation rules (apply exactly)

1. Judge only from the passages, never from outside knowledge.
2. Supported means one passage states every factual part of the statement. A compound
   statement is supported if each part is stated in some passage.
3. Unit conversions, number words and formats count as the same value ("two weeks" = "14 days",
   "€1,200" = "1200 euros", "8 am" = "08:00"). Any other arithmetic (sums, differences) does not.
4. Limit words carry meaning: "minimum 14" = "at least 14"; "at least 14" ≠ "at most 14" ≠
   "exactly 14". Dropping or reversing a limit word makes a statement unsupported. Avoid
   borderline cases.
5. Any unsupported part makes the whole statement unsupported.
6. A citation is correct when every cited label points to a passage that states part of the
   statement, and the cited passages together state all of it.
7. An answer addresses the question when it states the asked item: the value or person for value
   questions, a clear yes or no for yes/no questions, the steps or the reason for how and why
   questions.
8. A disagreement is two passages from different documents that give incompatible information
   about the same thing for the same group and period. Different stated periods → `different_dates`.
   Only edit dates, or a date on one side only → `conflict`. Different groups or plans, the same
   value written differently, a value equal to one end of a range, and a topic with no value on
   one side → `none`.
9. Every case has exactly one defensible label. If two labels are reasonable, rewrite or replace
   the case.

### Clarifications after the first label audit (added 2026-10-16)

The first blind label audit found no label that contradicts the subtype tables above. It did
find that rules 7 and 8 were easy to misread. The rules below restate the subtype tables and
change no label. Where the wording of rule 7 or 8 seems to differ from them, they decide.
Cases written after the audit follow them. All examples are invented.

7a. **Is the asked item in the passages?** Decide this before reading the answer. Does any passage
    state the asked item for exactly what the question names: the same plan, period, group or
    item?
    - If one does, an answer that gives only other facts is `off_target`.
    - If none does, the case is `unanswerable_answered`, or `correct_refusal` when the answer says
      the documents do not give it. This holds even when the passages give the same kind of fact
      for something else.
    - Example: "How much is the museum's audio guide?" The passages give only entry prices, and
      the answer is "Entry costs £11.80" → `unanswerable_answered`. If a passage gave the audio
      guide's price, the same answer would be `off_target`.

8a. **Disagreement, in two steps.**
    1. **Do the two passages, from different documents, give different information about the
       same thing for the same group?**
       - The same thing may be named with different words ("fee" / "charge").
       - "Required" against "not required" is different information.
       - They do not differ, and the label is `none`, when:
         - they concern different groups, plans, days, places or classes;
         - they give the same value or the same range in other words or formats;
         - one gives a single value equal to one end of the other's range;
         - one gives a one-sided limit ("at least 3", "by 15:45") equal to the matching end;
         - only one of them gives a value;
         - they give the same value although their documents carry different dates.
       - Compare only the ends that both passages state. Two ranges that differ at either end do
         differ, even when one lies inside the other ("60–90 minutes" / "75–90 minutes").
    2. **Only if they differ, look at the dates.**
       - If both documents say which dates their information applies to ("valid from 1 March
         2026", "2025 price list"), and those dates differ → `different_dates`.
       - Otherwise → `conflict`. This covers only edit dates ("last updated"), a date on one side
         only, and no dates.
       - Here "dates" means the dates a document applies to, not a duration that is itself the
         value.
    - Examples:

      | Passage 1 | Passage 2 | Label |
      |---|---|---|
      | "2025 list: £11.80" | "2026 list: 11.80 pounds" | `none` |
      | "2025 list: £11.80" | "2026 list: £12.60" | `different_dates` |
      | "Last updated 2025: £11.80" | "Last updated 2026: £12.60" | `conflict` |
      | "Last tour ends by 15:45" | "Tours run 13:15–15:45" | `none` |
      | "Tour takes 60–90 minutes" | "Tour takes 75 to 90 minutes" | `conflict` |

## Quality

- No repeated questions anywhere in the dataset; no templates. Vary length, wording and number
  formats.
- Write statements and answers as AI assistants do, with mixed citation styles: after sentences,
  at the start of a sentence, or (where the category allows) none.
- Give every case a one-sentence `rationale` explaining its label.
- Write down every decision you had to make where these rules were unclear, in your notes file.

## Counts (exact)

| Category | Subtype | dev | test |
|---|---|---|---|
| A | verbatim | 6 | 14 |
| A | paraphrase | 15 | 35 |
| A | number_form | 6 | 14 |
| A | wrong_value | 9 | 21 |
| A | wrong_subject | 6 | 14 |
| A | negation | 6 | 14 |
| A | absent | 6 | 14 |
| B | correct_single | 12 | 28 |
| B | correct_two | 6 | 14 |
| B | wrong_related | 9 | 21 |
| B | wrong_unrelated | 3 | 7 |
| B | one_of_two_wrong | 3 | 7 |
| B | nonexistent_label | 3 | 7 |
| C | correct_value | 12 | 28 |
| C | correct_other | 9 | 21 |
| C | off_target | 9 | 21 |
| C | unanswerable_answered | 9 | 21 |
| C | correct_refusal | 9 | 21 |
| D | plain_conflict | 6 | 14 |
| D | synonym_conflict | 6 | 14 |
| D | range_shared_endpoint | 6 | 14 |
| D | requirement_denial | 6 | 14 |
| D | revision_date_only | 3 | 7 |
| D | validity_dates | 6 | 14 |
| D | agreement | 6 | 14 |
| D | different_groups | 6 | 14 |
| D | incomplete | 6 | 14 |
| D | range_consistent | 6 | 14 |
| D | dated_agreement | 3 | 7 |

Totals: A 54 / 126, B 36 / 84, C 48 / 112, D 60 / 140; dev 198, test 462, all 660.

## File formats

All files are UTF-8 with LF line endings. JSONL files hold one JSON object per line.

Case files: `dev/A.jsonl`, `dev/B.jsonl`, `dev/C.jsonl`, `dev/D.jsonl` and the same under `test/`.
Case ids are `<category>-<D|T>-<number>`, e.g. `A-T-001`, numbered from 001 within each file.

Common fields: `id`, `category`, `split` (`dev` or `test`), `subtype`, `domain` (as written in
the domain table), `question`, `passages`, `gold`, `rationale`.

A, B and C: `passages` is a list of `{"label": "S1", "passage_id": "P-T-017"}` in label order
(S1, S2, … without gaps); every passage comes from the case's split.

```json
{"id": "A-T-001", "category": "A", "split": "test", "subtype": "paraphrase", "domain": "ferry operator",
 "question": "...", "passages": [{"label": "S1", "passage_id": "P-T-017"}, {"label": "S2", "passage_id": "P-T-040"}, {"label": "S3", "passage_id": "P-T-041"}],
 "statement": "... [S2]", "gold": {"supported": true, "supporting_labels": ["S2"]}, "rationale": "..."}
{"id": "B-T-001", "category": "B", "split": "test", "subtype": "wrong_related", "domain": "dental practice",
 "question": "...", "passages": [...], "statement": "... [S1]", "cited_labels": ["S1"],
 "gold": {"citation_correct": false, "supporting_labels": ["S3"]}, "rationale": "..."}
{"id": "C-T-001", "category": "C", "split": "test", "subtype": "off_target", "domain": "veterinary clinic",
 "question": "...", "passages": [...], "answer": "... [S2]",
 "gold": {"answer_kind": "off_target", "answerable": true, "question_type": "amount"}, "rationale": "..."}
```

D: `passages` is a list of `{"label": "S1", "document_id": "D-T-001-a", "title": "...", "text": "..."}`
in label order, from at least two different `document_id`s.

```json
{"id": "D-T-001", "category": "D", "split": "test", "subtype": "plain_conflict", "domain": "theatre ticketing",
 "question": "...", "passages": [{"label": "S1", "document_id": "D-T-001-a", "title": "...", "text": "..."}, ...],
 "gold": {"disagreement": true, "kind": "conflict", "pair": ["S1", "S3"]}, "rationale": "..."}
```

Notes: `notes/<category or documents>.md`, each decision as a short bullet.

## Hashes

For every file, SHA-256 of its bytes after replacing CRLF with LF. Write `manifest.json` with
`files` (relative path → hash, for every file except the manifest), the counts per category,
subtype and split, the list of documents (`document_id`, `split`, `domain`, `title`), the random
seed you used (if any), and `combined_sha256`: the SHA-256 of the text made of one line
`<relative path>\t<hash>\n` per file, sorted by path, with `/` as the separator.
