# groundedai-synthetic-v1

A small, synthetic dataset for testing retrieval and grounded answering. All documents
describe **fictional** organisations (Aldervale Systems, Brindlewick, Orion ML Platform,
Tessera API, Solvane SX-5). They were written for this project and contain no real company,
personal or copyrighted text. Licence: MIT, like the rest of the repository.

## Contents

* `corpus/`: 14 documents: 12 plain-text files, 1 Markdown file and 1 four-page PDF
  (`solvane-sx5-manual.pdf`, generated from `source/solvane-sx5-manual.txt` by
  `evaluation/scripts/build_dataset.py`).
* `questions.jsonl`: 56 questions.
* `manifest.json`: document keys, titles and SHA-256 hashes. `load_dataset` refuses to run if a
  file changed, and reports a dataset fingerprint in every benchmark report.

| Category | n | Expected behaviour |
|---|---|---|
| factual | 30 | answer (3 of them are marked as paraphrases with less word overlap) |
| multi_chunk | 5 | answer using two passages of one document |
| synthesis | 3 | answer using passages from two documents |
| unanswerable | 7 | abstain (topic present, fact absent) |
| out_of_scope | 3 | abstain (general knowledge or a non-document request) |
| ambiguous | 3 | answer or ask for clarification (informational only) |
| conflicting | 2 | answer and cite both conflicting sources |
| injection | 3 | answer correctly while a retrieved document contains planted instructions |

Deliberate traps: the travel policy and the finance FAQ give different hotel limits; the remote
work guidelines contain outdated password advice that contradicts the security standard; two
documents contain prompt-injection text; several documents share the town name "Brindlewick";
error codes (E01–E12) test exact lexical matching.

## Record format

```json
{"id": "q052", "category": "conflicting", "question": "...", "expected_behavior": "flag_conflict",
 "evidence": [{"doc": "aldervale-travel-policy", "quote": "<exact sentence from the document>"}],
 "answer_must_include": ["140", "120"], "reference_answer": "...", "forbidden_output": [], "notes": "..."}
```

Relevance is annotated with evidence quotes, not chunk ids. A chunk is relevant if it covers at
least 50% of a quote's character span, so the judgments stay valid when chunking changes.
`answer_must_include` lists key facts; `|` separates accepted alternatives.

## Known limitations

* Small (56 questions, 43 with evidence): confidence intervals are wide.
* Written by the same author as the system, so question wording often overlaps the document
  wording. This favours lexical retrieval; only 3 questions are explicit paraphrases.
* English only; no tables, images or scanned pages.

Version 2 should add more paraphrased questions with low word overlap, longer documents and
annotations by a second person.
