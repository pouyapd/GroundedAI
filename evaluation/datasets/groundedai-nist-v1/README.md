# groundedai-nist-v1 (test) and groundedai-nist-val-v1 (validation)

Question sets on **real PDFs**: publications of the U.S. National Institute of Standards and
Technology (NIST). They complement the synthetic datasets, whose documents were written for this
project and are short and lexically easy. NIST publications are works of the U.S. Government and
are not subject to copyright in the United States; they are not committed here.

```bash
python scripts/fetch_nist_corpus.py      # downloads the PDFs and checks their SHA-256 hashes
python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-val-v1 --generator extractive --out val.json
```

## Splits

Documents never appear in more than one split, so no question shares a document with another
split's questions.

| Split | Dataset | Documents | Questions |
|---|---|---|---|
| test | `groundedai-nist-v1` | SP 800-63B (June 2017, 80 pages), SP 800-63B-4 (July 2025, 129 pages), AI 100-1 / AI RMF 1.0 (January 2023, 48 pages) | 53 |
| validation | `groundedai-nist-val-v1` | AI 600-1 (Generative AI Profile, 64 pages), NISTIR 8312 (Four Principles of Explainable AI, 43 pages) | 18 |
| training (fine-tuning only) | `experiments/finetune/corpus.json` | CSF 2.0, SP 800-61r3, SP 800-88r1, SP 800-207, SP 800-218 | generated, see `experiments/` |

The validation split is for choosing settings. The test split is run only with settings fixed
in advance; every reported test number names its settings.

## How the questions were written

All questions, expected answers and evidence quotes were written on 2026-11-09 by the project
author from the PDF text, **before** any retrieval or model run on these documents. Each evidence
quote is copied from the text that GroundedAI extracts from the PDF and was checked to occur in
it (with both text clean-up versions, v1 and v2). `evidence_pages` records the page of each quote.

Each question has a `scope`: the documents it is asked against, as a user would choose them
under "Search in". Conflict questions are scoped to both versions of SP 800-63B.

Test categories (53): direct and numerical facts (30, tags `numeric`, `date`, `entity`,
`terminology`), multi-passage (4), missing information (6, expected to be refused), ambiguous
(3, answer or ask), conflicting versions (4: the 2017 and 2025 guidelines give different
values or a different requirement strength), and paraphrases (6, tag `paraphrase:<id>`: the same
question as an earlier one in other words, e.g. "password" where the document says "memorized
secret").

## Limitations

- One author wrote the questions and the labels; there is no second annotator.
- 53 test questions give wide confidence intervals; reports include 95% bootstrap intervals.
- Key facts are matched as strings ("8 characters|eight characters"), which can miss a correct
  answer worded differently.
- The documents are English normative guidelines from one publisher; results do not transfer
  to other kinds of documents.
