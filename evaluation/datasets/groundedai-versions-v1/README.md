# groundedai-versions-v1

Development and validation cases written on 2026-11-13, **before** the version-aware
disagreement check, the extractive relevance gate and per-clause claim checking were built. None
of it is test data; the protected test sets stay `groundedai-nist-v1` and
`groundedai-claims-nist-v1`.

**Documents** (`source/*.txt`, built into `corpus/*.pdf` by `scripts/build_versions_corpus.py`;
`--check` verifies them): four pairs of editions of fictional documents, five pages each with a
running header and page numbers, plus the repository's two example PDFs as unrelated documents.

| Family | Editions | Question split |
|---|---|---|
| Kestrel Valley Water, customer service standards | 2019, 2024 | dev |
| Northgate Library, lending rules | Version 2.0 (2020), 3.0 (2025) | dev |
| Ashby Council, residents' parking permits | 2021, 2023 | validation |
| Corvane Systems, access control standard CS-AC-01 | Revision A (2018), B (2024) | validation |

The editions differ in values (4 hours / 2 hours), in requirement wording (SHOULD / SHALL,
SHOULD NOT / SHALL NOT, MUST / SHOULD), in terminology (service interruption / planned outage,
fee / charge), and agree in other places in different words (21 days / 3 weeks, 2 minutes /
120 seconds, 12 months / one year). Corvane has two assurance levels, AL1 and AL2, whose values
must not be compared with each other.

**`questions.jsonl`** (52): tags give the split (`split:dev`, `split:validation`) and the
expected disagreement outcome (`expect-conflict:versions` for 18 questions whose two editions
differ, `expect-conflict:none` for 34 where they agree, only one edition states the value, the
question names another organisation, or the question is a relevance case). 16 questions tagged
`relevance` are asked against one edition: answerable paraphrased and multi-part questions, and
unanswerable questions that share words with the document (`related-words`) or are out of scope.

**`claims.jsonl`** (40): one evidence sentence and a hand-written claim with a label
(`supported` 11, `partial` 11, `contradicted` 13, `nee` 5). The `partial` claims add a plausible
detail the evidence does not state ("... and includes a free hot meal on board").

The `dev` split was used while building the checks; the `validation` split was scored only after
they were fixed.
