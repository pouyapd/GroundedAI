# groundedai-claims-nist-v1: claim verification test set

108 claims, each paired with an evidence quote copied from one of the three **test** PDFs of
`groundedai-nist-v1` (SP 800-63B 2017, SP 800-63B-4 2025, AI RMF 1.0), with one of four labels:

| Label | Meaning | Claims |
|---|---|---:|
| `supported` | everything the claim says is stated by the evidence (paraphrases allowed) | 31 |
| `partial` | part of the claim is stated; another part (an extra detail) is not | 22 |
| `contradicted` | the evidence states something incompatible: another number, a negation, a weaker or stronger requirement ("may" for "SHALL"), another entity | 34 |
| `nee` | not enough evidence: the evidence neither supports nor contradicts the claim | 21 |

`kind` says what makes the label (`paraphrase`, `number`, `negation`, `modal`, `entity`,
`meaning`, `extra detail`, `related topic`, `not stated`, ...). Claims and labels were written by
hand on 2026-11-09 by the project author, before any verifier was run on them; every evidence
quote was checked to occur in the PDF text (text clean-up v1 and v2). It is used only as a test
set: training pairs for the fine-tuned verifier come from other documents
(`experiments/verifier/`). One annotator, no adjudication; some borderline cases (for example a
claim that adds a condition) could be argued either way between `partial` and `contradicted`.
