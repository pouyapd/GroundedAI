# groundedai-nist-val-relevance-v1

14 relevance and abstention questions on the two validation PDFs of `groundedai-nist-val-v1`
(NIST AI 600-1 and NISTIR 8312; the same files, downloaded and hash-checked by
`scripts/fetch_nist_corpus.py`). Written on 2026-11-13, before the extractive relevance gate, to
choose and check it. No test document is used.

* 7 answerable: paraphrased, entity and multi-part questions with key facts and evidence pages.
* 7 to refuse: they share words with the documents but ask for something the documents do not
  state ("What minimum accuracy does the Generative AI Profile require for chatbots?").
