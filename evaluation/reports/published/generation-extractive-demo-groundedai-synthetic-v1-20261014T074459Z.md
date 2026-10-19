# Answer evaluation: extractive-demo (extractive-demo-v1, answer-v2)

Generated 2026-10-14T07:44:59+00:00 · report `generation-extractive-demo-groundedai-synthetic-v1-20261014T074459Z`

- Provider: extractive-demo / extractive-demo-v1 (extractive baseline, not an LLM)
- Retrieval: dense (-), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-synthetic-v1 v1.0.0, 56 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.660 |
| abstention recall | 0.500 |
| abstention precision | 0.278 |
| citation validity | 1.000 |
| grounded rate | 1.000 |
| evidence coverage | 0.663 |
| key fact recall | 0.581 |
| injection resistance | 1.000 |
| failure rate | 0.000 |
| claim match rate | 1.000 |
| cited claim precision | 1.000 |
| wrong source rate | 0.000 |
| unsupported claim rate | 0.000 |
| answers verified rate | 0.895 |
| key fact attribution | 0.800 |
| conflict surfaced rate | 0.500 |
| false conflict rate | 0.018 |
| latency p50 / p95 (ms) | 122 / 195 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| ambiguous | 3 | – | – | – | – |
| conflicting | 2 | 0.000 | 0.000 | 0.000 | – |
| factual | 30 | 0.833 | 0.800 | 0.700 | – |
| injection | 3 | 0.667 | 0.667 | 0.667 | 1.000 |
| multi_chunk | 5 | 0.600 | 0.500 | 0.400 | – |
| out_of_scope | 3 | 0.667 | – | – | – |
| synthesis | 3 | 0.000 | 0.000 | 0.000 | 1.000 |
| unanswerable | 7 | 0.429 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
- The provider is the extractive demo generator, not a language model; these numbers describe that baseline only.
