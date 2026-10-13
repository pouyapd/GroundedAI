# Answer evaluation: ollama (qwen2.5:1.5b)

Generated 2026-10-13T11:36:18+00:00 · report `generation-ollama-groundedai-synthetic-v1-20261013T113618Z`

- Provider: ollama / qwen2.5:1.5b (language model)
- Retrieval: hybrid (rrf), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-synthetic-v1 v1.0.0, 56 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.906 |
| abstention recall | 0.700 |
| abstention precision | 1.000 |
| citation validity | 1.000 |
| grounded rate | 0.143 |
| evidence coverage | 0.209 |
| key fact recall | 0.942 |
| injection resistance | 1.000 |
| failure rate | 0.000 |
| latency p50 / p95 (ms) | 10039 / 16445 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| ambiguous | 3 | – | – | – | – |
| conflicting | 2 | 0.000 | 0.500 | 0.250 | – |
| factual | 30 | 1.000 | 0.233 | 1.000 | – |
| injection | 3 | 1.000 | 0.333 | 1.000 | 1.000 |
| multi_chunk | 5 | 1.000 | 0.000 | 1.000 | – |
| out_of_scope | 3 | 1.000 | – | – | – |
| synthesis | 3 | 1.000 | 0.000 | 0.667 | 1.000 |
| unanswerable | 7 | 0.571 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
