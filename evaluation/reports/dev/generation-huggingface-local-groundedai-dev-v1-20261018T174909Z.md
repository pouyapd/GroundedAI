# Answer evaluation: huggingface-local (Qwen/Qwen2.5-0.5B-Instruct, answer-v2)

Generated 2026-10-18T17:49:09+00:00 · report `generation-huggingface-local-groundedai-dev-v1-20261018T174909Z`

- Provider: huggingface-local / Qwen/Qwen2.5-0.5B-Instruct (language model)
- Retrieval: hybrid (rrf), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-dev-v1 v1.0.0, 25 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: battery
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.360 |
| abstention recall | 0.833 |
| abstention precision | 0.312 |
| citation validity | 1.000 |
| grounded rate | 0.167 |
| evidence coverage | 0.053 |
| key fact recall | 0.105 |
| injection resistance | 1.000 |
| failure rate | 0.000 |
| claim match rate | 0.600 |
| cited claim precision | 1.000 |
| wrong source rate | 0.000 |
| unsupported claim rate | 0.400 |
| answers verified rate | 0.500 |
| key fact attribution | 0.300 |
| conflict surfaced rate | 1.000 |
| false conflict rate | 0.000 |
| latency p50 / p95 (ms) | 61522 / 83385 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| conflicting | 2 | 0.000 | 0.000 | 0.250 | – |
| factual | 12 | 0.167 | 0.000 | 0.000 | – |
| injection | 2 | 0.500 | 0.500 | 0.500 | 1.000 |
| multi_chunk | 2 | 0.500 | 0.000 | 0.250 | – |
| out_of_scope | 2 | 1.000 | – | – | – |
| synthesis | 1 | 0.000 | 0.000 | 0.000 | – |
| unanswerable | 4 | 0.750 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
