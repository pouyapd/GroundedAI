# Answer evaluation: ollama (qwen2.5:1.5b)

Generated 2026-10-13T11:09:21+00:00 · report `generation-ollama-groundedai-dev-v1-20261013T110921Z`

- Provider: ollama / qwen2.5:1.5b (language model)
- Retrieval: hybrid (rrf), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-dev-v1 v1.0.0, 25 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.760 |
| abstention recall | 0.833 |
| abstention precision | 0.625 |
| citation validity | 1.000 |
| grounded rate | 0.118 |
| evidence coverage | 0.289 |
| key fact recall | 0.737 |
| injection resistance | 1.000 |
| failure rate | 0.000 |
| latency p50 / p95 (ms) | 10820 / 13383 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| conflicting | 2 | 0.000 | 0.000 | 0.750 | – |
| factual | 12 | 0.833 | 0.417 | 0.750 | – |
| injection | 2 | 0.500 | 0.000 | 0.500 | 1.000 |
| multi_chunk | 2 | 1.000 | 0.000 | 0.750 | – |
| out_of_scope | 2 | 1.000 | – | – | – |
| synthesis | 1 | 1.000 | 0.500 | 1.000 | – |
| unanswerable | 4 | 0.750 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
