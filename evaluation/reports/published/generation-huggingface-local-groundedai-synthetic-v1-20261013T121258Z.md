# Answer evaluation: huggingface-local (Qwen/Qwen2.5-0.5B-Instruct)

Generated 2026-10-13T12:12:58+00:00 · report `generation-huggingface-local-groundedai-synthetic-v1-20261013T121258Z`

- Provider: huggingface-local / Qwen/Qwen2.5-0.5B-Instruct (language model)
- Retrieval: hybrid (rrf), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-synthetic-v1 v1.0.0, 56 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.528 |
| abstention recall | 0.700 |
| abstention precision | 0.304 |
| citation validity | 1.000 |
| grounded rate | 0.200 |
| evidence coverage | 0.186 |
| key fact recall | 0.302 |
| injection resistance | 0.750 |
| failure rate | 0.000 |
| latency p50 / p95 (ms) | 37980 / 46982 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| ambiguous | 3 | – | – | – | – |
| conflicting | 2 | 0.000 | 0.000 | 0.000 | – |
| factual | 30 | 0.533 | 0.200 | 0.400 | – |
| injection | 3 | 0.000 | 0.000 | 0.000 | 1.000 |
| multi_chunk | 5 | 0.600 | 0.200 | 0.200 | – |
| out_of_scope | 3 | 1.000 | – | – | – |
| synthesis | 3 | 0.667 | 0.333 | 0.000 | 0.000 |
| unanswerable | 7 | 0.571 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
