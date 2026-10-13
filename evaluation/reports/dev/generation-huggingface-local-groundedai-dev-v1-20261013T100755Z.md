# Answer evaluation: huggingface-local (Qwen/Qwen2.5-0.5B-Instruct)

Generated 2026-10-13T10:07:55+00:00 · report `generation-huggingface-local-groundedai-dev-v1-20261013T100755Z`

- Provider: huggingface-local / Qwen/Qwen2.5-0.5B-Instruct (language model)
- Retrieval: hybrid (rrf), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-dev-v1 v1.0.0, 25 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.320 |
| abstention recall | 0.667 |
| abstention precision | 0.222 |
| citation validity | 1.000 |
| grounded rate | 0.000 |
| evidence coverage | 0.053 |
| key fact recall | 0.210 |
| injection resistance | 1.000 |
| failure rate | 0.000 |
| latency p50 / p95 (ms) | 38652 / 51301 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| conflicting | 2 | 0.000 | 0.000 | 0.000 | – |
| factual | 12 | 0.167 | 0.000 | 0.167 | – |
| injection | 2 | 0.500 | 0.000 | 0.500 | 1.000 |
| multi_chunk | 2 | 0.500 | 0.500 | 0.500 | – |
| out_of_scope | 2 | 0.000 | – | – | – |
| synthesis | 1 | 0.000 | 0.000 | 0.000 | – |
| unanswerable | 4 | 1.000 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
