# Recorded outputs re-checked: ollama (qwen2.5:1.5b, answer-v1)

Generated 2026-10-14T07:40:58+00:00 · report `revalidated-generation-ollama-groundedai-synthetic-v1-20261013T112505Z-20261014T074058Z`

- Provider: ollama / qwen2.5:1.5b (language model)
- Retrieval: hybrid (rrf), top-k 5, embeddings sentence-transformers:BAAI/bge-small-en-v1.5, reranker none
- Dataset: groundedai-synthetic-v1 v1.0.0, 56 questions
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

| Metric | Value |
|---|---|
| behavior accuracy | 0.830 |
| abstention recall | 0.900 |
| abstention precision | 0.474 |
| citation validity | 1.000 |
| grounded rate | 0.108 |
| evidence coverage | 0.221 |
| key fact recall | 0.767 |
| injection resistance | 1.000 |
| failure rate | 0.000 |
| claim match rate | 0.864 |
| cited claim precision | 0.903 |
| wrong source rate | 0.065 |
| unsupported claim rate | 0.082 |
| answers verified rate | 0.811 |
| key fact attribution | 0.861 |
| conflict surfaced rate | 0.500 |
| false conflict rate | 0.018 |
| latency p50 / p95 (ms) | 10920 / 20076 |

## By category

| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |
|---|---|---|---|---|---|
| ambiguous | 3 | – | – | – | – |
| conflicting | 2 | 0.000 | 0.250 | 0.000 | – |
| factual | 30 | 0.800 | 0.267 | 0.767 | – |
| injection | 3 | 1.000 | 0.000 | 1.000 | 1.000 |
| multi_chunk | 5 | 1.000 | 0.200 | 1.000 | – |
| out_of_scope | 3 | 0.667 | – | – | – |
| synthesis | 3 | 1.000 | 0.000 | 0.667 | 1.000 |
| unanswerable | 7 | 1.000 | – | – | – |

## Limitations

- Deterministic string and overlap checks only; no human or model judge.
- Key-fact recall can miss correct paraphrases and credit wrong conclusions.
- The support check is lexical: it detects missing words and invented numbers, not entailment.
- Recorded outputs replayed through the current checks; the model was not run.
