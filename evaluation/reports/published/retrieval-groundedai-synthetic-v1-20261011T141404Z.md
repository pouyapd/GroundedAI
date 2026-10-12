# Retrieval benchmark (full profile: local sentence-transformers models)

Generated 2026-10-11T14:14:04+00:00 · report `retrieval-groundedai-synthetic-v1-20261011T141404Z`

- Dataset: groundedai-synthetic-v1 v1.0.0 (fingerprint `5a04e91e09086368`), 14 documents, 87 chunks, 43 questions with evidence
- Chunking: 400 chars, overlap 60; candidates per retriever: 30
- Backend: SQLite: FTS5 BM25 (porter stemmer) + exact cosine scan in NumPy
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

Mean over questions with 95% bootstrap interval (2000 resamples, seed 13).

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| A-lexical-bm25 | 0.861 [0.77, 0.94] | 0.965 [0.91, 1.00] | 0.934 [0.86, 0.99] | 0.931 [0.87, 0.98] | 3.6 | 10.0 |
| B0-dense-hashing | 0.709 [0.58, 0.84] | 0.861 [0.77, 0.95] | 0.807 [0.70, 0.90] | 0.799 [0.70, 0.90] | 5.6 | 12.4 |
| B1-dense-bge-small | 0.779 [0.65, 0.90] | 0.942 [0.87, 1.00] | 0.879 [0.80, 0.95] | 0.886 [0.80, 0.96] | 103.9 | 201.5 |
| B2-dense-minilm | 0.767 [0.65, 0.87] | 0.930 [0.85, 0.99] | 0.880 [0.80, 0.95] | 0.881 [0.79, 0.95] | 41.6 | 86.9 |
| C1-hybrid-rrf-bge-small | 0.826 [0.72, 0.92] | 0.965 [0.91, 1.00] | 0.915 [0.84, 0.98] | 0.920 [0.85, 0.97] | 84.4 | 114.0 |
| C2-hybrid-weighted-bge-small | 0.826 [0.72, 0.92] | 0.942 [0.87, 1.00] | 0.915 [0.84, 0.98] | 0.912 [0.84, 0.97] | 77.8 | 181.1 |
| D1-hybrid-rrf-bge-small+rerank | 0.849 [0.74, 0.94] | 0.942 [0.88, 0.99] | 0.920 [0.85, 0.98] | 0.917 [0.84, 0.98] | 809.6 | 989.6 |

## Paired differences against `A-lexical-bm25`

| Configuration | Metric | Mean difference | 95% interval | Interval excludes 0 |
|---|---|---|---|---|
| B0-dense-hashing | recall@5 | -0.105 | [-0.198, -0.012] | yes |
| B0-dense-hashing | mrr | -0.127 | [-0.220, -0.046] | yes |
| B0-dense-hashing | ndcg@5 | -0.132 | [-0.223, -0.051] | yes |
| B1-dense-bge-small | recall@5 | -0.023 | [-0.081, +0.023] | no |
| B1-dense-bge-small | mrr | -0.056 | [-0.129, +0.010] | no |
| B1-dense-bge-small | ndcg@5 | -0.045 | [-0.110, +0.012] | no |
| B2-dense-minilm | recall@5 | -0.035 | [-0.128, +0.058] | no |
| B2-dense-minilm | mrr | -0.054 | [-0.142, +0.032] | no |
| B2-dense-minilm | ndcg@5 | -0.050 | [-0.141, +0.040] | no |
| C1-hybrid-rrf-bge-small | recall@5 | +0.000 | [-0.035, +0.035] | no |
| C1-hybrid-rrf-bge-small | mrr | -0.019 | [-0.058, +0.011] | no |
| C1-hybrid-rrf-bge-small | ndcg@5 | -0.011 | [-0.049, +0.021] | no |
| C2-hybrid-weighted-bge-small | recall@5 | -0.023 | [-0.081, +0.023] | no |
| C2-hybrid-weighted-bge-small | mrr | -0.020 | [-0.058, +0.011] | no |
| C2-hybrid-weighted-bge-small | ndcg@5 | -0.019 | [-0.063, +0.021] | no |
| D1-hybrid-rrf-bge-small+rerank | recall@5 | -0.023 | [-0.093, +0.035] | no |
| D1-hybrid-rrf-bge-small+rerank | mrr | -0.014 | [-0.072, +0.040] | no |
| D1-hybrid-rrf-bge-small+rerank | ndcg@5 | -0.014 | [-0.069, +0.039] | no |

## Limitations

- Small synthetic corpus written for this project; results are indicative only and do not transfer to other domains.
- 95% intervals are percentile bootstrap over questions; with this sample size many differences between configurations are not distinguishable.
- Latency is single-process wall-clock time on the listed CPU, including query embedding and reranking; the SQLite vector search is an exact linear scan.
