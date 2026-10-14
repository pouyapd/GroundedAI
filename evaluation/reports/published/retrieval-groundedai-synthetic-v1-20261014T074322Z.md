# Retrieval benchmark at the application's default settings (800/120-character chunks, 20 candidates)

Generated 2026-10-14T07:43:22+00:00 · report `retrieval-groundedai-synthetic-v1-20261014T074322Z`

- Dataset: groundedai-synthetic-v1 v1.0.0 (fingerprint `fa56ebd64e3b1eee`), 14 documents, 41 chunks, 43 questions with evidence
- Chunking: 800 chars, overlap 120; candidates per retriever: 20
- Backend: SQLite: FTS5 BM25 (porter stemmer) + exact cosine scan in NumPy
- Hardware: 12th Gen Intel(R) Core(TM) i7-1255U (12 logical CPUs), 15.6 GB RAM, no GPU; power source: ac
- Software: Python 3.12.7 on Windows 11

Mean over questions with 95% bootstrap interval (2000 resamples, seed 13).

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| A-lexical-bm25 | 0.919 [0.85, 0.98] | 1.000 [1.00, 1.00] | 0.985 [0.95, 1.00] | 0.990 [0.97, 1.00] | 2.8 | 5.2 |
| B0-dense-hashing | 0.826 [0.71, 0.93] | 0.942 [0.87, 0.99] | 0.915 [0.84, 0.98] | 0.910 [0.84, 0.97] | 3.4 | 6.8 |
| C0-hybrid-rrf-hashing | 0.872 [0.78, 0.95] | 0.977 [0.93, 1.00] | 0.950 [0.88, 1.00] | 0.956 [0.90, 1.00] | 4.1 | 7.8 |
| B1-dense-bge-small | 0.756 [0.63, 0.86] | 0.954 [0.90, 1.00] | 0.884 [0.81, 0.95] | 0.892 [0.82, 0.95] | 58.0 | 73.7 |
| C1-hybrid-rrf-bge-small | 0.872 [0.78, 0.95] | 1.000 [1.00, 1.00] | 0.959 [0.91, 1.00] | 0.969 [0.93, 1.00] | 56.9 | 98.2 |

## Paired differences against `A-lexical-bm25`

| Configuration | Metric | Mean difference | 95% interval | Interval excludes 0 |
|---|---|---|---|---|
| B0-dense-hashing | recall@5 | -0.058 | [-0.128, -0.012] | yes |
| B0-dense-hashing | mrr | -0.070 | [-0.140, -0.023] | yes |
| B0-dense-hashing | ndcg@5 | -0.080 | [-0.146, -0.027] | yes |
| C0-hybrid-rrf-hashing | recall@5 | -0.023 | [-0.070, +0.000] | no |
| C0-hybrid-rrf-hashing | mrr | -0.035 | [-0.093, +0.000] | no |
| C0-hybrid-rrf-hashing | ndcg@5 | -0.034 | [-0.089, +0.000] | no |
| B1-dense-bge-small | recall@5 | -0.046 | [-0.105, +0.000] | no |
| B1-dense-bge-small | mrr | -0.100 | [-0.170, -0.038] | yes |
| B1-dense-bge-small | ndcg@5 | -0.098 | [-0.163, -0.041] | yes |
| C1-hybrid-rrf-bge-small | recall@5 | +0.000 | [+0.000, +0.000] | no |
| C1-hybrid-rrf-bge-small | mrr | -0.025 | [-0.060, +0.000] | no |
| C1-hybrid-rrf-bge-small | ndcg@5 | -0.021 | [-0.046, -0.002] | yes |

## Limitations

- Small synthetic corpus written for this project; results are indicative only and do not transfer to other domains.
- 95% intervals are percentile bootstrap over questions; with this sample size many differences between configurations are not distinguishable.
- Latency is single-process wall-clock time on the listed CPU, including query embedding and reranking; the SQLite vector search is an exact linear scan.
