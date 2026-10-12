# Resource profile

Two ways to run GroundedAI. Memory figures were measured on the development laptop
(Windows 11, i7-1255U, 15.6 GB RAM, no GPU) with the 14-document sample corpus indexed;
they are indicative, not guarantees.

| | Lightweight mode (default) | Full local mode (optional) |
|---|---|---|
| Embeddings | `hashing` (no model) | `sentence-transformers`, `BAAI/bge-small-en-v1.5` |
| Reranker | off | optional `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | `extractive-demo` (not an LLM) | `ollama` (`qwen2.5:1.5b`) or `huggingface-local` (`Qwen/Qwen2.5-0.5B-Instruct`) |
| Database | SQLite in `./data` | SQLite or PostgreSQL + pgvector (Docker Compose) |
| Python packages | core extras only (`.venv`, ~350 MB) | plus the `[ml]` extra (PyTorch CPU + transformers, ~1 GB if PyTorch is not already installed) |
| Model downloads | none | ~130 MB (bge-small) + ~90 MB (reranker) + ~1 GB (LLM) |
| API process memory | ~120 MB (measured) | ~630 MB with bge-small (measured); ~1.5–1.8 GB working set with Qwen2.5-0.5B loaded (measured) |
| Retrieval latency (p50, published benchmark) | BM25 3.6 ms; hybrid with hashing ~5 ms (offline profile) | hybrid RRF with bge-small 84 ms; with reranker 810 ms |
| Generation latency | tens of milliseconds (no model) | seconds to tens of seconds per answer on CPU, depending on model and answer length |
| Answer quality | extractive baseline; see `docs/EVALUATION.md` | depends on the model; see `docs/EVALUATION.md` |

## Choosing settings for a 16 GB CPU-only machine

* Keep `GROUNDEDAI_MAX_CONCURRENT_GENERATIONS` at 1–2. Each CPU generation uses most cores.
* Prefer `qwen2.5:1.5b` (Ollama, Q4) over larger models. The 0.5B model is fast but weak at
  following the citation format (measured, see the evaluation document).
* The reranker improves ordering slightly on this dataset but adds ~700 ms per query on this CPU
  (`D1` vs `C1` in the benchmark). Leave it off unless precision at rank 1 matters more than
  latency.
* `GROUNDEDAI_EMBEDDING_BATCH_SIZE` (default 32) bounds memory during ingestion; lower it on
  machines with less RAM.
* The SQLite vector search scans all chunk vectors of the active model per query. That is
  fine for thousands of chunks; for larger collections use PostgreSQL with the HNSW index.

## Bounded processing

* Uploads are limited to `GROUNDEDAI_MAX_UPLOAD_BYTES` (10 MB) and rejected while streaming if
  the body is larger.
* PDFs are limited to `GROUNDEDAI_MAX_PDF_PAGES` (300) and `GROUNDEDAI_MAX_DOCUMENT_CHARS`
  (2 million characters).
* Embeddings are computed in batches; chunks are written in one transaction per document.
* Context sent to the model is capped at `GROUNDEDAI_MAX_CONTEXT_CHARS` (6000) and answers at
  `GROUNDEDAI_MAX_NEW_TOKENS` (384).
