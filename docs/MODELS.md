# Models

GroundedAI works without any model (hashing embedder + extractive demo generator). The models
below are optional. None is downloaded automatically: use `scripts/download_models.py`
(Hugging Face models) or `ollama pull` (Ollama models). Licences were taken from the model cards.

| Model | Role | Size on disk | Licence | Notes |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | Dense embeddings (384 dims) | ~130 MB | MIT | Uses the query instruction `Represent this sentence for searching relevant passages: ` (set `GROUNDEDAI_EMBEDDING_QUERY_PREFIX`) |
| `sentence-transformers/all-MiniLM-L6-v2` | Dense embeddings (384 dims) | ~90 MB (175 MB cache incl. extra formats) | Apache-2.0 | No query prefix |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker | ~90 MB | Apache-2.0 | Trained on MS MARCO; check the MS MARCO data terms before commercial use |
| `Qwen/Qwen2.5-0.5B-Instruct` | Generation via transformers (`huggingface-local`) | ~1 GB | Apache-2.0 | Fast on CPU but weak at following the citation format (see evaluation) |
| `qwen2.5:1.5b` (Ollama, Q4_K_M, digest `65ec06548149`) | Generation via Ollama (default Ollama model in the settings) | 986 MB on disk; about 1.3 GB in memory with a 4096-token context | Apache-2.0 (Qwen2.5-1.5B-Instruct) | Measured on 2026-10-13 with Ollama 0.40.3: 10 s per answer (median) on an i7-1255U without a GPU; gets most facts right but rarely uses the `[S#]` citation format (see `docs/EVALUATION.md`) |

Other Ollama models work by setting `GROUNDEDAI_OLLAMA_MODEL`. Check each model's licence;
some Qwen2.5 sizes and other model families use more restrictive licences.

## Activating models

```bash
# Python ML extra (sentence-transformers, transformers, langchain-huggingface)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements/ml.txt
pip install --no-deps -e backend

# download, after a size/licence summary and confirmation
python scripts/download_models.py --preset recommended    # bge-small + reranker (~220 MB)
python scripts/download_models.py --generation Qwen/Qwen2.5-0.5B-Instruct
```

```bash
# .env for real embeddings + reranker + Ollama
GROUNDEDAI_EMBEDDING_PROVIDER=sentence-transformers
GROUNDEDAI_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
GROUNDEDAI_EMBEDDING_DIM=384
GROUNDEDAI_EMBEDDING_QUERY_PREFIX="Represent this sentence for searching relevant passages: "
GROUNDEDAI_RERANKER=cross-encoder
GROUNDEDAI_GENERATION_PROVIDER=ollama
GROUNDEDAI_OLLAMA_MODEL=qwen2.5:1.5b
```

Ollama: install it from https://ollama.com, run `ollama serve`, then `ollama pull qwen2.5:1.5b`.
The UI and `/api/v1/system/status` show whether the configured model is reachable. If it is not,
questions fail with `503 model_unavailable`; GroundedAI never switches to another model by itself.

After changing the embedding model, re-ingest documents (Documents → Re-ingest, or
`POST /api/v1/documents/{id}/reingest`). Documents indexed with another model are excluded from
dense search until then, and responses carry a warning saying so.

## Verified in this repository

* sentence-transformers embeddings (bge-small, MiniLM) and the cross-encoder reranker: used in
  the published retrieval benchmark.
* `huggingface-local` with Qwen2.5-0.5B-Instruct: used in the generation evaluation (see
  `docs/EVALUATION.md`).
* Ollama: the `ChatOllama` integration is tested against a fake Ollama HTTP server (request
  options, streaming, token counts, missing model, unreachable server). It was **not** run against
  a real Ollama server here, because Ollama is not installed on the development machine.
* `scripts/download_models.py`: its listing, cache detection and confirmation prompt were run;
  the download step itself was not exercised because all models were already cached.
