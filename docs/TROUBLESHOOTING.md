# Troubleshooting

**The UI says "The GroundedAI API is not reachable".**
Start the API (`groundedai serve`) and reload. With `npm run dev`, the UI proxies `/api` to
`http://127.0.0.1:8000`; start the API on that port.

**Questions fail with `503 model_unavailable`.**
The configured model provider cannot be used. The message says why:
* Ollama not running → `ollama serve`; model missing → `ollama pull <model>`.
* `huggingface-local` or `sentence-transformers` model not cached →
  `python scripts/download_models.py ...` (GroundedAI does not download implicitly).
* `[ml]` extra not installed → `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements/ml.txt`.
Or switch back to the demo generator: `GROUNDEDAI_GENERATION_PROVIDER=extractive-demo`.

**Ollama works in the browser/curl but GroundedAI gets HTTP 503.**
An HTTP proxy is probably configured (`HTTP_PROXY`). GroundedAI bypasses proxies for loopback
URLs (`localhost`, `127.0.0.1`); for another host name add it to `NO_PROXY`.

**`504 model_timeout` or very slow answers.**
CPU generation is slow for large models. Use a smaller model, lower
`GROUNDEDAI_MAX_NEW_TOKENS`, or raise `GROUNDEDAI_GENERATION_TIMEOUT_S`. Laptops on battery
are much slower.

**`503 server_busy`.**
More concurrent questions than `GROUNDEDAI_MAX_CONCURRENT_GENERATIONS`. Wait, or raise the
limit if the hardware allows.

**A document is `failed`.**
The table shows the error. `document_parse_failed`: damaged or encrypted PDF.
`empty_document`: no text layer (scanned PDFs need OCR, which is not supported).
`interrupted`: the server stopped during processing. Fix the cause and use **Re-ingest**.

**Answers ignore some documents after changing the embedding model.**
Documents remember the model they were indexed with and are excluded from dense search for
other models. Re-ingest them. On PostgreSQL the vector dimension is fixed at the first migration;
a different dimension needs a fresh database.

**`/ready` returns 503 with `vector_dimension` failing.**
The PostgreSQL column was created for another `GROUNDEDAI_EMBEDDING_DIM`. Use the original
dimension or recreate the database volume (`docker compose down -v`).

**Uploads fail with `413`.**
Files above `GROUNDEDAI_MAX_UPLOAD_BYTES` (10 MB by default) are rejected.

**PowerShell does not expand `*` in `groundedai ingest corpus/*`.**
Pass the directory instead: `groundedai ingest evaluation/datasets/groundedai-synthetic-v1/corpus`.

**Windows console shows `�` in CLI output.**
Set `PYTHONIOENCODING=utf-8` (or use Windows Terminal). Report files are always written as UTF-8.

**Tests pick up my local settings.**
The test suite clears all `GROUNDEDAI_*` variables and ignores `.env`; if a test still behaves
differently, check `HF_HUB_OFFLINE` and proxy variables.

**`ollama` is not recognised after installing it (Windows).**
The installer adds Ollama to the user `PATH` and starts the tray app; open a new terminal. The
API is at `http://127.0.0.1:11434` (`curl http://127.0.0.1:11434/api/version`). Models are
stored in `OLLAMA_MODELS` if that variable is set, otherwise in your profile.

**Ollama runs but answers are refused or very short.**
Check that the pulled model is the configured one: `ollama list` must show
`GROUNDEDAI_OLLAMA_MODEL` (default `qwen2.5:1.5b`). Very small models (0.5B) often refuse
answerable questions; see `docs/EVALUATION.md`.

**A correct statement is shown as "Partly found".**
The claim check compares words and figures, not meaning. A paraphrase with other words, or a
statement that adds a detail the passage does not give ("young children" for "children under
5"), is only partly found. Read the quoted passage; `GROUNDEDAI_CLAIM_CHECK=claims-v2` reads
some equivalent wordings but is still lexical.

**`scripts/e2e_browser.py` fails with `No module named 'playwright'`.**
The browser check needs Playwright, which is not a project dependency:
`pip install playwright && python -m playwright install chromium` (about 150 MB), in the
environment that has the backend installed.
