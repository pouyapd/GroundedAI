# Third-party software and models

GroundedAI's own code and the synthetic evaluation dataset are released under the MIT License
(`LICENSE`). The project depends on, but does not include, the following software. Each package
keeps its own licence; versions are pinned in `backend/requirements/*.txt` and
`frontend/package-lock.json`.

## Main runtime dependencies

| Package | Licence |
|---|---|
| FastAPI, Pydantic, pydantic-settings, Uvicorn, Starlette | MIT / BSD-3-Clause |
| SQLAlchemy, Alembic | MIT |
| pypdf | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| LangChain (langchain-core, langchain-text-splitters, langchain-ollama, langchain-huggingface) | MIT |
| httpx | BSD-3-Clause |
| psycopg | LGPL-3.0 (used as a separately installed library) |
| pgvector (Python package and PostgreSQL extension) | MIT / PostgreSQL License |
| sentence-transformers | Apache-2.0 |
| transformers, huggingface_hub | Apache-2.0 |
| PyTorch | BSD-3-Clause |
| React, React DOM | MIT |
| Vite, Vitest, TypeScript, ESLint | MIT / Apache-2.0 |

## Optional models (not distributed with this repository)

| Model | Licence | Note |
|---|---|---|
| BAAI/bge-small-en-v1.5 | MIT | |
| sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Apache-2.0 | trained on MS MARCO, which has its own data terms |
| Qwen/Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct (`qwen2.5:1.5b` in Ollama) | Apache-2.0 | |

## Online demo (GitHub Pages)

The demo site redistributes unmodified copies of these wheels, pinned with their SHA-256 in
`frontend/browser-runtime.lock.json`, and bundles WebLLM into its JavaScript:

| Package | Licence |
|---|---|
| alembic, fastapi, langchain-core, langchain-protocol, langchain-text-splitters, langsmith, mako, pydantic-settings | MIT |
| httpx2, httpx2-jsfetch, pypdf, python-dotenv, starlette, uuid-utils, websockets | BSD-3-Clause |
| opentelemetry-api, python-multipart, requests-toolbelt, tenacity | Apache-2.0 |
| @mlc-ai/web-llm (WebLLM) | Apache-2.0 |

Loaded by the visitor's browser from their publishers, not redistributed by this project:
Pyodide and its packages (MPL-2.0 for Pyodide; each package keeps its licence) from the
jsDelivr CDN, and, only when the visitor chooses it, `mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC`
(Qwen2.5-1.5B-Instruct converted by the MLC project; Apache-2.0) from Hugging Face with its
WebGPU library from the MLC project.

Container images referenced by `Dockerfile` and `docker-compose.yml` (python, node,
pgvector/pgvector, ollama/ollama) are pulled from their publishers and keep their own licences.
