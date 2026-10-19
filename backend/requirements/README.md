# Pinned dependency locks

Generated with `uv pip compile --universal` from `backend/pyproject.toml`, so each file
works on Linux, macOS and Windows (platform differences are expressed as markers).

| File | Contents | Used by |
|---|---|---|
| `prod.txt` | runtime + PostgreSQL driver | Docker image (default) |
| `dev.txt` | runtime + PostgreSQL + test/lint tools | CI, local development |
| `ml.txt` | runtime + PostgreSQL + sentence-transformers/transformers with **CPU-only** PyTorch | Docker image with `INSTALL_ML=true`, optional local ML setup |

`ml.txt` needs the PyTorch CPU index:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements/ml.txt
```

## Regenerating after changing `pyproject.toml`

```bash
pip install uv
uv pip compile backend/pyproject.toml --extra postgres --universal --python-version 3.12 --no-header -o backend/requirements/prod.txt
uv pip compile backend/pyproject.toml --extra postgres --extra dev --universal --python-version 3.12 --no-header -o backend/requirements/dev.txt
uv pip compile backend/pyproject.toml --extra postgres --extra ml --universal --python-version 3.12 --torch-backend cpu --no-header -o backend/requirements/ml.txt
```

Then run the full test suite before committing the new locks.
