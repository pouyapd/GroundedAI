# syntax=docker/dockerfile:1.7
# GroundedAI: API + built web UI in one image.
#
#   docker build -t groundedai .                         # lightweight (no PyTorch, ~250 MB)
#   docker build -t groundedai:ml --build-arg INSTALL_ML=true .   # + sentence-transformers, CPU-only PyTorch
#
# Models are never baked into the image. With INSTALL_ML=true, mount a Hugging Face
# cache (see docker-compose.yml) or download models explicitly with
# scripts/download_models.py.

# ---------------------------------------------------------------- frontend build
FROM node:24.21.0-alpine3.24 AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------- runtime
FROM python:3.12.15-slim-bookworm AS runtime

ARG INSTALL_ML=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GROUNDEDAI_DATA_DIR=/data \
    GROUNDEDAI_AUTO_MIGRATE=false \
    GROUNDEDAI_EVALUATION_DIR=/app/evaluation \
    GROUNDEDAI_FRONTEND_DIST=/app/frontend/dist \
    HF_HOME=/models/huggingface

RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /app app

WORKDIR /app
COPY backend/requirements/ backend/requirements/
# ml.txt pins the CPU-only PyTorch wheel (served from the PyTorch index), which
# avoids several GB of CUDA libraries on Linux.
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements/ml.txt; \
    else \
      pip install -r backend/requirements/prod.txt; \
    fi

COPY backend/ backend/
RUN pip install --no-deps ./backend
COPY evaluation/ evaluation/
COPY scripts/ scripts/
COPY --from=frontend /build/dist frontend/dist
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 0755 /usr/local/bin/entrypoint.sh \
    && mkdir -p /data /models \
    && chown -R app:app /data /models

USER app
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "groundedai.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
