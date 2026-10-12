"""FastAPI application factory.

Run with ``uvicorn groundedai.main:create_app --factory`` (see README).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from groundedai import __version__
from groundedai.api.errors import register_exception_handlers
from groundedai.api.routes import documents, evaluation, health, metrics, query
from groundedai.core.config import Settings, get_settings
from groundedai.observability.logging import configure_logging
from groundedai.observability.metrics import MetricsRegistry
from groundedai.observability.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from groundedai.services.container import Services, build_services

logger = logging.getLogger("groundedai")

API_PREFIX = "/api/v1"
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

DESCRIPTION = """
Grounded question answering over your own documents.

* Upload PDF / text documents, which are parsed, chunked and indexed.
* Ask questions: answers cite numbered sources, and every citation is checked
  against the passages the model actually received.
* Inspect retrieval (lexical, dense, hybrid, optional reranking) and evaluation reports.

When `GROUNDEDAI_API_KEY` is set, `/api/v1` endpoints (except health, readiness and
system status) require the `X-API-Key` header.
"""


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    registry = services.metrics if services else MetricsRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = services or build_services(settings, metrics=registry)
        app.state.services = active
        recovered = active.ingestion.recover_interrupted()
        active.evaluation.recover_interrupted()
        logger.info(
            "startup",
            extra={
                "version": __version__,
                "env": settings.env.value,
                "database": active.engine.dialect.name,
                "embedding_provider": settings.embedding_provider,
                "generation_provider": settings.generation_provider,
                "auth_enabled": settings.auth_enabled,
                "recovered_documents": recovered,
            },
        )
        try:
            yield
        finally:
            active.close()
            logger.info("shutdown")

    app = FastAPI(
        title="GroundedAI API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    # Middleware: the last one added is the outermost.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
        allow_credentials=False,
    )
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
    )
    app.add_middleware(RequestContextMiddleware, metrics=registry)
    register_exception_handlers(app)

    # Health endpoints also exist without the prefix for container orchestrators.
    app.include_router(health.router, include_in_schema=False)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(query.router, prefix=API_PREFIX)
    app.include_router(evaluation.router, prefix=API_PREFIX)
    app.include_router(metrics.router, prefix=API_PREFIX)

    _mount_frontend(app, settings.frontend_dist)
    return app


def _mount_frontend(app: FastAPI, dist: Path | None) -> None:
    if dist is not None and (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
        return

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": "GroundedAI API", "version": __version__, "docs": "/docs"}
