"""Service wiring (manual dependency injection).

Heavy components (embedding model, reranker, chat model) are created lazily
on first use and can be replaced for tests via :func:`build_services`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from groundedai.core.config import Settings
from groundedai.generation.providers import GenerationProvider, build_generation_provider
from groundedai.ingestion.chunking import ChunkingConfig
from groundedai.observability.metrics import MetricsRegistry
from groundedai.persistence.db import create_db_engine, create_session_factory
from groundedai.persistence.migrate import upgrade_database
from groundedai.persistence.search import SearchBackend, backend_for_dialect
from groundedai.persistence.storage import FileStore
from groundedai.retrieval.embeddings import EmbeddingProvider, build_embedding_provider
from groundedai.retrieval.engine import RetrievalEngine
from groundedai.retrieval.rerank import CrossEncoderReranker, Reranker
from groundedai.security.rate_limit import TokenBucketLimiter
from groundedai.services.evaluation_service import EvaluationService
from groundedai.services.ingestion_service import IngestionLimits, IngestionService
from groundedai.services.qa_service import QAService
from groundedai.services.status_service import StatusService


class Lazy[T]:
    """Thread-safe lazily-built value. Failed builds are retried on the next call."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._value: T | None = None
        self._built = False
        self._lock = threading.Lock()

    def get(self) -> T:
        with self._lock:
            if not self._built:
                self._value = self._factory()
                self._built = True
            return cast(T, self._value)

    @property
    def loaded(self) -> bool:
        return self._built


@dataclass
class Services:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    file_store: FileStore
    search_backend: SearchBackend
    metrics: MetricsRegistry
    rate_limiter: TokenBucketLimiter
    embedding: Lazy[EmbeddingProvider]
    reranker: Lazy[Reranker | None]
    generation: Lazy[GenerationProvider]
    retrieval: RetrievalEngine
    ingestion: IngestionService
    qa: QAService
    evaluation: EvaluationService
    status: StatusService

    def close(self) -> None:
        self.qa.close()
        self.engine.dispose()


def build_services(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    generation_provider: GenerationProvider | None = None,
    reranker: Reranker | None = None,
    metrics: MetricsRegistry | None = None,
    run_migrations: bool | None = None,
) -> Services:
    url = settings.resolved_database_url
    if run_migrations if run_migrations is not None else settings.auto_migrate:
        upgrade_database(url, settings.embedding_dim)
    engine = create_db_engine(url)
    session_factory = create_session_factory(engine)
    metrics = metrics or MetricsRegistry()

    embedding: Lazy[EmbeddingProvider] = Lazy(
        (lambda: embedding_provider)
        if embedding_provider
        else lambda: build_embedding_provider(settings)
    )

    def make_reranker() -> Reranker | None:
        if reranker is not None:
            return reranker
        if settings.reranker == "cross-encoder":
            return CrossEncoderReranker(settings.reranker_model, offline=settings.hf_offline)
        return None

    reranker_lazy: Lazy[Reranker | None] = Lazy(make_reranker)
    generation: Lazy[GenerationProvider] = Lazy(
        (lambda: generation_provider)
        if generation_provider
        else lambda: build_generation_provider(settings)
    )

    search_backend = backend_for_dialect(engine.dialect.name)
    file_store = FileStore(settings.uploads_dir)
    retrieval = RetrievalEngine(search_backend, embedding.get, reranker_lazy.get)
    ingestion = IngestionService(
        session_factory=session_factory,
        file_store=file_store,
        embedding_provider=embedding.get,
        chunking=ChunkingConfig(settings.chunk_size, settings.chunk_overlap),
        limits=IngestionLimits(
            max_upload_bytes=settings.max_upload_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            max_document_chars=settings.max_document_chars,
        ),
        metrics=metrics,
    )
    qa = QAService(settings, session_factory, retrieval, generation.get, metrics)
    evaluation = EvaluationService(settings, session_factory, embedding.get, reranker_lazy.get)
    status = StatusService(settings, engine, session_factory, embedding.get, metrics)
    return Services(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        file_store=file_store,
        search_backend=search_backend,
        metrics=metrics,
        rate_limiter=TokenBucketLimiter(),
        embedding=embedding,
        reranker=reranker_lazy,
        generation=generation,
        retrieval=retrieval,
        ingestion=ingestion,
        qa=qa,
        evaluation=evaluation,
        status=status,
    )
