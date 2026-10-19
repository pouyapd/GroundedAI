"""Readiness checks, system status and operational metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

from groundedai import __version__
from groundedai.core.config import Settings
from groundedai.core.errors import GroundedAIError
from groundedai.generation.providers import ProviderStatus, check_generation_provider
from groundedai.observability.metrics import MetricsRegistry, summarize
from groundedai.persistence.db import check_connection, session_scope
from groundedai.persistence.migrate import current_revision, head_revision
from groundedai.persistence.repositories import (
    ChunkRepository,
    DocumentRepository,
    QueryEventRepository,
)
from groundedai.retrieval.embeddings import EmbeddingProvider
from groundedai.schemas.system import (
    CheckOut,
    ComponentStatusOut,
    HourlyBucketOut,
    LatencySummaryOut,
    OperationalMetricsOut,
    ReadinessOut,
    SystemStatusOut,
)

PROVIDER_STATUS_TTL_S = 15.0


class StatusService:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        session_factory: sessionmaker[Session],
        embedding_provider: Callable[[], EmbeddingProvider],
        metrics: MetricsRegistry,
        generation_check: Callable[[Settings], ProviderStatus] = check_generation_provider,
    ):
        self.settings = settings
        self.engine = engine
        self.session_factory = session_factory
        self._embedding_provider = embedding_provider
        self.metrics = metrics
        self._generation_check = generation_check
        self._generation_status: ProviderStatus | None = None

    # ------------------------------------------------------------- readiness
    def readiness(self) -> ReadinessOut:
        checks: dict[str, CheckOut] = {}
        try:
            check_connection(self.engine)
            checks["database"] = CheckOut(ok=True, detail=self.engine.dialect.name)
        except Exception:
            checks["database"] = CheckOut(ok=False, detail="Database is not reachable.")
        if checks["database"].ok:
            current, head = current_revision(self.engine), head_revision()
            checks["migrations"] = CheckOut(
                ok=current == head, detail=f"current={current} head={head}"
            )
            if self.engine.dialect.name == "postgresql":
                checks["vector_dimension"] = self._check_pg_dimension()
        try:
            provider = self._embedding_provider()
            checks["embedding"] = CheckOut(ok=True, detail=provider.model_id)
        except GroundedAIError as exc:
            checks["embedding"] = CheckOut(ok=False, detail=exc.message)
        ready = all(check.ok for check in checks.values())
        return ReadinessOut(status="ready" if ready else "not_ready", checks=checks)

    def _check_pg_dimension(self) -> CheckOut:
        with self.engine.connect() as connection:
            column_type = connection.execute(
                text(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
                )
            ).scalar()
        expected = f"vector({self.settings.embedding_dim})"
        return CheckOut(
            ok=column_type == expected,
            detail=f"column={column_type} configured={expected}",
        )

    # ---------------------------------------------------------- system status
    def generation_status(self, refresh: bool = False) -> ProviderStatus:
        cached = self._generation_status
        if refresh or cached is None or time.time() - cached.checked_at > PROVIDER_STATUS_TTL_S:
            cached = self._generation_status = self._generation_check(self.settings)
        return cached

    def system_status(self) -> SystemStatusOut:
        settings = self.settings
        generation = self.generation_status()
        try:
            provider = self._embedding_provider()
            embedding = ComponentStatusOut(
                provider=provider.provider,
                model=provider.model_id,
                available=True,
                detail=(
                    "Deterministic hashing embedder (word overlap only, not semantic)."
                    if provider.provider == "hashing"
                    else "Loaded."
                ),
            )
        except GroundedAIError as exc:
            embedding = ComponentStatusOut(
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                available=False,
                detail=exc.message,
            )
        reranker = ComponentStatusOut(
            provider=settings.reranker,
            model=settings.reranker_model if settings.reranker != "none" else None,
            available=None if settings.reranker == "none" else True,
            detail="Disabled." if settings.reranker == "none" else "Loaded on first use.",
        )
        return SystemStatusOut(
            version=__version__,
            environment=settings.env.value,
            demo_mode=not generation.is_llm,
            auth_required=settings.auth_enabled,
            database_backend=self.engine.dialect.name,
            embedding=embedding,
            generation=ComponentStatusOut(
                provider=generation.name,
                model=generation.model,
                is_llm=generation.is_llm,
                available=generation.available,
                detail=generation.detail,
            ),
            reranker=reranker,
            retrieval_defaults={
                "mode": settings.retrieval_mode,
                "fusion": settings.fusion,
                "top_k": settings.top_k,
                "candidate_k": settings.candidate_k,
                "rrf_k": settings.rrf_k,
                "hybrid_alpha": settings.hybrid_alpha,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            limits={
                "max_upload_bytes": settings.max_upload_bytes,
                "allowed_extensions": [".pdf", ".txt", ".md"],
                "max_pdf_pages": settings.max_pdf_pages,
                "max_question_chars": settings.max_question_chars,
                "rate_limit_query_per_minute": settings.rate_limit_query_per_minute
                if settings.rate_limit_enabled
                else None,
            },
        )

    # ------------------------------------------------------ operational metrics
    def operational_metrics(self, window_hours: int = 24) -> OperationalMetricsOut:
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=window_hours)
        with session_scope(self.session_factory) as session:
            documents = DocumentRepository(session).status_counts()
            chunks_total = ChunkRepository(session).count()
            events_repo = QueryEventRepository(session)
            queries_total = events_repo.total()
            events = list(events_repo.since(cutoff))

        by_status: dict[str, int] = {}
        for event in events:
            by_status[event.status] = by_status.get(event.status, 0) + 1
        answered = by_status.get("answered", 0)
        abstained = by_status.get("insufficient_evidence", 0)
        with_citation_check = [e for e in events if e.citations_valid is not None]

        def latency(values: list[float]) -> LatencySummaryOut:
            return LatencySummaryOut(**summarize(values))

        buckets: dict[str, HourlyBucketOut] = {}
        for offset in range(window_hours - 1, -1, -1):
            hour = (now - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0)
            key = hour.isoformat()
            buckets[key] = HourlyBucketOut(hour=key, queries=0, answered=0, abstained=0, failed=0)
        for event in events:
            key = event.created_at.replace(minute=0, second=0, microsecond=0).isoformat()
            bucket = buckets.get(key)
            if bucket is None:
                continue
            bucket.queries += 1
            if event.status == "answered":
                bucket.answered += 1
            elif event.status == "insufficient_evidence":
                bucket.abstained += 1
            else:
                bucket.failed += 1

        snapshot = self.metrics.snapshot()
        return OperationalMetricsOut(
            uptime_s=float(snapshot["uptime_s"]),  # type: ignore[arg-type]
            documents_by_status=documents,
            documents_total=sum(documents.values()),
            chunks_total=chunks_total,
            queries_total=queries_total,
            queries_window_hours=window_hours,
            queries_in_window=len(events),
            queries_by_status=by_status,
            abstention_rate=round(abstained / (answered + abstained), 4)
            if answered + abstained
            else None,
            citation_valid_rate=round(
                sum(1 for e in with_citation_check if e.citations_valid) / len(with_citation_check),
                4,
            )
            if with_citation_check
            else None,
            latency_ms={
                "total": latency([e.total_ms for e in events]),
                "retrieval": latency(
                    [e.retrieval_ms for e in events if e.retrieval_ms is not None]
                ),
                "generation": latency(
                    [e.generation_ms for e in events if e.generation_ms is not None]
                ),
            },
            hourly=list(buckets.values()),
            http_requests_total=self.metrics.counter_total("http_requests"),
            http_server_errors_total=self.metrics.counter_total(
                "http_requests", status_class="5xx"
            ),
            process_counters=list(snapshot["counters"]),  # type: ignore[call-overload]
        )
