"""Evaluation runs triggered through the API.

A run evaluates the instance's *current* retrieval configuration on the
bundled dataset, in a temporary isolated index, and stores the report in the
``evaluation_runs`` table. Only one run may be active at a time.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from groundedai.core.config import Settings
from groundedai.core.errors import Conflict, NotFound
from groundedai.evaluation.dataset import load_dataset
from groundedai.evaluation.reporting import list_reports, load_report
from groundedai.evaluation.retrieval_benchmark import (
    BenchmarkConfig,
    ConfigSpec,
    EmbedderSpec,
    run_retrieval_benchmark,
)
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import EvaluationRunRow, utcnow
from groundedai.persistence.repositories import EvaluationRunRepository
from groundedai.retrieval.embeddings import EmbeddingProvider
from groundedai.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        embedding_provider: Callable[[], EmbeddingProvider],
        reranker: Callable[[], Reranker | None],
    ):
        self.settings = settings
        self.session_factory = session_factory
        self._embedding_provider = embedding_provider
        self._reranker = reranker

    # --------------------------------------------------------- published reports
    def published_reports(self) -> list[dict[str, Any]]:
        return list_reports(self.settings.published_reports_dir)

    def published_report(self, report_id: str) -> dict[str, Any]:
        report = load_report(self.settings.published_reports_dir, report_id)
        if report is None:
            raise NotFound("Report not found.")
        return report

    # ------------------------------------------------------------- live runs
    def current_benchmark_config(self) -> BenchmarkConfig:
        settings = self.settings
        embedder = EmbedderSpec(
            key="configured",
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            query_prefix=settings.embedding_query_prefix,
        )
        configurations = [
            ConfigSpec(name="lexical", mode="lexical", description="FTS5 BM25"),
            ConfigSpec(name="dense", mode="dense", embedder="configured"),
            ConfigSpec(
                name=f"hybrid-{settings.fusion}",
                mode="hybrid",
                embedder="configured",
                fusion=settings.fusion,
                alpha=settings.hybrid_alpha,
                rrf_k=settings.rrf_k,
            ),
        ]
        if settings.reranker == "cross-encoder":
            configurations.append(
                ConfigSpec(
                    name=f"hybrid-{settings.fusion}+rerank",
                    mode="hybrid",
                    embedder="configured",
                    fusion=settings.fusion,
                    alpha=settings.hybrid_alpha,
                    rrf_k=settings.rrf_k,
                    reranker=settings.reranker_model,
                )
            )
        return BenchmarkConfig(
            title="Retrieval evaluation of this instance's configuration",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            k_values=(1, 3, 5, 10),
            candidate_k=max(settings.candidate_k, 10),
            rerank_candidates=settings.rerank_candidates,
            bootstrap_samples=1000,
            seed=0,
            baseline="lexical",
            embedders={"configured": embedder},
            configurations=configurations,
        )

    def create_run(self) -> EvaluationRunRow:
        config = self.current_benchmark_config()
        with session_scope(self.session_factory) as session:
            repository = EvaluationRunRepository(session)
            if repository.running_count():
                raise Conflict("An evaluation run is already in progress.")
            row = EvaluationRunRow(
                id=f"run_{uuid.uuid4().hex[:16]}",
                kind="retrieval",
                status="queued",
                config={**config.as_dict(), "dataset": self.settings.evaluation_dataset},
            )
            repository.add(row)
            return row

    def execute_run(self, run_id: str) -> None:
        """Run synchronously (called from a background task)."""
        with session_scope(self.session_factory) as session:
            EvaluationRunRepository(session).update(run_id, status="running")
        try:
            dataset = load_dataset(self.settings.dataset_dir)
            config = self.current_benchmark_config()
            reranker = self._reranker()
            with tempfile.TemporaryDirectory(
                prefix="groundedai-eval-", ignore_cleanup_errors=True
            ) as workdir:
                report = run_retrieval_benchmark(
                    dataset,
                    config,
                    workdir=Path(workdir),
                    provider_factory=lambda _spec: self._embedding_provider(),
                    reranker_factory=lambda _model: _require(reranker),
                )
            report["report_id"] = run_id
            summary = {
                item["name"]: {
                    name: item["metrics"][name]["mean"] for name in ("recall@5", "mrr", "ndcg@5")
                }
                for item in report["configurations"]
            }
            with session_scope(self.session_factory) as session:
                EvaluationRunRepository(session).update(
                    run_id, status="completed", finished_at=utcnow(), summary=summary, report=report
                )
            logger.info("evaluation_completed", extra={"run_id": run_id})
        except Exception as exc:
            logger.exception("evaluation_failed", extra={"run_id": run_id})
            message = getattr(exc, "message", None) or "Evaluation failed. See server logs."
            with session_scope(self.session_factory) as session:
                EvaluationRunRepository(session).update(
                    run_id, status="failed", finished_at=utcnow(), error_message=str(message)[:500]
                )

    def recover_interrupted(self) -> None:
        with session_scope(self.session_factory) as session:
            repository = EvaluationRunRepository(session)
            rows, _ = repository.page(offset=0, limit=1000)
            for row in rows:
                if row.status in ("queued", "running"):
                    row.status = "failed"
                    row.error_message = "Interrupted by a server restart."
                    row.finished_at = utcnow()


def _require(reranker: Reranker | None) -> Reranker:
    if reranker is None:
        raise Conflict("A reranker configuration was requested but none is configured.")
    return reranker
