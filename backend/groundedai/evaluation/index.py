"""Build an isolated, temporary index of an evaluation corpus.

Evaluation never touches the application's own database: each run ingests the
corpus through the real ingestion pipeline into a fresh SQLite file.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from groundedai.evaluation.dataset import EvalDataset
from groundedai.evaluation.relevance import ChunkSpan, chunks_covering, evidence_span
from groundedai.ingestion.chunking import ChunkingConfig, join_pages
from groundedai.ingestion.parsers import parse_document
from groundedai.persistence.db import create_db_engine, create_session_factory, session_scope
from groundedai.persistence.migrate import upgrade_database
from groundedai.persistence.models import ChunkRow, DocumentStatus
from groundedai.persistence.storage import FileStore
from groundedai.retrieval.embeddings import EmbeddingProvider
from groundedai.services.ingestion_service import IngestionLimits, IngestionService

EVAL_LIMITS = IngestionLimits(
    max_upload_bytes=50 * 1024 * 1024, max_pdf_pages=2000, max_document_chars=10_000_000
)


@dataclass
class IndexedCorpus:
    name: str
    database_url: str
    engine: Engine
    session_factory: sessionmaker[Session]
    embedding_provider: EmbeddingProvider
    doc_ids: dict[str, str]
    doc_texts: dict[str, str]
    chunks: list[ChunkSpan]
    index_seconds: float

    def close(self) -> None:
        self.engine.dispose()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name)[:60]


def build_index(
    dataset: EvalDataset,
    provider: EmbeddingProvider,
    chunking: ChunkingConfig,
    workdir: Path,
    name: str,
    progress: Callable[[str], None] | None = None,
) -> IndexedCorpus:
    workdir.mkdir(parents=True, exist_ok=True)
    database = workdir / f"index-{_safe_name(name)}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)
    url = f"sqlite:///{database.as_posix()}"
    upgrade_database(url, provider.dim)
    engine = create_db_engine(url)
    session_factory = create_session_factory(engine)
    service = IngestionService(
        session_factory=session_factory,
        file_store=FileStore(workdir / f"files-{_safe_name(name)}"),
        embedding_provider=lambda: provider,
        chunking=chunking,
        limits=EVAL_LIMITS,
    )

    doc_ids: dict[str, str] = {}
    doc_texts: dict[str, str] = {}
    started = perf_counter()
    for document in dataset.manifest.documents:
        data = dataset.document_bytes(document.key)
        filename = Path(document.file).name
        row = service.ingest_bytes(filename, None, data, title=document.title)
        if row.status != DocumentStatus.READY:
            raise RuntimeError(
                f"could not index {document.key}: {row.error_code} {row.error_message}"
            )
        doc_ids[document.key] = row.id
        parsed = parse_document(
            data,
            Path(filename).suffix.lower(),
            max_pages=EVAL_LIMITS.max_pdf_pages,
            max_chars=EVAL_LIMITS.max_document_chars,
        )
        doc_texts[document.key] = join_pages(parsed.pages)[0]
        if progress:
            progress(f"  indexed {document.key}: {row.chunk_count} chunks")
    index_seconds = perf_counter() - started

    key_by_id = {document_id: key for key, document_id in doc_ids.items()}
    with session_scope(session_factory) as session:
        rows = session.execute(
            select(ChunkRow.id, ChunkRow.document_id, ChunkRow.char_start, ChunkRow.char_end)
        ).all()
    chunks = [
        ChunkSpan(
            str(row.id), key_by_id[str(row.document_id)], int(row.char_start), int(row.char_end)
        )
        for row in rows
    ]
    return IndexedCorpus(
        name=name,
        database_url=url,
        engine=engine,
        session_factory=session_factory,
        embedding_provider=provider,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        chunks=chunks,
        index_seconds=round(index_seconds, 3),
    )


def judge_dataset(dataset: EvalDataset, corpus: IndexedCorpus) -> dict[str, list[set[str]]]:
    """Relevant chunk ids per evidence item, for every question that has evidence."""
    judgments: dict[str, list[set[str]]] = {}
    for question in dataset.with_evidence():
        evidence_sets = []
        for item in question.evidence:
            span = evidence_span(corpus.doc_texts[item.doc], item.quote)
            relevant = chunks_covering(item.doc, span, corpus.chunks)
            if not relevant:
                raise ValueError(f"{question.id}: no chunk covers the evidence {item.quote[:50]!r}")
            evidence_sets.append(relevant)
        judgments[question.id] = evidence_sets
    return judgments
