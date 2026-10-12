"""ORM models.

Search-specific structures are created by the migration, not mapped here:
an FTS5 virtual table on SQLite, and a generated ``tsvector`` column plus GIN
and HNSW indexes on PostgreSQL.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, ClassVar

import numpy as np
from sqlalchemy import JSON, BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from groundedai.persistence.column_types import EmbeddingVector, UTCDateTime


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON, datetime: UTCDateTime}


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), index=True, default=DocumentStatus.PENDING)
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    error_message: Mapped[str | None] = mapped_column(String(500), default=None)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    char_count: Mapped[int | None] = mapped_column(Integer, default=None)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str | None] = mapped_column(String(200), default=None)
    chunking: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChunkRow.chunk_index",
    )


class ChunkRow(Base):
    __tablename__ = "chunks"

    # Integer surrogate key: SQLite FTS5 external-content tables need a stable rowid.
    pk: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer, default=None)
    page_end: Mapped[int | None] = mapped_column(Integer, default=None)
    text_sha256: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[np.ndarray | None] = mapped_column(EmbeddingVector(), default=None)

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


class QueryEventRow(Base):
    """One answered (or failed) question. Stores timings and outcomes, never the text."""

    __tablename__ = "query_events"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    status: Mapped[str] = mapped_column(String(40))
    error_code: Mapped[str | None] = mapped_column(String(64), default=None)
    retrieval_mode: Mapped[str] = mapped_column(String(20))
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str | None] = mapped_column(String(200), default=None)
    question_chars: Mapped[int] = mapped_column(Integer)
    n_retrieved: Mapped[int] = mapped_column(Integer, default=0)
    n_citations: Mapped[int] = mapped_column(Integer, default=0)
    citations_valid: Mapped[bool | None] = mapped_column(default=None)
    grounding: Mapped[str | None] = mapped_column(String(30), default=None)
    retrieval_ms: Mapped[float | None] = mapped_column(default=None)
    generation_ms: Mapped[float | None] = mapped_column(default=None)
    total_ms: Mapped[float] = mapped_column(default=0.0)


class EvaluationRunRow(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(default=None)
    config: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    summary: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    report: Mapped[dict[str, Any] | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(String(500), default=None)
