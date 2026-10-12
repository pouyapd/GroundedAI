"""Initial schema: documents, chunks (+ full-text and vector indexes), query events, eval runs.

Revision ID: 0001
Revises:
Create Date: 2026-10-11
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import context, op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    return bool(op.get_context().dialect.name == "postgresql")


def _embedding_dim() -> int:
    dim = context.config.attributes.get("embedding_dim")
    if dim is None:
        from groundedai.core.config import get_settings

        dim = get_settings().embedding_dim
    return int(dim)


def upgrade() -> None:
    postgres = _is_postgres()
    timestamp = sa.DateTime(timezone=True)
    big_int = sa.BigInteger() if postgres else sa.Integer()

    if postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=True),
        sa.Column("chunking", sa.JSON(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
        sa.Column("updated_at", timestamp, nullable=False),
    )
    op.create_index("ix_documents_sha256", "documents", ["sha256"], unique=True)
    op.create_index("ix_documents_status", "documents", ["status"])

    if postgres:
        from pgvector.sqlalchemy import Vector

        embedding_type: sa.types.TypeEngine[Any] = Vector(_embedding_dim())
    else:
        embedding_type = sa.LargeBinary()

    op.create_table(
        "chunks",
        sa.Column("pk", big_int, primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(100), nullable=False),
        sa.Column(
            "document_id",
            sa.String(64),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("text_sha256", sa.String(64), nullable=False),
        sa.Column("embedding", embedding_type, nullable=True),
    )
    op.create_index("ix_chunks_id", "chunks", ["id"], unique=True)
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_text_sha256", "chunks", ["text_sha256"])

    if postgres:
        op.execute(
            "ALTER TABLE chunks ADD COLUMN tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
        )
        op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    else:
        op.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "text, content='chunks', content_rowid='pk', tokenize='porter unicode61')"
        )
        op.execute(
            "CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN "
            "INSERT INTO chunks_fts(rowid, text) VALUES (new.pk, new.text); END"
        )
        op.execute(
            "CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN "
            "INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.pk, old.text); "
            "END"
        )
        op.execute(
            "CREATE TRIGGER chunks_fts_au AFTER UPDATE OF text ON chunks BEGIN "
            "INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.pk, old.text); "
            "INSERT INTO chunks_fts(rowid, text) VALUES (new.pk, new.text); END"
        )

    op.create_table(
        "query_events",
        sa.Column("id", big_int, primary_key=True, autoincrement=True),
        sa.Column("created_at", timestamp, nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("retrieval_mode", sa.String(20), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(200), nullable=True),
        sa.Column("question_chars", sa.Integer(), nullable=False),
        sa.Column("n_retrieved", sa.Integer(), nullable=False),
        sa.Column("n_citations", sa.Integer(), nullable=False),
        sa.Column("citations_valid", sa.Boolean(), nullable=True),
        sa.Column("grounding", sa.String(30), nullable=True),
        sa.Column("retrieval_ms", sa.Float(), nullable=True),
        sa.Column("generation_ms", sa.Float(), nullable=True),
        sa.Column("total_ms", sa.Float(), nullable=False),
    )
    op.create_index("ix_query_events_created_at", "query_events", ["created_at"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", timestamp, nullable=False),
        sa.Column("finished_at", timestamp, nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("report", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
    )
    op.create_index("ix_evaluation_runs_status", "evaluation_runs", ["status"])
    op.create_index("ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"])


def downgrade() -> None:
    op.drop_table("evaluation_runs")
    op.drop_table("query_events")
    if not _is_postgres():
        op.execute("DROP TRIGGER IF EXISTS chunks_fts_au")
        op.execute("DROP TRIGGER IF EXISTS chunks_fts_ad")
        op.execute("DROP TRIGGER IF EXISTS chunks_fts_ai")
        op.execute("DROP TABLE IF EXISTS chunks_fts")
    op.drop_table("chunks")
    op.drop_table("documents")
