"""Database-backed lexical and vector search.

Two interchangeable backends implement the same protocol:

* :class:`SqliteSearchBackend` - FTS5 with BM25 ranking for lexical search and
  an exact cosine scan in NumPy for vector search. No server needed.
* :class:`PostgresSearchBackend` - a generated ``tsvector`` column ranked with
  ``ts_rank_cd`` and pgvector cosine distance (HNSW index).

The two lexical rankers are not identical (BM25 vs. cover-density ranking),
so absolute scores are not comparable across backends. Published benchmark
numbers state which backend produced them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from groundedai.core.text import STOPWORDS, tokenize
from groundedai.persistence.models import DocumentStatus

MAX_QUERY_TERMS = 32


@dataclass(frozen=True, slots=True)
class ScoredChunkId:
    chunk_id: str
    score: float


class SearchBackend(Protocol):
    name: str

    def lexical_search(
        self, session: Session, query: str, k: int, document_ids: Sequence[str] | None = None
    ) -> list[ScoredChunkId]: ...

    def dense_search(
        self,
        session: Session,
        query_embedding: np.ndarray,
        k: int,
        embedding_model: str,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunkId]: ...


def query_terms(query: str) -> list[str]:
    """Search terms from free text: word characters only, no stop words, de-duplicated.

    Restricting terms to word characters means user input can never inject
    FTS5 or ``tsquery`` operators.
    """
    seen: dict[str, None] = {}
    for token in tokenize(query):
        if token not in STOPWORDS and token not in seen:
            seen[token] = None
        if len(seen) >= MAX_QUERY_TERMS:
            break
    return list(seen)


def _document_filter_sql(document_ids: Sequence[str] | None) -> str:
    return " AND c.document_id IN :document_ids" if document_ids else ""


class SqliteSearchBackend:
    name = "sqlite"

    def lexical_search(
        self, session: Session, query: str, k: int, document_ids: Sequence[str] | None = None
    ) -> list[ScoredChunkId]:
        terms = query_terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{term}"' for term in terms)
        statement = text(
            "SELECT c.id AS chunk_id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.pk = chunks_fts.rowid "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE chunks_fts MATCH :match AND d.status = :ready"
            f"{_document_filter_sql(document_ids)} "
            "ORDER BY rank, c.id LIMIT :k"
        )
        params: dict[str, object] = {"match": match, "ready": DocumentStatus.READY.value, "k": k}
        if document_ids:
            statement = statement.bindparams(bindparam("document_ids", expanding=True))
            params["document_ids"] = list(document_ids)
        rows = session.execute(statement, params).all()
        # FTS5 bm25() returns lower-is-better negative numbers; flip the sign.
        return [ScoredChunkId(str(row.chunk_id), -float(row.rank)) for row in rows]

    def dense_search(
        self,
        session: Session,
        query_embedding: np.ndarray,
        k: int,
        embedding_model: str,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunkId]:
        statement = text(
            "SELECT c.id AS chunk_id, c.embedding AS embedding "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.status = :ready AND d.embedding_model = :model AND c.embedding IS NOT NULL"
            f"{_document_filter_sql(document_ids)}"
        )
        params: dict[str, object] = {
            "ready": DocumentStatus.READY.value,
            "model": embedding_model,
        }
        if document_ids:
            statement = statement.bindparams(bindparam("document_ids", expanding=True))
            params["document_ids"] = list(document_ids)
        rows = session.execute(statement, params).all()
        if not rows:
            return []
        query = np.asarray(query_embedding, dtype=np.float32)
        matrix = np.vstack([np.frombuffer(row.embedding, dtype=np.float32) for row in rows])
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"embedding dimension mismatch: index has {matrix.shape[1]}, "
                f"query has {query.shape[0]}"
            )
        scores = cosine_scores(matrix, query)
        ids = [str(row.chunk_id) for row in rows]
        return top_k_scored(ids, scores, k)


class PostgresSearchBackend:
    name = "postgresql"

    def lexical_search(
        self, session: Session, query: str, k: int, document_ids: Sequence[str] | None = None
    ) -> list[ScoredChunkId]:
        terms = query_terms(query)
        if not terms:
            return []
        statement = text(
            "SELECT c.id AS chunk_id, ts_rank_cd(c.tsv, q.query) AS rank "
            "FROM chunks c JOIN documents d ON d.id = c.document_id, "
            "to_tsquery('english', :tsquery) AS q(query) "
            "WHERE c.tsv @@ q.query AND d.status = :ready"
            f"{_document_filter_sql(document_ids)} "
            "ORDER BY rank DESC, c.id LIMIT :k"
        )
        params: dict[str, object] = {
            "tsquery": " | ".join(terms),
            "ready": DocumentStatus.READY.value,
            "k": k,
        }
        if document_ids:
            statement = statement.bindparams(bindparam("document_ids", expanding=True))
            params["document_ids"] = list(document_ids)
        rows = session.execute(statement, params).all()
        return [ScoredChunkId(str(row.chunk_id), float(row.rank)) for row in rows]

    def dense_search(
        self,
        session: Session,
        query_embedding: np.ndarray,
        k: int,
        embedding_model: str,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunkId]:
        from pgvector.sqlalchemy import Vector

        statement = text(
            "SELECT c.id AS chunk_id, 1 - (c.embedding <=> :query_vector) AS score "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.status = :ready AND d.embedding_model = :model AND c.embedding IS NOT NULL"
            f"{_document_filter_sql(document_ids)} "
            "ORDER BY c.embedding <=> :query_vector, c.id LIMIT :k"
        ).bindparams(bindparam("query_vector", type_=Vector()))
        params: dict[str, object] = {
            "query_vector": np.asarray(query_embedding, dtype=np.float32),
            "ready": DocumentStatus.READY.value,
            "model": embedding_model,
            "k": k,
        }
        if document_ids:
            statement = statement.bindparams(bindparam("document_ids", expanding=True))
            params["document_ids"] = list(document_ids)
        rows = session.execute(statement, params).all()
        return [ScoredChunkId(str(row.chunk_id), float(row.score)) for row in rows]


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``matrix`` and ``query`` (zero-safe)."""
    row_norms = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominator = row_norms * query_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, (matrix @ query) / denominator, 0.0)
    return scores.astype(np.float64)


def top_k_scored(ids: Sequence[str], scores: np.ndarray, k: int) -> list[ScoredChunkId]:
    """Top-k by score, ties broken by id so results are deterministic."""
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:k]
    return [ScoredChunkId(ids[i], float(scores[i])) for i in order]


def backend_for_dialect(dialect_name: str) -> SearchBackend:
    if dialect_name == "postgresql":
        return PostgresSearchBackend()
    if dialect_name == "sqlite":
        return SqliteSearchBackend()
    raise ValueError(f"unsupported database dialect: {dialect_name}")
