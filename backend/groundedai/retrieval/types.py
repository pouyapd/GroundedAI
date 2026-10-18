"""Retrieval data structures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

RetrievalModeName = Literal["lexical", "dense", "hybrid"]
FusionName = Literal["rrf", "weighted"]


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    mode: RetrievalModeName = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    document_ids: Sequence[str] | None = None
    fusion: FusionName = "rrf"
    alpha: float = 0.5
    rrf_k: int = 60
    use_reranker: bool = False
    rerank_candidates: int = 20


@dataclass(slots=True)
class ChunkScores:
    """Why a chunk was retrieved: component scores and ranks (1-based)."""

    lexical: float | None = None
    lexical_rank: int | None = None
    dense: float | None = None
    dense_rank: int | None = None
    fused: float | None = None
    rerank: float | None = None


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    scores: ChunkScores
    rank: int = 0
    matched_terms: list[str] = field(default_factory=list)
    page_breaks: list[list[int]] | None = None


@dataclass(slots=True)
class RetrievalDiagnostics:
    mode: str
    fusion: str | None
    reranker: str | None
    embedding_model: str | None
    lexical_candidates: int = 0
    dense_candidates: int = 0
    fused_candidates: int = 0
    duplicates_removed: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics
