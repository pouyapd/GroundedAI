"""The retrieval engine: lexical, dense or hybrid search with optional reranking.

The engine is independent of any particular embedding model or database:
it receives a :class:`~groundedai.persistence.search.SearchBackend` and
callables that return the embedding provider and reranker (so heavy models
load only when a request needs them).
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from sqlalchemy.orm import Session

from groundedai.core.text import content_tokens
from groundedai.persistence.repositories import ChunkRepository, DocumentRepository
from groundedai.persistence.search import ScoredChunkId, SearchBackend
from groundedai.retrieval.embeddings import EmbeddingProvider
from groundedai.retrieval.fusion import (
    order_by_score,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from groundedai.retrieval.rerank import Reranker
from groundedai.retrieval.types import (
    ChunkScores,
    RetrievalDiagnostics,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 3)


class RetrievalEngine:
    def __init__(
        self,
        backend: SearchBackend,
        embedding_provider: Callable[[], EmbeddingProvider],
        reranker: Callable[[], Reranker | None],
    ):
        self.backend = backend
        self._embedding_provider = embedding_provider
        self._reranker = reranker

    def retrieve(self, session: Session, request: RetrievalRequest) -> RetrievalResult:
        started = perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []
        candidate_k = max(request.candidate_k, request.top_k)
        document_ids = list(request.document_ids) if request.document_ids else None

        lexical: list[ScoredChunkId] = []
        dense: list[ScoredChunkId] = []
        embedding_model_id: str | None = None

        if request.mode in ("lexical", "hybrid"):
            step = perf_counter()
            lexical = self.backend.lexical_search(session, request.query, candidate_k, document_ids)
            timings["lexical_ms"] = _elapsed_ms(step)

        if request.mode in ("dense", "hybrid"):
            provider = self._embedding_provider()
            embedding_model_id = provider.model_id
            step = perf_counter()
            query_vector = provider.embed_query(request.query)
            timings["embed_query_ms"] = _elapsed_ms(step)
            step = perf_counter()
            dense = self.backend.dense_search(
                session, query_vector, candidate_k, provider.model_id, document_ids
            )
            timings["dense_ms"] = _elapsed_ms(step)
            warnings.extend(self._embedding_mismatch_warnings(session, provider.model_id))

        step = perf_counter()
        fused = self._fuse(request, lexical, dense)
        ordered = order_by_score(fused)[:candidate_k]
        timings["fusion_ms"] = _elapsed_ms(step)

        lexical_positions = {item.chunk_id: (r, item.score) for r, item in enumerate(lexical, 1)}
        dense_positions = {item.chunk_id: (r, item.score) for r, item in enumerate(dense, 1)}
        rows = ChunkRepository(session).get_many(ordered)

        chunks: list[RetrievedChunk] = []
        seen_hashes: set[str] = set()
        duplicates = 0
        for chunk_id in ordered:
            if chunk_id not in rows:  # deleted between search and fetch
                continue
            row, title = rows[chunk_id]
            if row.text_sha256 in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(row.text_sha256)
            lexical_hit = lexical_positions.get(chunk_id)
            dense_hit = dense_positions.get(chunk_id)
            chunks.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_title=title,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    char_start=row.char_start,
                    char_end=row.char_end,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    scores=ChunkScores(
                        lexical=lexical_hit[1] if lexical_hit else None,
                        lexical_rank=lexical_hit[0] if lexical_hit else None,
                        dense=dense_hit[1] if dense_hit else None,
                        dense_rank=dense_hit[0] if dense_hit else None,
                        fused=fused[chunk_id],
                    ),
                )
            )

        reranker_name: str | None = None
        if request.use_reranker:
            reranker = self._reranker()
            if reranker is None:
                warnings.append("Reranking was requested but no reranker is configured.")
            else:
                step = perf_counter()
                chunks = self._rerank(reranker, request, chunks)
                timings["rerank_ms"] = _elapsed_ms(step)
                reranker_name = reranker.name

        final = chunks[: request.top_k]
        query_terms = set(content_tokens(request.query))
        for rank, chunk in enumerate(final, start=1):
            chunk.rank = rank
            chunk.matched_terms = sorted(query_terms & set(content_tokens(chunk.text)))

        timings["total_ms"] = _elapsed_ms(started)
        diagnostics = RetrievalDiagnostics(
            mode=request.mode,
            fusion=request.fusion if request.mode == "hybrid" else None,
            reranker=reranker_name,
            embedding_model=embedding_model_id,
            lexical_candidates=len(lexical),
            dense_candidates=len(dense),
            fused_candidates=len(ordered),
            duplicates_removed=duplicates,
            timings_ms=timings,
            warnings=warnings,
        )
        return RetrievalResult(chunks=final, diagnostics=diagnostics)

    @staticmethod
    def _fuse(
        request: RetrievalRequest, lexical: list[ScoredChunkId], dense: list[ScoredChunkId]
    ) -> dict[str, float]:
        if request.mode == "lexical":
            return {item.chunk_id: item.score for item in lexical}
        if request.mode == "dense":
            return {item.chunk_id: item.score for item in dense}
        ranked = {"lexical": lexical, "dense": dense}
        weights = {"lexical": 1.0 - request.alpha, "dense": request.alpha}
        if request.fusion == "weighted":
            return weighted_score_fusion(ranked, weights)
        return reciprocal_rank_fusion(ranked, weights, k=request.rrf_k)

    @staticmethod
    def _rerank(
        reranker: Reranker, request: RetrievalRequest, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        head = chunks[: request.rerank_candidates]
        tail = chunks[request.rerank_candidates :]
        scores = reranker.score(request.query, [chunk.text for chunk in head])
        for chunk, score in zip(head, scores, strict=True):
            chunk.scores.rerank = score
        order = sorted(range(len(head)), key=lambda i: (-(head[i].scores.rerank or 0.0), i))
        return [head[i] for i in order] + tail

    @staticmethod
    def _embedding_mismatch_warnings(session: Session, model_id: str) -> list[str]:
        in_use = DocumentRepository(session).embedding_models_in_use()
        other = sum(count for model, count in in_use.items() if model != model_id)
        if not other:
            return []
        return [
            f"{other} document(s) were indexed with a different embedding model and are "
            "excluded from dense search. Re-ingest them to include them."
        ]
