"""Rank fusion for hybrid retrieval.

* Reciprocal Rank Fusion (Cormack et al., 2009): ``score = sum_i w_i / (k + rank_i)``.
  Uses ranks only, so it needs no score calibration between BM25 and cosine.
* Weighted score fusion: min-max normalise each list, then a weighted sum.
  Sensitive to score distributions; included as a comparison baseline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from groundedai.persistence.search import ScoredChunkId


def reciprocal_rank_fusion(
    ranked: Mapping[str, Sequence[ScoredChunkId]],
    weights: Mapping[str, float],
    k: int = 60,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    for name, results in ranked.items():
        weight = weights.get(name, 1.0)
        for rank, item in enumerate(results, start=1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + weight / (k + rank)
    return fused


def min_max_normalize(results: Sequence[ScoredChunkId]) -> dict[str, float]:
    if not results:
        return {}
    scores = [item.score for item in results]
    low, high = min(scores), max(scores)
    if high == low:
        return {item.chunk_id: 1.0 for item in results}
    return {item.chunk_id: (item.score - low) / (high - low) for item in results}


def weighted_score_fusion(
    ranked: Mapping[str, Sequence[ScoredChunkId]], weights: Mapping[str, float]
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for name, results in ranked.items():
        weight = weights.get(name, 1.0)
        for chunk_id, score in min_max_normalize(results).items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight * score
    return fused


def order_by_score(scores: Mapping[str, float]) -> list[str]:
    """Descending score; ties broken by chunk id for determinism."""
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
