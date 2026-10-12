import pytest

from groundedai.persistence.search import ScoredChunkId, top_k_scored
from groundedai.retrieval.fusion import (
    min_max_normalize,
    order_by_score,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)

LEXICAL = [ScoredChunkId("a", 9.0), ScoredChunkId("b", 5.0), ScoredChunkId("c", 1.0)]
DENSE = [ScoredChunkId("c", 0.9), ScoredChunkId("a", 0.8), ScoredChunkId("d", 0.1)]


def test_rrf_matches_hand_computed_scores() -> None:
    fused = reciprocal_rank_fusion({"lexical": LEXICAL, "dense": DENSE}, {}, k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert order_by_score(fused) == ["a", "c", "b", "d"]


def test_rrf_weights_shift_the_ranking() -> None:
    fused = reciprocal_rank_fusion(
        {"lexical": LEXICAL, "dense": DENSE}, {"lexical": 0.1, "dense": 0.9}, k=60
    )
    assert order_by_score(fused)[0] == "c"


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"x": LEXICAL}, {}, k=0)


def test_min_max_normalisation() -> None:
    assert min_max_normalize(LEXICAL) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert min_max_normalize([ScoredChunkId("x", 3.0), ScoredChunkId("y", 3.0)]) == {
        "x": 1.0,
        "y": 1.0,
    }
    assert min_max_normalize([]) == {}


def test_weighted_fusion() -> None:
    fused = weighted_score_fusion(
        {"lexical": LEXICAL, "dense": DENSE}, {"lexical": 0.5, "dense": 0.5}
    )
    assert fused["a"] == pytest.approx(0.5 * 1.0 + 0.5 * (0.7 / 0.8))
    assert fused["d"] == pytest.approx(0.0)


def test_ties_are_broken_by_id() -> None:
    assert order_by_score({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]


def test_top_k_scored_is_deterministic() -> None:
    import numpy as np

    result = top_k_scored(["b", "a", "c"], np.array([0.5, 0.5, 0.9]), 2)
    assert [item.chunk_id for item in result] == ["c", "a"]
