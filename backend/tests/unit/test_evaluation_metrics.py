import math

import pytest

from groundedai.evaluation.metrics import (
    bootstrap_ci,
    contains_any,
    hit_at_k,
    key_fact_recall,
    ndcg_at_k,
    paired_bootstrap_difference,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from groundedai.evaluation.relevance import ChunkSpan, chunks_covering

RANKED = ["x", "a", "y", "b", "z"]


def test_recall_at_k_is_evidence_level() -> None:
    evidence = [{"a"}, {"b", "q"}]
    assert recall_at_k(RANKED, evidence, 1) == 0.0
    assert recall_at_k(RANKED, evidence, 2) == 0.5
    assert recall_at_k(RANKED, evidence, 4) == 1.0
    with pytest.raises(ValueError, match="evidence"):
        recall_at_k(RANKED, [], 3)


def test_hit_and_reciprocal_rank() -> None:
    assert hit_at_k(RANKED, {"b"}, 3) == 0.0
    assert hit_at_k(RANKED, {"b"}, 4) == 1.0
    assert reciprocal_rank(RANKED, {"y", "b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"missing"}) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    # Relevant at ranks 2 and 4; ideal has them at ranks 1 and 2.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"a", "b"}, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, {"a", "b"}, 1) == 0.0


def test_bootstrap_interval_is_reproducible_and_contains_mean() -> None:
    values = [0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0]
    first = bootstrap_ci(values, samples=500, seed=3)
    second = bootstrap_ci(values, samples=500, seed=3)
    assert first == second
    assert first is not None
    assert first[0] <= sum(values) / len(values) <= first[1]
    assert bootstrap_ci([1.0]) is None


def test_paired_difference() -> None:
    better = [1.0] * 20
    worse = [0.0] * 10 + [1.0] * 10
    result = paired_bootstrap_difference(better, worse, samples=500, seed=0)
    assert result["mean_diff"] == pytest.approx(0.5)
    assert result["distinguishable"] is True
    same = paired_bootstrap_difference(better, better, samples=200, seed=0)
    assert same["distinguishable"] is False
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])


def test_token_f1() -> None:
    assert token_f1("The limit is €140.", "limit is €140") == pytest.approx(1.0)
    assert token_f1("completely different", "limit is €140") == 0.0
    assert token_f1("", "") == 1.0


def test_key_fact_recall_with_alternatives() -> None:
    answer = "Reimbursement is €45 per day; receipts above 25 euros."
    assert key_fact_recall(answer, ["€45|45 euros", "€25|25 euros"]) == 1.0
    assert key_fact_recall(answer, ["€45", "€99"]) == 0.5
    assert key_fact_recall(answer, []) is None


def test_contains_any_is_case_insensitive() -> None:
    assert contains_any("All invoices are PAID WITHIN 3 DAYS", ["paid within 3 days"])
    assert not contains_any("net 45 days", ["within 3 days"])


def test_chunk_coverage_threshold() -> None:
    chunks = [
        ChunkSpan("c1", "doc", 0, 100),
        ChunkSpan("c2", "doc", 80, 180),
        ChunkSpan("c3", "other", 0, 500),
    ]
    assert chunks_covering("doc", (85, 105), chunks) == {"c1", "c2"}  # c1 covers 15/20
    assert chunks_covering("doc", (60, 100), chunks) == {"c1", "c2"}  # c2 covers 20/40 = 0.5
    assert chunks_covering("doc", (55, 100), chunks) == {"c1"}  # c2 covers 20/45 < 0.5
    assert chunks_covering("doc", (55, 100), chunks, min_coverage=0.4) == {"c1", "c2"}
