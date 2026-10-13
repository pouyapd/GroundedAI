"""Deterministic evaluation metrics.

Retrieval metrics follow the usual IR definitions with binary relevance:

* ``recall@k`` - fraction of a question's evidence items covered by at least
  one of the top-k chunks (evidence-level recall, robust to chunking).
* ``hit@k`` - 1 if any evidence item is covered in the top k.
* ``MRR`` - reciprocal rank of the first relevant chunk (0 if none retrieved).
* ``nDCG@k`` - binary-gain DCG over chunks, normalised by the ideal ranking.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

import numpy as np

from groundedai.core.text import normalize_for_match


def recall_at_k(ranked: Sequence[str], evidence_sets: Sequence[set[str]], k: int) -> float:
    if not evidence_sets:
        raise ValueError("recall@k needs at least one evidence item")
    top = set(ranked[:k])
    return sum(1 for relevant in evidence_sets if relevant & top) / len(evidence_sets)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("nDCG needs at least one relevant item")
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked[:k], start=1)
        if chunk_id in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def bootstrap_ci(
    values: Sequence[float], *, samples: int = 2000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval of the mean."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def paired_bootstrap_difference(
    a: Sequence[float], b: Sequence[float], *, samples: int = 2000, seed: int = 0
) -> dict[str, float | bool]:
    """Mean of (a - b) over paired questions with a 95% bootstrap interval.

    ``distinguishable`` is True when the interval excludes zero. With small
    datasets most differences will not be distinguishable; reports say so.
    """
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired comparison needs two equal-length samples (n >= 2)")
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    interval = bootstrap_ci(list(diffs), samples=samples, seed=seed)
    if interval is None:  # pragma: no cover - guarded by the length check above
        raise ValueError("bootstrap interval unavailable")
    low, high = interval
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "distinguishable": bool(low > 0 or high < 0),
    }


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation) - {"€", "$", "%"}


def _normalize_answer(text: str) -> str:
    text = normalize_for_match(text)
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between normalised strings."""
    predicted = _normalize_answer(prediction).split()
    gold = _normalize_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    common = Counter(predicted) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def fact_in_text(fact: str, text: str) -> bool:
    """Whether one key fact ('|'-separated alternatives) occurs in ``text``."""
    normalized = _normalize_answer(text)
    alternatives = [_normalize_answer(option) for option in fact.split("|")]
    return any(option and option in normalized for option in alternatives)


def key_fact_recall(answer: str, facts: Sequence[str]) -> float | None:
    """Fraction of key facts present in the answer; each fact may list '|' alternatives."""
    if not facts:
        return None
    return sum(1 for fact in facts if fact_in_text(fact, answer)) / len(facts)


def contains_any(answer: str, phrases: Sequence[str]) -> bool:
    normalized = normalize_for_match(answer)
    return any(normalize_for_match(phrase) in normalized for phrase in phrases if phrase)
