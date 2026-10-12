"""Mapping evidence quotes to chunk-level relevance judgments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from groundedai.core.text import find_normalized_span

MIN_EVIDENCE_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    chunk_id: str
    doc_key: str
    start: int
    end: int


def evidence_span(document_text: str, quote: str) -> tuple[int, int]:
    span = find_normalized_span(document_text, quote)
    if span is None:
        raise ValueError(f"evidence quote not found in document: {quote[:60]!r}")
    return span


def chunks_covering(
    doc_key: str,
    span: tuple[int, int],
    chunks: Sequence[ChunkSpan],
    min_coverage: float = MIN_EVIDENCE_COVERAGE,
) -> set[str]:
    """Chunks of ``doc_key`` that contain at least ``min_coverage`` of the evidence span."""
    start, end = span
    length = max(1, end - start)
    relevant = set()
    for chunk in chunks:
        if chunk.doc_key != doc_key:
            continue
        overlap = min(end, chunk.end) - max(start, chunk.start)
        if overlap > 0 and overlap / length >= min_coverage:
            relevant.add(chunk.chunk_id)
    return relevant
