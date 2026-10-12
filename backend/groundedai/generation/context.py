"""Bounded context assembly with stable citation labels."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.documents import Document

from groundedai.retrieval.types import RetrievedChunk

_TAG_LIKE = re.compile(r"<\s*/?\s*(source|question|system|assistant|instructions?)\b", re.I)
MIN_TRUNCATED_CHARS = 200


def neutralize_tags(text: str) -> str:
    """Stop retrieved text from opening/closing prompt blocks (``<source>`` etc.)."""
    return _TAG_LIKE.sub(lambda match: match.group(0).replace("<", "‹"), text)


def page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


@dataclass(frozen=True, slots=True)
class SourceBlock:
    label: str
    chunk: RetrievedChunk
    text: str
    truncated: bool


@dataclass(slots=True)
class AssembledContext:
    sources: list[SourceBlock]
    text: str
    dropped_chunk_ids: list[str] = field(default_factory=list)

    @property
    def labels(self) -> dict[str, SourceBlock]:
        return {block.label: block for block in self.sources}

    def as_documents(self) -> list[Document]:
        return [
            Document(
                page_content=block.text,
                id=block.chunk.chunk_id,
                metadata={
                    "label": block.label,
                    "chunk_id": block.chunk.chunk_id,
                    "document_title": block.chunk.document_title,
                    "pages": page_label(block.chunk.page_start, block.chunk.page_end),
                },
            )
            for block in self.sources
        ]


def format_source(label: str, title: str, pages: str, body: str) -> str:
    attributes = f'id="{label}" title="{html.escape(title, quote=True)}"'
    if pages:
        attributes += f' pages="{pages}"'
    return f"<source {attributes}>\n{body}\n</source>"


def format_documents(documents: Sequence[Document]) -> str:
    """Render LangChain documents (from :meth:`AssembledContext.as_documents`)."""
    return "\n\n".join(
        format_source(
            str(doc.metadata["label"]),
            str(doc.metadata["document_title"]),
            str(doc.metadata.get("pages") or ""),
            doc.page_content,
        )
        for doc in documents
    )


def assemble_context(chunks: Sequence[RetrievedChunk], max_chars: int) -> AssembledContext:
    """Label chunks S1..Sn in rank order until the character budget is used.

    The top chunk is truncated rather than dropped when it alone exceeds the
    budget; later chunks that do not fit are dropped and reported.
    """
    blocks: list[SourceBlock] = []
    dropped: list[str] = []
    used = 0
    for chunk in chunks:
        label = f"S{len(blocks) + 1}"
        body = neutralize_tags(chunk.text)
        pages = page_label(chunk.page_start, chunk.page_end)
        overhead = len(format_source(label, chunk.document_title, pages, "")) + 2
        truncated = False
        if used + overhead + len(body) > max_chars:
            room = max_chars - used - overhead
            if blocks or room < MIN_TRUNCATED_CHARS:
                dropped.append(chunk.chunk_id)
                continue
            body = body[:room].rsplit(" ", 1)[0] + " …"
            truncated = True
        blocks.append(SourceBlock(label=label, chunk=chunk, text=body, truncated=truncated))
        used += overhead + len(body)
    text = "\n\n".join(
        format_source(
            block.label,
            block.chunk.document_title,
            page_label(block.chunk.page_start, block.chunk.page_end),
            block.text,
        )
        for block in blocks
    )
    return AssembledContext(sources=blocks, text=text, dropped_chunk_ids=dropped)
