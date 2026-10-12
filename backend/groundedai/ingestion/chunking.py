"""Deterministic chunking with source-location tracking.

Pages are joined into one text so that chunks may cross page boundaries; each
chunk keeps its character span in that text and the first/last page it covers.
Splitting uses LangChain's ``RecursiveCharacterTextSplitter`` (paragraph, then
line, then sentence, then word boundaries).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from groundedai.ingestion.parsers import PageText

PAGE_SEPARATOR = "\n\n"
CHUNKER_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "splitter": "langchain_text_splitters.RecursiveCharacterTextSplitter",
            "version": CHUNKER_VERSION,
        }


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PageSpan:
    start: int
    end: int
    number: int | None


def join_pages(pages: Sequence[PageText]) -> tuple[str, list[PageSpan]]:
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for page in pages:
        if not page.text:
            continue
        if parts:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)
        spans.append(PageSpan(cursor, cursor + len(page.text), page.number))
        parts.append(page.text)
        cursor += len(page.text)
    return "".join(parts), spans


def chunk_id_for(document_id: str, index: int) -> str:
    return f"{document_id}:{index:04d}"


def _make_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator="end",
        strip_whitespace=True,
        add_start_index=True,
    )


def chunk_pages(pages: Sequence[PageText], config: ChunkingConfig) -> list[ChunkSpec]:
    full_text, spans = join_pages(pages)
    if not full_text.strip():
        return []
    documents = _make_splitter(config).create_documents([full_text])
    chunks: list[ChunkSpec] = []
    search_from = 0
    for document in documents:
        text = document.page_content
        start = int(document.metadata.get("start_index", -1))
        if start < 0 or full_text[start : start + len(text)] != text:
            # Recover the offset ourselves if the splitter's estimate is off.
            start = full_text.find(text, search_from)
            if start < 0:
                start = full_text.find(text)
            if start < 0:  # pragma: no cover - would indicate a splitter bug
                raise RuntimeError("chunk text not found in source text")
        end = start + len(text)
        search_from = start + 1
        covered = [span.number for span in spans if span.start < end and span.end > start]
        numbered = [number for number in covered if number is not None]
        chunks.append(
            ChunkSpec(
                index=len(chunks),
                text=text,
                char_start=start,
                char_end=end,
                page_start=min(numbered) if numbered else None,
                page_end=max(numbered) if numbered else None,
            )
        )
    return chunks
