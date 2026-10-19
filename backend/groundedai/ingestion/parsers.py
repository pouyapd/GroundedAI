"""Text extraction from PDF and plain-text files.

Document content is treated as data only: nothing in a file is executed,
followed as an instruction or used to build a path. PDF parsing is bounded by
a page limit and a total character limit.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from groundedai.core.errors import DocumentParseError, EmptyDocument, GroundedAIError
from groundedai.core.text import clean_extracted_text

logger = logging.getLogger(__name__)

MAX_TITLE_CHARS = 300


@dataclass(frozen=True, slots=True)
class PageText:
    number: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    pages: list[PageText]
    title: str | None
    page_count: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


def parse_document(
    data: bytes, extension: str, *, max_pages: int, max_chars: int
) -> ParsedDocument:
    if extension == ".pdf":
        return parse_pdf(data, max_pages=max_pages, max_chars=max_chars)
    if extension in {".txt", ".md"}:
        return parse_text(data, markdown=extension == ".md", max_chars=max_chars)
    raise DocumentParseError(f"No parser for '{extension}' files.")


def parse_text(data: bytes, *, markdown: bool, max_chars: int) -> ParsedDocument:
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("Text files must be UTF-8 encoded.") from exc
    text = clean_extracted_text(raw)
    if len(text) > max_chars:
        raise DocumentParseError(
            f"The document has more than {max_chars} characters of text.",
            details={"max_chars": max_chars},
        )
    if not text:
        raise EmptyDocument()
    return ParsedDocument(
        pages=[PageText(number=None, text=text)],
        title=_markdown_title(text) if markdown else _first_line_title(text),
        page_count=None,
    )


def _first_line_title(text: str) -> str | None:
    """A short first line that does not read like a sentence is used as the title."""
    first = text.split("\n", 1)[0].strip()
    if 3 <= len(first) <= 120 and not first.endswith((".", ":", ";", ",")) and "\n" in text:
        return first
    return None


def parse_pdf(data: bytes, *, max_pages: int, max_chars: int) -> ParsedDocument:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        # Some PDFs are "encrypted" with an empty user password; allow those only.
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentParseError("Password-protected PDFs are not supported.")
        page_count = len(reader.pages)
        if page_count > max_pages:
            raise DocumentParseError(
                f"The PDF has {page_count} pages; the limit is {max_pages}.",
                details={"max_pages": max_pages},
            )
        pages: list[PageText] = []
        total_chars = 0
        for number, page in enumerate(reader.pages, start=1):
            text = clean_extracted_text(page.extract_text() or "")
            total_chars += len(text)
            if total_chars > max_chars:
                raise DocumentParseError(
                    f"The document has more than {max_chars} characters of text.",
                    details={"max_chars": max_chars},
                )
            pages.append(PageText(number=number, text=text))
        title = _pdf_title(reader)
    except GroundedAIError:
        raise
    except Exception as exc:  # pypdf raises many exception types on malformed input
        logger.info("pdf_parse_failed", extra={"error_type": type(exc).__name__})
        raise DocumentParseError(
            "The PDF could not be read. It may be damaged or malformed."
        ) from exc

    if not any(page.text for page in pages):
        raise EmptyDocument(
            "No extractable text was found. Scanned PDFs need OCR, which is not supported."
        )
    return ParsedDocument(pages=pages, title=title, page_count=page_count)


def _pdf_title(reader: Any) -> str | None:
    try:
        metadata = reader.metadata
        title = metadata.title if metadata else None
    except Exception:  # metadata is optional and sometimes malformed
        return None
    if not isinstance(title, str):
        return None
    title = clean_extracted_text(title).replace("\n", " ")[:MAX_TITLE_CHARS]
    return title or None


def _markdown_title(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()[:MAX_TITLE_CHARS] or None
