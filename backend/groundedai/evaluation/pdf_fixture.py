"""A minimal, dependency-free PDF writer for test fixtures and the synthetic corpus.

Produces simple text-only PDFs (Helvetica, WinAnsi encoding) that pypdf can
parse. It exists so that tests and the evaluation dataset do not depend on a
PDF-generation library; it is not used by the application at runtime.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

PAGE_WIDTH = 612
PAGE_HEIGHT = 792


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _wrap(page_text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in page_text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(paragraph, width, break_on_hyphens=False, break_long_words=False)
        )
    return lines


def make_pdf(
    pages: Sequence[str],
    *,
    title: str | None = None,
    font_size: int = 10,
    line_height: int = 13,
    wrap_width: int = 95,
) -> bytes:
    """Return the bytes of a PDF with one text page per entry in ``pages``."""
    streams = []
    for page_text in pages:
        operations = [
            b"BT",
            f"/F1 {font_size} Tf".encode(),
            f"{line_height} TL".encode(),
            f"50 {PAGE_HEIGHT - 40} Td".encode(),
        ]
        for line in _wrap(page_text, wrap_width):
            operations.append(b"(" + _escape(line) + b") Tj T*")
        operations.append(b"ET")
        streams.append(b"\n".join(operations))

    objects: dict[int, bytes] = {}
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(len(streams)))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode()
    objects[3] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    objects[4] = (
        b"<< /Title (" + _escape(title or "") + b") /Producer (GroundedAI fixture writer) >>"
    )
    for index, stream in enumerate(streams):
        page_number, content_number = 5 + 2 * index, 6 + 2 * index
        objects[page_number] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>"
        ).encode()
        objects[content_number] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref_position = len(output)
    size = max(objects) + 1
    output += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for number in range(1, size):
        output += f"{offsets[number]:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {size} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(output)
