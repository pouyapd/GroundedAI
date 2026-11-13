"""Optional clean-up of text extracted from PDFs (text clean-up ``v2``).

Real publications carry layout text that is not part of the content, and PDF text extraction
keeps it. Three things are repaired, each conservatively:

* **Ligatures.** Typeset PDFs often store "fi", "fl", "ff" as one character (U+FB00-FB06),
  so "Arti\\ufb01cial" does not match a search for "artificial". They are expanded.
* **Words hyphenated at a line break.** "trustwor-" + newline + "thy" becomes "trustworthy".
  Only a hyphen between two lower-case letters at the end of a line is removed; "800-" before
  "63" and "Self-" before "Assessment" are kept.
* **Running headers and footers.** A line among the first or last four lines of a page that
  recurs on at least half of the pages is removed. Digits are ignored, so "Page 21" and
  "Page 22" count as the same line (and so do two edge lines that differ only in their
  numbers). An edge line that holds only a number ("21") is removed only when it follows the
  page order: the number minus the page's position in the file is the same on at least half of
  the pages, as printed page numbers are. A number at a page edge that does not follow the page
  order (a table value, a list item) is kept. Lower-case roman numerals ("iv", "Page ii", the
  front matter) and rules of underscores at page edges are removed too. Short documents (fewer
  than four pages with text) are left alone.

Page numbers are unaffected: each page keeps its own text, only shortened. Version ``v1`` (the
default) applies none of this and reproduces every earlier index and result.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal

TextCleanup = Literal["v1", "v2"]

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}
_LIGATURE_RE = re.compile("[" + "".join(LIGATURES) + "]")
_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_DIGITS_RE = re.compile(r"\d+")
_ARABIC_PAGE_RE = re.compile(r"^(?:page\s+)?(\d{1,4})$", re.IGNORECASE)
# A valid lower-case roman numeral up to 399 (front-matter page numbers), optionally "Page ".
_ROMAN_PAGE_RE = re.compile(
    r"^(?:[Pp]age\s+)?(?=[ivxlc])c{0,3}(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$"
)
_RULE_RE = re.compile(r"^[_\-=—–\s]{8,}$")
EDGE_LINES = 4
MIN_PAGES = 4
REPEAT_SHARE = 0.5


def expand_ligatures(text: str) -> str:
    return _LIGATURE_RE.sub(lambda match: LIGATURES[match.group(0)], text)


def join_hyphenated_lines(text: str) -> str:
    return _HYPHEN_BREAK_RE.sub("", text)


def _key(line: str) -> str:
    return _DIGITS_RE.sub("#", line.strip().casefold())


def _edge_indices(lines: Sequence[str]) -> list[int]:
    filled = [index for index, line in enumerate(lines) if line.strip()]
    return sorted(set(filled[:EDGE_LINES] + filled[-EDGE_LINES:]))


def repeated_edge_lines(pages: Sequence[str]) -> set[str]:
    """Keys of lines that recur at the top or bottom of at least half of the pages."""
    with_text = [page for page in pages if page.strip()]
    if len(with_text) < MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for page in with_text:
        lines = page.split("\n")
        counts.update({_key(lines[index]) for index in _edge_indices(lines)})
    threshold = max(MIN_PAGES - 1, REPEAT_SHARE * len(with_text))
    return {key for key, count in counts.items() if count >= threshold and key.strip("#")}


def page_number_offsets(pages: Sequence[str]) -> set[int]:
    """Offsets (printed number minus position in the file) shared by most pages' edge numbers."""
    with_text = [page for page in pages if page.strip()]
    if len(with_text) < MIN_PAGES:
        return set()
    counts: Counter[int] = Counter()
    for position, page in enumerate(pages, start=1):
        lines = page.split("\n")
        found = set()
        for index in _edge_indices(lines):
            match = _ARABIC_PAGE_RE.match(lines[index].strip())
            if match:
                found.add(int(match.group(1)) - position)
        counts.update(found)
    threshold = max(MIN_PAGES - 1, REPEAT_SHARE * len(with_text))
    return {offset for offset, count in counts.items() if count >= threshold}


def is_page_number(line: str, position: int, offsets: set[int]) -> bool:
    """A printed page number: arabic and in the page order, or a lower-case roman numeral."""
    match = _ARABIC_PAGE_RE.match(line)
    if match:
        return int(match.group(1)) - position in offsets
    return bool(_ROMAN_PAGE_RE.match(line))


def strip_edge_lines(
    page: str, repeated: set[str], position: int = 0, offsets: set[int] | None = None
) -> str:
    lines = page.split("\n")
    drop = set()
    for index in _edge_indices(lines):
        stripped = lines[index].strip()
        if (
            _key(stripped) in repeated
            or is_page_number(stripped, position, offsets or set())
            or _RULE_RE.match(stripped)
        ):
            drop.add(index)
    return "\n".join(line for index, line in enumerate(lines) if index not in drop).strip()


def clean_pdf_pages(pages: Sequence[str]) -> list[str]:
    """Apply the v2 clean-up to the text of every page of one PDF (order preserved)."""
    expanded = [expand_ligatures(page) for page in pages]
    repeated = repeated_edge_lines(expanded)
    offsets = page_number_offsets(expanded)
    return [
        join_hyphenated_lines(strip_edge_lines(page, repeated, position, offsets))
        for position, page in enumerate(expanded, start=1)
    ]
