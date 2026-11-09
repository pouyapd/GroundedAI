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
  numbers). Lines that only hold a page number or a rule of underscores are removed too.
  Short documents (fewer than four pages with text) are left alone.

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
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,6})$", re.IGNORECASE)
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


def strip_edge_lines(page: str, repeated: set[str]) -> str:
    lines = page.split("\n")
    drop = set()
    for index in _edge_indices(lines):
        stripped = lines[index].strip()
        if (
            _key(stripped) in repeated
            or _PAGE_NUMBER_RE.match(stripped)
            or _RULE_RE.match(stripped)
        ):
            drop.add(index)
    return "\n".join(line for index, line in enumerate(lines) if index not in drop).strip()


def clean_pdf_pages(pages: Sequence[str]) -> list[str]:
    """Apply the v2 clean-up to the text of every page of one PDF (order preserved)."""
    expanded = [expand_ligatures(page) for page in pages]
    repeated = repeated_edge_lines(expanded)
    return [join_hyphenated_lines(strip_edge_lines(page, repeated)) for page in expanded]
