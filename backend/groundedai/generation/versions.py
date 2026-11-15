"""Which retrieved documents are editions of the same document, and which edition each one is.

Two documents count as versions of one document when their titles, without edition words
("Edition 2019", "Version 3.0", "Revision B", years, version numbers), share at least
:data:`FAMILY_SIMILARITY` of their words. Each one's edition, for display only, is an edition
label or identifier from the title or first passage, and a publication date: the year in the
title, or a date that follows "published", "publication date", "issued" or "approved".

The rule reads titles and the opening text only. It can join two different documents with
very similar generic titles ("Travel Policy" and "Travel Policy for Contractors"), and it
misses versions whose titles were rewritten. It never decides which edition is current: the
disagreement check only uses it to say that two statements come from different editions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from groundedai.core.quantities import MONTHS
from groundedai.core.text import content_tokens

FAMILY_SIMILARITY = 0.6
OPENING_CHARS = 1500

_MONTH = "|".join(MONTHS)
_LABEL_RE = re.compile(
    r"\b(edition|version|revision|rev\.|release|issue)\s+([A-Za-z0-9][A-Za-z0-9.]*)", re.IGNORECASE
)
# An identifier that ends in a hyphenated number: "SP 800-63B", "NIST.SP.800-63B-4" (a DOI).
_SERIES_RE = re.compile(r"\b(?:[A-Z][A-Za-z]{1,12}[ .])+\d{2,5}(?:-[0-9A-Za-z]+)+\b")
_CURRENCIES = {"gbp", "eur", "usd", "chf", "jpy", "cad", "aud"}
_FIELD_NAMES = {"number", "series", "title", "document", "identifier", "no"}
# A date counts only after a word that says it is the publication date of this document.
_DATE_RE = re.compile(
    rf"\b(?:published|publication date\(?s?\)?|issued|approved|dated|effective)[:\s]+(?:on\s+)?"
    rf"((?:\d{{1,2}}\s+)?(?:{_MONTH})\s+(?:\d{{1,2}},\s+)?(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VERSION_NUMBER_RE = re.compile(r"\b[vV]?\d+(?:\.\d+)+\b")
_EDITION_WORDS = {"edition", "version", "revision", "rev", "release", "issue", "draft", "final"}


@dataclass(frozen=True, slots=True)
class Edition:
    document_id: str
    title: str
    family_words: frozenset[str]
    label: str | None  # "Edition 2019", "Version 3.0", "Revision B", "SP 800-63B-4"
    date: str | None  # "March 2019"

    def describe(self) -> str:
        date = self.date if self.date and not (self.label and self.date in self.label) else None
        parts = [part for part in (self.label, date) if part]
        return ", ".join(parts) if parts else "no edition given"


def _family_words(title: str) -> frozenset[str]:
    text = _LABEL_RE.sub(" ", title)
    text = _VERSION_NUMBER_RE.sub(" ", _YEAR_RE.sub(" ", text))
    return frozenset(
        token
        for token in content_tokens(text)
        if token not in _EDITION_WORDS and not token.isdigit()
    )


def edition_of(document_id: str, title: str, opening: str = "") -> Edition:
    """The edition a document names in its title or opening text (first match wins)."""
    head = f"{title}\n{opening[:OPENING_CHARS]}"
    label_match = _LABEL_RE.search(head)
    label = f"{label_match.group(1).title()} {label_match.group(2)}" if label_match else None
    if label is None:
        for series in _SERIES_RE.finditer(head):
            words = series.group(0).replace(".", " ").split()
            while words and words[0].casefold() in _FIELD_NAMES:
                words = words[1:]  # "Series/Number NIST Special Publication 800-63B"
            if words and words[0].casefold() not in _CURRENCIES:
                label = " ".join(words)
                break
    date_match = _DATE_RE.search(head)
    date = " ".join(date_match.group(1).split()) if date_match else None
    title_year = _YEAR_RE.search(f"{title} {label or ''}")
    if title_year and (date is None or title_year.group(0) not in date):
        date = title_year.group(0)  # the title's year wins over a date in the text
    return Edition(document_id, title, _family_words(title), label, date)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def same_family(a: Edition, b: Edition) -> bool:
    """Two different documents whose titles, without edition words, are near-identical.

    Identical files are refused at upload, so two documents with the same title are two
    editions (or two copies saved at different times); either way their differences are
    shown as differences between versions.
    """
    if a.document_id == b.document_id:
        return False
    return similarity(a.family_words, b.family_words) >= FAMILY_SIMILARITY


def families(editions: Mapping[str, Edition]) -> list[list[str]]:
    """Groups of document ids that are editions of the same document (two or more each)."""
    ids = sorted(editions)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for index, first in enumerate(ids):
        if first in seen:
            continue
        group = [first] + [
            other for other in ids[index + 1 :] if same_family(editions[first], editions[other])
        ]
        if len(group) > 1:
            groups.append(group)
            seen.update(group)
    return groups
