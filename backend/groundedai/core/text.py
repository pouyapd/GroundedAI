"""Small, dependency-free text utilities shared by retrieval, generation and evaluation.

These helpers are deliberately simple and deterministic. They are used for
the hashing test embedder, the extractive demo generator, the lexical
support heuristic and evaluation matching, so all of them must agree on how
text is tokenised.
"""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_WS_RE = re.compile(r"\s+")
# Sentence boundary: end punctuation followed by whitespace and an uppercase
# letter, digit, quote or bracket; or a blank line / bullet line break.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[A-Z0-9])|\n\s*\n|\n(?=\s*[-*•]\s)")
# A heading on its own line (as PDF text keeps section titles): short, capitalised, without
# sentence punctuation, followed by a line that starts a sentence. Wrapped lines of running
# text are longer or end in a connecting word, and stay in their sentence.
_HEADING_LINE_RE = re.compile(r"(?m)^([A-Z][^\n.!?;:,]{0,58}[^\s.!?;:,])\n(?=[\"'“(\[]?[A-Z0-9])")
_HEADING_MAX_WORDS = 8
_CONNECTING_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "per",
        "than",
        "that",
        "the",
        "to",
        "via",
        "with",
    ]
)

_STOPWORD_TEXT = """
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves out
over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves also may must shall
per via etc please tell give list explain describe according document documents
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_extracted_text(text: str) -> str:
    """Normalise text extracted from files without changing its meaning.

    Removes NUL and other control characters (except newlines and tabs),
    applies Unicode NFC normalisation and normalises line endings.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    # Collapse runs of spaces/tabs but keep paragraph structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def light_stem(token: str) -> str:
    """A tiny suffix stripper; applied identically to queries and passages."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, lightly stemmed tokens."""
    return [light_stem(token) for token in tokenize(text) if token not in STOPWORDS]


def _heading_break(match: re.Match[str]) -> str:
    line = match.group(1)
    words = line.split()
    if len(words) > _HEADING_MAX_WORDS or words[-1].lower() in _CONNECTING_WORDS:
        return match.group(0)
    return line + "\n\n"


def split_sentences(text: str) -> list[str]:
    text = _HEADING_LINE_RE.sub(_heading_break, text)
    parts = (normalize_whitespace(part) for part in _SENTENCE_SPLIT_RE.split(text))
    return [part for part in parts if part]


def extract_numbers(text: str) -> list[str]:
    """Numbers in a canonical form (thousands separators removed)."""
    numbers = []
    for match in NUMBER_RE.findall(text):
        value = match
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", value):
            value = value.replace(",", "")
        numbers.append(value)
    return numbers


def normalize_for_match(text: str) -> str:
    """Case-folded text with collapsed whitespace, used for evaluation matching."""
    return normalize_whitespace(unicodedata.normalize("NFKC", text)).casefold()


def find_normalized_span(haystack: str, needle: str) -> tuple[int, int] | None:
    """Locate ``needle`` in ``haystack`` ignoring case and whitespace differences.

    Returns the ``(start, end)`` character span in the *original* haystack, or
    ``None`` if the needle does not occur.
    """
    norm_chars: list[str] = []
    index_map: list[int] = []
    previous_space = True
    for position, char in enumerate(haystack):
        if char.isspace():
            if previous_space:
                continue
            norm_chars.append(" ")
            index_map.append(position)
            previous_space = True
        else:
            for folded in unicodedata.normalize("NFKC", char).casefold():
                norm_chars.append(folded)
                index_map.append(position)
            previous_space = False
    normalized_haystack = "".join(norm_chars)
    target = normalize_for_match(needle)
    if not target:
        return None
    found = normalized_haystack.find(target)
    if found < 0:
        return None
    start = index_map[found]
    end = index_map[found + len(target) - 1] + 1
    return start, end
