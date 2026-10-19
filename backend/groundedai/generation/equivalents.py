"""Equivalent wordings for version 2 of the claim and disagreement checks.

Two small, fixed lists, applied the same way to an answer's statements and to the passages:

* **Phrases** with one meaning are rewritten to one word before words are compared, so a
  negated wording and its positive equivalent read alike: "do not have to pay", "at no cost",
  "without paying" and "free of charge" all become "free". The rewrite is only used for
  comparing; quotes are always taken from the original text.
* **Words** with one meaning in policy documents share a key: fee / charge / cost / price,
  annual / yearly, begin / start, purchase / buy.

They were written for general wording, not for any document, after the paraphrase errors
seen on the development split and the PDF journey. They do not make the check understand
meaning: a synonym that is not listed is still a different word, and a listed word is
treated as equivalent even where the context gives it another sense.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_COST = r"(?:cost|charge|fee|payment)"
_EXTRA = r"(?:extra |additional |further )?"
_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:"
            rf"free of charge|for free|free of {_COST}"
            rf"|at no {_EXTRA}{_COST}"
            rf"|(?:with )?no {_EXTRA}{_COST}s?(?: is| are)? (?:charged|payable|due)"
            rf"|with no {_EXTRA}{_COST}s?"
            rf"|without (?:any )?{_EXTRA}{_COST}s?"
            r"|without (?:having to )?pay(?:ing)?(?: anything)?"
            r"|(?:do|does|did|will|would) not (?:have|need) to pay(?: anything)?"
            r"|(?:don't|doesn't|didn't|won't) (?:have|need) to pay(?: anything)?"
            r"|need not pay(?: anything)?"
            r"|pays? nothing|costs? nothing"
            r")\b",
            re.IGNORECASE,
        ),
        "free",
    ),
    (re.compile(r"\b(?:given|paid|handed) back\b|\brepaid\b|\breimbursed\b", re.I), "refunded"),
    (re.compile(r"\b(?:give|pay|hand)s? back\b|\brepay\b|\breimburse\b", re.I), "refund"),
    (re.compile(r"\bin a row\b|\bback[- ]to[- ]back\b", re.I), "consecutive"),
    (re.compile(r"\b(?:turn|show)(?:s|ed)? up\b", re.I), "arrive"),
    (re.compile(r"\bpick(?:s|ed)?[- ]up\b", re.I), "collection"),
]

# Each group shares the key of its first word.
_GROUPS = [
    "cost charge fee price tariff",
    "annual yearly",
    "begin start commence",
    "end finish",
    "buy purchase",
    "allow permit",
    "collect collection",
    "child kid",
    "telephone phone",
]


def rewrite_phrases(text: str) -> str:
    """``text`` with each listed phrase replaced by its one-word equivalent."""
    for pattern, replacement in _PHRASES:
        text = pattern.sub(replacement, text)
    return text


def keyed_stemmer(stem: Callable[[str], str]) -> Callable[[str], str]:
    """A stemmer that gives the listed equivalent words one key."""
    keys: dict[str, str] = {}
    for group in _GROUPS:
        words = group.split()
        key = stem(words[0])
        for word in words:
            keys[stem(word)] = key

    def keyed(token: str) -> str:
        stemmed = stem(token)
        return keys.get(stemmed, stemmed)

    return keyed
