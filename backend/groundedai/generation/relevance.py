"""Lexical question relevance: does the evidence mention what the question asks about?

Evidence support and question relevance are different properties. A sentence can be copied
word for word from a passage (fully supported) and still not answer the question: asked "How
much is the late fee for e-books?", a passage that only states the late fee for adult books
supports a true sentence about adult books, but nothing in it is about e-books.

:func:`question_relevance` reads the question's content words and leaves out the ones that do
not name what is asked:

* words of the documents' own titles ("AI", "RMF" in a question about the AI RMF): every
  passage of a document is about its title;
* the word after "how" ("how long", "how many", "how often"), which asks for a kind of value;
* generic question verbs ("get", "happen", "say", "call") and numbers, which are worded
  differently in answers ("five" / "5").

What is left is matched against the retrieved passages with the same word-form rule as the
extractive generator (:func:`groundedai.generation.extractive.words_match`). An asked word that
no retrieved passage contains is reported as unmatched: the passages may not state what the
question asks for. It is a lexical signal with known limits: a passage that says it in other
words ("failed" for "wrong", a name without the word "author") counts as not mentioning it, and
a passage that mentions every asked word can still be about something else.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from groundedai.core.text import content_tokens

QUESTION_FORM_RE = re.compile(r"\bhow\s+(\w+)", re.IGNORECASE)
GENERIC_QUESTION_VERBS = frozenset(
    [
        "get",
        "gets",
        "got",
        "happen",
        "happens",
        "happened",
        "say",
        "says",
        "said",
        "call",
        "called",
        "name",
        "named",
        "mean",
        "means",
        "stand",
        "stands",
        "give",
        "gives",
        "do",
        "does",
        "did",
        "take",
        "takes",
        "last",
        "lasts",
    ]
)
NUMBER_WORDS = frozenset(
    [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "twenty",
        "hundred",
        "thousand",
        "once",
        "twice",
        "thrice",
        "first",
        "second",
        "third",
        "half",
    ]
)
MIN_SHARED_PREFIX = 5


def shared_prefix(a: str, b: str) -> int:
    length = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        length += 1
    return length


def words_match(a: str, b: str) -> bool:
    """Same word, or two forms of one word: a shared prefix of at least five letters that
    leaves at most three letters of the shorter word ("reauthenticated" / "reauthentication",
    "minutes" / "minute"). A heuristic; it also joins some unrelated words."""
    if a == b:
        return True
    return shared_prefix(a, b) >= max(MIN_SHARED_PREFIX, min(len(a), len(b)) - 3)


@dataclass(frozen=True, slots=True)
class Relevance:
    asked: frozenset[str]  # words that name what the question asks about
    left_out: frozenset[str]  # title words, question-form words, generic verbs, numbers
    unmatched: tuple[str, ...]  # asked words that no retrieved passage contains
    passage_words: frozenset[str] = field(default_factory=frozenset)
    # question words without the documents' title words: what a sentence is scored on
    scored: frozenset[str] = field(default_factory=frozenset)

    @property
    def all_mentioned(self) -> bool:
        return not self.unmatched

    def coverage(self, text: str) -> float:
        """Share of the asked words that ``text`` contains (word forms included)."""
        if not self.asked:
            return 1.0
        words = set(content_tokens(text))
        found = sum(1 for word in self.asked if any(words_match(word, other) for other in words))
        return found / len(self.asked)


def question_relevance(question: str, titles: Sequence[str], passages: Sequence[str]) -> Relevance:
    words = set(content_tokens(question))
    title_words = {token for title in titles for token in content_tokens(title)}
    form = {match.group(1).lower() for match in QUESTION_FORM_RE.finditer(question)}
    form_words = {token for word in form for token in content_tokens(word)}
    generic = {
        word
        for word in words
        if word in GENERIC_QUESTION_VERBS
        or word in NUMBER_WORDS
        or word.isdigit()
        or any(word == token for verb in GENERIC_QUESTION_VERBS for token in content_tokens(verb))
    }
    left_out = (words & title_words) | (words & form_words) | generic
    asked = words - left_out
    if not asked:  # a question made only of title words: keep them
        asked, left_out = words - form_words - generic, (words & form_words) | generic
    passage_words = frozenset(token for text in passages for token in content_tokens(text))
    unmatched = tuple(
        sorted(word for word in asked if not any(words_match(word, p) for p in passage_words))
    )
    scored = (words - title_words) or words
    return Relevance(
        frozenset(asked), frozenset(left_out), unmatched, passage_words, frozenset(scored)
    )
