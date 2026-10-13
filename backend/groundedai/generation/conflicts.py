"""Deterministic check for sources that state different figures for the same thing.

Before generation, sentences of the passages given to the model are compared in pairs.
A pair is reported when the two sentences

* come from different documents,
* each contain at least two of the question's content words,
* share at least two non-numeric content words with each other, and
* both state figures, with no figure in common (``€140 / €110`` vs ``€120``).

It is a cheap, conservative heuristic, not conflict detection in general: it misses
conflicts written in words ("twice" vs "three times") or without numbers, and two figures
about different things in similar sentences can still be paired. The result is shown as a
warning and, with the answer-v2 prompt, passed to the model as a short note; it never
changes the answer by itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from groundedai.core.text import content_tokens, extract_numbers, split_sentences
from groundedai.generation.context import SourceBlock

MIN_QUESTION_OVERLAP = 2
MIN_PAIR_OVERLAP = 2


@dataclass(frozen=True, slots=True)
class FigureConflict:
    label_a: str
    label_b: str
    figures_a: tuple[str, ...]
    figures_b: tuple[str, ...]

    def warning(self) -> str:
        return (
            f"Sources {self.label_a} and {self.label_b} state different figures "
            f"({', '.join(self.figures_a)} vs {', '.join(self.figures_b)}) in sentences about "
            "the same topic. They may disagree; check both."
        )

    def note(self) -> str:
        return (
            f"Application note: sources [{self.label_a}] and [{self.label_b}] state different "
            "figures. If both answer the question, report both values and cite both sources."
        )


@dataclass(frozen=True, slots=True)
class _Sentence:
    label: str
    document_id: str
    words: frozenset[str]
    figures: frozenset[str]


def _words(text: str) -> frozenset[str]:
    return frozenset(token for token in content_tokens(text) if not token[0].isdigit())


def find_figure_conflicts(question: str, sources: Sequence[SourceBlock]) -> list[FigureConflict]:
    question_words = _words(question)
    if len(question_words) < MIN_QUESTION_OVERLAP:
        return []
    sentences: list[_Sentence] = []
    for block in sources:
        for text in split_sentences(block.text):
            figures = frozenset(extract_numbers(text))
            words = _words(text)
            if figures and len(words & question_words) >= MIN_QUESTION_OVERLAP:
                sentences.append(_Sentence(block.label, block.chunk.document_id, words, figures))

    conflicts: dict[tuple[str, str], FigureConflict] = {}
    for first, second in combinations(sentences, 2):
        if first.document_id == second.document_id:
            continue
        if len(first.words & second.words) < MIN_PAIR_OVERLAP:
            continue
        if first.figures & second.figures:
            continue
        a, b = sorted((first, second), key=lambda sentence: int(sentence.label[1:]))
        key = (a.label, b.label)
        if key in conflicts:
            continue
        conflicts[key] = FigureConflict(
            label_a=a.label,
            label_b=b.label,
            figures_a=tuple(sorted(a.figures, key=_numeric_order)),
            figures_b=tuple(sorted(b.figures, key=_numeric_order)),
        )
    return list(conflicts.values())


def _numeric_order(value: str) -> tuple[float, str]:
    try:
        return (float(value.replace(",", "")), value)
    except ValueError:
        return (float("inf"), value)
