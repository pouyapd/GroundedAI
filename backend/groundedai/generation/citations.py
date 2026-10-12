"""Parsing and checking citations in generated answers.

Two different questions are answered here and kept apart:

* **Citation validity** - does every ``[S#]`` label refer to a source that was
  actually given to the model? This is exact.
* **Support** - do the cited sources contain the words and numbers of the
  sentence? This is a lexical heuristic. It catches invented numbers and
  off-topic citations but cannot judge meaning (negation, paraphrase,
  entailment), so a "supported" label is not proof of correctness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from groundedai.core.text import content_tokens, extract_numbers, split_sentences
from groundedai.generation.prompts import ABSTAIN_MARKER

CITATION_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_LABEL_RE = re.compile(r"S\d+", re.IGNORECASE)
_LEADING_CITATIONS_RE = re.compile(
    r"^((?:\s*\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\])+)[\s.!?]*(.*)$", re.IGNORECASE | re.DOTALL
)
# Only consulted when an answer has no valid citation. Small models paraphrase or
# misspell the abstention marker ("[Insuffcient Evidence: ...]"), so the patterns
# are deliberately tolerant.
_ABSTENTION_PHRASES = re.compile(
    r"\binsuff?i?c?ient[\s_-]*evidence\b"
    r"|\b(?:the )?(?:provided |given )?sources? (?:do|does) not (?:contain|provide|mention|include|"
    r"state|say|specify|cover)\b"
    r"|\bnot enough information\b"
    r"|\bno (?:specific |relevant |such )?information\b"
    r"|\b(?:is|are|was|were) not (?:mentioned|specified|provided|stated|given|found)\b"
    r"|\bi (?:do not|don't|cannot|can't) (?:know|answer|find)\b"
    r"|\bcannot be (?:answered|determined)\b",
    re.IGNORECASE,
)

_LEADING_ABSTENTION_RE = re.compile(
    r"^(?:\s*\[[^\]\n]{0,200}\])*\s*\[?\s*insuff?i?c?ient[\s_-]*evidence\b", re.IGNORECASE
)

SupportLabel = Literal["supported", "partial", "unsupported", "uncited"]
GroundingLabel = Literal["grounded", "partially_grounded", "ungrounded"]

SUPPORTED_RECALL = 0.7
PARTIAL_RECALL = 0.4


@dataclass(slots=True)
class SentenceAssessment:
    text: str
    labels: list[str]
    support: SupportLabel
    token_recall: float | None = None
    unsupported_numbers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CitationCheck:
    cited: list[str]
    valid: list[str]
    invalid: list[str]
    sentences_total: int
    sentences_cited: int

    @property
    def all_valid(self) -> bool:
        return not self.invalid


@dataclass(slots=True)
class ParsedAnswer:
    raw: str
    text: str
    is_abstention: bool
    abstention_reason: str | None
    sentences: list[SentenceAssessment]
    check: CitationCheck
    grounding: GroundingLabel | None


def labels_in(text: str) -> list[str]:
    labels: list[str] = []
    for group in CITATION_RE.findall(text):
        for label in _LABEL_RE.findall(group):
            normalized = label.upper()
            if normalized not in labels:
                labels.append(normalized)
    return labels


def strip_citations(text: str) -> str:
    without = re.sub(r"\s{2,}", " ", CITATION_RE.sub("", text))
    return re.sub(r"\s+([.,;:!?])", r"\1", without).strip()


def split_answer_sentences(answer: str) -> list[str]:
    """Split into sentences, moving citations that open a sentence to the previous one.

    Models often write "Fact one. [S1] Fact two. [S2]"; a plain sentence splitter
    would credit [S1] to "Fact two".
    """
    merged: list[str] = []
    for sentence in split_sentences(answer):
        match = _LEADING_CITATIONS_RE.match(sentence)
        if merged and match:
            merged[-1] = f"{merged[-1]} {match.group(1).strip()}"
            rest = match.group(2).strip()
            if rest:
                merged.append(rest)
        else:
            merged.append(sentence)
    return merged


def assess_support(sentence: str, cited_texts: Sequence[str]) -> SentenceAssessment:
    labels = labels_in(sentence)
    claim = strip_citations(sentence)
    tokens = set(content_tokens(claim))
    source_tokens: set[str] = set()
    source_numbers: set[str] = set()
    for source_text in cited_texts:
        source_tokens.update(content_tokens(source_text))
        source_numbers.update(extract_numbers(source_text))
    recall = len(tokens & source_tokens) / len(tokens) if tokens else 0.0
    unsupported_numbers = sorted(set(extract_numbers(claim)) - source_numbers)
    if unsupported_numbers or recall < PARTIAL_RECALL:
        label: SupportLabel = "unsupported"
    elif recall < SUPPORTED_RECALL:
        label = "partial"
    else:
        label = "supported"
    return SentenceAssessment(
        text=claim,
        labels=labels,
        support=label,
        token_recall=round(recall, 3),
        unsupported_numbers=unsupported_numbers,
    )


def parse_answer(raw: str, sources: Mapping[str, str]) -> ParsedAnswer:
    """Parse model output against the sources it was given (label -> text)."""
    text = raw.strip()
    cited = labels_in(text)
    valid = [label for label in cited if label in sources]
    invalid = [label for label in cited if label not in sources]

    marker_index = text.upper().find(ABSTAIN_MARKER)
    explicit_abstention = marker_index >= 0
    # A reply that *opens* with an abstention marker (after any bracketed labels or titles)
    # is an abstention even if it also cites sources, e.g. "[S1][S4]\nInsufficient evidence: ...".
    leading = _LEADING_ABSTENTION_RE.match(text)
    phrase_abstention = not valid and bool(_ABSTENTION_PHRASES.search(text))
    if explicit_abstention or leading or phrase_abstention:
        if explicit_abstention:
            tail = text[marker_index + len(ABSTAIN_MARKER) :]
        elif leading:
            tail = text[leading.end() :]
        else:
            tail = text
        reason = strip_citations(tail.strip(" :-\n[]()")) or None
        return ParsedAnswer(
            raw=raw,
            text=text,
            is_abstention=True,
            abstention_reason=reason,
            sentences=[],
            check=CitationCheck(cited, valid, invalid, 0, 0),
            grounding=None,
        )

    assessments: list[SentenceAssessment] = []
    for sentence in split_answer_sentences(text):
        sentence_labels = [label for label in labels_in(sentence) if label in sources]
        if not sentence_labels:
            assessments.append(
                SentenceAssessment(text=strip_citations(sentence), labels=[], support="uncited")
            )
            continue
        assessment = assess_support(sentence, [sources[label] for label in sentence_labels])
        assessment.labels = sentence_labels
        assessments.append(assessment)

    cited_sentences = sum(1 for item in assessments if item.labels)
    check = CitationCheck(cited, valid, invalid, len(assessments), cited_sentences)
    return ParsedAnswer(
        raw=raw,
        text=text,
        is_abstention=False,
        abstention_reason=None,
        sentences=assessments,
        check=check,
        grounding=_grounding(assessments, check),
    )


def _grounding(assessments: Sequence[SentenceAssessment], check: CitationCheck) -> GroundingLabel:
    if not assessments or not check.valid:
        return "ungrounded"
    if check.invalid:
        return "partially_grounded"
    if all(item.support == "supported" for item in assessments):
        return "grounded"
    if any(item.support in ("supported", "partial") for item in assessments):
        return "partially_grounded"
    return "ungrounded"
