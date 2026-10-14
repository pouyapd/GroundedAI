"""Deterministic check for retrieved passages that disagree about what the question asks.

Replaces the figure check of :mod:`groundedai.generation.conflicts` (kept to reproduce the
reports published on 2026-10-13). Steps:

1. **What is asked.** The question's content words, without names (capitalised words after
   the first) and without words that only ask for a value ("how many", "how long").
2. **Statements.** Sentences of the passages that state a value (a figure with its unit:
   money, durations, times, counts, percentages, ...) or that deny a requirement ("do not have
   to", "no fixed schedule"). Instruction-like text is skipped.
3. **About the question.** A statement counts only if it contains at least two of the asked
   words and at most one fewer than the best-matching statement.
4. **Pairs from different documents** are then classified:

   ``agreement``
       a value of one equals a value of the other ("08:00" and "8 am", "two" and "2");
   ``different context``
       both name a group, plan, day, season, class or place, and the names differ (students vs
       adults, weekdays vs Saturdays, free vs Pro plan);
   ``different_dates``
       the values differ and the statements, or their documents, carry different dates
       (an edition year in the title, "from 1 March 2026", "until 2025", "in 2024");
   ``conflict``
       the values differ with nothing that separates them, or one passage states a value or a
       requirement and the other denies it.

   Only ``conflict`` and ``different_dates`` are reported. Neither is resolved: both
   statements are returned with their quotes so a reader can check them.

Limits: a pattern-based reading, not language understanding. It misses disagreements that
need world knowledge or arithmetic, can pair statements about different things that share
words, and treats any shared value as agreement.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from groundedai.core.quantities import MONTHS, Quantity, read_quantities
from groundedai.core.text import find_normalized_span, split_sentences, tokenize
from groundedai.generation.claims import QUESTION_FILLERS, claim_stem, claim_words
from groundedai.generation.context import SourceBlock
from groundedai.generation.safety import detect_injection

DisagreementKind = Literal["conflict", "different_dates"]
MIN_FOCUS_WORDS = 2
MAX_REPORTED = 3

_MONTH = "|".join(MONTHS)
_DATE_PHRASE = (
    rf"(?:\d{{1,2}}\s+)?(?:{_MONTH})(?:\s+\d{{1,2}})?,?\s+(?:19|20)\d{{2}}"
    rf"|(?:19|20)\d{{2}}-\d{{2}}-\d{{2}}|(?:19|20)\d{{2}}"
)
# A date that scopes a statement: "from 1 March 2026", "until 31 December 2025", "In 2024,".
_SCOPED_DATE_RE = re.compile(
    rf"\b(?:from|since|until|till|before|after|as of|effective|valid from|valid until|in|by)\s+"
    rf"(?:the\s+)?(?:{_DATE_PHRASE})\b",
    re.IGNORECASE,
)
# A document that says when it applies or how old it is: "(2024 edition)", "Price List 2025",
# "effective January 2026", "This policy applies ... from 1 January 2026".
_EDITION_RE = re.compile(
    rf"(?:{_DATE_PHRASE})\s+(?:edition|tariff|version|price list|prices|rules)\b"
    rf"|\b(?:edition|tariff|version|price list|prices|updated|effective|valid from|valid until)"
    rf"\s+(?:{_DATE_PHRASE})\b"
    rf"|\b(?:applies|apply|applied|in force|valid|effective)\b[^.\n]{{0,80}}?\b(?:from|since|"
    rf"until|as of)\s+(?:{_DATE_PHRASE})\b",
    re.IGNORECASE,
)
_YEAR_IN_TITLE_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NEGATED_REQUIREMENT_RE = re.compile(
    r"\b(?:do|does|did)\s+not\s+(?:have|need)\s+to\b"
    r"|\b(?:don't|doesn't)\s+(?:have|need)\s+to\b"
    r"|\bneed\s+not\b"
    r"|\bno\s+longer\s+(?:need|have)\s+to\b"
    r"|\b(?:is|are)\s+(?:not|no longer)\s+(?:required|necessary|needed|mandatory)\b"
    r"|\bno\s+(?:fixed|set|regular)\s+(?:schedule|interval|period|limit|deadline)\b"
    r"|\bnot\s+(?:on\s+)?a\s+(?:fixed|set|regular)\s+(?:schedule|interval)\b"
    r"|\bthere\s+is\s+no\s+(?:fixed\s+|set\s+|regular\s+)?(?:schedule|limit|deadline|requirement)\b",
    re.IGNORECASE,
)
_AFFIRMED_REQUIREMENT_RE = re.compile(
    r"\b(?:must|has to|have to|needs? to|is required|are required|required to|remember to)\b",
    re.IGNORECASE,
)
_ASKS_FOR_DATE_RE = re.compile(r"^\s*when\b|\b(?:what|which)\s+date\b|\bby when\b", re.IGNORECASE)
_CAPITALISED_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
# What kind of value a question asks for. Only values of that kind are compared, so a question
# about a fee is not answered by two different deadlines. Checked in order; no match = any kind.
_ASKED_KINDS: list[tuple[re.Pattern[str], frozenset[str]]] = [
    (
        re.compile(r"\bhow many times\b|\bhow often\b|\bhow frequently\b", re.IGNORECASE),
        frozenset({"count", "duration"}),
    ),
    (
        re.compile(
            r"\bhow (?:long|soon|quickly)\b|\bhow far in advance\b"
            r"|\bhow many (?:seconds|minutes|hours|days|nights|working days|business days|"
            r"weeks|months|years)\b",
            re.IGNORECASE,
        ),
        frozenset({"duration"}),
    ),
    (
        re.compile(
            r"\bwhat time\b|\bopening hours\b|\bvisiting hours\b"
            r"|\bwhen (?:does|do|is|are)\b.*\b(?:open|opens|close|closes|start|starts|end|ends)\b",
            re.IGNORECASE,
        ),
        frozenset({"time_of_day"}),
    ),
    (
        re.compile(
            r"\b(?:cost|costs|price|prices|fee|fees|charge|charges|pay|paid|fine|fines|deposit|"
            r"reimburse\w*|refund\w*|compensation|allowance|stipend|discount)\b",
            re.IGNORECASE,
        ),
        frozenset({"money", "percent"}),
    ),
    (re.compile(r"\bhow many\b", re.IGNORECASE), frozenset({"count", "number"})),
]

# Words that name who or what a value applies to. Two statements whose names in the same group
# differ (students / adults) are about different things, not in conflict.
_QUALIFIER_GROUPS: dict[str, str] = {
    "audience": (
        "adult child kid student senior pensioner infant teen teenager youth junior resident "
        "nonresident visitor guest member nonmember employee contractor staff volunteer family "
        "individual group"
    ),
    "plan": "free basic standard plus pro premium business enterprise starter personal team trial",
    "day": (
        "weekday weekend monday tuesday wednesday thursday friday saturday sunday holiday workday"
    ),
    "season": "summer winter spring autumn",
    "class": "economy business first premium express standard priority regular overnight",
    "place": (
        "north south east west northern southern eastern western central capital rural urban "
        "domestic international indoor outdoor"
    ),
    "desk": "hot fixed dedicated shared private",
    "hours": "peak offpeak daytime evening",
}
_QUALIFIERS: dict[str, frozenset[str]] = {
    group: frozenset(claim_stem(word) for word in words.split())
    for group, words in _QUALIFIER_GROUPS.items()
}


@dataclass(frozen=True, slots=True)
class Statement:
    label: str
    document_id: str
    document_title: str
    sentence: str
    start: int
    end: int
    values: tuple[Quantity, ...]
    denies: bool  # "do not have to", "no fixed schedule"
    words: frozenset[str]
    focus: int  # asked words it contains
    qualifiers: dict[str, frozenset[str]] = field(default_factory=dict)
    dates: tuple[str, ...] = ()
    affirms: bool = False  # "must", "is required"

    def value_texts(self) -> list[str]:
        return [value.text for value in self.values]

    def stance(self) -> str:
        """What the statement says about the asked property, for the explanation."""
        if self.values:
            return ", ".join(self.value_texts())
        if self.denies:
            return "there is no such requirement"
        return "it is required"  # a statement without a value either denies or affirms


@dataclass(frozen=True, slots=True)
class Disagreement:
    kind: DisagreementKind
    first: Statement
    second: Statement

    @property
    def labels(self) -> tuple[str, str]:
        return self.first.label, self.second.label

    def _side(self, statement: Statement) -> str:
        return f"{statement.label} says {statement.stance()}"

    def explanation(self) -> str:
        if self.kind == "different_dates":
            dates = "; ".join(
                f"{s.label}: {', '.join(s.dates) or 'no date given'}"
                for s in (self.first, self.second)
            )
            return (
                f"Sources {self.first.label} and {self.second.label} give different values that "
                f"apply to different dates ({dates}). Check which date applies."
            )
        return (
            f"Sources {self.first.label} and {self.second.label} disagree: "
            f"{self._side(self.first)}, {self._side(self.second)}. GroundedAI does not decide "
            "which is right; check both documents."
        )

    def note(self) -> str:
        """Short instruction for the answer-v2 prompt (passed as an application note)."""
        a, b = self.labels
        if self.kind == "different_dates":
            return (
                f"Application note: sources [{a}] and [{b}] state different values for different "
                "dates. If both answer the question, report both values with their dates and "
                "cite both sources."
            )
        if (self.first.denies and not self.first.values) or (
            self.second.denies and not self.second.values
        ):
            return (
                f"Application note: sources [{a}] and [{b}] disagree. If both answer the "
                "question, report what each says and cite both sources."
            )
        return (
            f"Application note: sources [{a}] and [{b}] state different figures. If both "
            "answer the question, report both values and cite both sources."
        )


def asked_kinds(question: str) -> frozenset[str] | None:
    """The kinds of value the question asks for, or None when it is not clear."""
    for pattern, kinds in _ASKED_KINDS:
        if pattern.search(question):
            return kinds
    return None


def asked_words(question: str) -> set[str]:
    """Content words of the question, without names and without words that only ask."""
    names = {
        claim_stem(token)
        for match in _CAPITALISED_RE.finditer(question)
        if match.start() > 0
        for token in tokenize(match.group(0))
    }
    fillers = {claim_stem(word) for word in QUESTION_FILLERS}
    return claim_words(question) - names - fillers


def _document_dates(sources: Sequence[SourceBlock]) -> dict[str, tuple[str, ...]]:
    """Dates that say when a document applies, from its title and its passages in the context."""
    found: dict[str, list[str]] = {}
    for block in sources:
        dates = found.setdefault(block.chunk.document_id, [])
        title = block.chunk.document_title
        for match in _EDITION_RE.finditer(title):
            dates.append(match.group(0))
        if not _EDITION_RE.search(title):
            dates.extend(match.group(0) for match in _YEAR_IN_TITLE_RE.finditer(title))
        for match in _EDITION_RE.finditer(block.text):
            dates.append(match.group(0))
    return {key: tuple(dict.fromkeys(value)) for key, value in found.items()}


def _statements(question: str, focus: set[str], sources: Sequence[SourceBlock]) -> list[Statement]:
    dates_are_values = bool(_ASKS_FOR_DATE_RE.search(question))
    kinds = asked_kinds(question)
    statements = []
    for block in sources:
        title_words = claim_words(block.chunk.document_title)
        for sentence in split_sentences(block.text):
            if detect_injection(sentence):
                continue
            words = frozenset(claim_words(sentence))
            if not words - title_words:
                continue  # a heading that only repeats the title
            values = tuple(
                quantity
                for quantity in read_quantities(sentence).quantities
                if (dates_are_values or quantity.kind not in ("date", "year"))
                and (kinds is None or quantity.kind in kinds)
            )
            denies = bool(_NEGATED_REQUIREMENT_RE.search(sentence))
            # "do not have to" contains "have to": a requirement counts only outside a denial.
            affirms = bool(
                _AFFIRMED_REQUIREMENT_RE.search(_NEGATED_REQUIREMENT_RE.sub(" ", sentence))
            )
            if not values and not denies and not affirms:
                continue
            span = find_normalized_span(block.text, sentence) or (0, 0)
            statements.append(
                Statement(
                    label=block.label,
                    document_id=block.chunk.document_id,
                    document_title=block.chunk.document_title,
                    sentence=sentence,
                    start=span[0],
                    end=span[1],
                    values=values,
                    denies=denies,
                    words=words,
                    focus=len(focus & words),
                    qualifiers={
                        group: words & names
                        for group, names in _QUALIFIERS.items()
                        if words & names
                    },
                    dates=tuple(match.group(0) for match in _SCOPED_DATE_RE.finditer(sentence)),
                    affirms=affirms,
                )
            )
    return statements


def _about_the_question(statements: list[Statement], focus: set[str]) -> list[Statement]:
    if not statements:
        return []
    needed = min(MIN_FOCUS_WORDS, len(focus))
    best = max(statement.focus for statement in statements)
    return [s for s in statements if s.focus >= max(needed, best - 1, 1)]


def _different_context(first: Statement, second: Statement) -> bool:
    for group, names in first.qualifiers.items():
        other = second.qualifiers.get(group)
        if other and not (names & other):
            return True
    return False


def _classify(
    first: Statement, second: Statement, document_dates: dict[str, tuple[str, ...]]
) -> Disagreement | None:
    comparable = [(a, b) for a in first.values for b in second.values if a.comparable(b)]
    if comparable:
        if any(a.same_value(b) for a, b in comparable):
            return None  # agreement on at least one value
    elif first.denies == second.denies:
        return None  # nothing to compare, and not one stating what the other denies
    if _different_context(first, second):
        return None
    first = replace(first, dates=first.dates or document_dates.get(first.document_id, ()))
    second = replace(second, dates=second.dates or document_dates.get(second.document_id, ()))
    a, b = sorted((first, second), key=lambda statement: int(statement.label[1:]))
    if (a.dates or b.dates) and set(a.dates) != set(b.dates):
        return Disagreement("different_dates", a, b)
    return Disagreement("conflict", a, b)


def find_disagreements(question: str, sources: Sequence[SourceBlock]) -> list[Disagreement]:
    """Pairs of passages from different documents that disagree about what is asked."""
    focus = asked_words(question)
    if not focus:
        return []
    candidates = _about_the_question(_statements(question, focus, sources), focus)
    document_dates = _document_dates(sources)
    found: dict[tuple[str, str], Disagreement] = {}
    ranked = sorted(candidates, key=lambda statement: -statement.focus)
    for index, first in enumerate(ranked):
        for second in ranked[index + 1 :]:
            if first.document_id == second.document_id:
                continue
            disagreement = _classify(first, second, document_dates)
            if disagreement is None:
                continue
            key = (disagreement.first.document_id, disagreement.second.document_id)
            if key not in found:
                found[key] = disagreement
    ordered = sorted(
        found.values(), key=lambda d: (int(d.first.label[1:]), int(d.second.label[1:]))
    )
    return ordered[:MAX_REPORTED]
