"""Deterministic check for retrieved passages that disagree about what the question asks.

Replaces the figure check of :mod:`groundedai.generation.conflicts` (kept to reproduce the
reports published on 2026-10-13). Steps:

1. **What is asked.** The question's content words, without names (capitalised words after
   the first) and without words that only ask for a value ("how many", "how long").
2. **Statements.** Sentences of the passages that state a value (a figure with its unit:
   money, durations, times, counts, percentages, ...) or that deny a requirement ("do not have
   to", "no fixed schedule"). Instruction-like text is skipped.
3. **About the question.** A statement counts only if it contains at least two of the asked
   words (one, when the question has a single asked word) and at most one fewer than the
   best-matching statement.
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

Version 2 (``disagreement-v2``, chosen with :func:`find_disagreements`'s ``version``) keeps
these steps and changes three things:

* **Ranges as a whole.** "06:00–16:00" and "07:00–16:00" differ although they share an end. A
  single value agrees with a range when it equals one of its ends, is not compared when it
  lies inside, and differs when it lies outside.
* **Two kinds of date.** Dates that say when a value applies ("valid from 1 March 2026",
  "2025 price list", "autumn term 2026") are kept apart from dates that only say when a
  document was edited ("last updated", "revised", "published"). ``different_dates`` needs
  differing dates of the first kind on *both* sides; edit dates only, or a date on one side
  only, give ``conflict``, and the dates are shown in the explanation.
* **Word forms.** The asked words are matched with :func:`claim_stem_v2`
  ("emptied" = "empty").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from groundedai.core.quantities import MONTHS, Quantity, Range, read_quantities, read_ranges
from groundedai.core.text import find_normalized_span, split_sentences, tokenize
from groundedai.generation.claims import (
    QUESTION_FILLERS,
    Stemmer,
    claim_stem,
    claim_stem_v2,
    claim_words,
)
from groundedai.generation.context import SourceBlock
from groundedai.generation.safety import detect_injection

DisagreementKind = Literal["conflict", "different_dates"]
DisagreementVersion = Literal["disagreement-v1", "disagreement-v2"]
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
# disagreement-v2: a date that only says when a document was written or edited.
_REVISION_RE = re.compile(
    rf"\b(?:last\s+)?(?:updated|revised|edited|reviewed|modified|amended|published|issued|"
    rf"printed|dated)\b\s*(?:on|in|:)?\s*(?:{_DATE_PHRASE})",
    re.IGNORECASE,
)
_SEASON = r"(?:spring|summer|autumn|fall|winter)"
# disagreement-v2: a date that says when a value applies.
_VALIDITY_RE = re.compile(
    rf"\b(?:valid|effective|applies|apply|applied|applicable|in force|in effect)\b[^.\n]{{0,60}}?"
    rf"\b(?:from|since|until|till|as of|to)\s+(?:the\s+)?(?:{_DATE_PHRASE})\b"
    rf"|\b(?:from|since|until|till|as of|effective|starting|beginning|valid from|valid until|"
    rf"in|for)\s+(?:the\s+)?(?:{_SEASON}\s+)?(?:term\s+|season\s+)?(?:{_DATE_PHRASE})\b"
    rf"|(?:{_DATE_PHRASE})\s+(?:{_SEASON}\s+)?(?:edition|tariff|version|price list|prices|rates|"
    rf"fees|charges|season|term|timetable|schedule)\b"
    rf"|\b(?:edition|tariff|version|price list|season|term|rates|prices)\s+(?:{_DATE_PHRASE})\b"
    rf"|\b{_SEASON}\s+(?:term|season|semester)\s+(?:{_DATE_PHRASE})\b",
    re.IGNORECASE,
)
_SEASON_RE = re.compile(rf"\b{_SEASON}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
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
_QUALIFIERS_V2: dict[str, frozenset[str]] = {
    group: frozenset(claim_stem_v2(word) for word in words.split())
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
    # disagreement-v2 only
    ranges: tuple[Range, ...] = ()
    validity_dates: tuple[str, ...] = ()
    revision_dates: tuple[str, ...] = ()

    def value_texts(self) -> list[str]:
        if not self.ranges:
            return [value.text for value in self.values]
        in_ranges = {id(end) for value in self.ranges for end in (value.low, value.high)}
        texts = [_range_text(value) for value in self.ranges]
        return texts + [value.text for value in self.values if id(value) not in in_ranges]

    def stance(self) -> str:
        """What the statement says about the asked property, for the explanation."""
        if self.values:
            return ", ".join(self.value_texts())
        if self.denies:
            return "there is no such requirement"
        return "it is required"  # a statement without a value either denies or affirms


def _range_text(value: Range) -> str:
    """ "3 days" and "5 days" -> "3–5 days"; times and amounts keep both ends as written."""
    low, high = value.low.text, value.high.text
    if value.low.kind == value.high.kind and low.split()[1:] == high.split()[1:] and " " in low:
        low = low.split()[0]
    return f"{low}–{high}"


@dataclass(frozen=True, slots=True)
class Disagreement:
    kind: DisagreementKind
    first: Statement
    second: Statement
    show_dates: bool = False  # disagreement-v2: name each side's dates in a conflict

    @property
    def labels(self) -> tuple[str, str]:
        return self.first.label, self.second.label

    def _side(self, statement: Statement) -> str:
        side = f"{statement.label} says {statement.stance()}"
        dated = statement.validity_dates + statement.revision_dates
        if self.show_dates and dated:
            side += f" ({'; '.join(dated)})"
        return side

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


def asked_words(question: str, stem: Stemmer = claim_stem) -> set[str]:
    """Content words of the question, without names and without words that only ask."""
    names = {
        stem(token)
        for match in _CAPITALISED_RE.finditer(question)
        if match.start() > 0
        for token in tokenize(match.group(0))
    }
    fillers = {stem(word) for word in QUESTION_FILLERS}
    return claim_words(question, stem) - names - fillers


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


def _statements(
    question: str, focus: set[str], sources: Sequence[SourceBlock], v2: bool = False
) -> list[Statement]:
    dates_are_values = bool(_ASKS_FOR_DATE_RE.search(question))
    kinds = asked_kinds(question)
    stem = claim_stem_v2 if v2 else claim_stem
    qualifier_groups = _QUALIFIERS_V2 if v2 else _QUALIFIERS
    statements = []
    for block in sources:
        title_words = claim_words(block.chunk.document_title, stem)
        for sentence in split_sentences(block.text):
            if detect_injection(sentence):
                continue
            words = frozenset(claim_words(sentence, stem))
            if not words - title_words:
                continue  # a heading that only repeats the title
            found_quantities = read_quantities(sentence).quantities
            values = tuple(
                quantity
                for quantity in found_quantities
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
            ranges: tuple[Range, ...] = ()
            validity: tuple[str, ...] = ()
            revision: tuple[str, ...] = ()
            if v2:
                kept = {id(value) for value in values}
                ranges = tuple(
                    found
                    for found in read_ranges(sentence, found_quantities)
                    if id(found.low) in kept and id(found.high) in kept
                )
                revision = tuple(m.group(0) for m in _REVISION_RE.finditer(sentence))
                validity = tuple(m.group(0) for m in _VALIDITY_RE.finditer(sentence))
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
                        for group, names in qualifier_groups.items()
                        if words & names
                    },
                    dates=(
                        validity
                        if v2
                        else tuple(match.group(0) for match in _SCOPED_DATE_RE.finditer(sentence))
                    ),
                    affirms=affirms,
                    ranges=ranges,
                    validity_dates=validity,
                    revision_dates=revision,
                )
            )
    return statements


def _document_dates_v2(
    sources: Sequence[SourceBlock],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """disagreement-v2: per document, the dates its title and passages give, as (dates that
    say when it applies, dates that only say when it was edited). A bare year in a title
    counts as the first kind unless it is part of an edit date."""
    found: dict[str, tuple[list[str], list[str]]] = {}
    for block in sources:
        validity, revision = found.setdefault(block.chunk.document_id, ([], []))
        title = block.chunk.document_title
        title_revisions = [m.group(0) for m in _REVISION_RE.finditer(title)]
        revision.extend(title_revisions)
        title_validity = [m.group(0) for m in _VALIDITY_RE.finditer(title)]
        validity.extend(title_validity)
        if not title_validity:
            stripped = _REVISION_RE.sub(" ", title)
            years = [m.group(0) for m in _YEAR_IN_TITLE_RE.finditer(stripped)]
            seasons = [m.group(0) for m in _SEASON_RE.finditer(stripped)]
            validity.extend(" ".join([*seasons, year]) for year in years)
        for match in _VALIDITY_RE.finditer(block.text):
            validity.append(match.group(0))
        for match in _REVISION_RE.finditer(block.text):
            revision.append(match.group(0))
    return {
        key: (tuple(dict.fromkeys(value[0])), tuple(dict.fromkeys(value[1])))
        for key, value in found.items()
    }


def _date_key(dates: Sequence[str]) -> tuple[frozenset[str], frozenset[str], frozenset[float]]:
    """Years, season words and full dates named in ``dates``, for comparing two sides."""
    text = " ".join(dates)
    years = frozenset(m.group(0) for m in _YEAR_RE.finditer(text))
    seasons = frozenset(m.group(0).lower() for m in _SEASON_RE.finditer(text))
    full = frozenset(
        q.value for q in read_quantities(text).quantities if q.kind == "date" and q.value > 10000
    )
    return years, seasons, full


def _dates_differ(first: Sequence[str], second: Sequence[str]) -> bool:
    years_a, seasons_a, full_a = _date_key(first)
    years_b, seasons_b, full_b = _date_key(second)
    if years_a and years_b and years_a != years_b:
        return True
    if seasons_a and seasons_b and seasons_a != seasons_b:
        return True
    return bool(full_a and full_b and not full_a & full_b)


def _relation(first: Statement, second: Statement) -> str:
    """disagreement-v2: "agree", "differ" or "none" (nothing comparable), ranges as a whole.

    When a range can be compared with the other side, only comparisons that involve a range
    decide: a stated range is the value asked about, and an incidental shared figure (a year,
    a count) elsewhere in the sentences does not make two different ranges agree.
    """

    def items(statement: Statement) -> list[Range | Quantity]:
        ends = {id(end) for value in statement.ranges for end in (value.low, value.high)}
        singles = [value for value in statement.values if id(value) not in ends]
        return [*statement.ranges, *singles]

    def comparable(a: Range | Quantity, b: Range | Quantity) -> bool:
        if isinstance(a, Range):
            return a.comparable(b)
        if isinstance(b, Range):
            return b.comparable(a)
        return a.comparable(b)

    pairs = [(a, b) for a in items(first) for b in items(second) if comparable(a, b)]
    ranged = [(a, b) for a, b in pairs if isinstance(a, Range) or isinstance(b, Range)]
    differ = False
    for a, b in ranged or pairs:
        if isinstance(a, Range) and isinstance(b, Range):
            if a.same_range(b):
                return "agree"
            differ = True
        elif isinstance(a, Range) or isinstance(b, Range):
            span, value = (a, b) if isinstance(a, Range) else (b, a)
            if not isinstance(span, Range) or not isinstance(value, Quantity):
                continue
            if span.has_end(value):
                return "agree"
            if not span.contains(value):
                differ = True
        elif isinstance(a, Quantity) and isinstance(b, Quantity):
            if a.same_value(b):
                return "agree"
            differ = True
    return "differ" if differ else "none"


def _classify_v2(
    first: Statement,
    second: Statement,
    document_dates: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
) -> Disagreement | None:
    relation = _relation(first, second)
    if relation == "agree":
        return None
    if relation == "none" and first.denies == second.denies:
        return None
    sides = []
    for statement in (first, second):
        validity, revision = document_dates.get(statement.document_id, ((), ()))
        sides.append(
            replace(
                statement,
                validity_dates=statement.validity_dates or validity,
                revision_dates=statement.revision_dates or revision,
                dates=statement.validity_dates or validity,
            )
        )
    a, b = sorted(sides, key=lambda statement: int(statement.label[1:]))
    if a.validity_dates and b.validity_dates and _dates_differ(a.validity_dates, b.validity_dates):
        return Disagreement("different_dates", a, b, show_dates=True)
    if _different_context(first, second):
        return None
    return Disagreement("conflict", a, b, show_dates=True)


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


def find_disagreements(
    question: str,
    sources: Sequence[SourceBlock],
    version: DisagreementVersion = "disagreement-v1",
) -> list[Disagreement]:
    """Pairs of passages from different documents that disagree about what is asked.

    ``version`` chooses the method; ``disagreement-v1`` reproduces every earlier result.
    """
    v2 = version == "disagreement-v2"
    focus = asked_words(question, claim_stem_v2 if v2 else claim_stem)
    if not focus:
        return []
    candidates = _about_the_question(_statements(question, focus, sources, v2), focus)
    document_dates = _document_dates(sources)
    dates_v2 = _document_dates_v2(sources) if v2 else {}
    found: dict[tuple[str, str], Disagreement] = {}
    ranked = sorted(candidates, key=lambda statement: -statement.focus)
    for index, first in enumerate(ranked):
        for second in ranked[index + 1 :]:
            if first.document_id == second.document_id:
                continue
            disagreement = (
                _classify_v2(first, second, dates_v2)
                if v2
                else _classify(first, second, document_dates)
            )
            if disagreement is None:
                continue
            key = (disagreement.first.document_id, disagreement.second.document_id)
            if key not in found:
                found[key] = disagreement
    ordered = sorted(
        found.values(), key=lambda d: (int(d.first.label[1:]), int(d.second.label[1:]))
    )
    return ordered[:MAX_REPORTED]
