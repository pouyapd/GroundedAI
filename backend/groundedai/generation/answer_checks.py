"""Three separate verdicts on a checked answer (``claims-v2``).

Version 1 of the claim check gives one verdict (:attr:`ClaimReport.verdict`): whether every
statement of the answer was found in a retrieved passage, whatever its citation. Version 2
keeps that verdict, with the same meaning, and adds three results so a reader can see what
is wrong:

``evidence``
    ``all_found``, ``some_uncertain``, ``some_not_found``, ``none_found`` or
    ``not_applicable``. A statement that is only partly found counts as uncertain, never as not
    found.
``citations``
    ``all_correct``, ``some_wrong``, ``some_unconfirmed``, ``none_given`` or
    ``not_applicable``, from each statement's citation status:

    * ``correct``: every cited passage states at least part of it, and they state all of it;
    * ``wrong``: the cited passage does not state it, but another passage does;
    * ``partly_wrong``: one of its cited passages states nothing of it;
    * ``invalid``: a cited label is not among the passages;
    * ``unconfirmed``: it cites a passage, but the statement was not found in any passage;
    * ``none``: it cites nothing.
``addresses_question``
    ``yes``, ``no``, ``declined`` or ``not_checked``. The question's wording says what kind of
    value it asks for (an amount for "how much" or a price, a duration for "how long", a time
    for "what time", a person or role for "who", an identifier for "which form" or "which
    code", ...). The answer gets ``yes`` when a statement gives a value of that kind and shares
    the question's subject words, ``no`` when none does, and ``declined`` when it says the
    documents do not give what was asked. Yes/no, how, why and where questions are
    ``not_checked``.

All of this is lexical, like the claim check. A value of the right kind for the wrong item can
still pass when it shares the question's words, and an answer that names the asked item with
other words can be marked ``no``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from groundedai.core.quantities import read_quantities
from groundedai.core.text import split_sentences, tokenize
from groundedai.generation.claims import (
    MATCHED,
    QUESTION_FILLERS,
    Claim,
    ClaimReport,
    claim_stem_v2,
    claim_words,
)

ANSWER_CHECK_VERSION = "claims-v2"

EvidenceVerdict = Literal[
    "all_found", "some_uncertain", "some_not_found", "none_found", "not_applicable"
]
CitationVerdict = Literal[
    "all_correct", "some_wrong", "some_unconfirmed", "none_given", "not_applicable"
]
CitationStatus = Literal["correct", "wrong", "partly_wrong", "invalid", "unconfirmed", "none"]
Addresses = Literal["yes", "no", "declined", "not_checked"]
AnswerType = Literal[
    "amount", "percentage", "duration", "time", "date", "count", "person_or_role", "code"
]

# ------------------------------------------------------------------ verdicts


def evidence_verdict(report: ClaimReport) -> EvidenceVerdict:
    if not report.claims:
        return "not_applicable"
    statuses = [claim.status for claim in report.claims]
    if all(status in MATCHED for status in statuses):
        return "all_found"
    if all(status == "unsupported" for status in statuses):
        return "none_found"
    if "unsupported" in statuses:
        return "some_not_found"
    return "some_uncertain"


def citation_status(claim: Claim) -> CitationStatus:
    if claim.invalid_citations:
        return "invalid"
    if not claim.cited:
        return "none"
    if claim.status == "supported":
        return (
            "correct"
            if all(claim.label_support.get(label, True) for label in claim.cited)
            else ("partly_wrong")
        )
    if claim.status == "wrong_source":
        return "wrong"
    return "unconfirmed"


def citation_verdict(report: ClaimReport) -> CitationVerdict:
    if not report.claims:
        return "not_applicable"
    statuses = [citation_status(claim) for claim in report.claims]
    if any(status in ("wrong", "partly_wrong", "invalid") for status in statuses):
        return "some_wrong"
    if "unconfirmed" in statuses:
        return "some_unconfirmed"
    if all(status == "none" for status in statuses):
        return "none_given"
    return "all_correct"


# ----------------------------------------------------------- the asked item
_YES_NO_RE = re.compile(
    r"^\s*(?:is|are|am|was|were|do|does|did|can|could|will|would|should|shall|may|might|must|"
    r"has|have|had|if\b[^?]*,\s*(?:is|are|do|does|can|will|would|should|may|must))\b",
    re.IGNORECASE,
)
_NOT_CHECKED_RE = re.compile(
    r"^\s*(?:why|where)\b|\bhow (?:do|does|can|could|should|would|must|did) (?:i|we|you|they|"
    r"one|parents|students|residents|visitors|members|patients|owners|customers)\b|^\s*how to\b"
    r"|\bwhat (?:do|should|must) (?:i|we|you) (?:do|bring|need)\b",
    re.IGNORECASE,
)
# Checked in order; the first match decides.
_ANSWER_TYPES: list[tuple[re.Pattern[str], AnswerType]] = [
    (re.compile(r"\bwhat percentage\b|\bwhat share\b|\bhow many per ?cent\b", re.I), "percentage"),
    (
        re.compile(
            r"\bhow long\b|\bhow soon\b|\bhow quickly\b|\bhow far in advance\b|\bhow much notice\b"
            r"|\bhow many (?:seconds|minutes|hours|days|nights|weeks|months|years|working days|"
            r"business days)\b|\bfor how many (?:days|nights|weeks|months|years)\b",
            re.I,
        ),
        "duration",
    ),
    (
        re.compile(
            r"\b(?:what|until what|by what|from what) time\b|\bat what time\b|\bwhich hours\b"
            r"|\bwhat hours\b|\bopening hours\b|\bopening times?\b"
            r"|\bwhen (?:does|do|is|are|will)\b[^?]*\b(?:open|opens|close|closes|start|starts|"
            r"end|ends|begin|begins|served)\b",
            re.I,
        ),
        "time",
    ),
    (
        re.compile(
            r"\b(?:what|which) date\b|\bon what date\b|\bwhich day\b|\bwhat day\b|\bby when\b"
            r"|^\s*when\b",
            re.I,
        ),
        "date",
    ),
    (re.compile(r"\bhow many\b|\bwhat (?:is the )?(?:\w+ )?number of\b", re.I), "count"),
    (
        re.compile(
            r"\bhow much\b|\b(?:cost|costs|price|prices|fee|fees|charge|charges|charged|fine|"
            r"fines|deposit|surcharge|tariff|rate|rates|refund|pay|paid|payable)\b",
            re.I,
        ),
        "amount",
    ),
    (
        re.compile(
            r"^\s*who\b|\bwhom\b|\bwhose\b|\bwhich (?:person|staff member|member of staff|"
            r"officer|manager|teacher|doctor|nurse|vet|dentist)\b|\bwho (?:is|are|should|do|does|"
            r"can|must|will|chairs|leads|runs|handles|arranges|approves)\b",
            re.I,
        ),
        "person_or_role",
    ),
    (
        re.compile(
            r"\b(?:which|what) (?:form|code|reference|prefix|letters|identifier|permit number|"
            r"form number)\b|\breference number\b",
            re.I,
        ),
        "code",
    ),
]
_KINDS: dict[AnswerType, frozenset[str]] = {
    "amount": frozenset({"money"}),
    "percentage": frozenset({"percent"}),
    "duration": frozenset({"duration"}),
    "time": frozenset({"time_of_day"}),
    "date": frozenset({"date", "year"}),
    "count": frozenset({"count", "number"}),
}
# Words that only name the kind of value asked for ("on what date", "what time").
_TYPE_WORDS = frozenset(
    [
        "date",
        "dates",
        "time",
        "times",
        "cost",
        "costs",
        "price",
        "prices",
        "much",
        "many",
        "number",
        "numbers",
        "long",
        "name",
        "called",
        "day",
        "days",
        "amount",
    ]
)
_CAPITALISED_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’]*(?:-[A-Za-z0-9]+)*")
_ROLE_WORDS = frozenset(
    [
        "manager",
        "officer",
        "coordinator",
        "director",
        "lead",
        "head",
        "administrator",
        "assistant",
        "nurse",
        "doctor",
        "dentist",
        "vet",
        "veterinarian",
        "receptionist",
        "secretary",
        "chair",
        "chairperson",
        "president",
        "supervisor",
        "teacher",
        "tutor",
        "instructor",
        "master",
        "warden",
        "inspector",
        "adviser",
        "advisor",
        "counsellor",
        "clerk",
        "registrar",
        "treasurer",
        "trustee",
        "owner",
        "team",
        "department",
        "office",
        "desk",
        "deputy",
        "principal",
        "keyholder",
        "ranger",
        "captain",
        "steward",
        "leader",
        "organiser",
        "organizer",
        "contact",
    ]
)
_PERSON_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+[A-Z]"
    r"|\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b"
    r"|,\s+[A-Z][a-z]+\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b"
)
_CODE_RE = re.compile(r"\b[A-Z]{2,}(?:[-/]?[A-Z0-9]+)*\b|[\"'“‘][A-Z0-9][A-Z0-9-]{1,}[\"'”’]")
_COMMON_CAPITALS = frozenset({"VAT", "UK", "EU", "US", "USA", "PDF", "FAQ", "ID", "AM", "PM", "OK"})


def expected_answer_type(question: str) -> AnswerType | None:
    """The kind of value the question asks for, or None for yes/no, how, why and where
    questions and for questions whose wording gives no kind."""
    if _YES_NO_RE.search(question) or _NOT_CHECKED_RE.search(question):
        return None
    for pattern, answer_type in _ANSWER_TYPES:
        if pattern.search(question):
            return answer_type
    return None


def gives_value(text: str, answer_type: AnswerType) -> bool:
    """Whether ``text`` states a value of the asked kind."""
    if answer_type in _KINDS:
        return any(q.kind in _KINDS[answer_type] for q in read_quantities(text).quantities)
    if answer_type == "person_or_role":
        words = {claim_stem_v2(token) for token in tokenize(text)}
        return bool(_PERSON_RE.search(text) or words & {claim_stem_v2(w) for w in _ROLE_WORDS})
    codes = [m.group(0) for m in _CODE_RE.finditer(text)]
    return bool(read_quantities(text).identifiers) or any(
        code.strip("\"'“”‘’") not in _COMMON_CAPITALS for code in codes
    )


def subject_words(question: str) -> set[str]:
    """The question's subject words: content words without names and without words that only
    say what kind of value is asked for."""
    names = {
        claim_stem_v2(token)
        for match in _CAPITALISED_RE.finditer(question)
        if match.start() > 0
        for token in tokenize(match.group(0))
    }
    fillers = {claim_stem_v2(word) for word in QUESTION_FILLERS | _TYPE_WORDS}
    return claim_words(question, claim_stem_v2) - names - fillers


# A sentence saying the documents do not give something.
_DOCUMENT_NOUNS = (
    r"(?:documents?|sources?|passages?|texts?|handbook|guide|policy|policies|faq|faqs|notice|"
    r"information(?:\s+provided)?|materials?|brochure|leaflet|website|page|sheet|terms|"
    r"agreement|pack|letter|records?|newsletter|rules|schedule|price list|tariff)"
)
_STATING_VERBS = (
    r"(?:say|state|give|mention|specify|list|include|contain|provide|name|explain|show|tell|"
    r"cover|set out|describe|indicate|confirm|quote|identify|offer|detail)"
)
_DECLINE_RE = re.compile(
    rf"\b{_DOCUMENT_NOUNS}\b[^.;]{{0,30}}?\b(?:do|does|did)(?:\s+not|n['’]t)\s+(?:\w+\s+)?"
    rf"{_STATING_VERBS}"
    rf"|\bneither\s+(?:of\s+the\s+)?(?:documents?|sources?|passages?|texts?)\s+\w*\s*"
    rf"{_STATING_VERBS}?"
    rf"|\bthere(?:['’]s|\s+is|\s+are)\s+no\b[^.;]{{0,80}}?\b(?:in|from)\s+(?:the|these|those|"
    rf"either|both)\s+{_DOCUMENT_NOUNS}"
    rf"|\b(?:is|are|was|were)\s+not\s+(?:stated|mentioned|given|specified|listed|included|"
    rf"provided|covered|named|explained|shown)\s+(?:in|by|anywhere)\b"
    rf"|\bno\s+(?:information|details?|figure|price|date|time|name)\s+(?:is|are)\s+"
    rf"(?:given|provided|stated|available|mentioned)\b"
    rf"|\bI\s+(?:can['’]t|cannot|can\s+not|am\s+unable\s+to)\s+(?:tell|find|say|confirm)\b",
    re.IGNORECASE,
)


def declines(answer: str) -> bool:
    """Whether the answer says the documents do not give what was asked."""
    return any(_DECLINE_RE.search(sentence) for sentence in split_sentences(answer))


@dataclass(frozen=True, slots=True)
class AnswerChecks:
    evidence: EvidenceVerdict
    citations: CitationVerdict
    addresses_question: Addresses
    expected_answer_type: AnswerType | None
    citation_statuses: list[CitationStatus] = field(default_factory=list)
    version: str = ANSWER_CHECK_VERSION


SUBJECT_SHARE = 0.5  # share of the question's subject words a value statement must contain
SUBJECT_WORDS = 2  # or this many of them


def addresses_question(
    question: str, answer: str, claims: Sequence[Claim]
) -> tuple[Addresses, AnswerType | None]:
    """Whether the answer gives the kind of value the question asks for, about its subject."""
    answer_type = expected_answer_type(question)
    if answer_type is not None:
        subject = subject_words(question)
        for claim in claims:
            if not gives_value(claim.text, answer_type):
                continue
            shared = len(subject & claim_words(claim.text, claim_stem_v2))
            if not subject or shared >= SUBJECT_WORDS or shared / len(subject) >= SUBJECT_SHARE:
                return "yes", answer_type
    if declines(answer):
        return "declined", answer_type
    if answer_type is None:
        return "not_checked", None
    return "no", answer_type


def check_answer(question: str, answer: str, report: ClaimReport) -> AnswerChecks:
    """The three verdicts for an answer whose statements ``report`` checked (claims-v2)."""
    addresses, answer_type = addresses_question(question, answer, report.claims)
    return AnswerChecks(
        evidence=evidence_verdict(report),
        citations=citation_verdict(report),
        addresses_question=addresses,
        expected_answer_type=answer_type,
        citation_statuses=[citation_status(claim) for claim in report.claims],
    )
