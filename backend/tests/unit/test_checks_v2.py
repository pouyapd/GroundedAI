"""Version 2 of the answer checks (claims-v2, disagreement-v2) and the unchanged version 1.

Every example is invented for these tests; none comes from an evaluation set.
"""

from __future__ import annotations

import pytest

from groundedai.core.quantities import limit_of, read_quantities, read_ranges
from groundedai.core.text import light_stem
from groundedai.generation.answer_checks import (
    addresses_question,
    check_answer,
    citation_status,
    declines,
    expected_answer_type,
)
from groundedai.generation.claims import (
    EvidenceSource,
    check_claims,
    claim_stem,
    claim_stem_v2,
)
from groundedai.generation.context import SourceBlock
from groundedai.generation.disagreement import find_disagreements
from groundedai.retrieval.types import ChunkScores, RetrievedChunk

# ---------------------------------------------------------------- stemming

FAMILIES = [
    ["change", "changes", "changed", "changing"],
    ["require", "requires", "required", "requiring"],
    ["update", "updated", "updating"],
    ["date", "dates", "dated"],
    ["base", "based"],
    ["use", "used", "uses", "using"],
    ["apply", "applies", "applied", "applying"],
    ["empty", "emptied"],
    ["cancel", "cancelled", "canceled", "cancelling"],
    ["month", "monthly"],
    ["plan", "planned", "planning"],
    ["agree", "agreed"],
    ["install", "installed"],
]


@pytest.mark.parametrize("family", FAMILIES, ids=lambda family: family[0])
def test_claim_stem_v2_joins_word_forms(family: list[str]) -> None:
    assert len({claim_stem_v2(word) for word in family}) == 1


@pytest.mark.parametrize("pair", [("plan", "plane"), ("hop", "hope"), ("rat", "rate")])
def test_claim_stem_v2_keeps_short_words_apart(pair: tuple[str, str]) -> None:
    assert claim_stem_v2(pair[0]) != claim_stem_v2(pair[1])


def test_version_one_stemmers_are_unchanged() -> None:
    # light_stem feeds the published retrieval baseline and claims-v1; pinned word for word.
    pinned = {
        "change": "change", "changed": "chang", "changes": "change", "changing": "chang",
        "require": "require", "required": "requir", "update": "update", "updated": "updat",
        "date": "date", "dated": "dat", "base": "base", "based": "bas", "use": "use",
        "used": "used", "apply": "apply", "applied": "appli", "cancelled": "cancell",
        "policies": "policy", "classes": "class", "booking": "book", "bookings": "booking",
        "days": "day", "fees": "fee", "plane": "plane", "planned": "plann", "morning": "morn",
    }  # fmt: skip
    assert {word: light_stem(word) for word in pinned} == pinned
    assert claim_stem("changed") != claim_stem("change")  # the v1 gap that v2 closes


# ----------------------------------------------------------- limits, ranges


@pytest.mark.parametrize(
    ("text", "limit"),
    [
        ("Give at least 14 days' notice.", "at_least"),
        ("Give a minimum of 14 days' notice.", "at_least"),
        ("Allow up to 14 days.", "at_most"),
        ("Stays must not exceed 14 days.", "at_most"),
        ("No more than 14 days are allowed.", "at_most"),
        ("Stays of 14 days or more need a permit.", "at_least"),
        ("Stays of 14 days need a permit.", None),
    ],
)
def test_limit_words_on_a_figure(text: str, limit: str | None) -> None:
    (figure,) = read_quantities(text).quantities
    assert limit_of(text, figure) == limit


def test_ranges_are_linked_and_codes_or_year_spans_are_not() -> None:
    assert [r.text for r in read_ranges("Open 10:00–20:00 daily.")] == ["10:00–20:00"]
    assert [r.text for r in read_ranges("Collections run between 7 am and 4 pm.")] == [
        "7 am and 4 pm"
    ]
    assert [r.text for r in read_ranges("Delivery takes 3 to 5 days.")] == ["3 to 5"]
    assert read_ranges("Model SX-5 is sold in the 2026–27 season.") == []
    assert read_ranges("Adults pay 10 and children 5.") == []  # "and" needs "between"


# ------------------------------------------------------------ claims-v2

SOURCES = [
    EvidenceSource("S1", "Hall rules", "Guests must give at least 14 days' notice to cancel."),
    EvidenceSource("S2", "Hall rules", "Fees are not refunded after the event has started."),
    EvidenceSource("S3", "Hall prices", "The main hall costs €200 per day."),
]


def test_a_reversed_limit_word_is_not_supported_in_v2_only() -> None:
    statement = "Guests must give at most 14 days' notice to cancel [S1]."
    assert check_claims(statement, SOURCES).claims[0].status == "supported"  # v1 gap
    assert check_claims(statement, SOURCES, version="claims-v2").claims[0].status != "supported"
    same = "Guests must give a minimum of 14 days' notice to cancel [S1]."
    assert check_claims(same, SOURCES, version="claims-v2").claims[0].status == "supported"


def test_a_negation_that_differs_from_the_evidence_is_uncertain() -> None:
    statement = "Fees are refunded after the event has started [S2]."
    v1 = check_claims(statement, SOURCES)
    v2 = check_claims(statement, SOURCES, version="claims-v2")
    assert v1.claims[0].status == "supported"
    assert v2.claims[0].status == "partial"
    assert "negation_differs" in v2.claims[0].flags
    assert check_answer("Are fees refunded?", statement, v2).evidence == "some_uncertain"


def test_citation_statuses_and_verdicts() -> None:
    def status(answer: str) -> str:
        return citation_status(check_claims(answer, SOURCES, version="claims-v2").claims[0])

    assert status("The main hall costs €200 per day [S3].") == "correct"
    assert status("The main hall costs €200 per day [S1].") == "wrong"
    assert status("The main hall costs €200 per day [S3][S1].") == "partly_wrong"
    assert status("The main hall costs €200 per day [S7].") == "invalid"
    assert status("The main hall costs €200 per day.") == "none"
    report = check_claims("The main hall costs €200 per day [S1].", SOURCES, version="claims-v2")
    checks = check_answer("How much is the main hall?", "", report)
    assert (checks.evidence, checks.citations) == ("all_found", "some_wrong")
    assert report.verdict == "verified"  # the v1 verdict keeps its meaning


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("How much does the main hall cost?", "amount"),
        ("How long is the notice period?", "duration"),
        ("What time does the hall open?", "time"),
        ("On what date does booking open?", "date"),
        ("How many guests are allowed?", "count"),
        ("Who approves a booking?", "person_or_role"),
        ("Which form do I need?", "code"),
        ("Can I bring a dog?", None),
        ("Why is the hall closed?", None),
        ("Where is the hall?", None),
    ],
)
def test_expected_answer_type(question: str, kind: str | None) -> None:
    assert expected_answer_type(question) == kind


def test_an_answer_must_give_the_asked_value_about_the_asked_item() -> None:
    def addresses(question: str, answer: str) -> str:
        report = check_claims(answer, SOURCES, question, "claims-v2")
        return addresses_question(question, answer, report.claims)[0]

    question = "How much does the main hall cost per day?"
    assert addresses(question, "The main hall costs €200 per day [S3].") == "yes"
    # the same kind of value for something else is not the asked item
    assert addresses("How much does the garden room cost?", "The main hall costs €200.") == "no"
    assert addresses(question, "Guests must give at least 14 days' notice [S1].") == "no"
    assert addresses(question, "The documents do not give a price for the hall.") == "declined"
    assert addresses("Can guests cancel?", "Yes, with 14 days' notice [S1].") == "not_checked"
    assert declines("Neither document gives the price.")
    assert not declines("The hall costs €200 per day.")


# ------------------------------------------------------- disagreement-v2


def _blocks(*passages: tuple[str, str, str]) -> list[SourceBlock]:
    blocks = []
    for index, (document, title, text) in enumerate(passages, start=1):
        chunk = RetrievedChunk(
            f"c{index}", document, title, 0, text, 0, len(text), None, None, ChunkScores()
        )
        blocks.append(SourceBlock(f"S{index}", chunk, text, False))
    return blocks


def _found(question: str, blocks: list[SourceBlock], version: str) -> list[tuple[str, ...]]:
    return [
        (d.kind, *d.labels)
        for d in find_disagreements(question, blocks, version)  # type: ignore[arg-type]
    ]


def test_ranges_that_share_one_end_disagree_in_v2() -> None:
    blocks = _blocks(
        ("a", "Fuel pontoon", "The fuel pontoon opens daily from 08:00 to 20:00."),
        ("b", "Services", "The fuel pontoon is open daily from 08:00 to 18:00."),
    )
    question = "What are the fuel pontoon opening hours?"
    assert _found(question, blocks, "disagreement-v1") == []  # shared end read as agreement
    assert _found(question, blocks, "disagreement-v2") == [("conflict", "S1", "S2")]
    (found,) = find_disagreements(question, blocks, "disagreement-v2")
    assert "08:00–20:00" in found.explanation()


def test_a_value_at_one_end_of_a_range_agrees() -> None:
    blocks = _blocks(
        ("a", "Tours", "Last tour ends by 15:45 every day."),
        ("b", "Tour times", "Tours run every day from 13:15 to 15:45."),
    )
    assert _found("When do the tours end?", blocks, "disagreement-v2") == []


def test_edit_dates_and_validity_dates_are_told_apart() -> None:
    edits = _blocks(
        ("a", "Fees (last updated 12 May 2025)", "The locker fee is €60 per month."),
        ("b", "Fees (last updated 3 March 2026)", "The locker fee is €75 per month."),
    )
    seasons = _blocks(
        ("a", "Rates 2025 season", "The locker fee is €60 per month."),
        ("b", "Rates 2026 season", "The locker fee is €75 per month."),
    )
    one_side = _blocks(
        ("a", "Rates 2026 season", "The locker fee is €60 per month."),
        ("b", "Locker guide", "The locker fee is €75 per month."),
    )
    question = "How much is the locker fee per month?"
    assert _found(question, edits, "disagreement-v1") == [("different_dates", "S1", "S2")]
    assert _found(question, edits, "disagreement-v2") == [("conflict", "S1", "S2")]
    assert _found(question, seasons, "disagreement-v2") == [("different_dates", "S1", "S2")]
    assert _found(question, one_side, "disagreement-v2") == [("conflict", "S1", "S2")]
    (edited,) = find_disagreements(question, edits, "disagreement-v2")
    assert edited.first.revision_dates
    assert not edited.first.validity_dates
    assert "last updated" in edited.explanation()
