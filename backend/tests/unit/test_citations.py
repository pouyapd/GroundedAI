import pytest

from groundedai.generation.citations import (
    assess_support,
    bracketed_only,
    labels_in,
    malformed_citations,
    parse_answer,
    split_answer_sentences,
    states_conflict,
    strip_citations,
)


def test_document_titles_in_brackets_are_reported_as_malformed_citations() -> None:
    text = "[Brindlewick Cycling Club - Ride Rules] Rides start at 08:00 [S1]. See [S2, S3]."
    assert malformed_citations(text) == ["Brindlewick Cycling Club - Ride Rules"]


def test_bracketed_abstentions_are_not_malformed_citations() -> None:
    assert malformed_citations("[Insufficient Evidence: no allowance is mentioned.]") == []
    assert malformed_citations("[S1][S2]") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The policy says 28 days [S1], but the FAQ says 14 days [S2].", True),
        ("The sources disagree: €140 [S1] and €120 [S2].", True),
        ("There is a conflict between the two documents.", True),
        ("The limit is €140 in capital cities and €110 elsewhere [S1].", False),
        ("Different rules apply to children [S1].", False),
    ],
)
def test_conflict_statements(text: str, expected: bool) -> None:
    assert states_conflict(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "The sources do not provide information about the boiling point of water. [S5]",
        "Sources [S1] and [S3] do not provide information about student discounts. "
        "[S1] states that the policy applies to all journeys.",
        "[No evidence]\n\nThe sources describe delays and refunds, not discounts.",
        "No relevant information is given about parking [S2].",
        "Kestrel Valley Rail does not explicitly state whether students get a discount. "
        "[S1] mentions that the policy applies to all journeys.",
    ],
)
def test_refusals_that_carry_a_citation_are_refusals(text: str) -> None:
    parsed = parse_answer(text, {**SOURCES, "S3": "x", "S5": "y"})
    assert parsed.is_abstention


@pytest.mark.parametrize(
    "text",
    [
        "The exact date is not mentioned, but claims must be made within 28 days [S1].",
        "The hotel limit is €140 per night in capital cities [S1].",
        "Sources agree that receipts are needed above €25 [S2].",
        "Advance tickets cannot be refunded, and the operator does not allow refunds [S1].",
    ],
)
def test_answers_and_hedged_answers_with_figures_stay_answers(text: str) -> None:
    assert not parse_answer(text, SOURCES).is_abstention


def test_fabricated_label_is_invalid_and_does_not_ground_the_answer() -> None:
    parsed = parse_answer("The hotel limit is €140 per night [S7].", SOURCES)
    assert parsed.check.invalid == ["S7"]
    assert parsed.grounding == "ungrounded"


def test_invented_number_with_a_valid_citation_is_unsupported() -> None:
    parsed = parse_answer("The hotel limit is €180 per night in capital cities [S1].", SOURCES)
    assert parsed.check.all_valid
    assert parsed.sentences[0].support == "unsupported"
    assert parsed.sentences[0].unsupported_numbers == ["180"]
    assert parsed.grounding == "ungrounded"


def test_answer_without_any_citation_is_ungrounded() -> None:
    parsed = parse_answer("The hotel limit is €140 per night in capital cities.", SOURCES)
    assert not parsed.is_abstention
    assert parsed.check.cited == []
    assert parsed.grounding == "ungrounded"


SOURCES = {
    "S1": "The hotel limit is €140 per night in capital cities.",
    "S2": "Receipts are required for any single expense above €25.",
}


def test_label_formats() -> None:
    assert labels_in("A [S1]. B [S2, S3]; C [s4][S1] (S9) [Source 5]") == ["S1", "S2", "S3", "S4"]


def test_strip_citations() -> None:
    assert strip_citations("Limit is €140 [S1] per night [S2, S3].") == "Limit is €140 per night."
    assert (
        strip_citations("Boil for 1 minute [S1] , then cool [S2] .")
        == "Boil for 1 minute, then cool."
    )


def test_citation_after_full_stop_belongs_to_previous_sentence() -> None:
    # Regression: "[S1]" used to be credited to the following sentence.
    assert split_answer_sentences("The limit is €140. [S1] Receipts above €25. [S2]") == [
        "The limit is €140. [S1]",
        "Receipts above €25. [S2]",
    ]
    assert split_answer_sentences("One fact [S1]. [S2] Second fact [S3].") == [
        "One fact [S1]. [S2]",
        "Second fact [S3].",
    ]


def test_valid_grounded_answer() -> None:
    parsed = parse_answer(
        "The hotel limit is €140 per night in capital cities [S1]. "
        "Receipts are required above €25 [S2].",
        SOURCES,
    )
    assert not parsed.is_abstention
    assert parsed.check.valid == ["S1", "S2"]
    assert parsed.check.invalid == []
    assert parsed.check.all_valid
    assert parsed.grounding == "grounded"
    assert [s.support for s in parsed.sentences] == ["supported", "supported"]


def test_invented_citation_label_is_invalid() -> None:
    parsed = parse_answer("The limit is €140 per night [S1]. Also see [S7].", SOURCES)
    assert parsed.check.invalid == ["S7"]
    assert not parsed.check.all_valid
    assert parsed.grounding == "partially_grounded"


def test_citation_existence_is_not_semantic_support() -> None:
    # The label is valid, but the number is not in the cited source.
    parsed = parse_answer("The hotel limit is €200 per night in capital cities [S1].", SOURCES)
    assert parsed.check.all_valid
    sentence = parsed.sentences[0]
    assert sentence.support == "unsupported"
    assert sentence.unsupported_numbers == ["200"]
    assert parsed.grounding == "ungrounded"


def test_uncited_answer_is_ungrounded() -> None:
    parsed = parse_answer("Hotels cost up to €140 per night.", SOURCES)
    assert parsed.grounding == "ungrounded"
    assert parsed.sentences[0].support == "uncited"


def test_explicit_abstention_marker() -> None:
    parsed = parse_answer("INSUFFICIENT_EVIDENCE: the sources do not mention contractors.", SOURCES)
    assert parsed.is_abstention
    assert parsed.abstention_reason == "the sources do not mention contractors."
    assert parsed.grounding is None


def test_phrase_abstention_without_citations() -> None:
    parsed = parse_answer("The provided sources do not contain this information.", SOURCES)
    assert parsed.is_abstention


def test_partial_answer_with_citation_is_not_an_abstention() -> None:
    parsed = parse_answer(
        "The sources do not mention Tokyo, but the hotel limit is €140 per night in capital "
        "cities [S1].",
        SOURCES,
    )
    assert not parsed.is_abstention


def test_support_partial_label() -> None:
    assessment = assess_support("The hotel limit applies to trips abroad [S1].", [SOURCES["S1"]])
    assert assessment.support in {"partial", "unsupported"}
    assert assessment.token_recall is not None


def test_tolerant_abstention_detection_for_small_models() -> None:
    # Regression: real Qwen2.5-0.5B outputs that were scored as "answered, ungrounded".
    for raw in (
        "[Insuffcient Evidence: No specific information provided about the policy.]",
        "[InsuffICIENT_EVIDENCE: No relevant information found.]",
        "The meal allowance is not mentioned in the sources.",
    ):
        assert parse_answer(raw, SOURCES).is_abstention, raw
    explicit = parse_answer("[InsuffICIENT_EVIDENCE: No relevant information found.]", SOURCES)
    assert explicit.abstention_reason == "No relevant information found."


def test_cited_answer_mentioning_missing_detail_is_not_an_abstention() -> None:
    parsed = parse_answer(
        "The hotel limit is €140 per night in capital cities [S1]. A Tokyo rate is not mentioned.",
        SOURCES,
    )
    assert not parsed.is_abstention


def test_abstention_after_leading_labels_is_recognised() -> None:
    # Regression: real Qwen2.5-0.5B output that cited sources and then declined.
    raw = (
        "[Orion ML Platform - Incident Response Runbook][S1][Data Retention][S2]\n"
        "Insufficient Evidence: No specific command is mentioned in the provided sources."
    )
    parsed = parse_answer(raw, SOURCES)
    assert parsed.is_abstention
    assert parsed.abstention_reason == "No specific command is mentioned in the provided sources."
    # A normal answer that merely mentions the phrase later is not an abstention.
    later = parse_answer(
        "The hotel limit is €140 per night in capital cities [S1]. There is insufficient "
        "evidence about Tokyo.",
        SOURCES,
    )
    assert not later.is_abstention


def test_bracketed_only_finds_replies_without_text_outside_brackets() -> None:
    assert bracketed_only("[21]")
    assert bracketed_only("[S1][S2]")
    assert bracketed_only("[Travel Policy] [Finance FAQ]")
    assert not bracketed_only("[Standard plan]: First response within 2 business days")
    assert not bracketed_only("The limit is EUR 140 [S1].")
    assert not bracketed_only("[S1] 12")  # a bare value outside brackets is checked as a claim
    assert not bracketed_only("")
