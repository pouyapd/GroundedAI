from groundedai.generation.citations import (
    assess_support,
    labels_in,
    parse_answer,
    split_answer_sentences,
    strip_citations,
)

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
