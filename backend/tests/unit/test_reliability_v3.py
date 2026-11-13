"""Question relevance, added details, document editions and disagreement-v3 (2026-11-13)."""

from __future__ import annotations

import pytest

from groundedai.generation.claims import EvidenceSource, check_claims
from groundedai.generation.context import assemble_context
from groundedai.generation.disagreement import find_disagreements
from groundedai.generation.extractive import ExtractiveDemoChatModel
from groundedai.generation.prompts import ABSTAIN_MARKER
from groundedai.generation.relevance import question_relevance
from groundedai.generation.versions import edition_of, families, same_family
from groundedai.retrieval.types import ChunkScores, RetrievedChunk


def _sources(*passages: tuple[str, str, str]):  # type: ignore[no-untyped-def]
    chunks = [
        RetrievedChunk(f"c{i}", doc, title, 0, text, 0, len(text), 1, 1, ChunkScores())
        for i, (doc, title, text) in enumerate(passages)
    ]
    return assemble_context(chunks, max_chars=6000).sources


# --- question relevance -------------------------------------------------------------------


def test_title_words_form_words_and_generic_verbs_do_not_name_what_is_asked() -> None:
    relevance = question_relevance(
        "How long does the AI RMF say a review should last?",
        ["Artificial Intelligence Risk Management Framework (AI RMF 1.0)"],
        ["The review of the framework is expected no later than 2028."],
    )
    assert "rmf" not in relevance.asked  # title word
    assert "long" not in relevance.asked  # "how long" asks for a kind of value
    assert "say" not in relevance.asked  # generic question verb
    assert "review" in relevance.asked


def test_an_asked_word_found_in_no_passage_is_unmatched() -> None:
    relevance = question_relevance(
        "What is the water hardness in the supply area?",
        ["Kestrel Valley Water"],
        ["We supply drinking water to 226,000 households in the supply area."],
    )
    assert relevance.asked == {"hardness", "area", "supply"}
    assert relevance.unmatched == ("hardness",)
    assert relevance.coverage("The supply area has 226,000 households.") == pytest.approx(2 / 3)


def test_the_extractive_generator_refuses_when_the_passages_do_not_mention_what_is_asked() -> None:
    # Regression (t38-like, written on validation data): "accuracy" and "chatbot" are in no
    # passage, but "require" and the title words are; v1 answered with an unrelated sentence.
    model = ExtractiveDemoChatModel(matching="prefix", relevance="v2")
    sources = [("S1", "AI RMF use-case profiles describe the requirements for a hiring profile.")]
    titles = ["Artificial Intelligence Risk Management Framework (AI RMF 1.0)"]
    question = "What minimum accuracy does the AI RMF require for chatbots?"
    reply = model.compose(question, sources, titles)
    assert reply.startswith(ABSTAIN_MARKER)
    assert '"accuracy"' in reply
    old = ExtractiveDemoChatModel(matching="prefix", relevance="v1")
    assert not old.compose(question, sources, titles).startswith(ABSTAIN_MARKER)


def test_a_heading_does_not_win_over_the_sentence_that_answers() -> None:
    # Regression: leaving generic verbs out of the scoring made "Error codes" (a heading)
    # outscore "E07 means fan failure." Generic words only decide relevance, not scoring.
    model = ExtractiveDemoChatModel(matching="prefix", relevance="v2")
    reply = model.compose(
        "What does error code E07 mean?",
        [("S1", "Error codes\nE07 means fan failure. E09 means over-temperature.")],
        ["Inverter Manual"],
    )
    assert "fan failure" in reply


# --- added details --------------------------------------------------------------------------

FERRY = (
    "A single adult ticket from Kessport to Mirrow costs 14 euros and includes a seat. "
    "Children under 5 travel free on board."
)


@pytest.mark.parametrize(
    ("claim", "status"),
    [
        ("A single adult ticket from Kessport to Mirrow costs 14 euros [S1].", "supported"),
        ("Children under 5 travel free [S1].", "supported"),
        (
            "A single adult ticket to Mirrow costs 14 euros and includes a free hot meal on board "
            "[S1].",
            "partial",
        ),
    ],
)
def test_added_details_make_a_matching_claim_partial(claim: str, status: str) -> None:
    report = check_claims(
        claim, [EvidenceSource("S1", "Ferry Guide", FERRY)], "", detail_check=True
    )
    assert [c.status for c in report.claims] == [status]
    if status == "partial":
        assert report.claims[0].added_details == ["hot meal"]
        assert "added_detail" in report.claims[0].flags


def test_one_unmatched_word_is_not_an_added_detail() -> None:
    # A single paraphrased word ("journey" for "ticket") is not counted.
    report = check_claims(
        "A single adult journey from Kessport to Mirrow costs 14 euros [S1].",
        [EvidenceSource("S1", "Ferry Guide", FERRY)],
        "",
        detail_check=True,
    )
    assert report.claims[0].added_details == []


def test_the_detail_check_is_off_by_request() -> None:
    claim = "A single adult ticket to Mirrow costs 14 euros and includes a free hot meal on board."
    report = check_claims(claim, [EvidenceSource("S1", "Ferry Guide", FERRY)], "")
    assert report.claims[0].status != "partial"  # without the check, the lexical match passes
    assert report.claims[0].added_details == []


# --- document editions ------------------------------------------------------------------------


def test_editions_of_one_document_are_one_family_and_other_documents_are_not() -> None:
    a = edition_of("a", "Kestrel Valley Water - Customer Service Standards (Edition 2019)")
    b = edition_of("b", "Kestrel Valley Water - Customer Service Standards (Edition 2024)")
    other = edition_of("c", "Kestrel Valley Water - Annual Report")
    rmf = edition_of("d", "Artificial Intelligence Risk Management Framework (AI RMF 1.0)")
    profile = edition_of(
        "e",
        "Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence "
        "Profile",
    )
    assert same_family(a, b)
    assert not same_family(a, other)
    assert not same_family(rmf, profile)  # a companion document, not an edition
    assert families({"a": a, "b": b, "c": other}) == [["a", "b"]]
    assert a.describe() == "Edition 2019"


def test_a_publication_date_counts_only_after_a_publication_word() -> None:
    opening = "Date updated: July 24, 2025\nPublication Date(s) June 2017 (includes updates)"
    edition = edition_of("x", "Digital Identity Guidelines", opening)
    assert edition.date == "June 2017"
    assert edition_of("y", "Digital Identity Guidelines", "Updated July 24, 2025").date is None


# --- disagreement-v3 ----------------------------------------------------------------------

OLD = ("old", "Corvane Systems - Access Control Standard (Revision A)")
NEW = ("new", "Corvane Systems - Access Control Standard (Revision B)")


def _v3(question: str, old_text: str, new_text: str):  # type: ignore[no-untyped-def]
    sources = _sources((*OLD, old_text), (*NEW, new_text))
    return find_disagreements(question, sources, "disagreement-v3")


def test_two_editions_with_different_values_are_reported_as_different_versions() -> None:
    found = _v3(
        "After how much inactivity must AL2 users be reauthenticated?",
        "At AL2, users SHALL be reauthenticated after 30 minutes of inactivity.",
        "At AL2, reauthentication SHALL occur after 1 hour of inactivity.",
    )
    assert [d.kind for d in found] == ["different_versions"]
    assert {found[0].first.edition, found[0].second.edition} == {"Revision A", "Revision B"}
    assert "does not decide which version applies" in found[0].explanation()
    # v1 drops "AL2" as a name and does not match the word forms: nothing found
    sources = _sources(
        (*OLD, "At AL2, users SHALL be reauthenticated after 30 minutes of inactivity."),
        (*NEW, "At AL2, reauthentication SHALL occur after 1 hour of inactivity."),
    )
    assert (
        find_disagreements("After how much inactivity must AL2 users be reauthenticated?", sources)
        == []
    )


def test_a_change_of_requirement_strength_is_a_difference() -> None:
    found = _v3(
        "May administrators share accounts?",
        "Administrators SHOULD NOT share accounts.",
        "Administrators SHALL NOT share accounts.",
    )
    assert [(d.kind, d.reason) for d in found] == [("different_versions", "requirement")]
    assert {found[0].first.modal, found[0].second.modal} == {"SHOULD NOT", "SHALL NOT"}


def test_the_same_value_in_other_units_is_agreement() -> None:
    assert (
        _v3(
            "How fast does the helpline answer calls?",
            "The helpline answers calls within 2 minutes on average.",
            "The helpline answers calls within 120 seconds on average.",
        )
        == []
    )


def test_a_statement_about_another_assurance_level_is_not_compared() -> None:
    assert (
        _v3(
            "What is the inactivity timeout at AL2?",
            "At AL2, the inactivity timeout SHALL be 30 minutes.",
            "At AL1, the inactivity timeout SHALL be no more than 4 hours.",
        )
        == []
    )


def test_ranges_are_compared_as_a_whole_across_editions() -> None:
    found = _v3(
        "What are the opening hours on weekdays?",
        "The office opens from 09:00 to 19:00 on weekdays and from 10:00 to 16:00 on Saturdays.",
        "The office opens from 09:00 to 20:00 on weekdays and from 10:00 to 16:00 on Saturdays.",
    )
    assert [d.kind for d in found] == ["different_versions"]


def test_a_section_number_glued_to_a_sentence_is_not_a_value() -> None:
    found = _v3(
        "Must staff show photo identification before entering a property?",
        "4 Visits Field staff SHOULD show photo identification before entering a property.",
        "4 Visits Field staff SHALL show photo identification before entering a property.",
    )
    assert [(d.kind, d.reason) for d in found] == [("different_versions", "requirement")]


def test_a_name_in_the_question_needs_all_its_words_in_a_document() -> None:
    # Regression (written on dev data, n10): "library" alone let another library's rules in.
    sources = _sources(
        ("orchard", "Orchard Lane Library Card Rules", "Members may borrow up to 12 items."),
        ("north", "Northgate Library - Lending Rules", "Members can borrow up to 10 items."),
    )
    question = "How many items can members borrow at Orchard Lane Library?"
    assert find_disagreements(question, sources, "disagreement-v3") == []
    assert find_disagreements(question, sources, "disagreement-v1") != []


def test_pairs_that_are_not_editions_are_judged_as_in_v1() -> None:
    sources = _sources(
        ("handbook", "Orbis Fitness - Member Handbook", "The monthly membership fee is €39."),
        ("faq", "Orbis Fitness - Website FAQ", "Membership costs €45 per month."),
    )
    question = "How much is the monthly membership fee at Orbis Fitness?"
    v1 = [(d.kind, d.labels) for d in find_disagreements(question, sources, "disagreement-v1")]
    v3 = [(d.kind, d.labels) for d in find_disagreements(question, sources, "disagreement-v3")]
    assert v3 == v1 == [("conflict", ("S1", "S2"))]
