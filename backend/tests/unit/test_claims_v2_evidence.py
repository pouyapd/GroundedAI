"""claims-v2: support from one sentence (or a sentence and its continuation), equivalent
wordings, negation and figures, citation labels, and conflicts.

Every example is invented for these tests; none comes from an evaluation set. Version 1 is
checked where its published behaviour must stay as it was.
"""

from __future__ import annotations

import pytest

from groundedai.generation.answer_checks import citation_status, citation_verdict
from groundedai.generation.claims import MATCHED, EvidenceSource, check_claims
from groundedai.generation.context import assemble_context
from groundedai.generation.disagreement import Disagreement, find_disagreements
from groundedai.ingestion.chunking import ChunkingConfig, chunk_pages, page_at
from groundedai.ingestion.parsers import PageText
from groundedai.retrieval.types import ChunkScores, RetrievedChunk

TITLE = "Larkspur Pool - Visitor Guide"
PRICES = EvidenceSource(
    "S1",
    TITLE,
    "Children under 5 swim free of charge. A 10-visit card costs €45. "
    "Towels are not provided, so please bring your own. "
    "The outdoor pool opens on 1 June 2026. Monthly fees must be paid by the 5th of each month.",
)
RULES = EvidenceSource(
    "S2",
    TITLE,
    "Each swimmer may bring one guest per visit. Lockers cost €1. "
    "Guide dogs are welcome in the café.",
)
SOURCES = [PRICES, RULES]


def _claim(answer: str, version: str = "claims-v2", sources: list[EvidenceSource] = SOURCES):
    report = check_claims(answer, sources, None, version)  # type: ignore[arg-type]
    (claim,) = report.claims
    return claim


# --------------------------------------------------------- 1. equivalent wordings
@pytest.mark.parametrize(
    "answer",
    [
        "Children under 5 do not have to pay to swim [S1].",
        "Children under 5 can swim without paying anything [S1].",
        "Children under 5 swim at no cost [S1].",
    ],
)
def test_a_paraphrase_with_the_same_meaning_is_supported(answer: str) -> None:
    claim = _claim(answer)
    assert claim.status == "supported"
    assert claim.evidence[0].quote == "Children under 5 swim free of charge."


# ------------------------------------------------------------------ 2. negation
@pytest.mark.parametrize(
    "answer",
    [
        "Towels are provided [S1].",
        "Children under 5 have to pay to swim [S1].",
    ],
)
def test_a_reversed_meaning_is_not_supported(answer: str) -> None:
    assert _claim(answer).status not in MATCHED


# ------------------------------------------------------------- 3. and 4. figures
def test_a_correct_number_is_supported_and_a_wrong_one_is_not() -> None:
    assert _claim("A 10-visit card costs €45 [S1].").status == "supported"
    assert _claim("A 10-visit card costs €40 [S1].").status == "unsupported"


def test_a_correct_date_is_supported_and_a_wrong_one_is_not() -> None:
    assert _claim("The outdoor pool opens on June 1, 2026 [S1].").status == "supported"
    assert _claim("The outdoor pool opens on 1 July 2026 [S1].").status == "unsupported"


def test_a_wrong_day_of_the_month_is_not_supported() -> None:
    assert _claim("Monthly fees must be paid by the 5th of each month [S1].").status == "supported"
    claim = _claim("Monthly fees must be paid by the 10th of each month [S1].")
    assert claim.status == "unsupported"
    assert claim.missing_figures == ["10th"]


def test_a_figure_from_the_question_must_be_found_when_the_claim_asserts_it() -> None:
    classes = EvidenceSource(
        "S1",
        "Larkspur Pool - Swimming Lessons",
        "The Minnow class is for ages 4 to 6, with one teacher for every 6 children. "
        "The Dolphin class is for ages 7 to 10, with one teacher for every 8 children.",
    )
    question = "Which class has one teacher for every 8 children?"
    asserted = check_claims(
        "The Minnow class, for ages 4 to 6, has one teacher for every 8 children [S1].",
        [classes], question, "claims-v2",
    )  # fmt: skip
    assert asserted.claims[0].status not in MATCHED
    premise = check_claims(
        "If a class has one teacher for every 8 children, it is the Dolphin class, for ages 7 "
        "to 10 [S1].",
        [classes], question, "claims-v2",
    )  # fmt: skip
    assert premise.claims[0].status == "supported"


def test_a_date_without_a_year_states_the_year_given_in_the_same_sentence() -> None:
    holidays = EvidenceSource(
        "S1",
        "Larkspur Pool - Holiday Opening",
        "On 1 January the pool is closed, and it reopens on Friday 2 January 2026.",
    )
    assert _claim("The pool is closed on 1 January 2026 [S1].", sources=[holidays]).status == (
        "supported"
    )
    assert _claim("The pool is closed on 1 January 2027 [S1].", sources=[holidays]).status != (
        "supported"
    )


# ------------------------------------------------- 5. keywords without a sentence
def test_words_from_different_sentences_do_not_support_a_claim() -> None:
    # "bring" is in one sentence and "dogs" in another: no sentence says a swimmer may bring a dog.
    claim = _claim("Each swimmer may bring a dog [S2].")
    assert claim.status not in MATCHED
    assert _claim("Each swimmer may bring a dog [S2].", "claims-v1").status == "supported"


def test_a_figure_from_a_neighbouring_sentence_does_not_support_a_claim() -> None:
    classes = EvidenceSource(
        "S1",
        "Larkspur Pool - Swimming Lessons",
        "The Minnow class is for ages 4 to 6, with one teacher for every 6 children. "
        "The Dolphin class is for ages 7 to 10, with one teacher for every 10 children.",
    )
    claim = _claim("The Minnow class has one teacher for every 10 children [S1].",
                   sources=[classes])  # fmt: skip
    assert claim.status not in MATCHED


def test_a_sentence_and_its_continuation_support_a_claim_together() -> None:
    card = EvidenceSource(
        "S1",
        "Larkspur Pool - Membership",
        "A replacement card can be ordered at reception. It costs €4 and is ready the next day.",
    )
    claim = _claim("A replacement card costs €4 [S1].", sources=[card])
    assert claim.status == "supported"
    assert [quote.quote for quote in claim.evidence] == [
        "A replacement card can be ordered at reception.",
        "It costs €4 and is ready the next day.",
    ]


# ------------------------------------------------- 6. an unsupported extra sentence
def test_an_unsupported_extra_sentence_keeps_the_answer_from_being_verified() -> None:
    report = check_claims(
        "Lockers cost €1 [S2]. Prices may have changed since this guide was written.",
        SOURCES,
        None,
        "claims-v2",
    )
    assert [claim.status for claim in report.claims] == ["supported", "unsupported"]
    assert report.verdict == "partially_verified"


# ------------------------------------------------- 7. the quote and its page
def test_the_quote_is_the_supporting_sentence_on_its_page() -> None:
    pages = [
        PageText(1, "Larkspur Pool - Visitor Guide\nCan I bring my dog to the pool?"),
        PageText(2, "Dogs are welcome in the café garden. Each swimmer may bring one guest."),
    ]
    (spec,) = chunk_pages(pages, ChunkingConfig(800, 120))
    source = EvidenceSource("S1", TITLE, spec.text)
    claim = _claim("Yes, dogs are welcome in the café garden [S1].", sources=[source])
    assert claim.status == "supported"
    (quote,) = claim.evidence
    assert quote.quote == "Dogs are welcome in the café garden."
    assert page_at(quote.start, spec.page_start, spec.page_end, spec.page_breaks) == 2


def test_a_question_in_the_source_is_never_the_evidence_on_its_own() -> None:
    faq = EvidenceSource(
        "S1",
        "Larkspur Pool - FAQ",
        "Can I take my dog to the café?\nDogs travel free of charge in the café garden. "
        "Each swimmer may bring one guest.",
    )
    claim = _claim("Yes, you can take your dog to the café for free [S1].", sources=[faq])
    assert claim.status == "supported"
    assert "Dogs travel free of charge" in claim.evidence[0].quote


def test_a_no_after_a_question_in_the_source_is_read_as_a_denial() -> None:
    faq = EvidenceSource(
        "S1",
        "Larkspur Pool - FAQ",
        "Do I need to show a membership card at the door? No. Staff check names at the desk.",
    )
    claim = _claim("You need to show a membership card at the door [S1].", sources=[faq])
    assert claim.status not in MATCHED


# ------------------------------------------------- 8. and 9. citation labels
def test_an_unknown_label_is_reported_as_invalid() -> None:
    report = check_claims("Lockers cost €1 [S7].", SOURCES, None, "claims-v2")
    (claim,) = report.claims
    assert claim.cited == []
    assert claim.invalid_citations == ["S7"]
    assert citation_status(claim) == "invalid"
    assert citation_verdict(report) == "some_wrong"
    assert [reference.valid for reference in report.references] == [False]


@pytest.mark.parametrize("label", ["[Source]", "[Source A]", "[Larkspur Leaflet]"])
def test_a_label_that_names_no_passage_is_reported_as_unresolved(label: str) -> None:
    report = check_claims(f"Lockers cost €1 {label}.", SOURCES, None, "claims-v2")
    (claim,) = report.claims
    assert claim.cited == []
    assert claim.invalid_citations == [label]
    assert [(r.form, r.valid) for r in report.references] == [("unresolved", False)]
    assert citation_status(claim) == "invalid"
    # The statement itself is still found, by the check rather than by the model's label.
    assert claim.status == "added_source"
    assert claim.evidence[0].quote == "Lockers cost €1."


def test_a_numbered_generic_label_outside_the_passages_is_invalid() -> None:
    report = check_claims("Lockers cost €1 [Source 4].", SOURCES, None, "claims-v2")
    (claim,) = report.claims
    assert claim.invalid_citations == ["S4"]
    assert citation_status(claim) == "invalid"


# ------------------------------------------------------------- 10. and 11. conflicts
def _chunk(index: int, document_id: str, text: str, title: str = "Doc") -> RetrievedChunk:
    return RetrievedChunk(
        f"c{index}", document_id, title, 0, text, 0, len(text), None, None, ChunkScores()
    )


def _disagreements(question: str, *chunks: RetrievedChunk) -> list[Disagreement]:
    context = assemble_context(list(chunks), max_chars=6000)
    return find_disagreements(question, context.sources, "disagreement-v2")


def test_a_requirement_and_a_no_longer_needed_wording_conflict() -> None:
    (found,) = _disagreements(
        "Do swimmers need to wear a swimming cap at Larkspur Pool?",
        _chunk(1, "rules", "All swimmers must wear a swimming cap in the main pool."),
        _chunk(2, "faq", "Swimming caps: no cap is needed in any of our pools."),
    )
    assert found.kind == "conflict"
    assert found.labels == ("S1", "S2")


def test_a_question_in_a_faq_is_not_a_statement_and_its_no_answer_is_a_denial() -> None:
    (found,) = _disagreements(
        "Do visitors have to show proof of address to get a swim card?",
        _chunk(1, "card-rules", "To get a swim card, visitors must show proof of address."),
        _chunk(
            2,
            "faq",
            "Do I have to show proof of address to get a swim card? No. We only ask for your name.",
        ),
    )
    assert found.kind == "conflict"


def test_the_same_fee_named_with_another_word_is_compared() -> None:
    (found,) = _disagreements(
        "What is the annual fee for a family swim pass at Larkspur Pool?",
        _chunk(1, "prices", "The yearly charge for a family swim pass is €320."),
        _chunk(2, "leaflet", "Family swim pass: the annual fee is €290."),
    )
    assert found.kind == "conflict"


def test_values_for_different_dates_or_groups_are_not_called_a_conflict() -> None:
    (dated,) = _disagreements(
        "How much does a locker cost at Larkspur Pool?",
        _chunk(1, "prices-2025", "2025 price list: a locker costs €1."),
        _chunk(2, "prices-2026", "From 1 June 2026, a locker costs €2."),
    )
    assert dated.kind == "different_dates"
    assert not _disagreements(
        "How much is a swim at Larkspur Pool?",
        _chunk(1, "members", "Members pay €3 for a swim."),
        _chunk(2, "visitors", "Visitors pay €6 for a swim."),
    )


def test_a_requirement_without_a_denial_is_not_a_conflict() -> None:
    assert not _disagreements(
        "Do swimmers need to wear a swimming cap at Larkspur Pool?",
        _chunk(1, "rules", "All swimmers must wear a swimming cap in the main pool."),
        _chunk(2, "faq", "Swimming caps are sold at reception for €3."),
    )
