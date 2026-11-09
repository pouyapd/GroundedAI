from groundedai.ingestion.chunking import ChunkingConfig
from groundedai.ingestion.pdf_cleanup import (
    clean_pdf_pages,
    expand_ligatures,
    join_hyphenated_lines,
    repeated_edge_lines,
)

HEADER = "NIST SP 800-63B DIGITAL IDENTITY GUIDELINES:"


def _page(number: int, body: str) -> str:
    notice = "This publication is available free of charge"
    return f"{HEADER}\n{number}\n{notice}\n{body}\nPage {number}"


def test_ligatures_are_expanded() -> None:
    assert expand_ligatures("Artiﬁcial eﬀort reﬂects") == "Artificial effort reflects"


def test_only_lowercase_line_break_hyphens_are_joined() -> None:
    text = "trustwor-\nthy systems; SP 800-\n63-2; Self-\nAssessment; well-known"
    assert join_hyphenated_lines(text) == (
        "trustworthy systems; SP 800-\n63-2; Self-\nAssessment; well-known"
    )


def test_running_headers_footers_and_page_numbers_are_removed() -> None:
    topics = ["passwords", "tokens", "sessions", "biometrics", "recovery", "federation"]
    pages = [_page(n, f"Body text about {topic}.") for n, topic in enumerate(topics, start=1)]
    cleaned = clean_pdf_pages(pages)
    assert cleaned[2] == "Body text about sessions."
    assert all(HEADER not in page and "Page" not in page for page in cleaned)


def test_a_line_that_recurs_in_the_body_is_kept() -> None:
    # Only the first and last four lines of a page are candidates.
    body = "\n".join(f"line {i}" for i in range(12))
    pages = [f"Top\n{body}\nThe verifier SHALL limit attempts.\n{body}\nBottom" for _ in range(6)]
    assert "The verifier SHALL limit attempts." in clean_pdf_pages(pages)[0]


def test_short_documents_are_left_alone() -> None:
    pages = ["Header\nOne", "Header\nTwo", "Header\nThree"]
    assert repeated_edge_lines(pages) == set()
    assert clean_pdf_pages(pages) == pages


def test_v1_chunking_record_is_unchanged() -> None:
    # Reports written before the v2 clean-up existed must keep the same chunking record.
    assert "text_cleanup" not in ChunkingConfig(800, 120).as_dict()
    assert ChunkingConfig(800, 120, "v2").as_dict()["text_cleanup"] == "v2"
