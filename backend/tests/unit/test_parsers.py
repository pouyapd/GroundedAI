import pytest

from groundedai.core.errors import DocumentParseError, EmptyDocument
from groundedai.evaluation.pdf_fixture import make_pdf
from groundedai.ingestion.parsers import parse_document, parse_pdf, parse_text


def test_pdf_pages_and_title_are_extracted(sample_pdf: bytes) -> None:
    parsed = parse_pdf(sample_pdf, max_pages=10, max_chars=10_000)
    assert parsed.page_count == 2
    assert [page.number for page in parsed.pages] == [1, 2]
    assert "150 to 850 V" in parsed.pages[0].text
    assert "E07 means fan failure" in parsed.pages[1].text
    assert parsed.title == "Inverter Manual"


def test_pdf_with_euro_and_degree_signs_round_trips() -> None:
    parsed = parse_pdf(make_pdf(["Limit €140 at 45 °C"]), max_pages=5, max_chars=1000)
    assert parsed.pages[0].text == "Limit €140 at 45 °C"


def test_malformed_pdf_raises_parse_error() -> None:
    with pytest.raises(DocumentParseError):
        parse_pdf(b"%PDF-1.4\nthis is not really a pdf", max_pages=10, max_chars=1000)


def test_truncated_pdf_raises_parse_error(sample_pdf: bytes) -> None:
    with pytest.raises(DocumentParseError):
        parse_pdf(sample_pdf[: len(sample_pdf) // 3], max_pages=10, max_chars=10_000)


def test_pdf_without_text_is_empty() -> None:
    with pytest.raises(EmptyDocument, match="OCR"):
        parse_pdf(make_pdf(["", ""]), max_pages=10, max_chars=1000)


def test_pdf_page_limit_is_enforced() -> None:
    pdf = make_pdf([f"page {i}" for i in range(5)])
    with pytest.raises(DocumentParseError, match="limit is 3"):
        parse_pdf(pdf, max_pages=3, max_chars=10_000)


def test_pdf_character_limit_is_enforced() -> None:
    pdf = make_pdf(["word " * 200])
    with pytest.raises(DocumentParseError, match="characters"):
        parse_pdf(pdf, max_pages=5, max_chars=100)


def test_text_with_bom_and_crlf() -> None:
    parsed = parse_text("﻿Line one\r\nLine two".encode(), markdown=False, max_chars=100)
    assert parsed.pages[0].text == "Line one\nLine two"
    assert parsed.pages[0].number is None
    assert parsed.title == "Line one"


def test_markdown_title_from_first_heading() -> None:
    parsed = parse_text(b"# Remote Work\n\nBody text.", markdown=True, max_chars=100)
    assert parsed.title == "Remote Work"


def test_whitespace_only_text_is_empty() -> None:
    with pytest.raises(EmptyDocument):
        parse_text(b"   \n\t  \n", markdown=False, max_chars=100)


def test_unknown_extension_has_no_parser() -> None:
    with pytest.raises(DocumentParseError):
        parse_document(b"x", ".docx", max_pages=1, max_chars=10)


def test_text_title_only_from_short_heading_like_first_line() -> None:
    assert (
        parse_text(b"Leave Policy\n\nBody.", markdown=False, max_chars=100).title == "Leave Policy"
    )
    assert parse_text(b"A full sentence.\nMore.", markdown=False, max_chars=100).title is None
    assert parse_text(b"Single line only", markdown=False, max_chars=100).title is None
    long_line = ("word " * 40).encode() + b"\nbody"
    assert parse_text(long_line, markdown=False, max_chars=1000).title is None
