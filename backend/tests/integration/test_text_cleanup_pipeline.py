"""Text clean-up v2 and prefix matching through ingestion and answering (Experiment A settings).

These pin what must not change when the two settings are used: the page of every chunk and
quote, each page's own content, citations and the claim check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from groundedai.core.config import Settings
from groundedai.evaluation.pdf_fixture import make_pdf
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import ChunkRow, DocumentStatus
from groundedai.schemas.query import AnswerResponse, QueryRequest
from groundedai.services.container import build_services
from tests.conftest import SettingsFactory

HEADER = "ACME PUMP SERIES 7 SERVICE MANUAL"
FACTS = {
    1: "Introduction. This manual covers the Series 7 pump.",
    2: "The inlet filter must be replaced every 400 operating hours.",
    3: "The maximum discharge pressure is 16 bar.",
    4: "The motor bearings are lubricated every 2000 operating hours.",
    5: "The pump must not run dry for more than 30 seconds.",
    6: "Spare impellers are ordered with part number IMP-7730.",
}


def _manual() -> bytes:
    pages = [f"{HEADER}\n{text}\nPage {number}" for number, text in FACTS.items()]
    return make_pdf(pages, title="Series 7 Service Manual")


def _settings(
    make_settings: SettingsFactory, tmp_path: Path, name: str, **values: object
) -> Settings:
    return make_settings(data_dir=tmp_path / name, **values)


def _chunks(settings: Settings) -> list[ChunkRow]:
    services = build_services(settings)
    try:
        row = services.ingestion.ingest_bytes("manual.pdf", "application/pdf", _manual())
        assert row.status == DocumentStatus.READY
        with session_scope(services.session_factory) as session:
            chunks = list(
                session.scalars(select(ChunkRow).where(ChunkRow.document_id == row.id)).all()
            )
            session.expunge_all()
            return chunks
    finally:
        services.close()


def _ask(settings: Settings, *questions: str) -> list[AnswerResponse]:
    services = build_services(settings)
    try:
        services.ingestion.ingest_bytes("manual.pdf", "application/pdf", _manual())
        return [services.qa.answer(QueryRequest(question=q)) for q in questions]
    finally:
        services.close()


def _pages_of(chunks: list[ChunkRow], sentence: str) -> tuple[int | None, int | None]:
    holder = next(c for c in chunks if sentence in " ".join(c.text.split()))
    return holder.page_start, holder.page_end


def test_v2_removes_the_running_header_and_keeps_every_fact_on_its_page(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    sizes = {"chunk_size": 200, "chunk_overlap": 0}
    v1 = _chunks(_settings(make_settings, tmp_path, "v1", text_cleanup="v1", **sizes))
    v2 = _chunks(_settings(make_settings, tmp_path, "v2", text_cleanup="v2", **sizes))
    assert any(HEADER in c.text for c in v1)
    assert not any(HEADER in c.text for c in v2)
    assert not any("Page " in c.text for c in v2)
    for number, sentence in FACTS.items():
        start, end = _pages_of(v2, sentence)
        assert start is not None
        assert end is not None
        assert start <= number <= end
        old_start, old_end = _pages_of(v1, sentence)
        assert old_start is not None
        assert old_end is not None
        assert old_start <= number <= old_end


@pytest.mark.parametrize("matching", ["exact", "prefix"])
def test_answers_cite_and_quote_the_right_page_with_the_new_settings(
    make_settings: SettingsFactory, tmp_path: Path, matching: str
) -> None:
    settings = _settings(
        make_settings, tmp_path, matching, text_cleanup="v2", extractive_matching=matching
    )
    (response,) = _ask(settings, "What is the maximum discharge pressure of the pump?")
    assert response.status == "answered"
    assert response.answer
    assert "16 bar" in response.answer
    assert response.citation_check.all_valid
    assert response.citation_check.valid
    assert response.claims
    assert all(claim.status == "supported" for claim in response.claims)
    quotes = [e for claim in response.claims for e in claim.evidence]
    pressure = [e for e in quotes if "16 bar" in e.quote]
    assert pressure
    assert all(e.page == 3 for e in pressure)
    assert all(HEADER not in e.quote for e in quotes)


QUESTIONS = [
    "How often are the bearings lubricated?",
    "When is lubrication of the motor bearings due?",
    "What is the maximum discharge pressure?",
    "How long may the pump run dry?",
    "Which part number do spare impellers have?",
    "How often must the inlet filter be replaced?",
]


def test_prefix_matching_gives_the_same_answers_where_exact_matching_already_answers(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    # Prefix matching may only add answers (other word forms); it must not change an answer
    # that exact matching already gives. The case where it adds one is in
    # tests/unit/test_safety_and_extractive.py (a question from the live demo test).
    exact = _ask(
        _settings(make_settings, tmp_path, "e", text_cleanup="v2", extractive_matching="exact"),
        *QUESTIONS,
    )
    prefix = _ask(
        _settings(make_settings, tmp_path, "p", text_cleanup="v2", extractive_matching="prefix"),
        *QUESTIONS,
    )
    answered = [(a, b) for a, b in zip(exact, prefix, strict=True) if a.status == "answered"]
    assert len(answered) >= 4
    for with_exact, with_prefix in answered:
        assert with_prefix.status == "answered"
        assert with_prefix.answer == with_exact.answer
        assert [c.status for c in with_prefix.claims] == [c.status for c in with_exact.claims]


def test_a_question_the_manual_does_not_answer_is_still_refused_with_prefix_matching(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    settings = _settings(
        make_settings, tmp_path, "refuse", text_cleanup="v2", extractive_matching="prefix"
    )
    (response,) = _ask(settings, "Who manufactures the control cable?")
    assert response.status == "insufficient_evidence"


def test_a_document_indexed_with_v1_keeps_working_and_reingest_applies_v2(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    # Documents indexed before the default changed keep their v1 text; questions still work,
    # and re-ingesting a document applies the current setting.
    old = build_services(make_settings(data_dir=tmp_path / "shared", text_cleanup="v1"))
    try:
        row = old.ingestion.ingest_bytes("manual.pdf", "application/pdf", _manual())
        assert row.chunking is not None
        assert "text_cleanup" not in row.chunking
    finally:
        old.close()
    new = build_services(make_settings(data_dir=tmp_path / "shared", text_cleanup="v2"))
    try:
        response = new.qa.answer(QueryRequest(question="What is the maximum discharge pressure?"))
        assert response.status == "answered"
        assert response.answer
        assert "16 bar" in response.answer
        assert all(claim.status == "supported" for claim in response.claims)
        with session_scope(new.session_factory) as session:
            texts = [c.text for c in session.scalars(select(ChunkRow)).all()]
        assert any(HEADER in text for text in texts)  # still the v1 text
        new.ingestion.prepare_reingest(row.id)
        again = new.ingestion.process(row.id)
        assert again.status == DocumentStatus.READY
        assert again.chunking is not None
        assert again.chunking["text_cleanup"] == "v2"
        with session_scope(new.session_factory) as session:
            texts = [c.text for c in session.scalars(select(ChunkRow)).all()]
        assert not any(HEADER in text for text in texts)
        pressure = new.qa.answer(QueryRequest(question="What is the maximum discharge pressure?"))
        quotes = [e for claim in pressure.claims for e in claim.evidence if "16 bar" in e.quote]
        assert quotes
        assert all(e.page == 3 for e in quotes)
    finally:
        new.close()
