"""Two editions of one document through ingestion, retrieval and answering (disagreement-v3)."""

from __future__ import annotations

from pathlib import Path

from groundedai.evaluation.pdf_fixture import make_pdf
from groundedai.persistence.db import session_scope
from groundedai.schemas.query import QueryRequest
from groundedai.services.container import Services, build_services
from tests.conftest import SettingsFactory

OLD = [
    "Corvane Systems Access Standard. Revision A, approved November 2018.\nScope of the standard.",
    "Sessions\nAt AL2, users SHALL be reauthenticated after 30 minutes of inactivity.",
    "Administrators\nAdministrators SHOULD NOT share accounts.",
]
NEW = [
    "Corvane Systems Access Standard. Revision B, approved March 2024.\nScope of the standard.",
    "Sessions\nAt AL2, reauthentication SHALL occur after 1 hour of inactivity.",
    "Administrators\nAdministrators SHALL NOT share accounts.",
]


def _services(make_settings: SettingsFactory, tmp_path: Path) -> tuple[Services, str, str]:
    services = build_services(make_settings(data_dir=tmp_path / "data"))
    old = services.ingestion.ingest_bytes(
        "rev-a.pdf", "application/pdf", make_pdf(OLD), title="Corvane Access Standard (Revision A)"
    )
    new = services.ingestion.ingest_bytes(
        "rev-b.pdf", "application/pdf", make_pdf(NEW), title="Corvane Access Standard (Revision B)"
    )
    return services, old.id, new.id


def test_both_editions_are_quoted_with_their_editions_and_pages(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    services, old, new = _services(make_settings, tmp_path)
    try:
        response = services.qa.answer(
            QueryRequest(question="After how much inactivity must AL2 users be reauthenticated?")
        )
        kinds = [conflict.kind for conflict in response.conflicts]
        assert kinds == ["different_versions"]
        sides = response.conflicts[0].statements
        assert {side.document_id for side in sides} == {old, new}
        assert {side.edition for side in sides} == {
            "Revision A, November 2018",
            "Revision B, March 2024",
        }
        assert all(side.page == 2 for side in sides)
        assert {tuple(side.values) for side in sides} == {("30 minutes",), ("1 hour",)}
        assert any("two versions of the same document" in w for w in response.warnings)
        assert "disagreement" in response.checks_run
    finally:
        services.close()


def test_a_missing_edition_is_added_to_the_retrieved_passages(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    services, old, new = _services(make_settings, tmp_path)
    try:
        request = services.qa.build_retrieval_request(
            "May administrators share accounts?", document_ids=[old, new]
        )
        with session_scope(services.session_factory) as session:
            retrieved = services.qa.retrieval_engine.retrieve(session, request).chunks
            only_old = [chunk for chunk in retrieved if chunk.document_id == old]
            chunks, editions = services.qa.ensure_editions_retrieved(session, request, only_old)
        assert {chunk.document_id for chunk in chunks} == {old, new}
        assert len(chunks) <= request.top_k
        assert set(editions) == {old, new}
    finally:
        services.close()


def test_the_checks_that_ran_are_listed_and_relevance_is_reported(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    services, _old, _new = _services(make_settings, tmp_path)
    try:
        answered = services.qa.answer(QueryRequest(question="May administrators share accounts?"))
        assert answered.status == "answered"
        assert {"citation_labels", "lexical_claims", "added_details", "relevance"} <= set(
            answered.checks_run
        )
        assert answered.relevance is not None
        assert answered.relevance.unmatched == []
        refused = services.qa.answer(QueryRequest(question="How long are audit logs kept?"))
        assert refused.status == "insufficient_evidence"
        assert refused.answer is not None
        assert '"audit"' in refused.answer
    finally:
        services.close()
