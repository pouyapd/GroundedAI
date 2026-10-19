"""PostgreSQL + pgvector integration (runs when GROUNDEDAI_TEST_POSTGRES_URL is set).

CI starts a pgvector/pgvector container for this job. Locally you can use
``python scripts/dev_postgres.py`` (bundled Postgres via the pgserver wheel)
or any Postgres 16 with the ``vector`` extension.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text

from groundedai.core.config import Settings
from groundedai.evaluation.sample_documents import FERRY_GUIDE
from groundedai.persistence.db import session_scope
from groundedai.retrieval.types import RetrievalRequest
from groundedai.schemas.query import QueryRequest
from groundedai.services.container import Services, build_services
from tests.conftest import INJECTION_TEXT, SECURITY_TEXT, TRAVEL_TEXT

pytestmark = pytest.mark.postgres
POSTGRES_URL = os.environ.get("GROUNDEDAI_TEST_POSTGRES_URL")


@pytest.fixture
def pg_services(make_settings) -> Iterator[Services]:  # type: ignore[no-untyped-def]
    if not POSTGRES_URL:
        pytest.skip("GROUNDEDAI_TEST_POSTGRES_URL is not set")
    engine = create_engine(POSTGRES_URL)
    with engine.begin() as connection:  # start from an empty schema
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    settings: Settings = make_settings(database_url=POSTGRES_URL)
    services = build_services(settings)
    try:
        yield services
    finally:
        services.close()


def test_readiness_checks_vector_dimension(pg_services: Services) -> None:
    readiness = pg_services.status.readiness()
    assert readiness.status == "ready", readiness
    assert readiness.checks["vector_dimension"].detail.startswith("column=vector(384)")


@pytest.mark.parametrize("mode", ["lexical", "dense", "hybrid"])
def test_search_modes(pg_services: Services, mode: str) -> None:
    travel = pg_services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
    pg_services.ingestion.ingest_bytes("security.txt", None, SECURITY_TEXT.encode())
    with session_scope(pg_services.session_factory) as session:
        result = pg_services.retrieval.retrieve(
            session,
            RetrievalRequest(query="hotel limit per night", mode=mode, top_k=2),  # type: ignore[arg-type]
        )
    assert result.chunks[0].document_id == travel.id


def test_hostile_lexical_query(pg_services: Services) -> None:
    pg_services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
    with session_scope(pg_services.session_factory) as session:
        for query in ["'; DROP TABLE chunks; --", "a & b | !c", "hotel:* <-> limit"]:
            pg_services.retrieval.retrieve(session, RetrievalRequest(query=query, mode="lexical"))


def test_answer_and_delete(pg_services: Services) -> None:
    supplier = pg_services.ingestion.ingest_bytes("s.txt", None, INJECTION_TEXT.encode())
    response = pg_services.qa.answer(
        QueryRequest(question="What are the payment terms for suppliers?")
    )
    assert response.status == "answered"
    assert "45 days" in (response.answer or "")
    pg_services.ingestion.delete(supplier.id)
    with session_scope(pg_services.session_factory) as session:
        remaining = session.execute(text("SELECT count(*) FROM chunks")).scalar()
    assert remaining == 0


def test_pdf_evidence_names_its_page(pg_services: Services) -> None:
    ferry = pg_services.ingestion.ingest_bytes(FERRY_GUIDE.filename, None, FERRY_GUIDE.pdf())
    response = pg_services.qa.answer(
        QueryRequest(question="How much does a single adult ticket cost?", document_ids=[ferry.id])
    )
    assert response.status == "answered"
    assert response.claims[0].evidence[0].page == 2  # page breaks round-trip through JSON
