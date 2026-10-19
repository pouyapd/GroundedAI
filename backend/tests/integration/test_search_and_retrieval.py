from collections.abc import Sequence

import pytest
from sqlalchemy import update

from groundedai.persistence.db import session_scope
from groundedai.persistence.models import DocumentRow
from groundedai.persistence.search import query_terms
from groundedai.retrieval.langchain_retriever import GroundedRetriever
from groundedai.retrieval.types import RetrievalRequest
from groundedai.services.container import Services
from tests.conftest import SECURITY_TEXT, TRAVEL_TEXT


@pytest.fixture
def indexed(services: Services) -> dict[str, str]:
    travel = services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
    security = services.ingestion.ingest_bytes("security.txt", None, SECURITY_TEXT.encode())
    return {"travel": travel.id, "security": security.id}


def _retrieve(services: Services, query: str, **kwargs: object) -> list[str]:
    with session_scope(services.session_factory) as session:
        result = services.retrieval.retrieve(session, RetrievalRequest(query=query, **kwargs))  # type: ignore[arg-type]
    return [chunk.document_id for chunk in result.chunks]


def test_query_terms_strip_operators_and_stopwords() -> None:
    assert query_terms('What is "the" hotel AND NOT limit* (night)?') == [
        "hotel",
        "limit",
        "night",
    ]
    assert query_terms("the of and") == []
    assert query_terms("???") == []


@pytest.mark.parametrize("mode", ["lexical", "dense", "hybrid"])
def test_each_mode_finds_the_relevant_document(
    services: Services, indexed: dict[str, str], mode: str
) -> None:
    ranked = _retrieve(services, "hotel limit per night", mode=mode, top_k=2)
    assert ranked[0] == indexed["travel"]


@pytest.mark.parametrize(
    "query",
    ['"unbalanced', "NEAR(a b) OR *", "col:value", "AND OR NOT", "'; DROP TABLE chunks; --"],
)
def test_hostile_queries_do_not_break_search(
    services: Services, indexed: dict[str, str], query: str
) -> None:
    _retrieve(services, query, mode="lexical")
    _retrieve(services, query, mode="hybrid")
    assert _retrieve(services, "hotel", mode="lexical")  # table still intact


def test_document_filter(services: Services, indexed: dict[str, str]) -> None:
    ranked = _retrieve(
        services, "hotel limit passwords", mode="hybrid", document_ids=[indexed["security"]]
    )
    assert set(ranked) == {indexed["security"]}


def test_non_ready_documents_are_not_searchable(
    services: Services, indexed: dict[str, str]
) -> None:
    with session_scope(services.session_factory) as session:
        session.execute(
            update(DocumentRow).where(DocumentRow.id == indexed["travel"]).values(status="failed")
        )
    assert indexed["travel"] not in _retrieve(services, "hotel limit", mode="hybrid")


def test_documents_from_another_embedding_model_are_excluded(
    services: Services, indexed: dict[str, str]
) -> None:
    with session_scope(services.session_factory) as session:
        session.execute(
            update(DocumentRow)
            .where(DocumentRow.id == indexed["travel"])
            .values(embedding_model="other:model")
        )
        result = services.retrieval.retrieve(
            session, RetrievalRequest(query="hotel limit", mode="dense")
        )
    assert indexed["travel"] not in {c.document_id for c in result.chunks}
    assert any("different embedding model" in w for w in result.diagnostics.warnings)


def test_diagnostics_explain_the_ranking(services: Services, indexed: dict[str, str]) -> None:
    with session_scope(services.session_factory) as session:
        result = services.retrieval.retrieve(
            session, RetrievalRequest(query="hotel limit per night", mode="hybrid", top_k=3)
        )
    top = result.chunks[0]
    assert top.rank == 1
    assert top.scores.lexical_rank == 1
    assert top.scores.fused is not None
    assert "hotel" in top.matched_terms
    timings = result.diagnostics.timings_ms
    assert {"lexical_ms", "embed_query_ms", "dense_ms", "fusion_ms", "total_ms"} <= set(timings)
    assert result.diagnostics.fusion == "rrf"


def test_identical_chunks_are_deduplicated(services: Services) -> None:
    services.ingestion.ingest_bytes("one.txt", None, b"Unique sentence about kayaks.")
    services.ingestion.ingest_bytes("two.txt", None, b"Unique sentence about kayaks.\n")
    with session_scope(services.session_factory) as session:
        result = services.retrieval.retrieve(
            session, RetrievalRequest(query="kayaks", mode="hybrid", top_k=5)
        )
    assert len(result.chunks) == 1
    assert result.diagnostics.duplicates_removed == 1


class ReverseReranker:
    name = "test:reverse"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(i) for i in range(len(passages))]


def test_reranker_reorders_candidates(settings) -> None:  # type: ignore[no-untyped-def]
    from groundedai.services.container import build_services

    services = build_services(settings, reranker=ReverseReranker())
    try:
        services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
        services.ingestion.ingest_bytes("security.txt", None, SECURITY_TEXT.encode())
        with session_scope(services.session_factory) as session:
            plain = services.retrieval.retrieve(
                session, RetrievalRequest(query="hotel passwords", mode="hybrid", top_k=2)
            )
            reranked = services.retrieval.retrieve(
                session,
                RetrievalRequest(
                    query="hotel passwords", mode="hybrid", top_k=2, use_reranker=True
                ),
            )
        assert [c.chunk_id for c in reranked.chunks] == [c.chunk_id for c in plain.chunks][::-1]
        assert reranked.diagnostics.reranker == "test:reverse"
        assert reranked.chunks[0].scores.rerank is not None
    finally:
        services.close()


def test_rerank_request_without_reranker_warns(services: Services, indexed: dict[str, str]) -> None:
    with session_scope(services.session_factory) as session:
        result = services.retrieval.retrieve(
            session, RetrievalRequest(query="hotel", mode="lexical", use_reranker=True)
        )
    assert result.diagnostics.reranker is None
    assert result.diagnostics.warnings


def test_langchain_retriever_adapter(services: Services, indexed: dict[str, str]) -> None:
    retriever = GroundedRetriever(
        engine=services.retrieval,
        session_factory=services.session_factory,
        request_template=RetrievalRequest(query="", mode="hybrid", top_k=2),
    )
    documents = retriever.invoke("hotel limit per night")
    assert documents[0].metadata["document_id"] == indexed["travel"]
    assert documents[0].metadata["scores"]["fused"] is not None
    assert len(documents) <= 2
