import pytest
from sqlalchemy import func, select, text

from groundedai.core.errors import Conflict, DuplicateDocument, NotFound, ProviderUnavailable
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import ChunkRow, DocumentRow, DocumentStatus
from groundedai.retrieval.embeddings import EmbeddingProvider, HashingEmbeddings
from groundedai.services.container import Services, build_services
from tests.conftest import SECURITY_TEXT, TRAVEL_TEXT


def _chunk_count(services: Services, document_id: str | None = None) -> int:
    with session_scope(services.session_factory) as session:
        query = select(func.count()).select_from(ChunkRow)
        if document_id:
            query = query.where(ChunkRow.document_id == document_id)
        return int(session.scalar(query) or 0)


def _fts_count(services: Services) -> int:
    with session_scope(services.session_factory) as session:
        return int(session.execute(text("SELECT count(*) FROM chunks_fts")).scalar() or 0)


def test_text_document_becomes_ready(services: Services) -> None:
    row = services.ingestion.ingest_bytes("travel.txt", "text/plain", TRAVEL_TEXT.encode())
    assert row.status == DocumentStatus.READY
    assert row.chunk_count >= 1
    assert row.embedding_model == "hashing:v1-384"
    assert row.title == "Travel Policy"
    assert row.chunking
    assert row.chunking["chunk_size"] == 800
    assert _chunk_count(services, row.id) == row.chunk_count
    assert services.file_store.exists(row.id, ".txt")


def test_pdf_pages_are_stored_on_chunks(services: Services, sample_pdf: bytes) -> None:
    row = services.ingestion.ingest_bytes("manual.pdf", "application/pdf", sample_pdf)
    assert row.status == DocumentStatus.READY
    assert row.page_count == 2
    assert row.title == "Inverter Manual"  # from PDF metadata
    with session_scope(services.session_factory) as session:
        chunks = session.scalars(select(ChunkRow).where(ChunkRow.document_id == row.id)).all()
        assert {c.page_start for c in chunks} <= {1, 2}
        assert max(c.page_end or 0 for c in chunks) == 2


def test_explicit_title_is_kept(services: Services, sample_pdf: bytes) -> None:
    row = services.ingestion.ingest_bytes("m.pdf", None, sample_pdf, title="My manual")
    assert row.title == "My manual"


def test_duplicate_upload_is_rejected_with_existing_id(services: Services) -> None:
    first = services.ingestion.ingest_bytes("a.txt", "text/plain", TRAVEL_TEXT.encode())
    with pytest.raises(DuplicateDocument) as error:
        services.ingestion.register_upload("renamed.txt", "text/plain", TRAVEL_TEXT.encode())
    assert error.value.details["document_id"] == first.id


def test_unparseable_pdf_fails_without_chunks(services: Services) -> None:
    row = services.ingestion.ingest_bytes("bad.pdf", "application/pdf", b"%PDF-1.4 garbage")
    assert row.status == DocumentStatus.FAILED
    assert row.error_code == "document_parse_failed"
    assert row.chunk_count == 0
    assert _chunk_count(services, row.id) == 0


class FlakyEmbeddings(HashingEmbeddings):
    def __init__(self) -> None:
        super().__init__(384)
        self.fail = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise ProviderUnavailable("embedding service down")
        return super().embed_documents(texts)


def test_failed_ingestion_can_be_retried(settings) -> None:  # type: ignore[no-untyped-def]
    embeddings = FlakyEmbeddings()
    provider = EmbeddingProvider("hashing", "v1-384", 384, embeddings)
    services = build_services(settings, embedding_provider=provider)
    try:
        row = services.ingestion.ingest_bytes("a.txt", "text/plain", TRAVEL_TEXT.encode())
        assert row.status == DocumentStatus.FAILED
        assert row.error_code == "model_unavailable"
        assert row.attempts == 1
        embeddings.fail = False
        services.ingestion.prepare_reingest(row.id)
        row = services.ingestion.process(row.id)
        assert row.status == DocumentStatus.READY
        assert row.attempts == 2
        assert row.error_code is None
    finally:
        services.close()


def test_reingest_replaces_chunks_without_duplicates(services: Services) -> None:
    row = services.ingestion.ingest_bytes("a.txt", "text/plain", TRAVEL_TEXT.encode())
    before = _chunk_count(services)
    services.ingestion.prepare_reingest(row.id)
    services.ingestion.process(row.id)
    assert _chunk_count(services) == before
    assert _fts_count(services) == before


def test_reingest_conflicts_while_pending(services: Services) -> None:
    row = services.ingestion.register_upload("a.txt", "text/plain", TRAVEL_TEXT.encode())
    with pytest.raises(Conflict):
        services.ingestion.prepare_reingest(row.id)


def test_delete_leaves_no_orphans(services: Services) -> None:
    keep = services.ingestion.ingest_bytes("keep.txt", "text/plain", SECURITY_TEXT.encode())
    gone = services.ingestion.ingest_bytes("gone.txt", "text/plain", TRAVEL_TEXT.encode())
    services.ingestion.delete(gone.id)
    assert _chunk_count(services, gone.id) == 0
    assert _fts_count(services) == _chunk_count(services) == keep.chunk_count
    assert not services.file_store.exists(gone.id, ".txt")
    with pytest.raises(NotFound):
        services.ingestion.get(gone.id)
    with pytest.raises(NotFound):
        services.ingestion.delete(gone.id)


def test_interrupted_documents_are_recovered(services: Services) -> None:
    row = services.ingestion.register_upload("a.txt", "text/plain", TRAVEL_TEXT.encode())
    with session_scope(services.session_factory) as session:
        stored = session.get(DocumentRow, row.id)
        assert stored is not None
        stored.status = DocumentStatus.PROCESSING
    assert services.ingestion.recover_interrupted() == 1
    recovered = services.ingestion.get(row.id)
    assert recovered.status == DocumentStatus.FAILED
    assert recovered.error_code == "interrupted"
    services.ingestion.prepare_reingest(row.id)
    assert services.ingestion.process(row.id).status == DocumentStatus.READY


def test_missing_source_file_is_reported(services: Services) -> None:
    row = services.ingestion.register_upload("a.txt", "text/plain", TRAVEL_TEXT.encode())
    services.file_store.delete(row.id, ".txt")
    processed = services.ingestion.process(row.id)
    assert processed.status == DocumentStatus.FAILED
    assert processed.error_code == "source_missing"
