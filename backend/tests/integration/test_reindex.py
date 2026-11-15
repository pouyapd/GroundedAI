"""Re-indexing documents that were indexed with older settings (issue G).

Documents are indexed with text clean-up v1 and then opened with the current default (v2), as
after an upgrade. What must hold: outdated documents are found; re-indexing gives exactly the
chunks a fresh ingestion gives; ids are kept; there are no duplicate or orphan chunks and no
stale full-text rows; a failure at any step keeps the previous index; a document changed or
deleted meanwhile is not overwritten; the bulk job copes with mixed states, partial failure,
an interrupted run and being started twice.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from groundedai.core.config import Settings
from groundedai.core.errors import ProviderUnavailable
from groundedai.evaluation.pdf_fixture import make_pdf
from groundedai.main import create_app
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import ChunkRow, DocumentRow, DocumentStatus
from groundedai.persistence.repositories import ChunkRepository
from groundedai.retrieval.embeddings import (
    EmbeddingProvider,
    HashingEmbeddings,
    build_embedding_provider,
    embedding_model_id,
)
from groundedai.services.container import Services, build_services
from groundedai.services.ingestion_service import IndexDifference
from tests.conftest import TRAVEL_TEXT, SettingsFactory

HEADER = "ACME PUMP SERIES 7 SERVICE MANUAL"
FACTS = [
    "Introduction. This manual covers the Series 7 pump.",
    "The inlet filter must be replaced every 400 operating hours.",
    "The maximum discharge pressure is 16 bar.",
    "The motor bearings are lubricated every 2000 operating hours.",
    "The pump must not run dry for more than 30 seconds.",
    "Spare impellers are ordered with part number IMP-7730.",
]
CLEANUP_V1_TO_V2 = IndexDifference("text clean-up", "v1", "v2")


def manual() -> bytes:
    """Six pages with a running header and page numbers, which clean-up v2 removes."""
    pages = [f"{HEADER}\n{fact}\nPage {number}" for number, fact in enumerate(FACTS, 1)]
    return make_pdf(pages, title="Series 7 Service Manual")


BOILER = [
    "The boiler must be serviced once a year.",
    "The water pressure should stay between one and two bar.",
    "Radiators are bled when they are cold.",
    "The flue must never be covered.",
    "The frost setting protects the pipes in winter.",
]


def other_manual() -> bytes:
    pages = [f"BOILER B2 GUIDE\n{line}\nPage {n}" for n, line in enumerate(BOILER, 1)]
    return make_pdf(pages, title="Boiler Guide")


def sparse_manual() -> bytes:
    """Every line is at a page edge and differs between pages only in its numbers, so clean-up
    v2 removes all of them as running headers: the v2 text is empty."""
    pages = [
        f"BOILER B2 GUIDE\nThe boiler setting {n} is {n * 5} degrees.\nPage {n}"
        for n in range(1, 6)
    ]
    return make_pdf(pages, title="Boiler Settings")


class SwitchableEmbeddings(HashingEmbeddings):
    """The hashing embedder (same model id), made to fail on demand."""

    def __init__(self) -> None:
        super().__init__(384)
        self.fail_when: str | None = None  # fail for any batch with a text containing this
        self.stop_when: str | None = None  # simulate the process stopping (not an Exception)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.stop_when and any(self.stop_when in t for t in texts):
            raise ProcessStopped()
        if self.fail_when and any(self.fail_when in t for t in texts):
            raise ProviderUnavailable("embedding service down")
        return super().embed_documents(texts)


class ProcessStopped(BaseException):
    """Stands in for the process being stopped in the middle of a job."""


def provider_with(embeddings: HashingEmbeddings) -> EmbeddingProvider:
    return EmbeddingProvider(provider="hashing", model="v1-384", dim=384, embeddings=embeddings)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def open_services(
    make_settings: SettingsFactory,
    data_dir: Path,
    embeddings: HashingEmbeddings | None = None,
    **values: Any,
) -> Services:
    settings: Settings = make_settings(data_dir=data_dir, **values)
    return build_services(
        settings, embedding_provider=provider_with(embeddings) if embeddings else None
    )


@pytest.fixture
def old_index(make_settings: SettingsFactory, data_dir: Path) -> dict[str, str]:
    """Two documents indexed with clean-up v1 (the behaviour before 2026-11)."""
    services = open_services(make_settings, data_dir, text_cleanup="v1")
    try:
        pdf = services.ingestion.ingest_bytes("manual.pdf", "application/pdf", manual())
        txt = services.ingestion.ingest_bytes("travel.txt", "text/plain", TRAVEL_TEXT.encode())
        assert pdf.status == txt.status == DocumentStatus.READY
        return {"pdf": pdf.id, "txt": txt.id}
    finally:
        services.close()


@pytest.fixture
def embeddings() -> SwitchableEmbeddings:
    return SwitchableEmbeddings()


@pytest.fixture
def current(
    make_settings: SettingsFactory,
    data_dir: Path,
    old_index: dict[str, str],
    embeddings: SwitchableEmbeddings,
) -> Iterator[Services]:
    """The same data opened with the current settings (clean-up v2)."""
    services = open_services(make_settings, data_dir, embeddings, text_cleanup="v2")
    try:
        yield services
    finally:
        services.close()


def chunk_snapshot(services: Services, document_id: str) -> list[tuple[Any, ...]]:
    with session_scope(services.session_factory) as session:
        rows = session.scalars(
            select(ChunkRow)
            .where(ChunkRow.document_id == document_id)
            .order_by(ChunkRow.chunk_index)
        ).all()
        return [
            (
                r.id,
                r.chunk_index,
                r.text,
                r.page_start,
                r.page_end,
                r.page_breaks,
                r.text_sha256,
                list(r.embedding),
            )
            for r in rows
        ]


def document(services: Services, document_id: str) -> DocumentRow:
    return services.ingestion.get(document_id)


def fts_matches(services: Services, word: str) -> int:
    with session_scope(services.session_factory) as session:
        return int(
            session.execute(
                text("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH :w"), {"w": word}
            ).scalar()
            or 0
        )


def counts(services: Services) -> tuple[int, int, int]:
    """(chunks, distinct chunk ids, full-text rows)."""
    with session_scope(services.session_factory) as session:
        chunks = int(session.scalar(select(func.count()).select_from(ChunkRow)) or 0)
        distinct = int(session.scalar(select(func.count(func.distinct(ChunkRow.id)))) or 0)
        fts = int(session.execute(text("SELECT count(*) FROM chunks_fts")).scalar() or 0)
        return chunks, distinct, fts


# ------------------------------------------------------------------------ detection
def test_embedding_model_id_matches_the_provider(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    assert embedding_model_id(settings) == build_embedding_provider(settings).model_id
    assert (
        embedding_model_id(make_settings(embedding_provider="ollama", embedding_model="m"))
        == "ollama:m"
    )


def test_documents_indexed_with_older_settings_are_found(
    current: Services, old_index: dict[str, str]
) -> None:
    for key in ("pdf", "txt"):
        # v1 records have no text_cleanup key; a missing key means v1
        assert document(current, old_index[key]).chunking.get("text_cleanup") is None  # type: ignore[union-attr]
        assert current.ingestion.index_differences(document(current, old_index[key])) == [
            CLEANUP_V1_TO_V2
        ]
    summary = current.reindex.summary()
    assert (summary.ready, summary.outdated, summary.current, summary.outdated_without_source) == (
        2,
        2,
        0,
        0,
    )
    assert [(d.setting, d.indexed, d.current, d.documents) for d in summary.differences] == [
        ("text clean-up", "v1", "v2", 2)
    ]


def test_documents_indexed_with_current_settings_are_current(
    make_settings: SettingsFactory, data_dir: Path, old_index: dict[str, str]
) -> None:
    services = open_services(make_settings, data_dir, text_cleanup="v1")
    try:
        assert services.reindex.summary().outdated == 0
        assert services.ingestion.reindex(old_index["pdf"]).result == "current"
    finally:
        services.close()


def test_other_recorded_differences(current: Services, old_index: dict[str, str]) -> None:
    with session_scope(current.session_factory) as session:
        row = session.get(DocumentRow, old_index["pdf"])
        assert row is not None
        row.embedding_model = "hashing:v0-384"
        row.chunking = {**(row.chunking or {}), "chunk_size": 1000}
        other = session.get(DocumentRow, old_index["txt"])
        assert other is not None
        other.chunking = None
    differences = current.ingestion.index_differences(document(current, old_index["pdf"]))
    assert [d.text for d in differences] == [
        "embedding model: hashing:v0-384 -> hashing:v1-384",
        "text clean-up: v1 -> v2",
        "chunk size: 1000 -> 800",
    ]
    assert [
        d.text for d in current.ingestion.index_differences(document(current, old_index["txt"]))
    ] == ["chunking: not recorded -> recorded"]


# ----------------------------------------------------------------------- re-indexing
def test_reindex_gives_the_chunks_of_a_fresh_ingestion(
    current: Services, old_index: dict[str, str], make_settings: SettingsFactory, tmp_path: Path
) -> None:
    pdf = old_index["pdf"]
    old_chunks = chunk_snapshot(current, pdf)
    assert fts_matches(current, "ACME") > 0  # the running header is in the v1 chunks

    outcome = current.ingestion.reindex(pdf)

    assert outcome.result == "reindexed"
    assert (outcome.chunks_before, outcome.chunks_after) == (
        len(old_chunks),
        len(chunk_snapshot(current, pdf)),
    )
    row = document(current, pdf)
    assert row.id == pdf
    assert row.status == DocumentStatus.READY
    assert row.chunking
    assert row.chunking["text_cleanup"] == "v2"
    assert current.ingestion.index_differences(row) == []
    fresh = open_services(make_settings, tmp_path / "fresh", text_cleanup="v2")
    try:
        fresh_row = fresh.ingestion.ingest_bytes("manual.pdf", "application/pdf", manual())
        assert fresh_row.id == pdf
        assert chunk_snapshot(current, pdf) == chunk_snapshot(fresh, pdf)
        assert row.chunk_count == fresh_row.chunk_count
        assert row.page_count == fresh_row.page_count
    finally:
        fresh.close()
    chunks, distinct, fts = counts(current)
    assert chunks == distinct == fts  # no duplicates, no orphan or stale full-text rows
    assert fts_matches(current, "ACME") == 0  # the old text is no longer searchable
    assert fts_matches(current, "impellers") == 1


def test_reindexing_again_changes_nothing(current: Services, old_index: dict[str, str]) -> None:
    assert current.ingestion.reindex(old_index["pdf"]).result == "reindexed"
    before = chunk_snapshot(current, old_index["pdf"])
    updated = document(current, old_index["pdf"]).updated_at
    again = current.ingestion.reindex(old_index["pdf"])
    assert again.result == "current"
    assert chunk_snapshot(current, old_index["pdf"]) == before
    assert document(current, old_index["pdf"]).updated_at == updated


def test_a_failed_reindex_keeps_the_previous_index(
    current: Services, old_index: dict[str, str], embeddings: SwitchableEmbeddings
) -> None:
    pdf = old_index["pdf"]
    before = chunk_snapshot(current, pdf)
    record = document(current, pdf).chunking
    embeddings.fail_when = "impellers"

    outcome = current.ingestion.reindex(pdf)

    assert outcome.result == "failed"
    assert outcome.code == "model_unavailable"
    assert "The previous index is still used." in outcome.detail
    row = document(current, pdf)
    assert row.status == DocumentStatus.READY
    assert row.chunking == record
    assert row.error_code is None
    assert chunk_snapshot(current, pdf) == before
    assert current.file_store.exists(pdf, ".pdf")
    assert current.ingestion.index_differences(row) == [CLEANUP_V1_TO_V2]  # still outdated
    embeddings.fail_when = None
    assert current.ingestion.reindex(pdf).result == "reindexed"


def test_a_failed_write_is_rolled_back(
    current: Services, old_index: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = old_index["pdf"]
    before = chunk_snapshot(current, pdf)
    totals = counts(current)

    def broken(self: ChunkRepository, rows: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(ChunkRepository, "add_many", broken)
    outcome = current.ingestion.reindex(pdf)
    assert (outcome.result, outcome.code) == ("failed", "reindex_write_failed")
    assert chunk_snapshot(current, pdf) == before  # the delete in the same transaction was undone
    assert counts(current) == totals


def test_a_missing_file_is_reported_and_the_index_kept(
    current: Services, old_index: dict[str, str]
) -> None:
    pdf = old_index["pdf"]
    before = chunk_snapshot(current, pdf)
    current.file_store.delete(pdf, ".pdf")
    summary = current.reindex.summary()
    assert (summary.outdated, summary.outdated_without_source) == (2, 1)
    assert list(summary.reindexable) == [old_index["txt"]]
    outcome = current.ingestion.reindex(pdf)
    assert (outcome.result, outcome.code) == ("failed", "source_missing")
    assert chunk_snapshot(current, pdf) == before
    assert document(current, pdf).status == DocumentStatus.READY


def test_a_document_changed_meanwhile_is_not_overwritten(
    current: Services, old_index: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = old_index["pdf"]
    before = chunk_snapshot(current, pdf)
    build = current.ingestion._build

    def build_then_reingest(document_id: str, extension: str) -> Any:
        built = build(document_id, extension)
        current.ingestion.prepare_reingest(document_id)  # someone presses "Re-ingest" meanwhile
        return built

    monkeypatch.setattr(current.ingestion, "_build", build_then_reingest)
    outcome = current.ingestion.reindex(pdf)
    assert outcome.result == "skipped"
    assert "changed while it was re-indexed" in outcome.detail
    assert document(current, pdf).status == DocumentStatus.PENDING  # left to the re-ingestion
    assert chunk_snapshot(current, pdf) == before


def test_a_document_deleted_meanwhile_leaves_no_chunks(
    current: Services, old_index: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = old_index["pdf"]
    build = current.ingestion._build

    def build_then_delete(document_id: str, extension: str) -> Any:
        built = build(document_id, extension)
        current.ingestion.delete(document_id)
        return built

    monkeypatch.setattr(current.ingestion, "_build", build_then_delete)
    outcome = current.ingestion.reindex(pdf)
    assert outcome.result == "skipped"
    assert chunk_snapshot(current, pdf) == []
    chunks, distinct, fts = counts(current)
    assert chunks == distinct == fts


def test_a_document_whose_new_text_is_empty_keeps_its_index_and_file(
    make_settings: SettingsFactory, data_dir: Path
) -> None:
    old = open_services(make_settings, data_dir, text_cleanup="v1")
    try:
        row = old.ingestion.ingest_bytes("settings.pdf", None, sparse_manual())
        assert row.status == DocumentStatus.READY
        assert row.chunk_count > 0
    finally:
        old.close()
    services = open_services(make_settings, data_dir, text_cleanup="v2")
    try:
        before = chunk_snapshot(services, row.id)
        outcome = services.ingestion.reindex(row.id)
        assert (outcome.result, outcome.code) == ("failed", "empty_document")
        assert chunk_snapshot(services, row.id) == before
        assert services.file_store.exists(row.id, ".pdf")  # never deleted by re-indexing
        assert document(services, row.id).status == DocumentStatus.READY
    finally:
        services.close()


def test_reingesting_a_document_that_was_indexed_keeps_its_file(
    make_settings: SettingsFactory, data_dir: Path
) -> None:
    """The other path ("Re-ingest" starts over): it may fail, but it must not discard a file
    that was readable with earlier settings."""
    old = open_services(make_settings, data_dir, text_cleanup="v1")
    try:
        row = old.ingestion.ingest_bytes("settings.pdf", None, sparse_manual())
    finally:
        old.close()
    services = open_services(make_settings, data_dir, text_cleanup="v2")
    try:
        services.ingestion.prepare_reingest(row.id)
        failed = services.ingestion.process(row.id)
        assert (failed.status, failed.error_code) == (DocumentStatus.FAILED, "empty_document")
        assert services.file_store.exists(row.id, ".pdf")
        assert "not kept" not in (failed.error_message or "")
    finally:
        services.close()
    back = open_services(make_settings, data_dir, text_cleanup="v1")
    try:
        back.ingestion.prepare_reingest(row.id)
        assert back.ingestion.process(row.id).status == DocumentStatus.READY
    finally:
        back.close()


def test_documents_that_are_not_ready_are_not_reindexed(current: Services) -> None:
    failed = current.ingestion.ingest_bytes("bad.pdf", "application/pdf", b"%PDF-1.4 garbage")
    pending = current.ingestion.register_upload("boiler.pdf", "application/pdf", other_manual())
    assert current.ingestion.reindex(failed.id).result == "skipped"
    assert current.ingestion.reindex(pending.id).result == "skipped"
    assert document(current, pending.id).status == DocumentStatus.PENDING


# -------------------------------------------------------------------------- bulk job
def test_bulk_job_with_mixed_states_a_failure_a_stop_and_a_restart(
    make_settings: SettingsFactory, data_dir: Path, embeddings: SwitchableEmbeddings
) -> None:
    old = open_services(make_settings, data_dir, text_cleanup="v1")
    try:
        manual_id = old.ingestion.ingest_bytes("manual.pdf", None, manual()).id
        travel_id = old.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode()).id
        boiler_id = old.ingestion.ingest_bytes("boiler.pdf", None, other_manual()).id
        lost_id = old.ingestion.ingest_bytes("lost.txt", None, b"Lost file\n\nIt is gone now.").id
        failed_id = old.ingestion.ingest_bytes("bad.pdf", None, b"%PDF-1.4 garbage").id
        old.file_store.delete(lost_id, ".txt")
    finally:
        old.close()

    services = open_services(make_settings, data_dir, embeddings, text_cleanup="v2")
    try:
        current_id = services.ingestion.ingest_bytes(
            "current.txt", None, b"Current\n\nIndexed with v2 already."
        ).id
        pending_id = services.ingestion.register_upload(
            "pending.txt", None, b"Pending\n\nNot processed."
        ).id
        summary = services.reindex.summary()
        assert (summary.ready, summary.outdated, summary.outdated_without_source) == (5, 4, 1)
        assert list(summary.reindexable) == [manual_id, travel_id, boiler_id]

        embeddings.fail_when = "Radiators"  # one document fails
        job, started = services.reindex.start()
        assert started
        assert job is not None
        again, started_again = services.reindex.start()
        assert again is job
        assert not started_again  # a running job is not started twice
        services.reindex.run(job)
        assert (job.state, job.total, job.done, job.reindexed) == ("finished", 3, 3, 2)
        assert [(f.document_id, f.code) for f in job.failed] == [(boiler_id, "model_unavailable")]
        boiler_chunks = chunk_snapshot(services, boiler_id)
        assert boiler_chunks  # kept

        # untouched: the failed, the pending, the current one and the one without its file
        assert document(services, failed_id).status == DocumentStatus.FAILED
        assert document(services, pending_id).status == DocumentStatus.PENDING
        assert document(services, current_id).chunking == services.ingestion.chunking.as_dict()
        assert document(services, lost_id).status == DocumentStatus.READY

        # the next run only touches what is still outdated, and stops in the middle
        embeddings.fail_when = None
        embeddings.stop_when = "Radiators"
        job, started = services.reindex.start()
        assert started
        assert job is not None
        assert list(job.titles) == [boiler_id]
        with pytest.raises(ProcessStopped):
            services.reindex.run(job)
        assert job.state == "finished"
        assert job.done == 0
        assert chunk_snapshot(services, boiler_id) == boiler_chunks
    finally:
        services.close()

    # restart: a new process on the same data; nothing was left half-written
    embeddings.stop_when = None
    restarted = open_services(make_settings, data_dir, embeddings, text_cleanup="v2")
    try:
        restarted.ingestion.recover_interrupted()  # what the server does on start-up
        assert document(restarted, boiler_id).status == DocumentStatus.READY
        assert list(restarted.reindex.summary().reindexable) == [boiler_id]
        job, started = restarted.reindex.start()
        assert started
        assert job is not None
        restarted.reindex.run(job)
        assert job.reindexed == 1
        final = restarted.reindex.summary()
        assert (final.outdated, final.outdated_without_source, final.reindexable) == (1, 1, {})
        nothing, started = restarted.reindex.start()
        assert not started
        assert nothing is job  # nothing to do: no new job
        chunks, distinct, fts = counts(restarted)
        assert chunks == distinct == fts
        for document_id in (manual_id, travel_id, boiler_id, current_id):
            row = document(restarted, document_id)
            assert row.chunk_count == len(chunk_snapshot(restarted, document_id))
    finally:
        restarted.close()


# ------------------------------------------------------------------------------- API
def test_reindex_api_and_document_list(
    make_settings: SettingsFactory, data_dir: Path, old_index: dict[str, str]
) -> None:
    app = create_app(make_settings(data_dir=data_dir, text_cleanup="v2"))
    with TestClient(app) as client:
        listed = client.get("/api/v1/documents").json()["items"]
        assert {d["id"]: d["index_differences"] for d in listed} == {
            old_index["pdf"]: [{"setting": "text clean-up", "indexed": "v1", "current": "v2"}],
            old_index["txt"]: [{"setting": "text clean-up", "indexed": "v1", "current": "v2"}],
        }
        status = client.get("/api/v1/documents/reindex").json()
        assert (status["ready"], status["outdated"], status["current"], status["job"]) == (
            2,
            2,
            0,
            None,
        )
        started = client.post("/api/v1/documents/reindex")
        assert started.status_code == 202
        job_id = started.json()["job"]["id"]
        # TestClient runs the background task before returning; the job is done
        done = client.get("/api/v1/documents/reindex").json()
        assert (done["outdated"], done["current"]) == (0, 2)
        assert done["job"]["id"] == job_id
        assert (done["job"]["state"], done["job"]["reindexed"], done["job"]["failed"]) == (
            "finished",
            2,
            [],
        )
        again = client.post("/api/v1/documents/reindex").json()
        assert again["job"]["id"] == job_id  # nothing outdated: no new job
        listed = client.get("/api/v1/documents").json()["items"]
        assert all(d["index_differences"] == [] for d in listed)
        assert client.get(f"/api/v1/documents/{old_index['pdf']}").json()["index_differences"] == []
