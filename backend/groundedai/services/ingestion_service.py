"""Document ingestion workflow: validate -> store -> parse -> chunk -> embed -> index.

State machine: ``pending -> processing -> ready | failed``. Slow work (parsing,
embedding) happens outside database transactions; the final write of chunks
and status is a single transaction, so a document is never searchable with a
partial set of chunks. A failed document keeps no chunks and can be retried.

Storage policy: the original file is kept while its document exists, so it can be
re-ingested after a configuration change, and is deleted with the document. A file
that cannot be read at all (damaged, password-protected, no extractable text) is
deleted as soon as that is known, because re-ingesting the same bytes cannot help;
the document keeps its error message, and uploading the file again replaces it. A file
that was indexed before is never deleted this way: if other settings later find no text
in it, it may still be readable with the earlier ones.

Re-indexing (:meth:`IngestionService.reindex`) is the safe path for ready documents after a
settings change: the document stays ready and searchable with its previous chunks while the
new ones are built, they are swapped in one transaction, and if anything fails the previous
index is kept unchanged (re-ingestion, by contrast, starts the document over).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any, Literal

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from groundedai.core.errors import (
    Conflict,
    DocumentParseError,
    DuplicateDocument,
    EmptyDocument,
    GroundedAIError,
    NotFound,
)
from groundedai.ingestion.chunking import ChunkingConfig, ChunkSpec, chunk_id_for, chunk_pages
from groundedai.ingestion.parsers import ParsedDocument, parse_document
from groundedai.ingestion.validation import MEDIA_TYPES, document_id_for, validate_upload
from groundedai.observability.metrics import MetricsRegistry
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import ChunkRow, DocumentRow, DocumentStatus, utcnow
from groundedai.persistence.repositories import ChunkRepository, DocumentRepository
from groundedai.persistence.storage import FileStore
from groundedai.retrieval.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

_EXTENSION_BY_MEDIA_TYPE = {media_type: ext for ext, media_type in MEDIA_TYPES.items()}
_LIMIT_DETAILS = frozenset({"max_pages", "max_chars"})
FILE_DISCARDED_NOTE = "The file was not kept; upload a readable copy to try again."


def _unreadable(exc: GroundedAIError) -> bool:
    """The file itself cannot be read; unlike a size limit, settings cannot change that."""
    if isinstance(exc, EmptyDocument):
        return True
    return isinstance(exc, DocumentParseError) and not _LIMIT_DETAILS & set(exc.details)


@dataclass(frozen=True, slots=True)
class IngestionLimits:
    max_upload_bytes: int
    max_pdf_pages: int
    max_document_chars: int


# How the chunking record names its settings; a record without a key was made with the value
# in _RECORD_DEFAULTS (text clean-up was recorded only once v2 existed).
_SETTING_NAMES = {
    "text_cleanup": "text clean-up",
    "chunk_size": "chunk size",
    "chunk_overlap": "chunk overlap",
    "splitter": "splitter",
    "version": "chunker version",
}
_RECORD_DEFAULTS: dict[str, object] = {"text_cleanup": "v1"}


@dataclass(frozen=True, slots=True)
class IndexDifference:
    """One setting a document was indexed with that differs from the current settings."""

    setting: str
    indexed: str
    current: str

    @property
    def text(self) -> str:
        return f"{self.setting}: {self.indexed} -> {self.current}"


ReindexResult = Literal["reindexed", "current", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class ReindexOutcome:
    document_id: str
    title: str
    result: ReindexResult
    detail: str = ""
    code: str | None = None
    chunks_before: int | None = None
    chunks_after: int | None = None


@dataclass(frozen=True, slots=True)
class _Built:
    parsed: ParsedDocument
    specs: list[ChunkSpec]
    vectors: Any
    model_id: str


def _fingerprint(row: DocumentRow) -> tuple[object, ...]:
    """Changes when anything else writes the document (re-ingestion, replacement, re-index)."""
    return (
        row.status,
        row.sha256,
        row.attempts,
        row.created_at,
        row.updated_at,
        row.embedding_model,
        json.dumps(row.chunking, sort_keys=True),
    )


def _shown(value: object) -> str:
    return "not recorded" if value is None else str(value)


def _chunk_rows(document_id: str, specs: Sequence[ChunkSpec], vectors: Any) -> list[ChunkRow]:
    return [
        ChunkRow(
            id=chunk_id_for(document_id, spec.index),
            document_id=document_id,
            chunk_index=spec.index,
            text=spec.text,
            char_start=spec.char_start,
            char_end=spec.char_end,
            page_start=spec.page_start,
            page_end=spec.page_end,
            page_breaks=[list(item) for item in spec.page_breaks] or None,
            text_sha256=spec.sha256,
            embedding=vector,
        )
        for spec, vector in zip(specs, vectors, strict=True)
    ]


class IngestionService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        file_store: FileStore,
        embedding_provider: Callable[[], EmbeddingProvider],
        chunking: ChunkingConfig,
        limits: IngestionLimits,
        metrics: MetricsRegistry | None = None,
        expected_embedding_model: str | None = None,
    ):
        self.session_factory = session_factory
        self.file_store = file_store
        self._embedding_provider = embedding_provider
        self.chunking = chunking
        self.limits = limits
        self.metrics = metrics or MetricsRegistry()
        # The model id new documents get; None: the embedding model is not compared.
        self.expected_embedding_model = expected_embedding_model

    # ------------------------------------------------------------------ upload
    def register_upload(
        self, filename: str, content_type: str | None, data: bytes, *, title: str | None = None
    ) -> DocumentRow:
        """Validate and store an upload; the document starts as ``pending``."""
        upload = validate_upload(filename, content_type, data, self.limits.max_upload_bytes)
        document_id = document_id_for(upload.sha256)
        with session_scope(self.session_factory) as session:
            repository = DocumentRepository(session)
            existing = repository.get_by_sha256(upload.sha256)
            if existing is not None and existing.status == DocumentStatus.FAILED:
                # Uploading a file again is the way to retry a failed document.
                repository.delete(existing)
                logger.info("failed_document_replaced", extra={"document_id": existing.id})
            elif existing is not None:
                raise DuplicateDocument(
                    details={"document_id": existing.id, "status": existing.status}
                )
        self.file_store.save(document_id, upload.extension, upload.data)
        row = DocumentRow(
            id=document_id,
            title=(title or PurePosixPath(upload.filename).stem)[:300],
            filename=upload.filename,
            media_type=upload.media_type,
            size_bytes=len(upload.data),
            sha256=upload.sha256,
            status=DocumentStatus.PENDING,
            chunk_count=0,
            attempts=0,
        )
        try:
            with session_scope(self.session_factory) as session:
                DocumentRepository(session).add(row)
        except IntegrityError as exc:  # a concurrent upload of the same file won the race
            raise DuplicateDocument(details={"document_id": document_id}) from exc
        self.metrics.inc("documents_registered")
        logger.info("document_registered", extra={"document_id": document_id, "bytes": len(data)})
        return row

    def ingest_bytes(
        self, filename: str, content_type: str | None, data: bytes, *, title: str | None = None
    ) -> DocumentRow:
        """Register and process synchronously (CLI and evaluation)."""
        row = self.register_upload(filename, content_type, data, title=title)
        return self.process(row.id)

    # -------------------------------------------------------------- processing
    def process(self, document_id: str) -> DocumentRow:
        """Run the pipeline for a registered document. Failures are recorded, not raised."""
        with session_scope(self.session_factory) as session:
            row = DocumentRepository(session).get(document_id)
            if row is None:
                raise NotFound(f"Document {document_id} does not exist.")
            row.status = DocumentStatus.PROCESSING
            row.attempts += 1
            row.error_code = None
            row.error_message = None
            extension = _EXTENSION_BY_MEDIA_TYPE[row.media_type]
            preferred_title = row.title
            original_stem = PurePosixPath(row.filename).stem
            # A file that was indexed before is readable; if the current settings find no
            # text in it (e.g. clean-up v2 removed every line), keep it for other settings.
            indexed_before = row.chunking is not None

        started = perf_counter()
        try:
            built = self._build(document_id, extension)
            parsed, specs = built.parsed, built.specs

            with session_scope(self.session_factory) as session:
                chunks = ChunkRepository(session)
                chunks.delete_for_document(document_id)
                session.flush()
                chunks.add_many(_chunk_rows(document_id, specs, built.vectors))
                row = DocumentRepository(session).get(document_id)
                if row is None:  # deleted while processing
                    raise NotFound(f"Document {document_id} was deleted during processing.")
                self._record_index(row, built, preferred_title, original_stem)
                row.status = DocumentStatus.READY
        except NotFound:
            logger.warning("document_vanished", extra={"document_id": document_id})
            raise
        except GroundedAIError as exc:
            message = exc.message
            if _unreadable(exc) and not indexed_before:
                self.file_store.delete(document_id, extension)
                message = f"{message} {FILE_DISCARDED_NOTE}"
            self._mark_failed(document_id, exc.code, message)
        except FileNotFoundError:
            self._mark_failed(
                document_id, "source_missing", "The stored original file is missing. Upload again."
            )
        except Exception:
            logger.exception("ingestion_failed", extra={"document_id": document_id})
            self._mark_failed(document_id, "ingestion_error", "Unexpected error during ingestion.")
        else:
            duration_ms = (perf_counter() - started) * 1000
            self.metrics.inc("ingestions", outcome="ready")
            self.metrics.observe("ingestion_latency", duration_ms)
            logger.info(
                "document_ready",
                extra={
                    "document_id": document_id,
                    "chunks": len(specs),
                    "pages": parsed.page_count,
                    "duration_ms": round(duration_ms, 1),
                },
            )
        return self.get(document_id)

    def _mark_failed(self, document_id: str, code: str, message: str) -> None:
        with session_scope(self.session_factory) as session:
            ChunkRepository(session).delete_for_document(document_id)
            row = DocumentRepository(session).get(document_id)
            if row is None:
                return
            row.status = DocumentStatus.FAILED
            row.error_code = code
            row.error_message = message[:500]
            row.chunk_count = 0
            row.updated_at = utcnow()
        self.metrics.inc("ingestions", outcome="failed")
        logger.info("document_failed", extra={"document_id": document_id, "error_code": code})

    def _build(self, document_id: str, extension: str) -> _Built:
        """Parse, chunk and embed the stored file with the current settings (no database)."""
        data = self.file_store.load(document_id, extension)
        parsed = parse_document(
            data,
            extension,
            max_pages=self.limits.max_pdf_pages,
            max_chars=self.limits.max_document_chars,
            cleanup=self.chunking.text_cleanup,
        )
        specs = chunk_pages(parsed.pages, self.chunking)
        if not specs:
            raise EmptyDocument()
        provider = self._embedding_provider()
        vectors = provider.embed_documents([spec.text for spec in specs])
        return _Built(parsed, specs, vectors, provider.model_id)

    def _record_index(
        self, row: DocumentRow, built: _Built, preferred_title: str, original_stem: str
    ) -> None:
        if built.parsed.title and preferred_title == original_stem:
            row.title = built.parsed.title[:300]
        row.page_count = built.parsed.page_count
        row.char_count = built.parsed.char_count
        row.chunk_count = len(built.specs)
        row.embedding_model = built.model_id
        row.chunking = self.chunking.as_dict()
        row.updated_at = utcnow()

    # ------------------------------------------------------------- re-indexing
    def index_differences(self, row: DocumentRow) -> list[IndexDifference]:
        """How a ready document's index differs from one built with the current settings.

        Only what is recorded on the document is compared: the chunking record (chunk size and
        overlap, splitter, chunker version, text clean-up) and the embedding model. A change
        to the parsing code that keeps these values the same is not detected. Documents that
        are not ready have no index to compare and return no differences.
        """
        if row.status != DocumentStatus.READY:
            return []
        differences: list[IndexDifference] = []
        if self.expected_embedding_model and row.embedding_model != self.expected_embedding_model:
            differences.append(
                IndexDifference(
                    "embedding model", _shown(row.embedding_model), self.expected_embedding_model
                )
            )
        current: dict[str, object] = {**_RECORD_DEFAULTS, **self.chunking.as_dict()}
        if not row.chunking:
            differences.append(IndexDifference("chunking", "not recorded", "recorded"))
            return differences
        recorded: dict[str, object] = {**_RECORD_DEFAULTS, **row.chunking}
        for key in [*current, *(key for key in recorded if key not in current)]:
            if recorded.get(key) != current.get(key):
                differences.append(
                    IndexDifference(
                        _SETTING_NAMES.get(key, key),
                        _shown(recorded.get(key)),
                        _shown(current.get(key)),
                    )
                )
        return differences

    def reindex(self, document_id: str) -> ReindexOutcome:
        """Rebuild a ready document's index with the current settings, keeping it safe.

        The document stays ready, and searchable with its previous chunks, while the new
        chunks are built. They replace the old ones in one transaction, under a row lock, and
        only if nothing else wrote the document in the meantime. If parsing, embedding or the
        write fails, nothing is changed: the previous index is kept and the stored file is
        never deleted. Document and chunk ids stay the same (chunk ids are derived from the
        document id and position). A document that is already current is left alone.
        """
        with session_scope(self.session_factory) as session:
            row = DocumentRepository(session).get(document_id)
            if row is None:
                return ReindexOutcome(document_id, "", "skipped", "The document no longer exists.")
            title = row.title
            if row.status != DocumentStatus.READY:
                return ReindexOutcome(
                    document_id,
                    title,
                    "skipped",
                    f"The document is {row.status}; only ready documents are re-indexed.",
                )
            chunks_before = row.chunk_count
            if not self.index_differences(row):
                return ReindexOutcome(
                    document_id,
                    title,
                    "current",
                    "Already indexed with the current settings.",
                    chunks_before=chunks_before,
                    chunks_after=chunks_before,
                )
            before = _fingerprint(row)
            extension = _EXTENSION_BY_MEDIA_TYPE[row.media_type]
            preferred_title = row.title
            original_stem = PurePosixPath(row.filename).stem

        def failed(code: str, message: str) -> ReindexOutcome:
            self.metrics.inc("reindex", outcome="failed")
            logger.info("reindex_failed", extra={"document_id": document_id, "error_code": code})
            return ReindexOutcome(
                document_id,
                title,
                "failed",
                f"{message} The previous index is still used.",
                code=code,
                chunks_before=chunks_before,
                chunks_after=chunks_before,
            )

        started = perf_counter()
        try:
            built = self._build(document_id, extension)
        except FileNotFoundError:
            return failed(
                "source_missing",
                "The original file is no longer stored; upload it again to rebuild its index.",
            )
        except GroundedAIError as exc:
            return failed(exc.code, exc.message)
        except Exception:
            logger.exception("reindex_error", extra={"document_id": document_id})
            return failed("reindex_error", "Unexpected error while re-indexing.")

        try:
            with session_scope(self.session_factory) as session:
                row = DocumentRepository(session).get_for_update(document_id)
                if row is None:
                    return ReindexOutcome(
                        document_id,
                        title,
                        "skipped",
                        "The document was deleted while it was re-indexed; nothing was written.",
                    )
                if _fingerprint(row) != before:
                    return ReindexOutcome(
                        document_id,
                        row.title,
                        "skipped",
                        "The document changed while it was re-indexed (re-ingested, replaced or "
                        "re-indexed elsewhere); nothing was written.",
                    )
                chunks = ChunkRepository(session)
                chunks.delete_for_document(document_id)
                session.flush()
                chunks.add_many(_chunk_rows(document_id, built.specs, built.vectors))
                self._record_index(row, built, preferred_title, original_stem)
                title = row.title
        except Exception:
            logger.exception("reindex_write_failed", extra={"document_id": document_id})
            return failed("reindex_write_failed", "The new index could not be saved.")
        duration_ms = (perf_counter() - started) * 1000
        self.metrics.inc("reindex", outcome="reindexed")
        logger.info(
            "document_reindexed",
            extra={
                "document_id": document_id,
                "chunks_before": chunks_before,
                "chunks": len(built.specs),
                "duration_ms": round(duration_ms, 1),
            },
        )
        return ReindexOutcome(
            document_id,
            title,
            "reindexed",
            chunks_before=chunks_before,
            chunks_after=len(built.specs),
        )

    # ------------------------------------------------------------- management
    def get(self, document_id: str) -> DocumentRow:
        with session_scope(self.session_factory) as session:
            row = DocumentRepository(session).get(document_id)
            if row is None:
                raise NotFound(f"Document {document_id} does not exist.")
            return row

    def source_stored(self, row: DocumentRow) -> bool:
        """Whether the original file is still stored (re-ingestion needs it)."""
        return self.file_store.exists(row.id, _EXTENSION_BY_MEDIA_TYPE[row.media_type])

    def prepare_reingest(self, document_id: str) -> DocumentRow:
        """Reset a document to ``pending`` so it can be processed again."""
        with session_scope(self.session_factory) as session:
            row = DocumentRepository(session).get(document_id)
            if row is None:
                raise NotFound(f"Document {document_id} does not exist.")
            if row.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
                raise Conflict("The document is already queued or being processed.")
            if not self.source_stored(row):
                raise Conflict(
                    "The original file is no longer stored. Upload it again.",
                    details={"reason": "source_missing"},
                )
            row.status = DocumentStatus.PENDING
            row.updated_at = utcnow()
            return row

    def delete(self, document_id: str) -> None:
        with session_scope(self.session_factory) as session:
            repository = DocumentRepository(session)
            row = repository.get(document_id)
            if row is None:
                raise NotFound(f"Document {document_id} does not exist.")
            if row.status == DocumentStatus.PROCESSING:
                raise Conflict("The document is being processed; delete it when processing ends.")
            extension = _EXTENSION_BY_MEDIA_TYPE[row.media_type]
            repository.delete(row)
        self.file_store.delete(document_id, extension)
        self.metrics.inc("documents_deleted")
        logger.info("document_deleted", extra={"document_id": document_id})

    def recover_interrupted(self) -> int:
        """Mark documents left pending/processing by a previous process as failed."""
        with session_scope(self.session_factory) as session:
            ids = DocumentRepository(session).ids_with_status(
                [DocumentStatus.PENDING, DocumentStatus.PROCESSING]
            )
        for document_id in ids:
            self._mark_failed(
                document_id,
                "interrupted",
                "Processing was interrupted by a server restart. Use re-ingest to retry.",
            )
        return len(ids)
