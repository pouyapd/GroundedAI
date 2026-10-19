"""Document ingestion workflow: validate -> store -> parse -> chunk -> embed -> index.

State machine: ``pending -> processing -> ready | failed``. Slow work (parsing,
embedding) happens outside database transactions; the final write of chunks
and status is a single transaction, so a document is never searchable with a
partial set of chunks. A failed document keeps no chunks and can be retried.

Storage policy: the original file is kept while its document exists, so it can be
re-ingested after a configuration change, and is deleted with the document. A file
that cannot be read at all (damaged, password-protected, no extractable text) is
deleted as soon as that is known, because re-ingesting the same bytes cannot help;
the document keeps its error message, and uploading the file again replaces it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from time import perf_counter

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
from groundedai.ingestion.chunking import ChunkingConfig, chunk_id_for, chunk_pages
from groundedai.ingestion.parsers import parse_document
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


class IngestionService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        file_store: FileStore,
        embedding_provider: Callable[[], EmbeddingProvider],
        chunking: ChunkingConfig,
        limits: IngestionLimits,
        metrics: MetricsRegistry | None = None,
    ):
        self.session_factory = session_factory
        self.file_store = file_store
        self._embedding_provider = embedding_provider
        self.chunking = chunking
        self.limits = limits
        self.metrics = metrics or MetricsRegistry()

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

        started = perf_counter()
        try:
            data = self.file_store.load(document_id, extension)
            parsed = parse_document(
                data,
                extension,
                max_pages=self.limits.max_pdf_pages,
                max_chars=self.limits.max_document_chars,
            )
            specs = chunk_pages(parsed.pages, self.chunking)
            if not specs:
                raise EmptyDocument()
            provider = self._embedding_provider()
            vectors = provider.embed_documents([spec.text for spec in specs])

            with session_scope(self.session_factory) as session:
                chunks = ChunkRepository(session)
                chunks.delete_for_document(document_id)
                session.flush()
                chunks.add_many(
                    [
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
                )
                row = DocumentRepository(session).get(document_id)
                if row is None:  # deleted while processing
                    raise NotFound(f"Document {document_id} was deleted during processing.")
                if parsed.title and preferred_title == original_stem:
                    row.title = parsed.title[:300]
                row.status = DocumentStatus.READY
                row.page_count = parsed.page_count
                row.char_count = parsed.char_count
                row.chunk_count = len(specs)
                row.embedding_model = provider.model_id
                row.chunking = self.chunking.as_dict()
                row.updated_at = utcnow()
        except NotFound:
            logger.warning("document_vanished", extra={"document_id": document_id})
            raise
        except GroundedAIError as exc:
            message = exc.message
            if _unreadable(exc):
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
