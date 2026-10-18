"""Document upload, listing, inspection, re-ingestion and deletion."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Path,
    Query,
    Response,
    UploadFile,
)

from groundedai.api.deps import ServicesDep, rate_limit, require_api_key
from groundedai.core.errors import NotFound
from groundedai.persistence.db import session_scope
from groundedai.persistence.models import DocumentRow
from groundedai.persistence.repositories import ChunkRepository, DocumentRepository
from groundedai.schemas.documents import (
    DOCUMENT_ID_PATTERN,
    ChunkList,
    ChunkOut,
    DocumentList,
    DocumentOut,
)
from groundedai.services.container import Services

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])

DocumentIdPath = Annotated[str, Path(pattern=DOCUMENT_ID_PATTERN)]
_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"description": "Missing or invalid API key"},
    404: {"description": "Unknown document"},
    409: {"description": "Duplicate document or document busy"},
    413: {"description": "File too large"},
    415: {"description": "Unsupported file type"},
    422: {"description": "Invalid or unparseable file"},
    429: {"description": "Rate limit exceeded"},
}


def _document_out(services: Services, row: DocumentRow) -> DocumentOut:
    out = DocumentOut.model_validate(row)
    out.source_stored = services.ingestion.source_stored(row)
    return out


def _process_in_background(services: Services, document_id: str) -> None:
    try:
        services.ingestion.process(document_id)
    except NotFound:
        logger.info("document_deleted_before_processing", extra={"document_id": document_id})


@router.post(
    "",
    status_code=202,
    response_model=DocumentOut,
    summary="Upload a PDF, .txt or .md document for ingestion",
    responses={k: v for k, v in _ERRORS.items() if k != 404},
    dependencies=[Depends(rate_limit("upload"))],
)
def upload_document(
    services: ServicesDep,
    background: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF, plain text or Markdown")],
    title: Annotated[str | None, Form(max_length=300)] = None,
) -> DocumentOut:
    # Read at most one byte past the limit; validation rejects anything larger.
    data = file.file.read(services.settings.max_upload_bytes + 1)
    row = services.ingestion.register_upload(
        file.filename or "", file.content_type, data, title=title.strip() if title else None
    )
    background.add_task(_process_in_background, services, row.id)
    return _document_out(services, row)


@router.get("", response_model=DocumentList, summary="List documents (newest first)")
def list_documents(
    services: ServicesDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status: Literal["pending", "processing", "ready", "failed"] | None = None,
) -> DocumentList:
    with session_scope(services.session_factory) as session:
        rows, total = DocumentRepository(session).page(offset=offset, limit=limit, status=status)
        items = [_document_out(services, row) for row in rows]
    return DocumentList(items=items, total=total, offset=offset, limit=limit)


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Document metadata and ingestion status",
    responses={404: _ERRORS[404]},
)
def get_document(services: ServicesDep, document_id: DocumentIdPath) -> DocumentOut:
    return _document_out(services, services.ingestion.get(document_id))


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkList,
    summary="Inspect the indexed chunks of a document",
    responses={404: _ERRORS[404]},
)
def list_chunks(
    services: ServicesDep,
    document_id: DocumentIdPath,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ChunkList:
    services.ingestion.get(document_id)
    with session_scope(services.session_factory) as session:
        rows, total = ChunkRepository(session).list_for_document(
            document_id, offset=offset, limit=limit
        )
        items = [ChunkOut.model_validate(row) for row in rows]
    return ChunkList(items=items, total=total, offset=offset, limit=limit)


@router.post(
    "/{document_id}/reingest",
    status_code=202,
    response_model=DocumentOut,
    summary="Re-run ingestion (e.g. after a failure or a configuration change)",
    responses={404: _ERRORS[404], 409: _ERRORS[409], 429: _ERRORS[429]},
    dependencies=[Depends(rate_limit("upload"))],
)
def reingest_document(
    services: ServicesDep, background: BackgroundTasks, document_id: DocumentIdPath
) -> DocumentOut:
    row = services.ingestion.prepare_reingest(document_id)
    background.add_task(_process_in_background, services, row.id)
    return _document_out(services, row)


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Delete a document, its chunks and its stored file",
    responses={404: _ERRORS[404], 409: _ERRORS[409]},
)
def delete_document(services: ServicesDep, document_id: DocumentIdPath) -> Response:
    services.ingestion.delete(document_id)
    return Response(status_code=204)
