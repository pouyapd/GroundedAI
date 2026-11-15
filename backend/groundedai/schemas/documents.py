"""Document and chunk schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DOCUMENT_ID_PATTERN = r"^doc_[0-9a-f]{20}$"
DocumentId = Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]


class IndexDifferenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    setting: str = Field(description="e.g. 'text clean-up', 'chunk size', 'embedding model'")
    indexed: str = Field(description="The value the document was indexed with")
    current: str = Field(description="The value the current settings use")


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    status: str = Field(description="pending | processing | ready | failed")
    error_code: str | None = None
    error_message: str | None = None
    page_count: int | None = None
    char_count: int | None = None
    chunk_count: int
    embedding_model: str | None = None
    chunking: dict[str, Any] | None = None
    attempts: int
    created_at: datetime
    updated_at: datetime
    source_stored: bool | None = Field(
        default=None,
        description="Whether the original file is still stored, so re-ingestion is possible. "
        "False after a file that could not be read was discarded.",
    )
    index_differences: list[IndexDifferenceOut] | None = Field(
        default=None,
        description="For a ready document: the recorded index settings that differ from the "
        "current ones (empty when it is current). Null for documents that are not ready.",
    )


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    offset: int
    limit: int


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None


class ChunkList(BaseModel):
    items: list[ChunkOut]
    total: int
    offset: int
    limit: int


class DifferenceCountOut(IndexDifferenceOut):
    documents: int


class ReindexOutcomeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    title: str
    result: Literal["reindexed", "current", "skipped", "failed"]
    detail: str
    code: str | None = None


class ReindexJobOut(BaseModel):
    id: str
    state: Literal["running", "finished"]
    started_at: datetime
    finished_at: datetime | None = None
    total: int
    done: int
    reindexed: int
    already_current: int
    skipped: list[ReindexOutcomeOut]
    failed: list[ReindexOutcomeOut] = Field(
        description="Documents whose re-indexing failed; each keeps its previous index."
    )
    processing_id: str | None = None
    processing_title: str | None = None


class ReindexStatus(BaseModel):
    """Ready documents indexed with older settings, and the bulk re-indexing job."""

    ready: int
    current: int
    outdated: int
    outdated_without_source: int = Field(
        description="Outdated documents whose original file is not stored (upload them again)."
    )
    differences: list[DifferenceCountOut]
    job: ReindexJobOut | None = Field(
        default=None, description="The running job, or the last one since the server started."
    )
