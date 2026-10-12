"""Document and chunk schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DOCUMENT_ID_PATTERN = r"^doc_[0-9a-f]{20}$"
DocumentId = Annotated[str, StringConstraints(pattern=DOCUMENT_ID_PATTERN)]


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
