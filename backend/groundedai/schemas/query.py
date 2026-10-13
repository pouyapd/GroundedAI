"""Question-answering and retrieval schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from groundedai.schemas.documents import DocumentId

RetrievalModeField = Literal["lexical", "dense", "hybrid"]
FusionField = Literal["rrf", "weighted"]
AnswerStatus = Literal["answered", "insufficient_evidence", "invalid_output"]


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "What is the hotel limit per night in the travel policy?", "top_k": 5}
            ]
        },
    )

    question: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[DocumentId] | None = Field(default=None, max_length=50)
    retrieval_mode: RetrievalModeField | None = None
    rerank: bool | None = None
    include_diagnostics: bool = False


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[DocumentId] | None = Field(default=None, max_length=50)
    retrieval_mode: RetrievalModeField | None = None
    fusion: FusionField | None = None
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank: bool | None = None


class ChunkScoresOut(BaseModel):
    lexical: float | None = None
    lexical_rank: int | None = None
    dense: float | None = None
    dense_rank: int | None = None
    fused: float | None = None
    rerank: float | None = None


class RetrievedChunkOut(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None = None
    page_end: int | None = None
    text: str
    scores: ChunkScoresOut
    matched_terms: list[str]


class RetrievalDiagnosticsOut(BaseModel):
    mode: str
    fusion: str | None = None
    reranker: str | None = None
    embedding_model: str | None = None
    lexical_candidates: int
    dense_candidates: int
    fused_candidates: int
    duplicates_removed: int
    timings_ms: dict[str, float]
    warnings: list[str]


class RetrieveResponse(BaseModel):
    results: list[RetrievedChunkOut]
    diagnostics: RetrievalDiagnosticsOut


class SourceOut(BaseModel):
    label: str = Field(description="Citation label used in the answer, e.g. S1")
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None = None
    page_end: int | None = None
    excerpt: str = Field(description="Short passage from the source that best matches the answer")
    text: str = Field(description="Text given to the model (may be truncated)")
    cited: bool
    truncated: bool
    injection_flags: list[str] = Field(
        description="Instruction-like patterns found in this source (treated as data)"
    )
    scores: ChunkScoresOut


class SentenceOut(BaseModel):
    text: str
    citations: list[str]
    support: Literal["supported", "partial", "unsupported", "uncited"]
    token_recall: float | None = None
    unsupported_numbers: list[str]


class CitationCheckOut(BaseModel):
    cited: list[str]
    valid: list[str]
    invalid: list[str] = Field(description="Labels the model cited that were not in its sources")
    all_valid: bool
    sentences_total: int
    sentences_cited: int


class ProviderOut(BaseModel):
    name: str
    model: str
    is_llm: bool = Field(description="False for the deterministic extractive demo generator")


class TimingsOut(BaseModel):
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float


class UsageOut(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cost_note: str


class AnswerResponse(BaseModel):
    status: AnswerStatus
    answer: str | None = Field(description="Answer text with [S#] citation markers")
    abstention_reason: str | None = None
    grounding: Literal["grounded", "partially_grounded", "ungrounded"] | None = None
    sources: list[SourceOut]
    sentences: list[SentenceOut]
    citation_check: CitationCheckOut
    warnings: list[str]
    provider: ProviderOut
    timings: TimingsOut
    usage: UsageOut
    retrieval: RetrievalDiagnosticsOut | None = None
    model_output: str | None = Field(
        default=None,
        description="The model's raw text. Filled only in evaluation runs, for error analysis.",
    )
