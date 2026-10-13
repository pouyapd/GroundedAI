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
    supports_claims: list[int] = Field(
        default_factory=list,
        description="Indices into `claims` of the claims this passage is evidence for",
    )


class SentenceOut(BaseModel):
    text: str
    citations: list[str]
    support: Literal["supported", "partial", "unsupported", "uncited"]
    token_recall: float | None = None
    unsupported_numbers: list[str]


ClaimStatusField = Literal["supported", "wrong_source", "added_source", "partial", "unsupported"]
VerdictField = Literal["verified", "partially_verified", "unverified", "not_applicable"]


class EvidenceOut(BaseModel):
    label: str = Field(description="Label of the passage, as in `sources`")
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None = None
    page_end: int | None = None
    quote: str = Field(description="The sentence(s) of the passage that carry the claim")
    quote_start: int = Field(description="Character offset of the quote in the source's `text`")
    quote_end: int


class ClaimOut(BaseModel):
    text: str = Field(description="The claim, without citation markers or attribution phrases")
    citations: list[str] = Field(
        description="Labels the model attached to the claim, in any citation form"
    )
    status: ClaimStatusField = Field(
        description=(
            "supported: the cited passage contains it; wrong_source: the cited passage does not, "
            "another retrieved passage does; added_source: the model cited nothing and a "
            "retrieved passage contains it; partial: some words match; unsupported: no "
            "retrieved passage contains its figures and most of its words"
        )
    )
    evidence: list[EvidenceOut]
    missing_figures: list[str] = Field(description="Figures that no retrieved passage states")
    word_overlap: float
    flags: list[str] = Field(description="instruction_text: the claim repeats an instruction")


class VerificationOut(BaseModel):
    verdict: VerdictField
    claims: int
    supported: int
    wrong_source: int
    added_source: int
    partial: int
    unsupported: int
    notes: int = Field(description="Sentences about the sources themselves, not checked")
    references_standard: int = Field(description="Citations written as [S#]")
    references_variant: int = Field(
        description="Citations written another way ([Source 1], (S1), a document title)"
    )
    method: str


class ConflictStatementOut(BaseModel):
    label: str
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int | None = None
    page_end: int | None = None
    quote: str = Field(description="The sentence that states the value")
    quote_start: int = Field(description="Character offset of the quote in the source's `text`")
    quote_end: int
    values: list[str] = Field(description="The values it states, as written")
    denies_requirement: bool = Field(description='It says there is none ("no fixed schedule")')
    dates: list[str] = Field(description="Dates that say when it applies, if any")


class ConflictOut(BaseModel):
    kind: Literal["conflict", "different_dates"] = Field(
        description=(
            "conflict: different values with nothing that separates them, or a requirement and "
            "its denial; different_dates: different values that apply to different dates"
        )
    )
    statements: list[ConflictStatementOut]
    explanation: str
    answer_reports: list[str] = Field(
        default_factory=list, description="Labels whose value the answer states"
    )


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
    sentences: list[SentenceOut] = Field(
        description="Strict [S#] sentence check (kept for comparison with earlier reports)"
    )
    claims: list[ClaimOut] = Field(
        default_factory=list,
        description="Every factual sentence checked against every retrieved passage",
    )
    verification: VerificationOut | None = None
    conflicts: list[ConflictOut] = Field(
        default_factory=list,
        description="Retrieved passages that disagree about what was asked (never resolved)",
    )
    unverified_answer: str | None = Field(
        default=None,
        description=(
            "The model's answer when it was withheld because no statement could be matched to "
            "a passage (setting GROUNDEDAI_UNVERIFIED_ANSWERS=withhold)"
        ),
    )
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
