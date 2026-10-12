"""Evaluation run and report schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["retrieval"] = Field(
        default="retrieval",
        description=(
            "Retrieval evaluation of the instance's current configuration on the bundled "
            "dataset, in an isolated temporary index."
        ),
    )


class EvaluationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    config: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    error_message: str | None = None


class EvaluationRunDetail(EvaluationRunOut):
    report: dict[str, Any] | None = None


class EvaluationRunList(BaseModel):
    items: list[EvaluationRunOut]
    total: int
    offset: int
    limit: int


class ReportListItem(BaseModel):
    id: str
    kind: str
    title: str
    generated_at: str | None = None
    dataset: str | None = None
    n_configurations: int


class ReportList(BaseModel):
    items: list[ReportListItem]
