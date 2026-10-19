"""Health, readiness, status and metrics schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str


class CheckOut(BaseModel):
    ok: bool
    detail: str


class ReadinessOut(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckOut]


class ComponentStatusOut(BaseModel):
    provider: str
    model: str | None = None
    is_llm: bool | None = None
    available: bool | None = None
    detail: str


class SystemStatusOut(BaseModel):
    version: str
    environment: str
    demo_mode: bool
    auth_required: bool
    database_backend: str
    embedding: ComponentStatusOut
    generation: ComponentStatusOut
    reranker: ComponentStatusOut
    retrieval_defaults: dict[str, Any]
    limits: dict[str, Any]


class LatencySummaryOut(BaseModel):
    n: int
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    max: float | None = None


class HourlyBucketOut(BaseModel):
    hour: str
    queries: int
    answered: int
    abstained: int
    failed: int


class OperationalMetricsOut(BaseModel):
    uptime_s: float
    documents_by_status: dict[str, int]
    documents_total: int
    chunks_total: int
    queries_total: int
    queries_window_hours: int
    queries_in_window: int
    queries_by_status: dict[str, int]
    abstention_rate: float | None = None
    citation_valid_rate: float | None = None
    latency_ms: dict[str, LatencySummaryOut]
    hourly: list[HourlyBucketOut]
    http_requests_total: int
    http_server_errors_total: int
    process_counters: list[dict[str, Any]]
