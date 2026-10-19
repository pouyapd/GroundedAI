"""Operational metrics (JSON for the dashboard, Prometheus text for scrapers)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from groundedai.api.deps import ServicesDep, require_api_key
from groundedai.schemas.system import OperationalMetricsOut

router = APIRouter(prefix="/metrics", tags=["operations"], dependencies=[Depends(require_api_key)])


@router.get(
    "", response_model=OperationalMetricsOut, summary="Operational metrics for the dashboard"
)
def operational_metrics(
    services: ServicesDep, window_hours: Annotated[int, Query(ge=1, le=168)] = 24
) -> OperationalMetricsOut:
    return services.status.operational_metrics(window_hours)


@router.get(
    "/prometheus",
    response_class=PlainTextResponse,
    summary="Process metrics in the Prometheus text exposition format",
)
def prometheus_metrics(services: ServicesDep) -> PlainTextResponse:
    return PlainTextResponse(
        services.metrics.prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
