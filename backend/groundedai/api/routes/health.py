"""Liveness, readiness and system status."""

from __future__ import annotations

from fastapi import APIRouter, Response

from groundedai import __version__
from groundedai.api.deps import ServicesDep
from groundedai.schemas.system import HealthOut, ReadinessOut, SystemStatusOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut, summary="Liveness probe")
def health() -> HealthOut:
    return HealthOut(status="ok", version=__version__)


@router.get(
    "/ready",
    response_model=ReadinessOut,
    summary="Readiness probe (database, migrations, embedding provider)",
    responses={503: {"model": ReadinessOut, "description": "A dependency is not ready"}},
)
def ready(services: ServicesDep, response: Response) -> ReadinessOut:
    result = services.status.readiness()
    if result.status != "ready":
        response.status_code = 503
    return result


@router.get(
    "/system/status",
    response_model=SystemStatusOut,
    summary="Configured providers, demo-mode flag and limits (no secrets)",
)
def system_status(services: ServicesDep) -> SystemStatusOut:
    return services.status.system_status()
