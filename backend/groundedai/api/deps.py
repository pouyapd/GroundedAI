"""FastAPI dependencies: service access, authentication and rate limits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from groundedai.security.auth import API_KEY_HEADER, client_identity, verify_api_key
from groundedai.security.rate_limit import Limit
from groundedai.services.container import Services

_api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Required on /api/v1 endpoints when GROUNDEDAI_API_KEY is set.",
)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_api_key(
    services: ServicesDep, api_key: Annotated[str | None, Security(_api_key_header)] = None
) -> None:
    verify_api_key(services.settings, api_key)


RateScope = Literal["query", "upload", "evaluation"]


def rate_limit(scope: RateScope) -> Callable[[Request, Services], None]:
    def dependency(request: Request, services: ServicesDep) -> None:
        settings = services.settings
        if not settings.rate_limit_enabled:
            return
        limits = {
            "query": Limit(settings.rate_limit_query_per_minute, 60.0),
            "upload": Limit(settings.rate_limit_upload_per_minute, 60.0),
            "evaluation": Limit(settings.rate_limit_eval_per_hour, 3600.0),
        }
        services.rate_limiter.check(scope, client_identity(request, settings), limits[scope])

    return dependency
