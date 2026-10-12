"""API-key authentication.

When ``GROUNDEDAI_API_KEY`` is set, every ``/api/v1`` endpoint except health,
readiness and system status requires the ``X-API-Key`` header. Production
mode refuses to start without a key (see :class:`~groundedai.core.config.Settings`).
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Request

from groundedai.core.config import Settings
from groundedai.core.errors import Unauthorized

API_KEY_HEADER = "X-API-Key"


def verify_api_key(settings: Settings, provided: str | None) -> None:
    if not settings.auth_enabled:
        return
    expected = settings.api_key.get_secret_value() if settings.api_key else ""
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise Unauthorized()


def client_identity(request: Request, settings: Settings) -> str:
    """Key used for rate limiting: a hash of the API key, else the client address."""
    provided = request.headers.get(API_KEY_HEADER)
    if settings.auth_enabled and provided:
        return "key:" + hashlib.sha256(provided.encode()).hexdigest()[:16]
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return "ip:" + first
    return "ip:" + (request.client.host if request.client else "unknown")
