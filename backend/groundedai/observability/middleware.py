"""Pure-ASGI middleware: request ids, access logs, metrics, security headers, body limits."""

from __future__ import annotations

import json
import logging
import re
import uuid
from time import perf_counter
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from groundedai.observability.logging import request_id_var
from groundedai.observability.metrics import MetricsRegistry

logger = logging.getLogger("groundedai.http")

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_QUIET_PATHS = frozenset({"/health", "/api/v1/health", "/ready", "/api/v1/ready"})
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
FRONTEND_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
API_CSP = "default-src 'none'; frame-ancestors 'none'"


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, metrics: MetricsRegistry):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = perf_counter()
        state: dict[str, Any] = {"status": 500, "duration_ms": None}
        path: str = scope.get("path", "")

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
                if not path.startswith(_DOCS_PATHS):
                    csp = API_CSP if path.startswith("/api") else FRONTEND_CSP
                    headers.setdefault("Content-Security-Policy", csp)
            elif message["type"] == "http.response.body" and not message.get("more_body"):
                # Measure when the response is complete, before any background task runs.
                state["duration_ms"] = round((perf_counter() - started) * 1000, 3)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = state["duration_ms"]
            if duration_ms is None:
                duration_ms = round((perf_counter() - started) * 1000, 3)
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            status = int(state["status"])
            self.metrics.inc(
                "http_requests",
                method=scope.get("method", ""),
                route=route,
                status_class=f"{status // 100}xx",
            )
            self.metrics.observe("http_request_latency", duration_ms, route=route)
            log_level = logging.DEBUG if path in _QUIET_PATHS else logging.INFO
            logger.log(
                log_level,
                "http_request",
                extra={
                    "method": scope.get("method"),
                    "route": route,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
            request_id_var.reset(token)


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes`` (declared or streamed)."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_bytes
            except ValueError:
                too_large = False
            if too_large:
                await _send_payload_too_large(send, self.max_bytes)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # HTTPException passes through FastAPI's body parsing unchanged.
                    raise HTTPException(status_code=413, detail="payload_too_large")
            return message

        await self.app(scope, limited_receive, send)


async def _send_payload_too_large(send: Send, max_bytes: int) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "payload_too_large",
                "message": "The request body is too large.",
                "details": {"max_bytes": max_bytes},
                "request_id": request_id_var.get(),
            }
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
