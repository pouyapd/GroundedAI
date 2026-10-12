"""Exception handlers producing one JSON error shape.

``{"error": {"code", "message", "details", "request_id"}}`` - never a stack
trace, internal path or the rejected input itself.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from groundedai.core.errors import GroundedAIError, RateLimited
from groundedai.observability.logging import request_id_var

logger = logging.getLogger("groundedai.errors")

_HTTP_CODES = {
    400: ("bad_request", "The request could not be processed."),
    401: ("unauthorized", "Authentication is required."),
    403: ("forbidden", "This operation is not allowed."),
    404: ("not_found", "The requested resource was not found."),
    405: ("method_not_allowed", "This HTTP method is not allowed here."),
    413: ("payload_too_large", "The request body is too large."),
    415: ("unsupported_media_type", "Unsupported media type."),
}


def error_payload(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id_var.get(),
        }
    }


async def _groundedai_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, GroundedAIError):  # pragma: no cover - registration guarantees it
        raise exc
    headers = {}
    if isinstance(exc, RateLimited):
        headers["Retry-After"] = str(exc.retry_after_s)
    if exc.status_code >= 500:
        logger.warning("request_failed", extra={"error_code": exc.code})
    return JSONResponse(
        error_payload(exc.code, exc.message, exc.details),
        status_code=exc.status_code,
        headers=headers,
    )


async def _validation_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    errors = [
        {
            "loc": [str(part) for part in error.get("loc", ())],
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        error_payload("invalid_input", "The request is invalid.", {"errors": errors}),
        status_code=422,
    )


async def _http_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        raise exc
    code, message = _HTTP_CODES.get(exc.status_code, ("http_error", "The request failed."))
    return JSONResponse(
        error_payload(code, message),
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def _unhandled_error(_request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_error", exc_info=exc)
    return JSONResponse(
        error_payload("internal_error", "An internal error occurred."), status_code=500
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GroundedAIError, _groundedai_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)
