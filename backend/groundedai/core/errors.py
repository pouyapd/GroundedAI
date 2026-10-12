"""Application error types.

Each error carries a stable machine-readable ``code`` and an HTTP status.
Messages are written for API clients: they never contain stack traces,
filesystem paths or secrets.
"""

from __future__ import annotations

from typing import Any


class GroundedAIError(Exception):
    code: str = "internal_error"
    status_code: int = 500
    default_message: str = "An internal error occurred."

    def __init__(self, message: str | None = None, *, details: dict[str, Any] | None = None):
        self.message = message or self.default_message
        self.details = details or {}
        super().__init__(self.message)


class InvalidInput(GroundedAIError):
    code = "invalid_input"
    status_code = 422
    default_message = "The request is invalid."


class UnsupportedFileType(GroundedAIError):
    code = "unsupported_file_type"
    status_code = 415
    default_message = "Only PDF, plain-text and Markdown files are supported."


class FileTooLarge(GroundedAIError):
    code = "file_too_large"
    status_code = 413
    default_message = "The uploaded file exceeds the size limit."


class DocumentParseError(GroundedAIError):
    code = "document_parse_failed"
    status_code = 422
    default_message = "The document could not be parsed."


class EmptyDocument(GroundedAIError):
    code = "empty_document"
    status_code = 422
    default_message = "No extractable text was found in the document."


class DuplicateDocument(GroundedAIError):
    code = "duplicate_document"
    status_code = 409
    default_message = "An identical document is already indexed."


class NotFound(GroundedAIError):
    code = "not_found"
    status_code = 404
    default_message = "The requested resource was not found."


class Conflict(GroundedAIError):
    code = "conflict"
    status_code = 409
    default_message = "The resource is in a state that does not allow this operation."


class ProviderUnavailable(GroundedAIError):
    code = "model_unavailable"
    status_code = 503
    default_message = "The configured model provider is not available."


class ProviderTimeout(GroundedAIError):
    code = "model_timeout"
    status_code = 504
    default_message = "The model provider did not respond in time."


class ServerBusy(GroundedAIError):
    code = "server_busy"
    status_code = 503
    default_message = "The server is busy with other generation requests. Try again shortly."


class RateLimited(GroundedAIError):
    code = "rate_limited"
    status_code = 429
    default_message = "Too many requests. Slow down and retry later."

    def __init__(self, retry_after_s: int, message: str | None = None):
        super().__init__(message, details={"retry_after_s": retry_after_s})
        self.retry_after_s = retry_after_s


class Unauthorized(GroundedAIError):
    code = "unauthorized"
    status_code = 401
    default_message = "A valid API key is required for this endpoint."


class PayloadTooLarge(GroundedAIError):
    code = "payload_too_large"
    status_code = 413
    default_message = "The request body is too large."


class GenerationFailed(GroundedAIError):
    code = "generation_failed"
    status_code = 502
    default_message = "The model provider returned an error."
