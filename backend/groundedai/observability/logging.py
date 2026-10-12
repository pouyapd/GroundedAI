"""Structured logging with request correlation ids."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else was passed via ``extra=``.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None)).keys()
    | {"message", "asctime", "taskName"}
)
_REDACTED_KEYS = frozenset({"api_key", "authorization", "password", "secret", "token"})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = "[redacted]" if key.lower() in _REDACTED_KEYS else value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        }
        request_id = request_id_var.get()
        prefix = f"[{request_id}] " if request_id else ""
        suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        return prefix + base + suffix


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_format else TextFormatter("%(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, "_groundedai", False):
            root.removeHandler(existing)
    handler._groundedai = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Uvicorn's own access log duplicates ours; keep its error log.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
