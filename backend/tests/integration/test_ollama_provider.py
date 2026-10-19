"""The Ollama provider against a fake Ollama HTTP server.

This exercises the real LangChain ``ChatOllama`` client, the ``ollama`` HTTP
client and GroundedAI's error mapping. The server returns canned text, so these
tests verify the integration, not model quality.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

from groundedai.core.errors import ProviderUnavailable
from groundedai.generation.providers import check_generation_provider, ollama_model_info
from groundedai.schemas.query import QueryRequest
from groundedai.services.container import build_services
from tests.conftest import TRAVEL_TEXT

MODEL = "qwen2.5:1.5b"


class FakeOllama(BaseHTTPRequestHandler):
    models: ClassVar[list[str]] = [MODEL]
    reply: ClassVar[str] = "The hotel limit is EUR 140 per night in capital cities [S1]."
    requests: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *args: Any) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            models = [
                {
                    "name": name,
                    "model": name,
                    "digest": "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
                    "details": {"quantization_level": "Q4_K_M"},
                }
                for name in self.models
            ]
            self._json(200, {"models": models})
        elif self.path == "/api/version":
            self._json(200, {"version": "0.40.3"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path != "/api/chat":
            self._json(404, {"error": "not found"})
            return
        FakeOllama.requests.append(body)
        if body.get("model") not in self.models:
            self._json(404, {"error": f"model '{body.get('model')}' not found"})
            return
        stamp = "2026-10-11T12:00:00Z"
        final = {
            "model": body["model"],
            "created_at": stamp,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 50,
            "eval_count": 12,
        }
        if body.get("stream", True):
            chunk = {
                "model": body["model"],
                "created_at": stamp,
                "message": {"role": "assistant", "content": self.reply},
                "done": False,
            }
            payload = (json.dumps(chunk) + "\n" + json.dumps(final) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            final["message"]["content"] = self.reply  # type: ignore[index]
            self._json(200, final)


@pytest.fixture
def fake_ollama() -> Iterator[str]:
    FakeOllama.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_answer_through_chat_ollama(make_settings, fake_ollama: str) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings(
        generation_provider="ollama", ollama_base_url=fake_ollama, max_new_tokens=128
    )
    services = build_services(settings)
    try:
        services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
        response = services.qa.answer(QueryRequest(question="hotel limit in capital cities"))
    finally:
        services.close()
    assert response.status == "answered"
    assert response.provider.name == "ollama"
    assert response.provider.is_llm is True
    assert response.citation_check.valid == ["S1"]
    assert response.usage.prompt_tokens == 50
    assert response.usage.completion_tokens == 12
    sent = FakeOllama.requests[-1]
    assert sent["model"] == MODEL
    assert sent["options"]["temperature"] == 0.0
    assert sent["options"]["seed"] == 0
    assert sent["options"]["num_ctx"] == 4096
    assert sent["options"]["num_predict"] == 128
    roles = [message["role"] for message in sent["messages"]]
    assert roles == ["system", "user"]  # policy and retrieved data are separate messages
    assert "untrusted data" in sent["messages"][1]["content"]


def test_missing_model_is_reported_not_replaced(make_settings, fake_ollama: str) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings(
        generation_provider="ollama", ollama_base_url=fake_ollama, ollama_model="not-pulled:1b"
    )
    services = build_services(settings)
    try:
        services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
        with pytest.raises(ProviderUnavailable, match="ollama pull not-pulled:1b"):
            services.qa.answer(QueryRequest(question="hotel limit in capital cities"))
    finally:
        services.close()


def test_unreachable_server_is_reported(make_settings) -> None:  # type: ignore[no-untyped-def]
    settings = make_settings(generation_provider="ollama", ollama_base_url=_closed_port_url())
    services = build_services(settings)
    try:
        services.ingestion.ingest_bytes("travel.txt", None, TRAVEL_TEXT.encode())
        with pytest.raises(ProviderUnavailable, match="Cannot reach Ollama"):
            services.qa.answer(QueryRequest(question="hotel limit in capital cities"))
    finally:
        services.close()


def test_status_check(make_settings, fake_ollama: str) -> None:  # type: ignore[no-untyped-def]
    ok = check_generation_provider(
        make_settings(generation_provider="ollama", ollama_base_url=fake_ollama)
    )
    assert ok.available
    assert ok.is_llm
    missing = check_generation_provider(
        make_settings(
            generation_provider="ollama", ollama_base_url=fake_ollama, ollama_model="x:1b"
        )
    )
    assert not missing.available
    assert "ollama pull x:1b" in missing.detail
    down = check_generation_provider(
        make_settings(generation_provider="ollama", ollama_base_url=_closed_port_url())
    )
    assert not down.available


def test_model_digest_and_server_version_are_read_for_reports(fake_ollama: str) -> None:
    info = ollama_model_info(fake_ollama, MODEL)
    assert info == {"digest": "65ec06548149", "quantization": "Q4_K_M", "server_version": "0.40.3"}
    assert ollama_model_info(fake_ollama, "missing-model") is None
    assert ollama_model_info(_closed_port_url(), MODEL) is None
