"""GroundedAI inside a web browser: the runtime of the online demo.

The online demo runs this package, unchanged, in the visitor's browser: Pyodide (CPython
compiled to WebAssembly) executes it in a Web Worker, and the React UI sends its HTTP requests
to :class:`BrowserDemo` instead of a server. Everything a server would do happens in the page:
upload validation, PDF text extraction, chunking, indexing in SQLite, hybrid retrieval, the
prompt, generation, the citation, claim and disagreement checks. Nothing leaves the browser.

Three things differ from a server, and all of them are handled here:

* **No threads.** WebAssembly Python cannot start threads, so the generation executor and
  FastAPI's thread pool run their work inline (:mod:`groundedai.webassembly`, which must run
  before this module is imported). The generation timeout therefore does not apply here.
* **No HTTP server.** Requests arrive from JavaScript as method, path, headers and body, and go
  through the same ASGI application with ``httpx.ASGITransport``.
* **A language model in the page.** The optional model (WebLLM on WebGPU) is asynchronous
  JavaScript, and the generation chain is synchronous Python. The page therefore asks in two
  steps: :meth:`BrowserDemo.prepare` runs the normal query once and records the exact
  messages the application sends to its model; the page generates a reply from those
  messages; :meth:`BrowserDemo.answer` runs the normal query again and hands the model's reply
  to the unchanged pipeline, which checks it like any other reply. The second run must send
  the same messages as the first, or the answer is refused.

Server deployments never import this module.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from groundedai.core.config import Settings
from groundedai.generation.providers import (
    GenerationProvider,
    ProviderStatus,
    build_generation_provider,
)
from groundedai.main import create_app
from groundedai.services.container import Services, build_services

DEFAULT_PAGE_MODEL = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
PAGE_PROVIDER = "webllm"
BASE_URL = "http://groundedai.browser"
_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


class PageModelReply(BaseChatModel):
    """Stands in for the model that runs in the page: records its input, returns its reply."""

    reply: str | None = None
    captured: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "browser-page-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured.append(list(messages))
        reply = AIMessage(content=self.reply or "")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def as_chat_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """LangChain messages as OpenAI-style chat messages (what WebLLM takes)."""
    return [{"role": _ROLES.get(m.type, "user"), "content": str(m.content)} for m in messages]


class BrowserDemo:
    """The FastAPI application with its services, driven by requests from JavaScript."""

    def __init__(
        self,
        data_dir: str | Path = "/data",
        page_model: str = DEFAULT_PAGE_MODEL,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings(
            _env_file=None,
            data_dir=Path(data_dir),
            generation_provider="extractive-demo",
        )
        self.services: Services = build_services(self.settings)
        self.extractive = build_generation_provider(self.settings)
        self.page_reply = PageModelReply()
        self.page_provider = GenerationProvider(
            name=PAGE_PROVIDER, model=page_model, is_llm=True, chat_model=self.page_reply
        )
        self.selected = "extractive"
        self._use_page_model = False
        self._prepared: dict[str, list[dict[str, str]]] = {}
        # The QA service asks for its provider on every question; the status service for the
        # configured one. Both are pointed at the generator chosen in the page.
        self.services.qa._generation_provider = self._provider
        self.services.status._generation_check = self._generation_status
        self.services.status._generation_status = None
        self.app = create_app(self.settings, services=self.services)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=BASE_URL, timeout=600.0
        )
        self._lifespan: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Run the application's startup (database migrations, recovery) and keep it open."""
        started = asyncio.Event()
        failure: list[str] = []
        sent_startup = False

        async def receive() -> dict[str, str]:
            nonlocal sent_startup
            if not sent_startup:
                sent_startup = True
                return {"type": "lifespan.startup"}
            await self._stop.wait()
            return {"type": "lifespan.shutdown"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "lifespan.startup.failed":
                failure.append(str(message.get("message", "startup failed")))
            if message["type"].startswith("lifespan.startup"):
                started.set()

        scope = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}
        self._lifespan = asyncio.ensure_future(self.app(scope, receive, send))  # type: ignore[arg-type]
        await started.wait()
        if failure:
            raise RuntimeError(failure[0])

    async def stop(self) -> None:
        self._stop.set()
        if self._lifespan is not None:
            await self._lifespan
        await self.client.aclose()

    # -------------------------------------------------------------------- requests
    async def request(
        self, method: str, path: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        """One HTTP request through the application, as a server would receive it."""
        response = await self.client.request(method, path, headers=headers, content=body)
        return response.status_code, list(response.headers.multi_items()), response.content

    def select_generator(self, name: str, model: str | None = None) -> None:
        """``extractive`` (the default demo generator) or ``webllm`` (a model in the page)."""
        if name not in ("extractive", PAGE_PROVIDER):
            raise ValueError(f"unknown generator: {name}")
        if model:
            self.page_provider.model = model
        self.selected = name
        self.services.status._generation_status = None

    async def prepare(self, body: bytes) -> dict[str, Any]:
        """Run the query once and return the messages the model would receive.

        ``generate`` is false when the application answers without a model (for example when
        no passage is retrieved); the page then calls :meth:`answer` without a reply.
        """
        self.page_reply.reply = None
        self.page_reply.captured.clear()
        recording, self.services.qa.record_events = self.services.qa.record_events, False
        try:
            status, _, _ = await self._query(body)
        finally:
            self.services.qa.record_events = recording
        if not self.page_reply.captured:
            self._prepared.pop(_key(body), None)
            return {"generate": False, "status": status}
        messages = as_chat_messages(self.page_reply.captured[-1])
        self._prepared[_key(body)] = messages
        return {
            "generate": True,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_new_tokens,
            "seed": 0,
        }

    async def answer(
        self, body: bytes, reply: str | None
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        """Run the query again with the page model's reply and return the checked answer."""
        self.page_reply.reply = reply
        self.page_reply.captured.clear()
        status, headers, content = await self._query(body)
        expected = self._prepared.pop(_key(body), None)
        if self.page_reply.captured and (
            expected is None or as_chat_messages(self.page_reply.captured[-1]) != expected
        ):
            error = {
                "error": {
                    "code": "context_changed",
                    "message": "The documents changed while the model was answering. "
                    "Ask the question again.",
                    "details": {},
                }
            }
            return 409, [("content-type", "application/json")], json.dumps(error).encode()
        return status, headers, content

    async def _query(self, body: bytes) -> tuple[int, list[tuple[str, str]], bytes]:
        self._use_page_model = True
        try:
            return await self.request(
                "POST", "/api/v1/query", {"content-type": "application/json"}, body
            )
        finally:
            self._use_page_model = False

    def _provider(self) -> GenerationProvider:
        return self.page_provider if self._use_page_model else self.extractive

    def _generation_status(self, settings: Settings) -> ProviderStatus:
        import time

        if self.selected == PAGE_PROVIDER:
            return ProviderStatus(
                name=PAGE_PROVIDER,
                model=self.page_provider.model,
                is_llm=True,
                available=True,
                detail="A language model running in this browser tab with WebGPU (WebLLM).",
                checked_at=time.time(),
            )
        return ProviderStatus(
            name=self.extractive.name,
            model=self.extractive.model,
            is_llm=False,
            available=True,
            detail="Deterministic extractive generator: copies sentences from the passages.",
            checked_at=time.time(),
        )


def _key(body: bytes) -> str:
    return body.decode("utf-8", "replace")
