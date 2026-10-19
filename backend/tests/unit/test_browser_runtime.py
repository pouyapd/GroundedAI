"""groundedai.browser: the runtime of the online demo, exercised with a normal Python.

The same code runs in the browser under Pyodide; there only ``patch_for_webassembly`` differs.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from groundedai.browser import (
    PAGE_PROVIDER,
    BrowserDemo,
    PageModelReply,
    as_chat_messages,
)
from groundedai.webassembly import InlineExecutor, patch_for_webassembly
from tests.conftest import REPO_ROOT

FERRY = REPO_ROOT / "examples" / "harbourline-ferry-guide.pdf"
LIBRARY = REPO_ROOT / "examples" / "orchard-library-rules.pdf"
TICKET = "How much does a single adult ticket to Mirrow cost?"


def _multipart(path: Path) -> tuple[dict[str, str], bytes]:
    boundary = "groundedai-test-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return {"content-type": f"multipart/form-data; boundary={boundary}"}, body


def _run(tmp_path: Path, scenario: Callable[[BrowserDemo], Awaitable[None]]) -> None:
    async def main() -> None:
        demo = BrowserDemo(data_dir=tmp_path / "data")
        await demo.start()
        try:
            await scenario(demo)
        finally:
            await demo.stop()

    asyncio.run(main())


async def _upload(demo: BrowserDemo, path: Path) -> str:
    headers, body = _multipart(path)
    status, _, content = await demo.request("POST", "/api/v1/documents", headers, body)
    assert status == 202, content
    return str(json.loads(content)["id"])


def _query(document_id: str, question: str = TICKET) -> bytes:
    return json.dumps({"question": question, "document_ids": [document_id]}).encode()


def test_the_extractive_demo_answers_with_the_quote_and_page(tmp_path: Path) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        ferry = await _upload(demo, FERRY)
        status, _, content = await demo.request(
            "POST", "/api/v1/query", {"content-type": "application/json"}, _query(ferry)
        )
        answer = json.loads(content)
        assert status == 200
        assert answer["status"] == "answered"
        assert answer["provider"]["name"] == "extractive-demo"
        assert "14 euros" in answer["answer"]
        evidence = answer["claims"][0]["evidence"][0]
        assert "14 euros" in evidence["quote"]
        assert evidence["page"] == 2

    _run(tmp_path, scenario)


def test_a_page_model_reply_is_checked_like_any_other(tmp_path: Path) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        ferry = await _upload(demo, FERRY)
        demo.select_generator(PAGE_PROVIDER)
        prepared = await demo.prepare(_query(ferry))
        assert prepared["generate"] is True
        roles = [message["role"] for message in prepared["messages"]]
        assert roles[0] == "system"
        assert roles[-1] == "user"
        assert "14 euros" in prepared["messages"][-1]["content"]
        assert prepared["max_tokens"] == demo.settings.max_new_tokens

        reply = "A single adult ticket from Kessport to Mirrow costs 14 euros [S1]."
        status, _, content = await demo.answer(_query(ferry), reply)
        answer = json.loads(content)
        assert status == 200
        assert answer["provider"]["name"] == PAGE_PROVIDER
        assert answer["provider"]["model"] == demo.page_provider.model
        assert answer["provider"]["is_llm"] is True
        assert answer["verification"]["verdict"] == "verified"
        assert answer["claims"][0]["evidence"][0]["page"] == 2

    _run(tmp_path, scenario)


def test_an_invented_figure_from_the_page_model_is_not_found(tmp_path: Path) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        ferry = await _upload(demo, FERRY)
        await demo.prepare(_query(ferry))
        reply = "A single adult ticket from Kessport to Mirrow costs 20 euros [S1]."
        _, _, content = await demo.answer(_query(ferry), reply)
        answer = json.loads(content)
        assert answer["verification"]["verdict"] != "verified"
        assert answer["claims"][0]["status"] not in ("supported", "wrong_source", "added_source")

    _run(tmp_path, scenario)


def test_a_refusal_needs_no_page_model(tmp_path: Path) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        ferry = await _upload(demo, FERRY)
        await _upload(demo, LIBRARY)
        question = _query(ferry, "How many items can members borrow?")
        prepared = await demo.prepare(question)
        if prepared["generate"]:
            # The model is asked; a refusal from it is recognised as one.
            _, _, content = await demo.answer(question, "INSUFFICIENT_EVIDENCE")
        else:
            _, _, content = await demo.answer(question, None)
        assert json.loads(content)["status"] == "insufficient_evidence"

    _run(tmp_path, scenario)


def test_an_answer_is_refused_when_the_passages_changed_between_the_two_steps(
    tmp_path: Path,
) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        await _upload(demo, FERRY)
        question = json.dumps({"question": TICKET}).encode()  # all documents
        prepared = await demo.prepare(question)
        assert prepared["generate"] is True
        # Each sample is one chunk and five passages are retrieved: the new one is among them.
        await _upload(demo, LIBRARY)
        status, _, content = await demo.answer(question, "It costs 14 euros [S1].")
        assert status == 409
        assert json.loads(content)["error"]["code"] == "context_changed"

    _run(tmp_path, scenario)


def test_an_answer_without_a_prepared_step_is_refused(tmp_path: Path) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        ferry = await _upload(demo, FERRY)
        status, _, content = await demo.answer(_query(ferry), "It costs 14 euros [S1].")
        assert status == 409
        assert json.loads(content)["error"]["code"] == "context_changed"

    _run(tmp_path, scenario)


def test_the_status_names_the_generator_chosen_in_the_page(tmp_path: Path) -> None:
    async def scenario(demo: BrowserDemo) -> None:
        async def generation() -> dict[str, Any]:
            _, _, content = await demo.request("GET", "/api/v1/system/status", {}, None)
            return dict(json.loads(content)["generation"])

        assert (await generation())["provider"] == "extractive-demo"
        demo.select_generator(PAGE_PROVIDER, "Qwen2.5-0.5B-Instruct-q4f16_1-MLC")
        after = await generation()
        assert after["provider"] == PAGE_PROVIDER
        assert after["model"] == "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"
        assert after["is_llm"] is True

    _run(tmp_path, scenario)


def test_the_page_model_records_messages_in_chat_format() -> None:
    model = PageModelReply(reply="ok")
    result = model.invoke([("system", "rules"), ("human", "question")])
    assert result.content == "ok"
    assert as_chat_messages(model.captured[0]) == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]


def test_threads_are_only_replaced_in_webassembly() -> None:
    assert patch_for_webassembly() is False


def test_the_inline_executor_runs_work_in_the_caller() -> None:
    with InlineExecutor() as executor:
        assert executor.submit(lambda x: x * 2, 21).result() == 42
        assert list(executor.map(str, [1, 2])) == ["1", "2"]
        failed = executor.submit(lambda: 1 / 0)
    assert isinstance(failed.exception(), ZeroDivisionError)
