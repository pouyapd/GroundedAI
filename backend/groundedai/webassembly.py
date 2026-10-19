"""Run thread-pool work inline when GroundedAI runs as WebAssembly (Pyodide).

WebAssembly Python cannot start threads. FastAPI runs synchronous endpoints, background tasks
and file reads in a thread pool, and the QA service and LangChain use thread-pool executors.
:func:`patch_for_webassembly` makes all of them run their work in the calling thread.

Call it before anything imports FastAPI, Starlette or LangChain: LangChain subclasses
``ThreadPoolExecutor`` when it is imported. This module imports only the standard library at
the top for that reason. On a normal Python it does nothing unless forced.
"""

from __future__ import annotations

import concurrent.futures
import sys
from collections.abc import Callable
from typing import Any


class InlineExecutor(concurrent.futures.Executor):
    """Runs submitted work at once, in the caller."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # delivered through the future
            future.set_exception(exc)
        return future


async def run_inline(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


async def run_sync_inline(func: Callable[..., Any], *args: Any, **_options: Any) -> Any:
    """Stands in for ``anyio.to_thread.run_sync`` (its keyword options do not apply)."""
    return func(*args)


_THREADPOOL_USERS = (
    "starlette.background",
    "starlette.datastructures",
    "starlette.routing",
    "fastapi.concurrency",
    "fastapi.dependencies.utils",
    "fastapi.routing",
)


def patch_for_webassembly(force: bool = False) -> bool:
    """Replace thread pools with inline execution; returns whether anything was patched."""
    if sys.platform != "emscripten" and not force:
        return False
    concurrent.futures.ThreadPoolExecutor = InlineExecutor  # type: ignore[misc,assignment]
    import importlib

    import anyio.to_thread
    import starlette.concurrency

    # Load every module that binds ``run_in_threadpool`` by name, then rebind it.
    for name in _THREADPOOL_USERS:
        importlib.import_module(name)
    anyio.to_thread.run_sync = run_sync_inline  # type: ignore[assignment]
    original = starlette.concurrency.run_in_threadpool
    for module in list(sys.modules.values()):
        if getattr(module, "run_in_threadpool", None) is original:
            module.run_in_threadpool = run_inline  # type: ignore[attr-defined]
    return True
