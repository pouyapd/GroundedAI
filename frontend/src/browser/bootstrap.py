# Runs inside the demo's Web Worker (Pyodide). It starts the unchanged GroundedAI application
# and exposes four calls to JavaScript. See backend/groundedai/browser.py.
import json
import warnings

import js
from pyodide.ffi import to_js

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from groundedai.webassembly import patch_for_webassembly  # noqa: E402

patch_for_webassembly()

from groundedai.browser import BrowserDemo  # noqa: E402

demo = BrowserDemo()
await demo.start()  # noqa: F704 - top-level await is allowed in Pyodide's runPythonAsync


def _missing(value):
    return value is None or type(value).__name__ in ("JsNull", "JsUndefined")


def _response(status, headers, content):
    return to_js(
        {"status": status, "headers": [list(pair) for pair in headers], "body": content},
        dict_converter=js.Object.fromEntries,
    )


async def handle_fetch(method, path, headers_json, body):
    raw = None if _missing(body) else bytes(body.to_py())
    status, headers, content = await demo.request(method, path, json.loads(headers_json), raw)
    return _response(status, headers, content)


async def handle_prepare(body):
    return to_js(await demo.prepare(body.encode()), dict_converter=js.Object.fromEntries)


async def handle_answer(body, reply):
    status, headers, content = await demo.answer(body.encode(), None if _missing(reply) else str(reply))
    return _response(status, headers, content)


def handle_select(generator, model):
    demo.select_generator(generator, None if _missing(model) else str(model))


to_js({"fetch": handle_fetch, "prepare": handle_prepare, "answer": handle_answer, "select": handle_select},
      dict_converter=js.Object.fromEntries)
