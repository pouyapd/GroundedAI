"""Browser end-to-end check of the online demo (GroundedAI running in the browser).

    python scripts/e2e_demo.py                      # the local build in frontend/dist
    python scripts/e2e_demo.py --url https://pouyapd.github.io/GroundedAI/
    python scripts/e2e_demo.py --llm --profile .cache/demo-browser --channel chrome   # + model

Without ``--url`` it serves ``frontend/dist`` (``npm run build:browser`` after
``python scripts/build_browser_runtime.py build``) under ``/GroundedAI/``, as GitHub Pages
does. It waits for the in-browser backend to start and index the two sample PDFs, asks the
sample questions, checks the answer, the quote and its page, a refusal, uploads a new PDF,
rejects a damaged one, resets the demo, and checks the phone layout.

``--llm`` also loads Qwen2.5-1.5B-Instruct with WebLLM (about 880 MB on the first run; give
``--profile`` a folder to keep the browser cache between runs) and asks a question with it.
WebGPU must be available to Chromium; the script starts it with WebGPU enabled.

Needs ``playwright`` with Chromium (not a project dependency). Exit status 0 means every
check passed.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.evaluation.pdf_fixture import make_pdf  # noqa: E402

BOOT_TIMEOUT_MS = 240_000
ANSWER_TIMEOUT_MS = 60_000
MODEL_TIMEOUT_MS = 3_600_000
LLM_ANSWER_TIMEOUT_MS = 600_000
# Headless Chromium needs these to expose a WebGPU adapter (on Windows through Direct3D 11).
WEBGPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU"] + (
    ["--use-angle=d3d11"] if sys.platform == "win32" else []
)
UPLOAD_TEXT = (
    "Kestrel Bay Kayak Club - Member Notes\n\n"
    "Kayak hire\nA single kayak costs 18 euros for two hours. "
    "Life jackets are included in the price."
)


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.results.append((name, condition, detail))
        print(
            f"{'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}",
            flush=True,
        )

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.results if not ok)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


def _serve(dist: Path) -> tuple[http.server.ThreadingHTTPServer, str, Path]:
    site = Path(tempfile.mkdtemp(prefix="groundedai-demo-site-"))
    shutil.copytree(dist, site / "GroundedAI")
    handler = functools.partial(_QuietHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/GroundedAI/", site


def _ask_example(page: Any, label: str, question: str, timeout: int = ANSWER_TIMEOUT_MS) -> str:
    chip = page.locator("button.chip-example", has_text=question).filter(has_text=label).first
    chip.click()
    page.wait_for_selector("text=Searched", timeout=timeout)
    page.wait_for_selector("h2:has-text('Answer'), h3:has-text('Answer')", timeout=timeout)
    return str(page.inner_text("main"))


def _answer_card(page: Any) -> str:
    return str(
        page.locator(".card", has=page.locator("text=Passages used as evidence")).first.inner_text()
    )


def run(page: Any, url: str, checks: Checks, shots: Path | None, work: Path) -> None:
    console: list[str] = []
    page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: console.append(str(e)))
    started = time.monotonic()
    page.goto(url)
    page.wait_for_selector("text=Live demo", timeout=BOOT_TIMEOUT_MS)
    checks.check("the demo explains that it runs in the browser", page.is_visible(".demo-panel"))
    page.wait_for_selector("button.chip-example", timeout=BOOT_TIMEOUT_MS)
    checks.check(
        "the in-browser backend starts and indexes the two sample PDFs",
        page.locator("#scope option").count() == 3,
        f"{time.monotonic() - started:.0f} s",
    )
    if shots:
        page.screenshot(path=str(shots / "demo-start.png"))

    text = _ask_example(page, "Ferry guide", "How much does a single adult ticket to Mirrow cost?")
    card = _answer_card(page)
    checks.check("a sample question is answered from the PDF", "14 euros" in card)
    checks.check("the quote names its page", "p. 2" in card)
    checks.check(
        "every statement is marked as found", "Every statement found in the sources" in text
    )
    checks.check("the answer says which generator wrote it", "not an LLM" in card)
    checks.check("the quote does not start with the section heading", "“Tickets" not in card)
    if shots:
        page.screenshot(path=str(shots / "demo-answer.png"), full_page=True)
        page.locator(".card", has=page.locator("text=Passages used as evidence")).first.screenshot(
            path=str(shots / "demo-answer-card.png")
        )

    _ask_example(page, "Ferry guide", "How many items can members borrow?")
    card = str(page.locator("main").inner_text())
    checks.check(
        "a question the selected PDF cannot answer is refused",
        "do not contain enough evidence" in card,
    )
    if shots:
        page.screenshot(path=str(shots / "demo-not-enough-evidence.png"), full_page=True)
    _ask_example(page, "Library rules", "How many items can members borrow?")
    checks.check("the other PDF answers it", "12 items" in _answer_card(page))

    page.click("button.tab:has-text('Documents')")
    page.wait_for_selector("table >> text=Harbourline Ferries Passenger Guide")
    damaged = work / "damaged.pdf"
    damaged.write_bytes(b"%PDF-1.4\nthis is not a real PDF")
    page.set_input_files("input[type=file]", str(damaged))
    page.click("button[type=submit]:has-text('Upload')")
    page.wait_for_selector("text=could not be read", timeout=ANSWER_TIMEOUT_MS)
    checks.check("a damaged PDF is refused with a reason", True)
    new_pdf = work / "kayak-club.pdf"
    new_pdf.write_bytes(make_pdf([UPLOAD_TEXT], title="Kestrel Bay Kayak Club - Member Notes"))
    page.set_input_files("input[type=file]", str(new_pdf))
    page.click("button[type=submit]:has-text('Upload')")
    page.wait_for_selector(
        "tr:has-text('Kestrel Bay Kayak Club') >> text=Ready", timeout=ANSWER_TIMEOUT_MS
    )
    checks.check("an uploaded PDF is indexed in the browser", True)
    page.locator("tr", has_text="Kestrel Bay Kayak Club").locator("button:has-text('Ask')").click()
    page.fill("#question", "How much does a single kayak cost?")
    page.click("button[type=submit]:has-text('Ask question')")
    page.wait_for_selector("text=Passages used as evidence", timeout=ANSWER_TIMEOUT_MS)
    checks.check("the uploaded PDF answers a question", "18 euros" in _answer_card(page))

    page.click("button:has-text('Reset demo')")
    page.wait_for_selector("button.chip-example", timeout=BOOT_TIMEOUT_MS)
    page.wait_for_function(
        "document.querySelectorAll('#scope option').length === 3", timeout=BOOT_TIMEOUT_MS
    )
    checks.check(
        "reset leaves only the two sample PDFs and no answer",
        "Kestrel Bay" not in page.inner_text("#scope")
        and page.locator("text=Passages used as evidence").count() == 0,
    )

    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    checks.check("no horizontal scrolling on a phone-sized screen", overflow <= 0, f"{overflow}px")
    _ask_example(page, "Ferry guide", "How much does it cost to take a bicycle?")
    checks.check("questions work on a phone-sized screen", "6 euros" in _answer_card(page))
    if shots:
        page.screenshot(path=str(shots / "demo-mobile.png"), full_page=True)
    page.set_viewport_size({"width": 1280, "height": 900})

    unexpected = [m for m in console if "WebGPU" not in m and "adapter" not in m.lower()]
    checks.check(
        "no unexpected browser console errors", not unexpected, "; ".join(unexpected)[:300]
    )


def run_llm(page: Any, checks: Checks, shots: Path | None) -> None:
    option = page.locator("label.option-card", has_text="on your GPU")
    if option.locator("input").is_disabled():
        checks.check("WebGPU is available for the in-page model", False, option.inner_text()[-120:])
        return
    started = time.monotonic()
    option.locator("input").click()  # selected only once the model has loaded
    selected = (
        "document.querySelector('label.option-card.is-selected')"
        "?.textContent.includes('on your GPU')"
    )
    deadline = time.monotonic() + MODEL_TIMEOUT_MS / 1000
    while not page.evaluate(selected):
        if time.monotonic() > deadline or page.locator("text=could not be loaded").count():
            checks.check(
                "the language model loads in the browser",
                False,
                page.inner_text(".demo-panel")[-200:],
            )
            return
        progress = page.locator(".model-progress")
        print(f"  model: {progress.inner_text() if progress.count() else 'starting'}", flush=True)
        page.wait_for_timeout(30_000)
    checks.check(
        "the language model loads in the browser", True, f"{time.monotonic() - started:.0f} s"
    )
    started = time.monotonic()
    _ask_example(
        page,
        "Ferry guide",
        "How much does a single adult ticket to Mirrow cost?",
        LLM_ANSWER_TIMEOUT_MS,
    )
    card = _answer_card(page)
    checks.check(
        "the model's answer is checked against the PDF",
        "14" in card and "p. 2" in card and "webllm" in card,
        f"{time.monotonic() - started:.0f} s",
    )
    if shots:
        page.screenshot(path=str(shots / "demo-llm-answer.png"), full_page=True)
        page.locator(".card", has=page.locator("text=Passages used as evidence")).first.screenshot(
            path=str(shots / "demo-llm-answer-card.png")
        )
    _ask_example(
        page, "Ferry guide", "Do young children have to pay for the ferry?", LLM_ANSWER_TIMEOUT_MS
    )
    card = _answer_card(page)
    verdicts = [
        v for v in ("Every statement found", "Some statements", "No statement") if v in card
    ]
    checks.check(
        "a paraphrased question gets an answer with a verdict", bool(verdicts), ", ".join(verdicts)
    )
    if shots:
        page.screenshot(path=str(shots / "demo-llm-paraphrase.png"), full_page=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", help="test this deployed demo instead of frontend/dist")
    parser.add_argument("--llm", action="store_true", help="also load and use the WebGPU model")
    parser.add_argument(
        "--profile", type=Path, help="browser profile folder (keeps the model cache)"
    )
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--channel",
        help="browser build, e.g. 'chrome' (Playwright's Chromium lacks the Direct3D shader "
        "compiler that WebGPU needs on Windows)",
    )
    args = parser.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("This check needs playwright: pip install playwright && playwright install chromium")
        return 2
    server = site = None
    url = args.url
    if not url:
        dist = ROOT / "frontend" / "dist"
        if not (dist / "runtime" / "manifest.json").is_file():
            print(
                "Build the demo first: python scripts/build_browser_runtime.py build && "
                "cd frontend && npm run build:browser"
            )
            return 2
        server, url, site = _serve(dist)
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    work = Path(tempfile.mkdtemp(prefix="groundedai-demo-e2e-"))
    try:
        with sync_playwright() as playwright:
            options: dict[str, Any] = {
                "headless": not args.headed,
                "args": WEBGPU_ARGS if args.llm else [],
            }
            if args.channel:
                options["channel"] = args.channel
            if args.profile:
                context = playwright.chromium.launch_persistent_context(
                    str(args.profile), viewport={"width": 1280, "height": 900}, **options
                )
                page = context.pages[0] if context.pages else context.new_page()
            else:
                browser = playwright.chromium.launch(**options)
                context = browser.new_context(viewport={"width": 1280, "height": 900})
                page = context.new_page()
            run(page, url, checks, args.screenshots, work)
            if args.llm:
                run_llm(page, checks, args.screenshots)
            context.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if server is not None:
            server.shutdown()
        if site is not None:
            shutil.rmtree(site, ignore_errors=True)
    print(f"\n{len(checks.results) - checks.failed} of {len(checks.results)} checks passed")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
