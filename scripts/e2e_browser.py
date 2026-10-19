"""Browser end-to-end check of the main user journey, against a real local server.

    python scripts/e2e_browser.py [--screenshots docs/screenshots] [--keep-data]

It starts ``groundedai serve`` with a fresh temporary data folder and the built UI
(``frontend/dist``; run ``npm run build`` first), then drives Chromium through the UI:
failure cases (unreadable PDF, PDF without text, unsupported type), uploading the two
sample PDFs, asking a question about one document, checking the answer's quote and page,
a question the selected document cannot answer, switching documents, repeated questions
and deleting a document. Every request goes through the real API, PDF parser, index and
checks. The server uses whatever the environment configures (default: hashing embedder and
the extractive demo generator, which is deterministic and not a language model), so the
expected answers below hold for the default configuration.

Needs the Python package ``playwright`` and its Chromium build. They are not project
dependencies (``pip install playwright && playwright install chromium``); CI does not run
this script. Exit status 0 means every check passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.evaluation.pdf_fixture import make_pdf  # noqa: E402
from groundedai.evaluation.sample_documents import FERRY_GUIDE, LIBRARY_RULES  # noqa: E402

TIMEOUT_MS = 20_000
# A local model on a laptop CPU can take a minute or more per answer (longer on battery).
ANSWER_TIMEOUT_MS = 300_000


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.results.append((name, condition, detail))
        print(f"{'PASS' if condition else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.results if not ok)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(url: str, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - local URL
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"server did not answer at {url}")


def _write_inputs(folder: Path) -> dict[str, Path]:
    files = {
        "ferry": folder / FERRY_GUIDE.filename,
        "library": folder / LIBRARY_RULES.filename,
        "corrupt": folder / "damaged.pdf",
        "no_text": folder / "scanned-without-text.pdf",
        "docx": folder / "notes.docx",
    }
    files["ferry"].write_bytes(FERRY_GUIDE.pdf())
    files["library"].write_bytes(LIBRARY_RULES.pdf())
    files["corrupt"].write_bytes(b"%PDF-1.7\n" + b"\x00 not a real pdf body " * 40)
    files["no_text"].write_bytes(make_pdf([""], title="Scan"))
    files["docx"].write_bytes(b"PK\x03\x04 a word document")
    return files


def run(page: Any, base: str, files: dict[str, Path], shots: Path | None) -> Checks:
    checks = Checks()
    errors: list[str] = []
    page.on(
        "console", lambda message: errors.append(message.text) if message.type == "error" else None
    )

    def shot(name: str) -> None:
        if shots:
            page.screenshot(path=str(shots / f"{name}.png"), full_page=True)

    def upload(path: Path) -> None:
        page.set_input_files("#file-input", str(path))
        page.get_by_role("button", name="Upload", exact=True).click()

    def row(title: str) -> Any:
        return page.locator("tbody tr", has_text=title)

    def ask(question: str) -> str:
        page.fill("#question", question)
        page.get_by_role("button", name="Ask question").click()
        # The previous answer is cleared on submit; wait for the new one to finish.
        page.get_by_text("Retrieving passages and generating an answer").wait_for(
            state="hidden", timeout=ANSWER_TIMEOUT_MS
        )
        page.get_by_text("Searched ", exact=False).first.wait_for()
        return str(page.locator("main").inner_text())

    page.goto(base)
    page.wait_for_selector("text=No document is ready to search yet", timeout=TIMEOUT_MS)
    checks.check(
        "empty state points to the upload page",
        page.get_by_role("link", name="Go to Documents").is_visible(),
    )
    page.fill("#question", "How much is a ticket?")
    checks.check(
        "asking is disabled without documents",
        page.get_by_role("button", name="Ask question").is_disabled(),
    )

    page.get_by_role("link", name="Go to Documents").click()
    page.wait_for_selector("text=Indexed documents")

    page.set_input_files("#file-input", str(files["docx"]))
    checks.check(
        "an unsupported file type is refused before upload",
        page.get_by_text("Unsupported file type").is_visible()
        and page.get_by_role("button", name="Upload", exact=True).is_disabled(),
    )

    upload(files["corrupt"])
    row("damaged").get_by_text("Failed").wait_for(timeout=TIMEOUT_MS)
    damaged = row("damaged").inner_text()
    checks.check(
        "a damaged PDF fails with a clear message",
        "could not be read" in damaged,
        damaged.splitlines()[1] if "\n" in damaged else damaged,
    )
    checks.check("the damaged file is not kept", "not kept" in damaged)
    checks.check(
        "re-ingest is disabled once the file is gone",
        row("damaged").get_by_role("button", name="Re-ingest").is_disabled(),
    )

    upload(files["no_text"])
    row("Scan").get_by_text("Failed").wait_for(timeout=TIMEOUT_MS)
    checks.check(
        "a PDF without text explains that OCR is not supported", "OCR" in row("Scan").inner_text()
    )

    upload(files["ferry"])
    row(FERRY_GUIDE.title).get_by_text("Ready").wait_for(timeout=TIMEOUT_MS)
    upload(files["library"])
    row(LIBRARY_RULES.title).get_by_text("Ready").wait_for(timeout=TIMEOUT_MS)
    ferry_cells = row(FERRY_GUIDE.title).locator("td").all_inner_texts()
    checks.check(
        "the PDF is indexed with its 3 pages",
        ferry_cells[2] == "3",
        f"pages cell {ferry_cells[2]!r}",
    )
    row(FERRY_GUIDE.title).get_by_role("button", name="Chunks").click()
    page.locator(".passage").first.wait_for()
    chunks = page.locator(".passage").all_inner_texts()
    checks.check(
        "the extracted text contains the PDF's facts",
        any("costs 14 euros" in text for text in chunks),
    )
    shot("documents")

    page.get_by_role("button", name=f"Ask about {FERRY_GUIDE.title}").click()
    page.wait_for_selector("#scope")
    selected = page.locator("#scope option:checked").inner_text()
    checks.check(
        "'Ask' opens the Ask page with that document selected",
        selected == FERRY_GUIDE.title,
        selected,
    )

    text = ask("How much does a single adult ticket to Mirrow cost?")
    checks.check("the answer gives the fact from the PDF", "14 euros" in text)
    evidence = page.locator(".evidence").first.inner_text()
    checks.check(
        "the evidence quote names the exact page",
        f"{FERRY_GUIDE.title} · p. 2" in evidence,
        evidence.splitlines()[0],
    )
    checks.check("the quote is the supporting sentence", "costs 14 euros" in evidence)
    checks.check(
        "the answer is marked as found in the sources",
        # claims-v1 says "Every statement found in the sources", claims-v2 "Every statement found".
        "Every statement found" in text,
    )
    label = page.locator(".evidence .cite").first.inner_text()
    page.locator(".evidence .cite").first.click()
    highlighted = page.locator(f"#source-{label}")
    checks.check(
        "the citation highlights its source passage",
        "highlight" in (highlighted.get_attribute("class") or "")
        and "costs 14 euros" in highlighted.inner_text(),
    )
    shot("answer")

    text = ask("How many items can members borrow?")
    checks.check(
        "a question the selected document cannot answer is not answered",
        "do not contain enough evidence" in text and "12 items" not in text,
    )
    shown = " ".join(page.locator(".source").all_inner_texts())
    checks.check("no passage of the other document is shown", LIBRARY_RULES.title not in shown)
    shot("insufficient-evidence")

    page.select_option("#scope", label=LIBRARY_RULES.title)
    checks.check(
        "changing the document is flagged against the shown answer",
        page.get_by_text("Your settings have changed").is_visible(),
    )
    text = ask("How many items can members borrow?")
    checks.check(
        "the other document answers it",
        "12 items" in text and "Searched “Orchard Lane Library Card Rules”" in text,
    )

    for question, expected in [
        ("How much do bicycles cost?", "6 euros"),
        ("Are dogs allowed on board?", "lead"),
        ("How much does a single adult ticket to Mirrow cost?", "14 euros"),
    ]:
        page.select_option("#scope", label=FERRY_GUIDE.title)
        text = ask(question)
        checks.check(
            f"repeated questions show the current answer ({expected})",
            expected in text and "12 items" not in text,
        )

    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Documents").click()
    row(LIBRARY_RULES.title).get_by_role("button", name="Delete").click()
    row(LIBRARY_RULES.title).wait_for(state="detached", timeout=TIMEOUT_MS)
    page.get_by_role("button", name="Ask", exact=True).click()
    page.select_option("#scope", "all")
    text = ask("How many items can members borrow?")
    checks.check("a deleted document no longer answers", "12 items" not in text)

    unexpected = [message for message in errors if "status of 4" not in message]
    checks.check(
        "no unexpected browser console errors", not unexpected, "; ".join(unexpected)[:200]
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--screenshots", type=Path, help="save full-page screenshots here")
    parser.add_argument("--keep-data", action="store_true", help="keep the temporary data folder")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("This check needs playwright: pip install playwright && playwright install chromium")
        return 2
    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        print("Build the UI first: cd frontend && npm run build")
        return 2

    work = Path(tempfile.mkdtemp(prefix="groundedai-e2e-"))
    files = _write_inputs(work)
    port = _free_port()
    env = {
        **os.environ,
        "GROUNDEDAI_DATA_DIR": str(work / "data"),
        "GROUNDEDAI_FRONTEND_DIST": str(dist),
    }
    env.setdefault("GROUNDEDAI_LOG_JSON", "false")
    server = subprocess.Popen(  # noqa: S603 - fixed arguments
        [sys.executable, "-m", "groundedai.cli", "serve", "--port", str(port)],
        cwd=ROOT / "backend",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for(f"{base}/health")
        if args.screenshots:
            args.screenshots.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not args.headed)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.set_default_timeout(TIMEOUT_MS)
            checks = run(page, base, files, args.screenshots)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
        if not args.keep_data:
            shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(checks.results) - checks.failed} of {len(checks.results)} checks passed")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
