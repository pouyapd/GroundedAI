"""Record a short animated preview of the live demo (real interface, real answer).

    python scripts/record_demo_preview.py [--url https://pouyapd.github.io/GroundedAI/]

Drives the online demo in Chromium (dark theme): the ferry guide is selected, the question is
typed and sent, and the answer appears with its claim check, quoted sentence and page. Every
frame is a screenshot of the page; nothing is drawn in afterwards. The first frame repeats
the final state, so the image tells the story before the animation plays.

Writes ``docs/images/demo-preview.gif`` and ``docs/images/demo-preview.png`` (the final
state, for viewers who prefer reduced motion). Needs ``playwright`` with Chromium.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
URL = "https://pouyapd.github.io/GroundedAI/"
QUESTION = "How much does a single adult ticket to Mirrow cost?"
DOCUMENT = "Harbourline Ferries Passenger Guide"
WIDTH = 960
PAGE_BACKGROUND = (13, 13, 13)  # --page in the dark theme


def record(url: str) -> tuple[list[Image.Image], list[int]]:
    from playwright.sync_api import sync_playwright

    shots: list[tuple[bytes, int]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 900}, color_scheme="dark")
        page.goto(url)
        page.wait_for_selector("button.chip-example", timeout=240_000)

        def shot(ms: int) -> None:
            page.evaluate("window.scrollTo(0, 0)")
            shots.append((page.screenshot(full_page=True), ms))

        shot(1200)
        page.select_option("#scope", label=DOCUMENT)
        shot(1000)
        typed = ""
        for part in ("How much does ", "a single adult ", "ticket to Mirrow ", "cost?"):
            typed += part
            page.fill("#question", typed)
            shot(260)
        page.wait_for_timeout(400)
        page.click("button[type=submit]:has-text('Ask question')")
        page.wait_for_selector(".evidence", timeout=60_000)
        page.evaluate("window.scrollTo(0, 0)")  # boxes are relative to the viewport
        page.wait_for_timeout(200)
        top = page.locator("form.card").bounding_box()
        bottom = page.locator(".evidence").first.bounding_box()
        if top is None or bottom is None:
            raise SystemExit("the answer did not appear")
        if "p. 2" not in page.locator(".evidence").first.inner_text():
            raise SystemExit("the quote does not name page 2; nothing was written")
        shot(4200)
        browser.close()
    box = (0, int(top["y"]) - 14, WIDTH, int(bottom["y"] + bottom["height"]) + 18)
    frames = []
    for png, _ in shots:
        image = Image.open(io.BytesIO(png)).convert("RGB")
        canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), PAGE_BACKGROUND)
        canvas.paste(image.crop((box[0], box[1], box[2], min(box[3], image.height))), (0, 0))
        frames.append(canvas)
    return frames, [ms for _, ms in shots]


def save(frames: list[Image.Image], durations: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    final = frames[-1]
    final.save(OUT / "demo-preview.png", optimize=True)
    sequence = [final, *frames]  # the first frame shows the result
    timing = [2600, *durations]
    palette = final.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in sequence]
    indexed[0].save(
        OUT / "demo-preview.gif",
        save_all=True,
        append_images=indexed[1:],
        duration=timing,
        loop=0,
        optimize=True,
        disposal=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    frames, durations = record(args.url)
    save(frames, durations)
    for name in ("demo-preview.gif", "demo-preview.png"):
        path = OUT / name
        size = Image.open(path).size
        print(f"{path.relative_to(ROOT).as_posix()}: {size}, {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
