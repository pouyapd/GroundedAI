"""Capture real answer cards from the live demo for the LinkedIn launch visual.

    python scripts/capture_launch_fragments.py --profile <browser profile folder> --out <folder>

Asks the ferry-guide question twice on https://pouyapd.github.io/GroundedAI/ in the dark theme:
once with the extractive generator and once with Qwen2.5-1.5B-Instruct running in the page
(WebLLM, WebGPU). For each answer it saves a screenshot of the answer card (twice the pixel
density) and the positions of its parts (answer text, warning, claims, quotes), so the visual
can point at them. It then widens the same page to 1280 pixels and saves the whole page
(``interface.png``), with the model's answer on it. The screenshots are not edited.
``--profile`` should be a Chrome profile that already holds the model (about 880 MB, cached
per origin); needs Chrome and ``playwright``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

URL = "https://pouyapd.github.io/GroundedAI/"
QUESTION = "How much does a single adult ticket to Mirrow cost?"
SCALE = 2
WEBGPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--use-angle=d3d11"]

PARTS = {
    "header": ".card-header",
    "answer": ".answer-text",
    "invalid_cite": ".cite-invalid",
    "banner": ".banner",
    "claims": ".claim",
    "badges": ".claim .claim-head",
    "evidence": ".evidence",
    "meta": ".answer-meta",
}


def ask(page: Any) -> None:
    chip = page.locator("button.chip-example", has_text=QUESTION).filter(has_text="Ferry guide")
    chip.first.click()


def card_shot(page: Any, out: Path, name: str) -> dict[str, Any]:
    page.evaluate("window.scrollTo(0, 0)")  # boxes are relative to the viewport
    page.wait_for_timeout(300)
    card = page.locator(".card", has=page.locator(".answer-text")).first
    box = card.bounding_box()
    if box is None:
        raise SystemExit("no answer card on the page")
    parts: dict[str, Any] = {}
    for key, selector in PARTS.items():
        found = []
        for i in range(card.locator(selector).count()):
            b = card.locator(selector).nth(i).bounding_box()
            if b:
                found.append(
                    {
                        k: round(b[k] - (box["x"] if k == "x" else box["y"] if k == "y" else 0), 1)
                        for k in ("x", "y", "width", "height")
                    }
                )
        parts[key] = found
    page.screenshot(
        path=str(out / f"{name}.png"),
        clip={"x": box["x"], "y": box["y"], "width": box["width"], "height": box["height"]},
        full_page=True,
    )
    text = card.inner_text()
    info = {"size": [box["width"], box["height"]], "scale": SCALE, "parts": parts, "text": text}
    (out / f"{name}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def interface_shot(page: Any, out: Path) -> None:
    """The same page and answer at desktop width: the whole page, plus where the warning ends."""
    page.set_viewport_size({"width": 1280, "height": 860})
    page.wait_for_timeout(800)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)
    banner = page.locator(".card .banner").first.bounding_box()
    if banner is None:
        raise SystemExit("no warning on the model answer")
    page.screenshot(path=str(out / "interface.png"), full_page=True)
    info = {"scale": SCALE, "width": 1280, "banner_bottom": banner["y"] + banner["height"]}
    (out / "interface.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("interface", info)


def main() -> int:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640, help="viewport width in CSS pixels")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(args.profile),
            channel="chrome",
            headless=True,
            args=WEBGPU_ARGS,
            viewport={"width": args.width, "height": 1100},
            device_scale_factor=SCALE,
            color_scheme="dark",
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(URL)
        page.wait_for_selector("button.chip-example", timeout=240_000)

        ask(page)
        page.wait_for_selector(".evidence", timeout=60_000)
        info = card_shot(page, args.out, "extractive-card")
        if "p. 2" not in info["text"]:
            raise SystemExit("extractive answer: no quote from page 2")
        print("extractive card", info["size"])

        option = page.locator("label.option-card", has_text="on your GPU")
        if option.locator("input").is_disabled():
            raise SystemExit("WebGPU is not available: " + option.inner_text()[-160:])
        option.locator("input").click()
        selected = (
            "document.querySelector('label.option-card.is-selected')"
            "?.textContent.includes('on your GPU')"
        )
        started = time.monotonic()
        while not page.evaluate(selected):
            if (
                time.monotonic() - started > 3600
                or page.locator("text=could not be loaded").count()
            ):
                raise SystemExit("the model did not load: " + page.inner_text(".demo-panel")[-200:])
            page.wait_for_timeout(5_000)
        print(f"model loaded in {time.monotonic() - started:.0f} s")
        ask(page)
        page.wait_for_function(
            "() => [...document.querySelectorAll('.card')].some(c => c.innerText.includes('webllm')"
            " && c.querySelector('.answer-text'))",
            timeout=600_000,
        )
        page.wait_for_timeout(500)
        info = card_shot(page, args.out, "model-card")
        print("model card", info["size"])
        print(info["text"][:600])
        interface_shot(page, args.out)
        context.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
