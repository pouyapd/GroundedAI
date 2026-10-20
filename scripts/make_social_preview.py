"""Make the repository's social preview image (1280 x 640): docs/assets/social-preview.png.

    python scripts/make_social_preview.py [--url https://pouyapd.github.io/GroundedAI/]

The right side shows two real fragments of one answer in the live demo, captured at twice the
pixel density in the dark theme: the answer with its verdict, and the claim with its status,
quoted sentence and page. Nothing in them is drawn or edited. The left side is the title, a
tagline and the demo URL, in the navy, cyan and blue of the author's profile. Uses Segoe UI
(Windows); needs ``playwright`` with Chromium and Pillow.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "social-preview.png"
URL = "https://pouyapd.github.io/GroundedAI/"
W, H = 1280, 640
NAVY, NAVY_2, DEEP = (10, 19, 48), (14, 27, 61), (5, 10, 25)
CYAN, BLUE, INK, INK_2, MUTED = (
    (94, 231, 255),
    (91, 140, 255),
    (238, 245, 255),
    (169, 188, 216),
    (111, 134, 171),
)
FONTS = Path("C:/Windows/Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def capture(url: str) -> tuple[Image.Image, Image.Image]:
    """The answer (header, verdict, text) and the first claim with its evidence, from the demo."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": 600, "height": 900}, device_scale_factor=2, color_scheme="dark"
        )
        page.goto(url)
        page.wait_for_selector("button.chip-example", timeout=240_000)
        page.select_option("#scope", label="Harbourline Ferries Passenger Guide")
        page.fill("#question", "How much does a single adult ticket to Mirrow cost?")
        page.click("button[type=submit]:has-text('Ask question')")
        page.wait_for_selector(".evidence", timeout=60_000)
        page.evaluate("window.scrollTo(0, 0)")  # boxes are relative to the viewport
        page.wait_for_timeout(200)
        card = page.locator(".card", has=page.locator(".answer-text")).first
        head = card.locator(".card-header").first.bounding_box()
        text = card.locator(".answer-text").first.bounding_box()
        claim = card.locator(".claim").first.bounding_box()
        evidence = card.locator(".evidence").first.bounding_box()
        box = card.bounding_box()
        if not (head and text and claim and evidence and box):
            raise SystemExit("the answer did not appear")
        if "p. 2" not in card.locator(".evidence").first.inner_text():
            raise SystemExit("the quote does not name page 2; nothing was written")
        shot = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
        browser.close()

    def crop(top: float, bottom: float) -> Image.Image:
        s = 2  # device scale factor
        return shot.crop(
            (int(box["x"] * s), int(top * s), int((box["x"] + box["width"]) * s), int(bottom * s))
        )

    answer = crop(head["y"] - 14, text["y"] + text["height"] + 8)
    proof = crop(claim["y"] - 10, evidence["y"] + evidence["height"] + 16)
    return answer, proof


def rounded(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1), radius=radius, fill=255
    )
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


def compose(answer: Image.Image, proof: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (W, H), DEEP)
    px = canvas.load()
    for y in range(H):  # diagonal navy gradient
        for x in range(0, W, 4):
            t = min(1.0, (x / W) * 0.6 + (y / H) * 0.4)
            c = tuple(int(DEEP[i] * (1 - t) + NAVY_2[i] * t) for i in range(3))
            for dx in range(4):
                if x + dx < W:
                    px[x + dx, y] = c
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((640, -120, 1400, 640), fill=(*CYAN, 34))
    canvas.paste(
        glow.filter(ImageFilter.GaussianBlur(120)),
        (0, 0),
        glow.filter(ImageFilter.GaussianBlur(120)),
    )
    d = ImageDraw.Draw(canvas)

    # right: the two real fragments in a framed panel
    panel_x, panel_w = 640, 584
    scale = (panel_w - 36) / answer.width
    a = answer.resize((panel_w - 36, round(answer.height * scale)), Image.LANCZOS)
    b = proof.resize((panel_w - 36, round(proof.height * scale)), Image.LANCZOS)
    panel_h = 18 + a.height + 14 + b.height + 18
    panel_y = (H - panel_h) // 2
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (panel_x - 6, panel_y - 6, panel_x + panel_w + 6, panel_y + panel_h + 6),
        radius=24,
        fill=(*CYAN, 60),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    canvas.paste(shadow, (0, 0), shadow)
    d.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        radius=20,
        fill=(17, 17, 17),
        outline=(29, 52, 97),
        width=2,
    )
    for x in range(panel_x + 20, panel_x + panel_w - 20):  # accent line, cyan to blue
        t = (x - panel_x - 20) / (panel_w - 40)
        d.line(
            [(x, panel_y + 1), (x, panel_y + 3)],
            fill=tuple(int(CYAN[i] * (1 - t) + BLUE[i] * t) for i in range(3)),
        )
    canvas.paste(rounded(a, 10), (panel_x + 18, panel_y + 18), rounded(a, 10))
    by = panel_y + 18 + a.height + 14
    canvas.paste(rounded(b, 10), (panel_x + 18, by), rounded(b, 10))

    # left: mark, name, tagline, three facts, URL
    x0 = 72
    tile = Image.new("RGBA", (64, 64))
    for yy in range(64):  # diagonal gradient, cyan to blue
        for xx in range(64):
            t = (xx + yy) / 126
            tile.putpixel(
                (xx, yy), (*tuple(int(CYAN[i] * (1 - t) + BLUE[i] * t) for i in range(3)), 255)
            )
    shape = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(shape).rounded_rectangle((0, 0, 63, 63), radius=15, fill=255)
    mark = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    mark.paste(tile, (0, 0), shape)
    ImageDraw.Draw(mark).line(
        [(17, 33), (28, 44), (47, 22)], fill=(5, 10, 25, 255), width=7, joint="curve"
    )
    canvas.paste(mark, (x0, 150), mark)
    d.text((x0 + 84, 140), "GroundedAI", font=font("segoeuib.ttf", 64), fill=INK)
    d.text((x0, 250), "Evidence-grounded document", font=font("segoeui.ttf", 32), fill=INK_2)
    d.text((x0, 290), "question answering", font=font("segoeui.ttf", 32), fill=INK_2)
    facts = (
        "Every statement checked against the source",
        "Exact quote and page for each claim",
        "Refuses when the evidence is not there",
    )
    for i, fact in enumerate(facts):
        y = 368 + i * 40
        d.ellipse((x0, y + 9, x0 + 10, y + 19), fill=CYAN)
        d.text((x0 + 24, y), fact, font=font("segoeui.ttf", 23), fill=(214, 230, 251))
    d.rounded_rectangle((x0, 520, x0 + 54, 522), radius=1, fill=CYAN)
    d.text(
        (x0, 534),
        "Live demo  ·  pouyapd.github.io/GroundedAI",
        font=font("seguisb.ttf", 22),
        fill=CYAN,
    )
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    image = compose(*capture(args.url))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, optimize=True)
    print(f"{OUT.relative_to(ROOT).as_posix()}: {image.size}, {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
