"""Make the LinkedIn launch visuals (1200 x 675, 16:9) from real answer cards of the live demo.

    python scripts/capture_launch_fragments.py --profile <chrome profile> --out <capture folder>
    python scripts/make_linkedin_visual.py --capture <capture folder>

Writes ``docs/images/linkedin-launch.png`` and ``docs/images/linkedin-launch-minimal.png``
(1200 x 675) and ``@2x`` versions (2400 x 1350) for uploading, and the post's screenshots
``linkedin-post-2-invalid-citation.png``, ``-3-verified-evidence.png`` and ``-4-interface.png``:
plain crops of the captured screenshots, without overlays (``--screenshots-only`` writes just
these).

The interface pieces are crops of one answer card captured from the live demo: Qwen2.5-1.5B-Instruct
running in the browser answered the ferry-guide question, cited a source (S2) it was never given,
and GroundedAI flagged it. Crops are not edited; the gaps between them are marked, and the
numbered rings only point at parts of the real interface. Layout, type and colour are HTML/CSS
rendered with Playwright (Chromium); text uses Inter from Google Fonts.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
W, H = 1200, 675

# Slices of the model answer card, in CSS pixels of the card (608 wide): top, bottom.
SLICES = {
    "answer": (10, 122),  # header with the verdict, answer text with S1 and the invalid S2
    "warning": (234, 362),  # "Check before relying on this answer"
    "claim": (468, 598),  # claim 1: status, statement, quote with document and page
}
X0, X1 = 12, 596  # inside the card's border

# The three points on the left; the numbers match the rings on the interface.
LEGEND = (
    ("Invented citations are flagged", "The model cited S2, a source it was never given"),
    ("Every statement is checked", "against the passages the model received"),
    ("Exact quote and page", "from the source PDF, for each statement it finds"),
)


def crop(card: Image.Image, scale: int, top: float, bottom: float) -> Image.Image:
    return card.crop((X0 * scale, int(top * scale), X1 * scale, int(bottom * scale)))


def data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


FONT = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
    '&display=swap" rel="stylesheet">'
)

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 1200px; height: 675px; overflow: hidden; }
body { font-family: Inter, "Segoe UI", sans-serif; color: #e8f0fb; position: relative;
  background:
    radial-gradient(620px 420px at 860px 330px, rgba(45, 212, 191, 0.13), transparent 70%),
    radial-gradient(900px 600px at 0px 0px, rgba(37, 99, 235, 0.10), transparent 70%),
    linear-gradient(160deg, #071022 0%, #0a1730 55%, #08132a 100%); }
.grid { position: absolute; inset: 0; opacity: 0.35;
  background-image: radial-gradient(rgba(148, 197, 255, 0.16) 1px, transparent 1px);
  background-size: 24px 24px;
  -webkit-mask-image: radial-gradient(520px 380px at 860px 340px, #000 0%, transparent 75%); }
.eyebrow { font-size: 13px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
  color: #5eead4; }
.eyebrow span { color: #64748b; margin: 0 8px; }
.brand { display: flex; align-items: center; gap: 14px; }
.mark { width: 46px; height: 46px; border-radius: 12px; display: grid; place-items: center;
  background: linear-gradient(135deg, #5eead4, #38bdf8);
  box-shadow: 0 6px 24px rgba(45, 212, 191, .25); }
.mark svg { width: 26px; height: 26px; }
.name { font-size: 50px; font-weight: 800; letter-spacing: -0.025em; color: #f8fbff;
  line-height: 1; }
.headline { font-weight: 600; color: #f1f6fd; letter-spacing: -0.015em; }
.headline em { font-style: normal; color: #5eead4; }
.url { font-size: 17px; font-weight: 600; color: #e2ecf8; }
.url b { color: #5eead4; font-weight: 600; }
.small { font-size: 13px; color: #8193ad; }
"""

MARK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#06121f" stroke-width="3.2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
)


def main_html(slices: dict[str, Image.Image], parts: dict) -> str:
    # panel geometry (CSS px of the visual); slices shown at 0.94 of their CSS size
    s = 0.94
    pad, gap = 16, 26
    panel_x, panel_y = 566, 92
    widths = (X1 - X0) * s
    tops, y = {}, panel_y + pad
    for key in ("answer", "warning", "claim"):
        tops[key] = y
        top, bottom = SLICES[key]
        y += (bottom - top) * s + gap
    panel_h = y - gap + pad - panel_y
    panel_w = widths + 2 * pad

    def at(key: str, x: float, yy: float) -> tuple[float, float]:
        """Card CSS coordinates inside a slice -> visual coordinates."""
        return panel_x + pad + (x - X0) * s, tops[key] + (yy - SLICES[key][0]) * s

    cite = parts["invalid_cite"][0]
    evidence = parts["evidence"][0]
    claim = parts["claims"][0]
    rings = [
        # (slice, x, y, w, h, marker number)
        ("answer", cite["x"] - 4, cite["y"] - 4, cite["width"] + 8, cite["height"] + 8, 1),
        ("claim", claim["x"] - 2, claim["y"] - 4, 210, 31, 2),
        (
            "claim",
            evidence["x"] - 4,
            evidence["y"] - 4,
            evidence["width"] + 6,
            evidence["height"] + 8,
            3,
        ),
    ]
    ring_html = ""
    for key, x, yy, w, h, n in rings:
        vx, vy = at(key, x, yy)
        ring_html += (
            f'<div class="ring" style="left:{vx:.1f}px;top:{vy:.1f}px;width:{w * s:.1f}px;'
            f'height:{h * s:.1f}px"></div>'
            f'<div class="dot" style="left:{vx + w * s + (6 if n == 1 else -13):.1f}px;'
            f'top:{vy - (2 if n == 1 else 13):.1f}px">{n}</div>'
        )
    imgs = ""
    for key in ("answer", "warning", "claim"):
        top, bottom = SLICES[key]
        imgs += (
            f'<img class="slice" src="{data_uri(slices[key])}" style="left:{panel_x + pad}px;'
            f'top:{tops[key]:.1f}px;width:{widths:.1f}px;height:{(bottom - top) * s:.1f}px">'
        )
        if key != "claim":
            gy = tops[key] + (bottom - top) * s + gap / 2 - 6
            imgs += (
                f'<div class="cut" style="left:{panel_x + pad}px;top:{gy:.1f}px;'
                f'width:{widths:.1f}px">· · ·</div>'
            )

    legend = "".join(
        f'<div class="item"><div class="n">{n}</div>'
        f"<div><b>{title}</b><span>{line}</span></div></div>"
        for n, (title, line) in enumerate(LEGEND, start=1)
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">{FONT}<style>{BASE_CSS}
.panel {{ position:absolute; left:{panel_x}px; top:{panel_y}px;
  width:{panel_w:.1f}px; height:{panel_h:.1f}px;
  border-radius:18px; background:#1a1a19; border:1px solid rgba(94,234,212,.28);
  box-shadow: 0 30px 80px rgba(0,0,0,.55), 0 0 0 6px rgba(94,234,212,.04); }}
.panel::before {{ content:""; position:absolute; left:24px; right:24px; top:-1px; height:2px;
  border-radius:2px; background: linear-gradient(90deg, #5eead4, #38bdf8); }}
.slice {{ position:absolute; border-radius:8px; }}
.cut {{ position:absolute; height:12px; text-align:center; font-size:12px; line-height:12px;
  letter-spacing:4px; color:#4b5563; }}
.ring {{ position:absolute; border:2px solid #5eead4; border-radius:9px;
  box-shadow: 0 0 0 4px rgba(94,234,212,.12), 0 0 18px rgba(94,234,212,.35); }}
.dot {{ position:absolute; width:26px; height:26px; border-radius:50%; background:#5eead4;
  color:#04211c; font-weight:800; font-size:14px; display:grid; place-items:center;
  box-shadow:0 0 0 3px #0a1730; }}
.left {{ position:absolute; left:64px; top:72px; width:450px; }}
.headline {{ font-size:31px; line-height:1.2; margin-top:26px; white-space:nowrap; }}
.legend {{ margin-top:34px; display:flex; flex-direction:column; gap:15px; }}
.item {{ display:flex; gap:14px; align-items:flex-start; }}
.item .n {{ flex:none; width:26px; height:26px; border-radius:50%; background:#5eead4;
  color:#04211c; font-weight:800; font-size:14px; display:grid; place-items:center;
  margin-top:1px; }}
.item b {{ display:block; font-size:19px; font-weight:600; color:#f1f6fd; }}
.item span {{ display:block; font-size:15px; color:#9fb1c9; margin-top:2px; }}
.foot {{ position:absolute; left:64px; bottom:50px; width:460px; }}
.cta {{ display:inline-flex; align-items:center; gap:10px; padding:10px 16px; border-radius:10px;
  background:rgba(94,234,212,.10); border:1px solid rgba(94,234,212,.35); }}
.cta i {{ font-style:normal; color:#5eead4; font-size:13px; }}
.caption {{ position:absolute; left:{panel_x}px; top:{panel_y + panel_h + 14:.1f}px;
  width:{panel_w:.1f}px; font-size:12.5px; color:#7d8ea8; text-align:center;
  white-space:nowrap; }}
</style></head><body><div class="grid"></div>
<div class="left">
  <div class="eyebrow">Open source<span>·</span>Release
    <i style="font-style:normal;text-transform:none">v0.1.0</i></div>
  <div class="brand" style="margin-top:18px"><div class="mark">{MARK_SVG}</div>
    <div class="name">GroundedAI</div></div>
  <div class="headline">Evidence-grounded<br>document <em>question answering</em></div>
  <div class="legend">{legend}</div>
</div>
<div class="foot">
  <div class="cta"><i>▶</i><span class="url">Live demo
    <b>pouyapd.github.io/GroundedAI</b></span></div>
  <div class="small" style="margin-top:10px">Runs in your browser · no install ·
    github.com/pouyapd/GroundedAI</div>
</div>
<div class="panel"></div>
{imgs}{ring_html}
<div class="caption">Real answer from the live demo (Qwen2.5-1.5B in the browser) ·
  crops of one answer card</div>
</body></html>"""


def minimal_html(claim: Image.Image, answer: Image.Image) -> str:
    s = 1.30
    w = (X1 - X0) * s
    ha = (SLICES["answer"][1] - SLICES["answer"][0]) * s
    hc = (SLICES["claim"][1] - SLICES["claim"][0]) * s
    x = (W - w) / 2
    y0 = 236
    return f"""<!doctype html><html><head><meta charset="utf-8">{FONT}<style>{BASE_CSS}
body {{ background:
    radial-gradient(700px 360px at 600px 470px, rgba(45, 212, 191, 0.12), transparent 70%),
    linear-gradient(165deg, #071022 0%, #0a1730 60%, #08132a 100%); }}
.top {{ position:absolute; left:0; right:0; top:58px; text-align:center; }}
.top .brand {{ justify-content:center; }}
.top .name {{ font-size:44px; }}
.top .headline {{ font-size:30px; margin-top:14px; }}
.card {{ position:absolute; left:{x - 18:.1f}px; top:{y0 - 18}px; width:{w + 36:.1f}px;
  height:{ha + hc + 30 + 36:.1f}px; border-radius:18px; background:#1a1a19;
  border:1px solid rgba(94,234,212,.28);
  box-shadow: 0 30px 80px rgba(0,0,0,.55); }}
.card::before {{ content:""; position:absolute; left:24px; right:24px; top:-1px; height:2px;
  background: linear-gradient(90deg, #5eead4, #38bdf8); }}
img {{ position:absolute; left:{x:.1f}px; width:{w:.1f}px; border-radius:8px; }}
.cut {{ position:absolute; left:{x:.1f}px; width:{w:.1f}px; text-align:center; font-size:13px;
  letter-spacing:4px; color:#4b5563; }}
.foot {{ position:absolute; left:0; right:0; bottom:30px; text-align:center; }}
</style></head><body>
<div class="top">
  <div class="brand"><div class="mark">{MARK_SVG}</div><div class="name">GroundedAI</div></div>
  <div class="headline">Evidence-grounded document <em>question answering</em></div>
</div>
<div class="card"></div>
<img src="{data_uri(answer)}" style="top:{y0}px;height:{ha:.1f}px">
<div class="cut" style="top:{y0 + ha + 8:.1f}px">· · ·</div>
<img src="{data_uri(claim)}" style="top:{y0 + ha + 30:.1f}px;height:{hc:.1f}px">
<div class="foot"><span class="url">Live demo <b>pouyapd.github.io/GroundedAI</b></span>
  <span class="small" style="margin-left:14px">Real answer from the in-browser model ·
  open source</span></div>
</body></html>"""


def render(html: str, name: str) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        page.set_content(html, wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(300)
        if not page.evaluate("document.fonts.check('600 20px Inter')"):
            raise SystemExit("Inter did not load; nothing was written")
        big = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
        browser.close()
    big.save(OUT / f"{name}@2x.png", optimize=True)
    big.resize((W, H), Image.LANCZOS).save(OUT / f"{name}.png", optimize=True)
    for f in (OUT / f"{name}.png", OUT / f"{name}@2x.png"):
        w, h = Image.open(f).size
        print(f"{f.relative_to(ROOT).as_posix()}: {w} x {h}, {f.stat().st_size / 1024:.0f} KB")


def screenshots(capture: Path, info: dict) -> None:
    """Plain crops of the captured screenshots for the post, after the cover. No overlays."""
    card = Image.open(capture / "model-card.png").convert("RGB")
    s = info["scale"]
    banner = info["parts"]["banner"][0]
    evidence = info["parts"]["evidence"][-1]
    cut = banner["y"] + banner["height"] + 14  # just below the warning
    shots = {
        # the answer with the invalid S2, the model and the warning
        "linkedin-post-2-invalid-citation": card.crop((0, 0, card.width, int(cut * s))),
        # the claim check: method note, both statements, their quotes and pages
        "linkedin-post-3-verified-evidence": card.crop(
            (0, int(cut * s), card.width, int((evidence["y"] + evidence["height"] + 16) * s))
        ),
    }
    page = json.loads((capture / "interface.json").read_text(encoding="utf-8"))
    full = Image.open(capture / "interface.png").convert("RGB")
    top = full.crop((0, 0, full.width, int((page["banner_bottom"] + 18) * page["scale"])))
    shots["linkedin-post-4-interface"] = top.resize(
        (1600, round(top.height * 1600 / top.width)), Image.LANCZOS
    )
    for name, image in shots.items():
        f = OUT / f"{name}.png"
        image.save(f, optimize=True)
        w, h = image.size
        print(f"{f.relative_to(ROOT).as_posix()}: {w} x {h}, {f.stat().st_size / 1024:.0f} KB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--screenshots-only",
        action="store_true",
        help="write only the post screenshots (2-4), not the cover images",
    )
    args = parser.parse_args()
    info = json.loads((args.capture / "model-card.json").read_text(encoding="utf-8"))
    for needle in (
        "S2. Those citations are invalid",
        "p. 2",
        "webllm",
        "Supported by the cited passage",
    ):
        if needle not in info["text"]:
            raise SystemExit(f"the captured model answer lacks {needle!r}; nothing was written")
    if (args.capture / "interface.png").exists():
        screenshots(args.capture, info)
    if args.screenshots_only:
        return 0
    card = Image.open(args.capture / "model-card.png").convert("RGB")
    scale = info["scale"]
    slices = {k: crop(card, scale, *v) for k, v in SLICES.items()}
    render(main_html(slices, info["parts"]), "linkedin-launch")
    render(minimal_html(slices["claim"], slices["answer"]), "linkedin-launch-minimal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
