"""README banners (light and dark SVG) and the social-preview image for GroundedAI.

Follows the visual recipe of the other pouyapd portfolio repositories (GitHub light/dark
palettes, series line, title, tagline, chips, a data motif on the right) with its own
accent colour. The motif is drawn from the committed retrieval benchmark report; no
number is drawn that is not in that file.

    python tools/make_readme_assets.py        # needs matplotlib

Outputs: docs/assets/banner-light.svg, docs/assets/banner-dark.svg,
docs/assets/social-preview.png
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "evaluation/reports/published/retrieval-groundedai-synthetic-v1-20261011T141404Z.json"
)
OUT = ROOT / "docs" / "assets"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "svg.fonttype": "path",  # text as outlines: renders the same in every browser
        "svg.hashsalt": "groundedai",  # stable ids, so regenerating gives identical files
        "path.simplify": True,
    }
)

THEMES = {
    "dark": dict(
        bg="#0d1117",
        panel="#151b23",
        border="#30363d",
        ink="#f0f6fc",
        ink2="#9198a1",
        muted="#3d444d",
        faint="#262c36",
    ),
    "light": dict(
        bg="#ffffff",
        panel="#f6f8fa",
        border="#d1d9e0",
        ink="#1f2328",
        ink2="#59636e",
        muted="#c8d1da",
        faint="#e6eaef",
    ),
}
ACCENT = {"light": "#bf3989", "dark": "#f778ba"}  # GitHub pink scale; unused by the other repos
SERIES = "POUYA BATHAEI POURMAND  ·  TRUSTWORTHY AI RESEARCH SOFTWARE"
TITLE = "GroundedAI"
TAGLINE = "Answers from your documents, with every citation checked"
CHIPS = ["hybrid retrieval", "checked citations", "measured with 95% CIs", "runs on a laptop CPU"]
LABELS = {
    "A-lexical-bm25": "BM25",
    "B0-dense-hashing": "hashing (baseline)",
    "B1-dense-bge-small": "dense bge-small",
    "B2-dense-minilm": "dense MiniLM",
    "C1-hybrid-rrf-bge-small": "hybrid RRF",
    "C2-hybrid-weighted-bge-small": "hybrid weighted",
    "D1-hybrid-rrf-bge-small+rerank": "hybrid + rerank",
}


def motif(ax: plt.Axes, th: dict[str, str], accent: str) -> str:
    """MRR with 95% bootstrap interval per retrieval configuration."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    configs = report["configurations"]
    for row, config in enumerate(configs):
        metric = config["metrics"]["mrr"]
        low, high = metric["ci95"]
        y = len(configs) - 1 - row
        is_bm25 = config["name"].startswith("A-")
        colour = accent if is_bm25 or config["name"].startswith("C1") else th["ink2"]
        ax.plot([low, high], [y, y], color=colour, lw=1.4, alpha=0.9, solid_capstyle="round")
        ax.plot(metric["mean"], y, "o", ms=5.2, color=colour, mec=th["panel"], mew=1.2)
        ax.text(
            0.665,
            y,
            LABELS.get(config["name"], config["name"]),
            fontsize=7.4,
            color=th["ink"] if colour == accent else th["ink2"],
            va="center",
            ha="right",
        )
    ax.set_xlim(0.40, 1.0)
    ax.set_ylim(-0.8, len(configs) - 0.2)
    for tick in (0.7, 0.8, 0.9, 1.0):
        ax.axvline(tick, color=th["faint"], lw=0.8, zorder=0)
        ax.text(tick, -0.95, f"{tick:.1f}", fontsize=6.6, color=th["ink2"], ha="center", va="top")
    ax.text(
        0.85,
        len(configs) - 0.05,
        "MRR (95% CI), 43 questions",
        fontsize=7.2,
        color=th["ink2"],
        ha="center",
        va="bottom",
    )
    return (
        "Source: evaluation/reports/published/retrieval-…-20261011T141404Z.json · "
        "hybrid matches but does not beat BM25 on this dataset"
    )


def panel(fig: plt.Figure, th: dict[str, str]) -> None:
    fig.patch.set_alpha(0.0)
    bg = fig.add_axes((0, 0, 1, 1))
    bg.set_axis_off()
    bg.add_patch(
        FancyBboxPatch(
            (0.004, 0.012),
            0.992,
            0.976,
            boxstyle="round,pad=0,rounding_size=0.018",
            transform=bg.transAxes,
            fc=th["panel"],
            ec=th["border"],
            lw=1.0,
            mutation_aspect=fig.get_figwidth() / fig.get_figheight(),
        )
    )


def chips(fig: plt.Figure, x: float, y: float, th: dict[str, str], size: float) -> None:
    renderer = fig.canvas.get_renderer()
    for chip in CHIPS:
        text = fig.text(
            x,
            y,
            chip,
            fontsize=size,
            color=th["ink2"],
            va="center",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.38,rounding_size=0.8", fc=th["bg"], ec=th["border"], lw=0.8
            ),
        )
        box = text.get_window_extent(renderer=renderer)
        x += box.width / fig.bbox.width + 0.012


def save_svg(fig: plt.Figure, out: Path) -> None:
    fig.savefig(out, transparent=True, metadata={"Date": None})
    plt.close(fig)
    lines = out.read_text(encoding="utf-8").splitlines()  # matplotlib leaves trailing spaces
    out.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def banner(theme: str) -> None:
    th, accent = THEMES[theme], ACCENT[theme]
    fig = plt.figure(figsize=(12, 2.7), dpi=100)
    panel(fig, th)
    fig.text(0.035, 0.80, SERIES, fontsize=8.2, color=accent, weight="semibold")
    fig.text(0.035, 0.535, TITLE, fontsize=29, color=th["ink"], weight="bold", va="center")
    fig.text(0.035, 0.33, TAGLINE, fontsize=12.5, color=th["ink2"], va="center")
    chips(fig, 0.037, 0.15, th, 9)
    ax = fig.add_axes((0.62, 0.17, 0.355, 0.70))
    ax.set_axis_off()
    source = motif(ax, th, accent)
    fig.text(
        0.975, 0.055, source, fontsize=6.3, color=th["ink2"], ha="right", va="center", alpha=0.85
    )
    save_svg(fig, OUT / f"banner-{theme}.svg")


def social_preview() -> None:
    th, accent = THEMES["dark"], ACCENT["dark"]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(th["bg"])
    bg = fig.add_axes((0, 0, 1, 1))
    bg.set_axis_off()
    bg.add_patch(
        FancyBboxPatch(
            (0.025, 0.04),
            0.95,
            0.92,
            boxstyle="round,pad=0,rounding_size=0.02",
            transform=bg.transAxes,
            fc=th["panel"],
            ec=th["border"],
            lw=1.2,
            mutation_aspect=2.0,
        )
    )
    fig.text(0.065, 0.855, SERIES, fontsize=10.5, color=accent, weight="semibold")
    fig.text(0.065, 0.70, TITLE, fontsize=36, color=th["ink"], weight="bold", va="center")
    fig.text(
        0.065,
        0.52,
        textwrap.fill(TAGLINE, 36),
        fontsize=16.5,
        color=th["ink2"],
        va="center",
        linespacing=1.3,
    )
    chips(fig, 0.067, 0.30, th, 12)
    ax = fig.add_axes((0.585, 0.40, 0.36, 0.46))
    ax.set_axis_off()
    motif(ax, th, accent)
    fig.text(0.065, 0.10, "github.com/pouyapd/GroundedAI", fontsize=12.5, color=th["ink2"])
    fig.savefig(
        OUT / "social-preview.png", format="png", facecolor=th["bg"], metadata={"Software": None}
    )
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    banner("light")
    banner("dark")
    social_preview()
    print(f"Wrote banners and social preview to {OUT}")


if __name__ == "__main__":
    main()
