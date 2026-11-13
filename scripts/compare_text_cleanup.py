"""Compare text clean-up v1 and v2 on PDFs: what v2 removes or changes, page by page.

    python scripts/compare_text_cleanup.py --out cleanup-report.json PDF [PDF ...]
    python scripts/compare_text_cleanup.py --out cleanup-report.json --datasets

For each PDF: pages, characters and chunks with v1 and v2; pages left empty by v2; how many
ligatures were expanded and line-break hyphens joined; and every distinct line v2 removed, with
how many pages it was removed from, so a reader can check that only layout text (running
headers, footers, page numbers) was dropped. It also checks two invariants of v2: every page
keeps its number, and every v2 page's text is made only of words that occur in the v1 page
(after ligature expansion and hyphen joining), so no text moves between pages.
``--datasets`` takes every PDF of the evaluation datasets, the examples and the demo samples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.ingestion.chunking import ChunkingConfig, chunk_pages  # noqa: E402
from groundedai.ingestion.parsers import parse_pdf  # noqa: E402
from groundedai.ingestion.pdf_cleanup import (  # noqa: E402
    LIGATURES,
    expand_ligatures,
    join_hyphenated_lines,
)

LIMITS = {"max_pages": 2000, "max_chars": 10_000_000}
WORD_RE = re.compile(r"\w+")


def dataset_pdfs() -> list[Path]:
    folders = [
        ROOT / "evaluation" / "datasets",
        ROOT / "examples",
        ROOT / "experiments" / "finetune" / "corpus",
        ROOT / "frontend" / "public" / "runtime" / "samples",
    ]
    found = {p.resolve() for folder in folders if folder.is_dir() for p in folder.rglob("*.pdf")}
    return sorted(found)


def compare(path: Path) -> dict:
    data = path.read_bytes()
    v1 = parse_pdf(data, cleanup="v1", **LIMITS)
    v2 = parse_pdf(data, cleanup="v2", **LIMITS)
    removed: Counter[str] = Counter()
    moved_pages = []
    for one, two in zip(v1.pages, v2.pages, strict=True):
        if one.number != two.number:
            raise SystemExit(f"{path.name}: page {one.number} became page {two.number}")
        before = join_hyphenated_lines(expand_ligatures(one.text))
        kept = {line.strip() for line in two.text.split("\n")}
        removed.update(
            line.strip() for line in before.split("\n") if line.strip() and line.strip() not in kept
        )
        if set(WORD_RE.findall(two.text)) - set(WORD_RE.findall(before)):
            moved_pages.append(one.number)
    v1_text = "\n".join(p.text for p in v1.pages)
    try:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        relative = path.name
    return {
        "file": relative,
        "pages": len(v1.pages),
        "chars_v1": sum(len(p.text) for p in v1.pages),
        "chars_v2": sum(len(p.text) for p in v2.pages),
        "chunks_v1": len(chunk_pages(v1.pages, ChunkingConfig(800, 120))),
        "chunks_v2": len(chunk_pages(v2.pages, ChunkingConfig(800, 120, "v2"))),
        "pages_changed": sum(a.text != b.text for a, b in zip(v1.pages, v2.pages, strict=True)),
        "pages_empty_v1": sum(not p.text for p in v1.pages),
        "pages_empty_v2": sum(not p.text for p in v2.pages),
        "ligatures_expanded": sum(v1_text.count(ch) for ch in LIGATURES),
        "hyphens_joined": len(re.findall(r"(?<=[a-z])-\n(?=[a-z])", expand_ligatures(v1_text))),
        "pages_with_new_words": moved_pages,
        "removed_lines": [{"line": line, "pages": count} for line, count in removed.most_common()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.pdfs) + (dataset_pdfs() if args.datasets else [])
    if not paths:
        parser.error("give PDF files or --datasets")
    rows = [compare(path) for path in paths]
    for row in rows:
        print(
            f"{row['file'][-58:]:58s} pages {row['pages']:4d}  changed {row['pages_changed']:4d}"
            f"  chunks {row['chunks_v1']:4d} -> {row['chunks_v2']:4d}"
            f"  empty {row['pages_empty_v1']} -> {row['pages_empty_v2']}"
            f"  removed lines {len(row['removed_lines']):4d}"
            f"  new words on {len(row['pages_with_new_words'])} pages"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
