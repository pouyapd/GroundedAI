"""Build the PDFs of evaluation/datasets/groundedai-versions-v1 from their source text.

    python scripts/build_versions_corpus.py           # write corpus/*.pdf and update hashes
    python scripts/build_versions_corpus.py --check   # fail if a PDF differs from its source

Each source file is one fictional document; a line holding only ``\\f`` separates pages. The
documents come in pairs: two editions of the same document whose values, requirement wording
or terminology differ in places, so that a disagreement check can be tested on document
versions. The two example PDFs in ``examples/`` are copied in as unrelated documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.evaluation.pdf_fixture import make_pdf  # noqa: E402

DATASET = ROOT / "evaluation" / "datasets" / "groundedai-versions-v1"
PAGE_BREAK = "\\f"


def pages_of(path: Path) -> list[str]:
    pages, current = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == PAGE_BREAK:
            pages.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    pages.append("\n".join(current).strip())
    return [page for page in pages if page]


def build(titles: dict[str, str]) -> dict[str, bytes]:
    out = {}
    for source in sorted((DATASET / "source").glob("*.txt")):
        name = f"corpus/{source.stem}.pdf"
        out[name] = make_pdf(pages_of(source), title=titles.get(name, source.stem))
    for example in ("harbourline-ferry-guide.pdf", "orchard-library-rules.pdf"):
        out[f"corpus/{example}"] = (ROOT / "examples" / example).read_bytes()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = DATASET / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_file = {doc["file"]: doc for doc in manifest["documents"]}
    files = build({name: doc["title"] for name, doc in by_file.items()})
    problems = []
    for name, data in files.items():
        digest = hashlib.sha256(data).hexdigest()
        doc = by_file.get(name)
        if doc is None:
            problems.append(f"{name} is not in the manifest")
            continue
        if args.check:
            target = DATASET / name
            if not target.is_file() or target.read_bytes() != data or doc["sha256"] != digest:
                problems.append(f"{name} differs from its source")
            continue
        (DATASET / name).parent.mkdir(parents=True, exist_ok=True)
        (DATASET / name).write_bytes(data)
        doc["sha256"] = digest
    if not args.check:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    for problem in problems:
        print(problem)
    print(f"{len(files)} documents {'checked' if args.check else 'written'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
