"""Download the NIST PDFs used by the real-document datasets and check their SHA-256 hashes.

    python scripts/fetch_nist_corpus.py            # every dataset manifest that has source URLs
    python scripts/fetch_nist_corpus.py --check    # only verify files already downloaded

The PDFs are U.S. Government works (not subject to copyright in the United States) and are not
committed; each manifest pins the exact file by its hash, so a changed upstream file is refused
instead of silently changing the evaluation. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = [
    ROOT / "evaluation" / "datasets" / "groundedai-nist-v1" / "manifest.json",
    ROOT / "evaluation" / "datasets" / "groundedai-nist-val-v1" / "manifest.json",
    ROOT / "experiments" / "finetune" / "corpus.json",
]
USER_AGENT = "GroundedAI-evaluation/0.1 (dataset download)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URLs
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify only, download nothing")
    args = parser.parse_args()
    problems = 0
    for manifest_path in MANIFESTS:
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for document in manifest["documents"]:
            url = document.get("source_url")
            if not url:
                continue
            target = manifest_path.parent / document["file"]
            if target.is_file() and sha256(target) == document["sha256"]:
                print(f"ok        {target.relative_to(ROOT)}")
                continue
            if args.check:
                print(f"MISSING   {target.relative_to(ROOT)}")
                problems += 1
                continue
            print(f"download  {url}")
            fetch(url, target)
            if sha256(target) != document["sha256"]:
                print(f"HASH MISMATCH {target.relative_to(ROOT)}: the upstream file changed")
                target.unlink()
                problems += 1
            else:
                print(f"ok        {target.relative_to(ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
