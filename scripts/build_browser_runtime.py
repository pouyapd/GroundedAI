"""Prepare the Python runtime of the online demo (GroundedAI in the browser, via Pyodide).

    python scripts/build_browser_runtime.py lock     # resolve pinned wheels on PyPI (network)
    python scripts/build_browser_runtime.py build    # write frontend/public/runtime/

``frontend/browser-runtime.lock.json`` pins everything the demo loads: the Pyodide release
(served by the jsDelivr CDN), the Pyodide-built packages it takes from that release, and every
other wheel with its PyPI URL and SHA-256. ``lock`` fills in URLs and hashes for the pinned
versions; ``build`` downloads those wheels (checking each hash), builds the GroundedAI wheel
from ``backend/``, copies the sample PDFs from ``examples/``, and writes ``manifest.json``.
The output is generated, git-ignored, and served next to the built UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "frontend" / "browser-runtime.lock.json"
OUT = ROOT / "frontend" / "public" / "runtime"
SAMPLES = ("harbourline-ferry-guide.pdf", "orchard-library-rules.pdf")


def _read_lock() -> dict[str, Any]:
    return dict(json.loads(LOCK.read_text(encoding="utf-8")))


def lock() -> None:
    data = _read_lock()
    abi = data["pyodide"]["abi"]
    python_tag = data["pyodide"]["python_tag"]
    for wheel in data["wheels"]:
        url = f"https://pypi.org/pypi/{wheel['name']}/{wheel['version']}/json"
        with urllib.request.urlopen(url, timeout=30) as response:
            files = json.load(response)["urls"]
        pure = [f for f in files if f["filename"].endswith("-none-any.whl")]
        built = [f for f in files if f["filename"].endswith(f"{python_tag}-{abi}_wasm32.whl")]
        chosen = (pure or built or [None])[0]
        if chosen is None:
            raise SystemExit(f"no browser wheel for {wheel['name']} {wheel['version']}")
        wheel.update(file=chosen["filename"], url=chosen["url"], sha256=chosen["digests"]["sha256"])
        print(f"{wheel['name']} {wheel['version']}: {chosen['filename']}")
    LOCK.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def _download(url: str, sha256: str, target: Path) -> None:
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned PyPI URL
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{target.name}: SHA-256 {digest} does not match the lock ({sha256})")
    target.write_bytes(payload)


def build() -> None:
    data = _read_lock()
    wheels = OUT / "wheels"
    if OUT.exists():
        shutil.rmtree(OUT)
    wheels.mkdir(parents=True)
    for wheel in data["wheels"]:
        if not wheel.get("sha256"):
            raise SystemExit(f"{wheel['name']} has no hash; run the 'lock' command first")
        _download(wheel["url"], wheel["sha256"], wheels / wheel["file"])
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", str(ROOT / "backend")]
    subprocess.run([*command, "-w", str(wheels)], check=True)  # noqa: S603 - fixed arguments
    own = sorted(path.name for path in wheels.glob("groundedai-*.whl"))
    if len(own) != 1:
        raise SystemExit(f"expected one GroundedAI wheel, found {own}")
    samples = OUT / "samples"
    samples.mkdir()
    for name in SAMPLES:
        shutil.copyfile(ROOT / "examples" / name, samples / name)
    manifest = {
        "pyodide": data["pyodide"],
        "packages": data["packages"],
        "wheels": [wheel["file"] for wheel in data["wheels"]] + own,
        "samples": list(SAMPLES),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    print(f"Wrote {OUT} ({len(manifest['wheels'])} wheels, {size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("lock", "build"))
    args = parser.parse_args()
    lock() if args.command == "lock" else build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
