"""Write the two sample PDFs used by the end-to-end checks to ``examples/``.

    python scripts/make_sample_pdfs.py

The output is deterministic, so the committed files can be checked against it.
"""

from __future__ import annotations

from pathlib import Path

from groundedai.evaluation.sample_documents import SAMPLES

OUT = Path(__file__).resolve().parents[1] / "examples"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for sample in SAMPLES:
        path = OUT / sample.filename
        path.write_bytes(sample.pdf())
        print(f"wrote {path.relative_to(OUT.parent)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
