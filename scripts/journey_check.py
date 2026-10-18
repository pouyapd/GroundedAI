"""Run the PDF question-answering journey through the HTTP API with the configured providers.

    GROUNDEDAI_GENERATION_PROVIDER=huggingface-local python scripts/journey_check.py --out r.json

The app runs in-process (FastAPI test client, temporary data folder) with the providers set
in the environment, so this is the way to check a real model on the journey: upload the two
sample PDFs, ask seven questions (four answerable from the ferry guide, one it cannot answer,
one answered only by the other document, and that one scoped to the other document), and
record each answer with its check results, evidence pages and timings. ``expected`` is a
plain substring test on the answer, a rough first look rather than a score; read the answers.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from groundedai.core.config import Settings  # noqa: E402
from groundedai.evaluation.environment import code_version  # noqa: E402
from groundedai.evaluation.sample_documents import FERRY_GUIDE, LIBRARY_RULES  # noqa: E402
from groundedai.main import create_app  # noqa: E402

# (question, scope, substring expected in an answer, or None when it should not be answered)
QUESTIONS: list[tuple[str, str, str | None]] = [
    ("How much does a single adult ticket from Kessport to Mirrow cost?", "ferry", "14"),
    ("Do young children have to pay for the ferry?", "ferry", "free"),
    ("How much does it cost to take a bicycle?", "ferry", "6"),
    ("Can I bring my dog on the ferry?", "ferry", "lead"),
    ("What time does the last ferry leave Mirrow?", "ferry", None),
    ("How many items can library members borrow?", "ferry", None),
    ("How many items can library members borrow?", "library", "12"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="groundedai-journey-") as work:
        settings = Settings(
            data_dir=Path(work) / "data", rate_limit_enabled=False, log_level="WARNING"
        )
        with TestClient(create_app(settings)) as client:
            status = client.get("/api/v1/system/status").json()
            ids = {}
            for key, sample in (("ferry", FERRY_GUIDE), ("library", LIBRARY_RULES)):
                files = {"file": (sample.filename, sample.pdf(), "application/pdf")}
                created = client.post("/api/v1/documents", files=files).json()
                document = client.get(f"/api/v1/documents/{created['id']}").json()
                if document["status"] != "ready":
                    print(f"{sample.filename} did not ingest: {document['error_message']}")
                    return 1
                ids[key] = created["id"]
            rows = []
            for question, scope, expected in QUESTIONS:
                started = time.perf_counter()
                response = client.post(
                    "/api/v1/query", json={"question": question, "document_ids": [ids[scope]]}
                )
                elapsed = time.perf_counter() - started
                body: dict[str, Any] = response.json()
                if response.status_code != 200:
                    rows.append({"question": question, "scope": scope, "error": body})
                    print(f"ERROR {question}: {body}")
                    continue
                answered = body["status"] == "answered"
                as_expected = (
                    (answered and expected.casefold() in body["answer"].casefold())
                    if expected
                    else not answered
                )
                verification = body.get("verification") or {}
                rows.append(
                    {
                        "question": question,
                        "scope": scope,
                        "expected": expected,
                        "as_expected": as_expected,
                        "status": body["status"],
                        "answer": body["answer"],
                        "verdict": verification.get("verdict"),
                        "claims": [
                            {
                                "text": claim["text"],
                                "status": claim["status"],
                                "pages": [item.get("page") for item in claim["evidence"]],
                                "documents": sorted({e["document_id"] for e in claim["evidence"]}),
                            }
                            for claim in body.get("claims", [])
                        ],
                        "source_documents": sorted({s["document_id"] for s in body["sources"]}),
                        "seconds": round(elapsed, 1),
                    }
                )
                print(
                    f"{'ok  ' if as_expected else 'MISS'} {body['status']:<22} "
                    f"{verification.get('verdict') or '-':<19} {elapsed:5.1f}s  {question}"
                )
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_version": code_version(),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "providers": {key: status[key] for key in ("embedding", "generation", "reranker")},
        "documents": {key: value for key, value in ids.items()},
        "as_expected": sum(1 for row in rows if row.get("as_expected")),
        "questions": len(rows),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{report['as_expected']} of {report['questions']} as expected; report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
