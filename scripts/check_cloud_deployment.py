"""End-to-end check of a deployed GroundedAI API (for example the Render + Neon deployment).

The API key is read from the GROUNDEDAI_API_KEY environment variable and is never printed.

    GROUNDEDAI_API_KEY=... python scripts/check_cloud_deployment.py https://groundedai-api-jyza.onrender.com

Checks: liveness, readiness (database dialect, migrations at head, pgvector column), API docs,
that requests without a key or with a wrong key are refused without internal details, upload
validation (a non-PDF named .pdf and an oversized file), upload and indexing of the two sample
PDFs in examples/, and three questions with known answers. The sample PDFs are uploaded only
once; later runs reuse them. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SAMPLES = ("harbourline-ferry-guide.pdf", "orchard-library-rules.pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="for example https://groundedai-api-jyza.onrender.com")
    parser.add_argument(
        "--max-upload-mb", type=float, default=5.0, help="the deployment's upload limit"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="seconds; a free instance may be asleep"
    )
    args = parser.parse_args()
    key = os.environ.get("GROUNDEDAI_API_KEY", "")
    if not key:
        print("set GROUNDEDAI_API_KEY", file=sys.stderr)
        return 2
    auth = {"X-API-Key": key}
    client = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout)
    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

    response = client.get("/health")  # also wakes a sleeping free instance
    check("GET /health", response.status_code == 200, response.text[:60])
    ready = client.get("/ready").json()["checks"]
    check("readiness: database", ready["database"]["ok"], ready["database"]["detail"])
    migrations = ready["migrations"]["detail"]
    current, head = (part.split("=")[1] for part in migrations.split())
    check(
        "readiness: migrations at head", ready["migrations"]["ok"] and current == head, migrations
    )
    vector = ready["vector_dimension"]
    check("readiness: vector column", vector["ok"], vector["detail"])
    check("GET /docs", client.get("/docs").status_code == 200)

    response = client.get("/api/v1/documents")
    check("no key: 401", response.status_code == 401)
    response = client.get("/api/v1/documents", headers={"X-API-Key": "x" * 32})
    check("wrong key: 401", response.status_code == 401)
    check(
        "401 body has no key or traceback",
        key not in response.text and "Traceback" not in response.text,
    )

    fake = {"file": ("fake.pdf", b"this is not a PDF", "application/pdf")}
    response = client.post("/api/v1/documents", headers=auth, files=fake)
    check("non-PDF named .pdf rejected", response.status_code == 415, str(response.status_code))
    size = int(args.max_upload_mb * 1024 * 1024) + 10
    big = {"file": ("big.pdf", b"%PDF-1.4\n" + b"0" * size, "application/pdf")}
    response = client.post("/api/v1/documents", headers=auth, files=big)
    check("oversized upload rejected", response.status_code == 413, str(response.status_code))

    listed = client.get("/api/v1/documents", headers=auth).json().get("items", [])
    ids = {doc["filename"]: doc["id"] for doc in listed if doc["filename"] in SAMPLES}
    for name in SAMPLES:
        if name in ids:
            continue
        with (EXAMPLES / name).open("rb") as handle:
            response = client.post(
                "/api/v1/documents", headers=auth, files={"file": (name, handle, "application/pdf")}
            )
        check(f"upload {name}: 202", response.status_code == 202, str(response.status_code))
        ids[name] = response.json()["id"]
    for name, doc_id in ids.items():
        status = ""
        for _ in range(60):
            status = client.get(f"/api/v1/documents/{doc_id}", headers=auth).json()["status"]
            if status in ("ready", "failed"):
                break
            time.sleep(2)
        check(f"{name} indexed", status == "ready", status)

    def ask(question: str, name: str) -> dict:
        body = {"question": question, "document_ids": [ids[name]]}
        response = client.post("/api/v1/query", headers=auth, json=body)
        check(f"query 200 ({name})", response.status_code == 200, str(response.status_code))
        return response.json()

    ferry = ask("How much does a single adult ticket to Mirrow cost?", SAMPLES[0])
    check("answer: 14 euros", "14 euros" in ferry.get("answer", ""), ferry.get("answer", "")[:100])
    pages = [
        quote.get("page")
        for claim in ferry.get("claims", [])
        for quote in claim.get("evidence", [])
    ]
    pages += [source.get("page") for source in ferry.get("sources", [])]
    check("evidence on page 2", 2 in pages, str(sorted({p for p in pages if p is not None})))
    refused = ask("How many items can members borrow?", SAMPLES[0])
    check("wrong document: insufficient evidence", refused.get("status") == "insufficient_evidence")
    library = ask("How many items can members borrow?", SAMPLES[1])
    check(
        "answer: 12 items", "12 items" in library.get("answer", ""), library.get("answer", "")[:100]
    )

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
