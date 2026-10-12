"""Lightweight API latency benchmark.

Sends sequential requests for the bundled dataset questions to /api/v1/retrieve and
/api/v1/query and reports client-side latency percentiles, together with the hardware and
power source (laptops are much slower on battery).

    # in-process: temporary SQLite index of the sample corpus, settings from the environment
    python scripts/bench_api.py --requests 100
    # against a running server (measures HTTP too)
    python scripts/bench_api.py --url http://127.0.0.1:8000 --requests 100 [--api-key KEY]

The in-process mode uses FastAPI's TestClient, so it measures application time without network
overhead. Results are printed as JSON and optionally written with --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from groundedai.evaluation.dataset import load_dataset  # noqa: E402
from groundedai.evaluation.environment import environment_info  # noqa: E402
from groundedai.observability.metrics import summarize  # noqa: E402


def run(
    client: Any, questions: list[str], requests: int, headers: dict[str, str]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, path, body_key in (
        ("retrieve", "/api/v1/retrieve", "query"),
        ("query", "/api/v1/query", "question"),
    ):
        latencies, statuses = [], {}
        for question in islice(cycle(questions), requests):
            started = time.perf_counter()
            response = client.post(path, json={body_key: question}, headers=headers)
            latencies.append((time.perf_counter() - started) * 1000)
            statuses[str(response.status_code)] = statuses.get(str(response.status_code), 0) + 1
        results[name] = {"latency_ms": summarize(latencies), "status_codes": statuses}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", help="benchmark a running server instead of an in-process app")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(REPO_ROOT / "evaluation" / "datasets" / "groundedai-synthetic-v1")
    questions = [question.question for question in dataset.questions]
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    if args.url:
        import httpx

        with httpx.Client(base_url=args.url, timeout=300, trust_env=False) as client:
            status = client.get("/api/v1/system/status", headers=headers).json()
            results = run(client, questions, args.requests, headers)
        mode = f"HTTP against {args.url}"
    else:
        from fastapi.testclient import TestClient

        from groundedai.core.config import get_settings
        from groundedai.main import create_app

        with tempfile.TemporaryDirectory(
            prefix="groundedai-bench-", ignore_cleanup_errors=True
        ) as tmp:
            settings = get_settings().model_copy(
                update={
                    "data_dir": Path(tmp),
                    "database_url": None,
                    "rate_limit_enabled": False,
                    "log_level": "WARNING",
                }
            )
            with TestClient(create_app(settings)) as client:
                for document in dataset.manifest.documents:
                    path = dataset.document_path(document.key)
                    client.post("/api/v1/documents", files={"file": (path.name, path.read_bytes())})
                status = client.get("/api/v1/system/status").json()
                client.post("/api/v1/query", json={"question": "warm up"})
                results = run(client, questions, args.requests, headers)
        mode = "in-process (FastAPI TestClient, no network)"

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "requests_per_endpoint": args.requests,
        "embedding": status["embedding"]["model"],
        "generation": f"{status['generation']['provider']}/{status['generation']['model']}",
        "database": status["database_backend"],
        "environment": environment_info(),
        "results": results,
        "note": "Sequential requests, one client. Not a load test.",
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
