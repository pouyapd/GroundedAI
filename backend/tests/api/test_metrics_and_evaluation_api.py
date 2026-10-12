import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import REPO_ROOT, TRAVEL_TEXT, upload


def test_operational_metrics_reflect_activity(client: TestClient) -> None:
    upload(client, "travel.txt", TRAVEL_TEXT.encode())
    client.post("/api/v1/query", json={"question": "What is the hotel limit in capital cities?"})
    client.post("/api/v1/query", json={"question": "Who won the 2018 FIFA World Cup?"})
    metrics = client.get("/api/v1/metrics").json()
    assert metrics["documents_total"] == 1
    assert metrics["documents_by_status"]["ready"] == 1
    assert metrics["chunks_total"] >= 1
    assert metrics["queries_in_window"] == 2
    assert sum(metrics["queries_by_status"].values()) == 2
    assert metrics["latency_ms"]["total"]["n"] == 2
    assert len(metrics["hourly"]) == 24
    assert metrics["http_requests_total"] >= 3  # the in-flight metrics request is not counted yet


def test_prometheus_endpoint(client: TestClient) -> None:
    client.get("/health")
    response = client.get("/api/v1/metrics/prometheus")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "groundedai_http_requests_total" in response.text


def test_published_reports_are_listed(client_factory, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    evaluation_dir = tmp_path / "evaluation"
    published = evaluation_dir / "reports" / "published"
    published.mkdir(parents=True)
    report = {
        "report_id": "retrieval-test-1",
        "kind": "retrieval",
        "title": "Test report",
        "generated_at": "2026-10-11T00:00:00+00:00",
        "dataset": {"name": "tiny"},
        "configurations": [{"name": "A"}, {"name": "B"}],
    }
    (published / "retrieval-test-1.json").write_text(json.dumps(report), encoding="utf-8")
    (published / "broken.json").write_text("{not json", encoding="utf-8")
    client = client_factory(evaluation_dir=evaluation_dir)
    listing = client.get("/api/v1/evaluations/reports").json()
    assert [item["id"] for item in listing["items"]] == ["retrieval-test-1"]
    assert listing["items"][0]["n_configurations"] == 2
    assert (
        client.get("/api/v1/evaluations/reports/retrieval-test-1").json()["title"] == "Test report"
    )
    assert client.get("/api/v1/evaluations/reports/missing").status_code == 404
    assert client.get("/api/v1/evaluations/reports/..%2F..%2Fsecret").status_code in (404, 422)


def test_evaluation_run_lifecycle(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(evaluation_dir=REPO_ROOT / "evaluation")
    created = client.post("/api/v1/evaluations/runs", json={})
    assert created.status_code == 202
    run_id = created.json()["id"]
    detail = client.get(f"/api/v1/evaluations/runs/{run_id}").json()
    assert detail["status"] == "completed", detail.get("error_message")
    assert set(detail["summary"]) == {"lexical", "dense", "hybrid-rrf"}
    assert detail["report"]["dataset"]["name"] == "groundedai-synthetic-v1"
    history = client.get("/api/v1/evaluations/runs").json()
    assert history["total"] == 1
    assert client.get("/api/v1/evaluations/runs/run_" + "0" * 16).status_code == 404
    # The live index of the instance is untouched by evaluation runs.
    assert client.get("/api/v1/documents").json()["total"] == 0
