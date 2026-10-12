from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    for path in ("/health", "/api/v1/health"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_readiness(client: TestClient) -> None:
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"database", "migrations", "embedding"}


def test_readiness_fails_when_embedding_model_is_missing(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(
        embedding_provider="sentence-transformers", embedding_model="no-such/model-for-tests"
    )
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["embedding"]["ok"] is False


def test_system_status_flags_demo_mode(client: TestClient) -> None:
    body = client.get("/api/v1/system/status").json()
    assert body["demo_mode"] is True
    assert body["generation"]["provider"] == "extractive-demo"
    assert body["generation"]["is_llm"] is False
    assert body["embedding"]["model"] == "hashing:v1-384"
    assert body["auth_required"] is False
    assert body["limits"]["allowed_extensions"] == [".pdf", ".txt", ".md"]
    assert "api_key" not in str(body).lower()


def test_request_id_is_generated_and_echoed(client: TestClient) -> None:
    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) == 32
    echoed = client.get("/health", headers={"X-Request-ID": "trace-123"}).headers["x-request-id"]
    assert echoed == "trace-123"
    rejected = client.get("/health", headers={"X-Request-ID": "bad id<script>"})
    assert rejected.headers["x-request-id"] != "bad id<script>"


def test_security_headers(client: TestClient) -> None:
    headers = client.get("/api/v1/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["content-security-policy"].startswith("default-src 'none'")


def test_openapi_lists_all_endpoints(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/system/status",
        "/api/v1/documents",
        "/api/v1/documents/{document_id}",
        "/api/v1/documents/{document_id}/chunks",
        "/api/v1/documents/{document_id}/reingest",
        "/api/v1/query",
        "/api/v1/retrieve",
        "/api/v1/evaluations/reports",
        "/api/v1/evaluations/reports/{report_id}",
        "/api/v1/evaluations/runs",
        "/api/v1/evaluations/runs/{run_id}",
        "/api/v1/metrics",
        "/api/v1/metrics/prometheus",
    } <= paths


def test_unknown_route_uses_error_format(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["request_id"] == response.headers["x-request-id"]
