from tests.conftest import TRAVEL_TEXT, upload

KEY = "k" * 32


def test_api_key_protects_data_endpoints(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(api_key=KEY)
    # Probes and status stay open for orchestrators and the UI's setup screen.
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/ready").status_code == 200
    assert client.get("/api/v1/system/status").json()["auth_required"] is True

    protected = [
        ("get", "/api/v1/documents", None),
        ("post", "/api/v1/query", {"question": "hotel limit"}),
        ("post", "/api/v1/retrieve", {"query": "hotel limit"}),
        ("get", "/api/v1/metrics", None),
        ("get", "/api/v1/evaluations/reports", None),
    ]
    for method, path, payload in protected:
        response = (
            getattr(client, method)(path, json=payload)
            if payload
            else getattr(client, method)(path)
        )
        assert response.status_code == 401, path
        assert response.json()["error"]["code"] == "unauthorized"
        wrong = client.request(method.upper(), path, json=payload, headers={"X-API-Key": "wrong"})
        assert wrong.status_code == 401, path

    assert upload(client, "a.txt", TRAVEL_TEXT.encode()).status_code == 401
    good = upload(client, "a.txt", TRAVEL_TEXT.encode(), headers={"X-API-Key": KEY})
    assert good.status_code == 202
    listing = client.get("/api/v1/documents", headers={"X-API-Key": KEY})
    assert listing.status_code == 200


def test_rate_limit_returns_429_with_retry_after(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(rate_limit_enabled=True, rate_limit_query_per_minute=2)
    for _ in range(2):
        assert client.post("/api/v1/query", json={"question": "hotel limit"}).status_code == 200
    limited = client.post("/api/v1/query", json={"question": "hotel limit"})
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.json()["error"]["code"] == "rate_limited"


def test_cors_allows_only_configured_origins(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(cors_origins=["https://ui.example"])
    allowed = client.options(
        "/api/v1/query",
        headers={"Origin": "https://ui.example", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "https://ui.example"
    denied = client.options(
        "/api/v1/query",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in denied.headers


def test_unhandled_errors_hide_internals(client_factory, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(raise_server_exceptions=False)
    services = client.app.state.services

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("secret internal detail at C:\\internal\\path.py")

    monkeypatch.setattr(services.status, "system_status", explode)
    response = client.get("/api/v1/system/status")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert "path.py" not in response.text
