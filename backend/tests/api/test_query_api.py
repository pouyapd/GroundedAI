import socket

from fastapi.testclient import TestClient

from tests.conftest import INJECTION_TEXT, TRAVEL_TEXT, upload


def test_query_returns_grounded_answer(client: TestClient) -> None:
    upload(client, "travel.txt", TRAVEL_TEXT.encode())
    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is the hotel limit in capital cities?",
            "include_diagnostics": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["provider"]["is_llm"] is False
    assert body["citation_check"]["all_valid"] is True
    assert body["retrieval"]["mode"] == "hybrid"
    cited = [source for source in body["sources"] if source["cited"]]
    assert cited[0]["document_title"] == "Travel Policy"  # first line of the text file
    assert body["timings"]["total_ms"] > 0


def test_query_on_empty_index_abstains(client: TestClient) -> None:
    body = client.post("/api/v1/query", json={"question": "What is the hotel limit?"}).json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer"] is None


def test_query_validation_errors_do_not_echo_input(client: TestClient) -> None:
    for payload in (
        {"question": ""},
        {"question": "ok question", "top_k": 0},
        {"question": "ok question", "unexpected": True},
        {"question": "ok question", "document_ids": ["../../etc"]},
        {"question": "ok question", "retrieval_mode": "magic"},
    ):
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 422, payload
        error = response.json()["error"]
        assert error["code"] == "invalid_input"
        assert "../../etc" not in response.text


def test_too_long_question_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"question": "word " * 300})
    assert response.status_code == 422


def test_retrieve_endpoint_returns_scores(client: TestClient) -> None:
    upload(client, "travel.txt", TRAVEL_TEXT.encode())
    for mode in ("lexical", "dense", "hybrid"):
        body = client.post(
            "/api/v1/retrieve", json={"query": "hotel limit", "retrieval_mode": mode, "top_k": 3}
        ).json()
        assert body["results"], mode
        assert body["diagnostics"]["mode"] == mode
        assert body["results"][0]["rank"] == 1


def test_injected_document_is_flagged_in_response(client: TestClient) -> None:
    upload(client, "supplier.txt", INJECTION_TEXT.encode())
    body = client.post(
        "/api/v1/query", json={"question": "What are the payment terms for suppliers?"}
    ).json()
    assert "PWNED" not in (body["answer"] or "")
    assert any(source["injection_flags"] for source in body["sources"])


def test_unavailable_model_returns_503_without_fallback(client_factory) -> None:  # type: ignore[no-untyped-def]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = client_factory(
        generation_provider="ollama", ollama_base_url=f"http://127.0.0.1:{port}"
    )
    upload(client, "travel.txt", TRAVEL_TEXT.encode())
    status = client.get("/api/v1/system/status").json()
    assert status["demo_mode"] is False
    assert status["generation"]["available"] is False
    response = client.post("/api/v1/query", json={"question": "hotel limit in capital cities"})
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_unavailable"
    assert "ollama" in error["message"].lower()
    assert "answer" not in response.json()


def test_unknown_document_filter_is_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query", json={"question": "hotel limit", "document_ids": ["doc_" + "a" * 20]}
    )
    assert response.status_code == 404
