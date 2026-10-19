from fastapi.testclient import TestClient

from tests.conftest import SECURITY_TEXT, TRAVEL_TEXT, upload


def test_upload_is_processed_and_listed(client: TestClient) -> None:
    response = upload(client, "travel.txt", TRAVEL_TEXT.encode())
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "pending"
    # TestClient runs background tasks before returning.
    document = client.get(f"/api/v1/documents/{created['id']}").json()
    assert document["status"] == "ready"
    assert document["chunk_count"] >= 1
    listing = client.get("/api/v1/documents").json()
    assert listing["total"] == 1
    assert listing["items"][0]["id"] == created["id"]


def test_pdf_upload_records_pages(client: TestClient, sample_pdf: bytes) -> None:
    created = upload(client, "manual.pdf", sample_pdf, "application/pdf").json()
    document = client.get(f"/api/v1/documents/{created['id']}").json()
    assert document["page_count"] == 2
    chunks = client.get(f"/api/v1/documents/{created['id']}/chunks").json()
    assert chunks["total"] >= 1
    assert chunks["items"][0]["page_start"] == 1


def test_pagination_and_status_filter(client: TestClient) -> None:
    for index in range(3):
        upload(client, f"doc{index}.txt", f"Document number {index} about kayaks.".encode())
    page = client.get("/api/v1/documents", params={"offset": 1, "limit": 1}).json()
    assert page["total"] == 3
    assert len(page["items"]) == 1
    assert client.get("/api/v1/documents", params={"status": "failed"}).json()["total"] == 0
    assert client.get("/api/v1/documents", params={"limit": 1000}).status_code == 422
    assert client.get("/api/v1/documents", params={"status": "weird"}).status_code == 422


def test_duplicate_upload_returns_conflict(client: TestClient) -> None:
    first = upload(client, "a.txt", TRAVEL_TEXT.encode()).json()
    response = upload(client, "b.txt", TRAVEL_TEXT.encode())
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "duplicate_document"
    assert error["details"]["document_id"] == first["id"]


def test_rejected_uploads(client: TestClient) -> None:
    cases = [
        (("evil.exe", b"MZ...", "application/octet-stream"), 415, "unsupported_file_type"),
        (("fake.pdf", b"not a pdf", "application/pdf"), 415, "unsupported_file_type"),
        (("empty.txt", b"", "text/plain"), 422, "empty_document"),
        (("binary.txt", b"\x00\x01\x02", "text/plain"), 415, "unsupported_file_type"),
    ]
    for (name, content, content_type), status, code in cases:
        response = upload(client, name, content, content_type)
        assert response.status_code == status, name
        assert response.json()["error"]["code"] == code
    assert client.get("/api/v1/documents").json()["total"] == 0


def test_upload_size_limit(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(max_upload_bytes=2048)
    response = upload(client, "big.txt", b"a" * 4096)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_request_body_limit_rejects_huge_requests(client_factory) -> None:  # type: ignore[no-untyped-def]
    client = client_factory(max_upload_bytes=2048)
    response = upload(client, "huge.txt", b"a" * (2 * 1024 * 1024))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_malformed_pdf_is_recorded_as_failed(client: TestClient) -> None:
    created = upload(client, "broken.pdf", b"%PDF-1.7 broken", "application/pdf").json()
    document = client.get(f"/api/v1/documents/{created['id']}").json()
    assert document["status"] == "failed"
    assert document["error_code"] == "document_parse_failed"
    assert "Traceback" not in (document["error_message"] or "")


def test_path_traversal_names_are_neutralised(client: TestClient) -> None:
    created = upload(client, "../../../../etc/passwd.txt", b"harmless text").json()
    assert created["filename"] == "passwd.txt"


def test_delete_and_reingest(client: TestClient) -> None:
    created = upload(client, "security.txt", SECURITY_TEXT.encode()).json()
    reingest = client.post(f"/api/v1/documents/{created['id']}/reingest")
    assert reingest.status_code == 202
    assert client.get(f"/api/v1/documents/{created['id']}").json()["attempts"] == 2
    assert client.delete(f"/api/v1/documents/{created['id']}").status_code == 204
    assert client.get(f"/api/v1/documents/{created['id']}").status_code == 404
    assert client.delete(f"/api/v1/documents/{created['id']}").status_code == 404


def test_invalid_document_ids_are_rejected(client: TestClient) -> None:
    assert client.get("/api/v1/documents/not-an-id").status_code == 422
    assert client.get("/api/v1/documents/doc_" + "0" * 20).status_code == 404
