"""End-to-end user workflow through the HTTP API only."""

from fastapi.testclient import TestClient

from groundedai.evaluation.pdf_fixture import make_pdf
from tests.conftest import upload


def test_upload_ask_inspect_delete(client: TestClient) -> None:
    pdf = make_pdf(
        [
            "Solvane SX-5 Manual\nThe DC input voltage range is 150 to 850 V.",
            "Error codes\nE07 means fan failure; check that the fan is not blocked.",
        ],
        title="SX-5 Manual",
    )
    assert client.get("/api/v1/system/status").json()["demo_mode"] is True

    created = upload(client, "sx5.pdf", pdf, "application/pdf").json()
    document = client.get(f"/api/v1/documents/{created['id']}").json()
    assert document["status"] == "ready"

    answer = client.post(
        "/api/v1/query", json={"question": "What does error code E07 mean?"}
    ).json()
    assert answer["status"] == "answered"
    assert "fan failure" in answer["answer"]
    cited = [source for source in answer["sources"] if source["cited"]]
    assert cited[0]["page_start"] <= 2 <= cited[0]["page_end"]  # both short pages share a chunk
    assert cited[0]["document_title"] == "SX-5 Manual"
    assert cited[0]["label"] in answer["answer"]

    unanswerable = client.post(
        "/api/v1/query", json={"question": "What is the retail price of the inverter?"}
    ).json()
    assert unanswerable["status"] == "insufficient_evidence"

    assert client.delete(f"/api/v1/documents/{created['id']}").status_code == 204
    after = client.post("/api/v1/query", json={"question": "What does error code E07 mean?"}).json()
    assert after["status"] == "insufficient_evidence"

    metrics = client.get("/api/v1/metrics").json()
    assert metrics["queries_in_window"] == 3
