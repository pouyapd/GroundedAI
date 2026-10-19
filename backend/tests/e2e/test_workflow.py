"""End-to-end user workflow through the HTTP API only."""

from typing import Any

from fastapi.testclient import TestClient

from groundedai.evaluation.pdf_fixture import make_pdf
from groundedai.evaluation.sample_documents import FERRY_GUIDE, LIBRARY_RULES
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


def _ask(client: TestClient, question: str, document_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"question": question}
    if document_id:
        body["document_ids"] = [document_id]
    response = client.post("/api/v1/query", json=body)
    assert response.status_code == 200
    result: dict[str, Any] = response.json()
    return result


def test_pdf_journey_cites_the_right_page_and_keeps_documents_apart(client: TestClient) -> None:
    ferry = upload(client, FERRY_GUIDE.filename, FERRY_GUIDE.pdf(), "application/pdf").json()
    library = upload(client, LIBRARY_RULES.filename, LIBRARY_RULES.pdf(), "application/pdf").json()
    document = client.get(f"/api/v1/documents/{ferry['id']}").json()
    assert (document["status"], document["page_count"]) == ("ready", 3)
    assert document["title"] == FERRY_GUIDE.title  # taken from the PDF's metadata
    chunks = client.get(f"/api/v1/documents/{ferry['id']}/chunks").json()["items"]
    extracted = " ".join(chunk["text"] for chunk in chunks)
    assert "costs 14 euros" in extracted
    assert "kept on a lead" in extracted

    # The three short pages share one chunk; the evidence still names the exact page.
    answer = _ask(client, "How much does a single adult ticket to Mirrow cost?", ferry["id"])
    assert answer["status"] == "answered"
    assert "14 euros" in answer["answer"]
    (claim,) = answer["claims"]
    assert claim["status"] == "supported"
    evidence = claim["evidence"][0]
    assert (evidence["document_id"], evidence["page"]) == (ferry["id"], 2)
    source = next(s for s in answer["sources"] if s["label"] == evidence["label"])
    assert source["text"][evidence["quote_start"] : evidence["quote_end"]] == evidence["quote"]
    assert "14 euros" in evidence["quote"]

    pets = _ask(client, "Are dogs allowed on the ferry?", ferry["id"])
    assert pets["claims"][0]["evidence"][0]["page"] == 3

    # A question the selected document cannot answer is not answered from another document.
    scoped = _ask(client, "How many items can members borrow?", ferry["id"])
    assert scoped["status"] == "insufficient_evidence"
    assert {s["document_id"] for s in scoped["sources"]} <= {ferry["id"]}
    library_answer = _ask(client, "How many items can members borrow?", library["id"])
    assert "12 items" in library_answer["answer"]
    assert {s["document_id"] for s in library_answer["sources"]} == {library["id"]}

    unsupported = _ask(client, "What time does the last ferry leave Mirrow?", ferry["id"])
    assert unsupported["status"] == "insufficient_evidence"
    assert unsupported["claims"] == []

    # Deleting a document removes it from later answers.
    assert client.delete(f"/api/v1/documents/{library['id']}").status_code == 204
    after = _ask(client, "How many items can members borrow?")
    assert all(s["document_id"] != library["id"] for s in after["sources"])
    assert "12 items" not in after["answer"]
