"""Documents, chunks and answers survive a restart on the same data directory."""

from fastapi.testclient import TestClient

from groundedai.main import create_app
from tests.conftest import SECURITY_TEXT, TRAVEL_TEXT, SettingsFactory, upload

QUESTION = {"question": "What is the hotel limit per night in capital cities?"}


def _ids(client: TestClient) -> set[str]:
    return {item["id"] for item in client.get("/api/v1/documents").json()["items"]}


def test_data_and_answers_survive_restarts(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with TestClient(create_app(settings)) as first:
        travel = upload(first, "travel.txt", TRAVEL_TEXT.encode()).json()
        security = upload(first, "security.txt", SECURITY_TEXT.encode()).json()
        chunks = first.get(f"/api/v1/documents/{travel['id']}/chunks").json()
        before = first.post("/api/v1/query", json=QUESTION).json()
    assert before["status"] == "answered"
    uploads = sorted(path.name for path in settings.uploads_dir.rglob("*") if path.is_file())
    assert len(uploads) == 2

    # Second process on the same data directory: everything is still there and answers match.
    with TestClient(create_app(make_settings())) as second:
        assert _ids(second) == {travel["id"], security["id"]}
        documents = second.get("/api/v1/documents").json()["items"]
        assert all(document["status"] == "ready" for document in documents)
        assert second.get(f"/api/v1/documents/{travel['id']}/chunks").json() == chunks
        after = second.post("/api/v1/query", json=QUESTION).json()
        assert after["answer"] == before["answer"]
        assert [s["chunk_id"] for s in after["sources"]] == [
            s["chunk_id"] for s in before["sources"]
        ]
        assert after["claims"] == before["claims"]
        assert after["conflicts"] == before["conflicts"]

        # Re-ingesting and deleting still work after the restart.
        assert second.post(f"/api/v1/documents/{travel['id']}/reingest").status_code == 202
        reingested = second.get(f"/api/v1/documents/{travel['id']}").json()
        assert (reingested["status"], reingested["attempts"]) == ("ready", 2)
        assert second.delete(f"/api/v1/documents/{security['id']}").status_code == 204

    # Third process: the deletion and the re-ingestion were stored, the deleted original is gone.
    with TestClient(create_app(make_settings())) as third:
        assert _ids(third) == {travel["id"]}
        assert third.get(f"/api/v1/documents/{travel['id']}").json()["attempts"] == 2
        assert third.post("/api/v1/query", json=QUESTION).json()["status"] == "answered"
        security_answer = third.post(
            "/api/v1/query", json={"question": "How long must passwords be?"}
        ).json()
        assert "14 characters" not in (security_answer["answer"] or "")
    remaining = [path for path in settings.uploads_dir.rglob("*") if path.is_file()]
    assert len(remaining) == 1
