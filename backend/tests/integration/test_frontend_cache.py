"""Cache headers on the built UI served by the API."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import ClientFactory


def test_pages_are_revalidated_and_hashed_assets_cached(
    client_factory: ClientFactory, tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "demo").mkdir()
    (dist / "index.html").write_text("<!doctype html><title>t</title>", "utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", "utf-8")
    (dist / "demo" / "manifest.json").write_text('{"exported_at": "x"}', "utf-8")
    client = client_factory(frontend_dist=dist)
    for path in ("/", "/index.html", "/demo/manifest.json"):
        assert client.get(path).headers["cache-control"] == "no-cache", path
    asset = client.get("/assets/index-abc123.js")
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert "immutable" not in health.headers.get("cache-control", "")
