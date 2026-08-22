"""A searched gallery says which rankings produced its order.

`/g?q=` fuses every ranking that answered; the page names them beside
the degraded note it already carried, so a person can tell a one-model
answer from an all-models one -- the distinction the retrieval
provenance exists to keep.
"""

from __future__ import annotations

from litestar.testing import TestClient
from PIL import Image

from db import retrieval
from sg_web.app import build_app


def test_the_grid_names_the_rankings_that_answered(tmp_path, monkeypatch):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (10 * i, 20, 30)).save(root / f"p{i}.png")

    def fused(conn, models_dir, phrase, k, now, *, offline=True, allowed=None):
        ids = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id")]
        return {
            "results": [{"file_id": file_id, "score": 1.0, "sources": {}} for file_id in ids],
            "participants": ["semantic.openclip.ViT-B-32.laion2b_s34b_b79k", "captions", "space.b"],
            "contributors": ["semantic.openclip.ViT-B-32.laion2b_s34b_b79k", "captions"],
            "missing": {"space.b": "not provisioned"},
        }

    monkeypatch.setattr(retrieval, "query", fused)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        page = client.get("/g", params={"q": "a banana"}, headers={"accept": "text/html"})
        assert page.status_code == 200, page.text[:300]
        assert 'data-answered="semantic.openclip.ViT-B-32.laion2b_s34b_b79k captions"' in page.text
        assert "data-degraded" in page.text
        assert 'title="not provisioned">space.b' in page.text
        plain = client.get("/g", headers={"accept": "text/html"}).text
        assert "data-answered" not in plain, "no phrase, no ranking to name"
