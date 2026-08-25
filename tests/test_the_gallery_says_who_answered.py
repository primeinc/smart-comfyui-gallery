"""A searched gallery says which rankings produced its order.

`/g?q=` fuses every ranking that responded; the page names them beside
the degraded note it already carried, so a person can tell a one-model
result set from an all-models one -- the distinction the retrieval
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


def test_a_phrase_no_caption_mentions_is_said_quietly_not_as_degraded(tmp_path, monkeypatch):
    """A word match that matched nothing is the ordinary outcome, not a
    model that failed: the grid notes it beside the result set and keeps the
    degraded note for spaces that could not respond."""
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(root / "p.png")

    def fused(conn, models_dir, phrase, k, now, *, offline=True, allowed=None):
        ids = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id")]
        return {
            "results": [{"file_id": file_id, "score": 1.0, "sources": {}} for file_id in ids],
            "participants": ["space.a", "captions"],
            "contributors": ["space.a"],
            "missing": {},
            "unmatched": {"captions": "no caption mentions a word of the phrase in this scope"},
        }

    monkeypatch.setattr(retrieval, "query", fused)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        page = client.get("/g", params={"q": "helicopter"}, headers={"accept": "text/html"}).text
        assert "data-captions-unmatched" in page
        assert "no caption mentions a word of the phrase" in page
        assert "data-degraded" not in page, "nothing failed to answer"


def test_the_gallery_opens_its_question_on_the_timeline(tmp_path):
    """Every query but a phrase has a timeline: the header's link is
    the canonical query; a semantic phrase ranks, it does not scope,
    so a searched gallery offers none."""
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "p.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        whole = client.get("/g", headers={"accept": "text/html"}).text
        assert 'data-timeline-link href="/timeline"' in whole
        scoped = client.get("/g", params={"folder": "lib", "kind": "image"}, headers={"accept": "text/html"}).text
        assert 'data-timeline-link href="/timeline?folder=lib&amp;kind=image"' in scoped
        assert client.get("/timeline", params={"folder": "lib", "kind": "image"}).status_code == 200
