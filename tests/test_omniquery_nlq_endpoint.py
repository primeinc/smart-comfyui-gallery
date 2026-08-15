"""Tests for the local OmniQuery NLQ endpoint (WI-31 wave 3):
POST /galleryout/api/omniquery/nlq -- natural language -> parser router ->
typed AST -> validated -> compiled read-only SELECT, persisted into
omniquery_sessions/omniquery_results exactly like the legacy manual-SQL
endpoint (execute_omniquery). Also covers the template smoke check for the
enable_ai_dam-guarded lightbox/aidam-panel markup.

Every test here pins the endpoint's lazily-built router singleton to a
heuristic-only Router so results are deterministic and the suite never
invokes the real (installed) needle2/fallback_qwen engines -- mirroring
tests/test_parsers_router.py's own policy of never calling .route() against
the real backends in a unit test.
"""

from __future__ import annotations

import pytest

from omniquery.parsers import get_backend
from omniquery.parsers.router import Router, load_thresholds

_NLQ_URL = "/galleryout/api/omniquery/nlq"
_EXEC_URL = "/galleryout/api/omniquery/execute"


def _heuristic_only_router() -> Router:
    return Router(primary=None, fallback=None, heuristic=get_backend("heuristic"),
                  thresholds=load_thresholds())


@pytest.fixture()
def nlq_router(smartgallery_app, monkeypatch):
    """Force the endpoint's module-level router singleton to a
    heuristic-only Router for deterministic, fast tests."""
    monkeypatch.setattr(smartgallery_app, "_omniquery_router", _heuristic_only_router())
    return smartgallery_app


@pytest.fixture()
def seeded_files(smartgallery_app):
    """Seed a handful of files rows directly into the app's real DB: two
    favorited videos, one non-favorited video, one favorited image, one
    non-favorited image. Cleaned up afterwards since this DB is shared
    (session-scoped) with other test modules."""
    file_ids = [
        "nlqtest-vid-fav-1", "nlqtest-vid-fav-2", "nlqtest-vid-plain",
        "nlqtest-img-fav", "nlqtest-img-plain",
    ]
    rows = [
        (file_ids[0], "/nlqtest/vid_fav_1.mp4", 1000.0, "vid_fav_1.mp4", "video", 1),
        (file_ids[1], "/nlqtest/vid_fav_2.mp4", 1001.0, "vid_fav_2.mp4", "video", 1),
        (file_ids[2], "/nlqtest/vid_plain.mp4", 1002.0, "vid_plain.mp4", "video", 0),
        (file_ids[3], "/nlqtest/img_fav.png", 1003.0, "img_fav.png", "image", 1),
        (file_ids[4], "/nlqtest/img_plain.png", 1004.0, "img_plain.png", "image", 0),
    ]
    with smartgallery_app.get_db_connection() as conn:
        conn.executemany(
            "INSERT INTO files (id, path, mtime, name, type, is_favorite) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    try:
        yield file_ids
    finally:
        with smartgallery_app.get_db_connection() as conn:
            conn.executemany("DELETE FROM omniquery_results WHERE file_id = ?", [(fid,) for fid in file_ids])
            conn.executemany("DELETE FROM files WHERE id = ?", [(fid,) for fid in file_ids])
            conn.commit()


# ---------------------------------------------------------------------------
# (A)/(D): the NLQ endpoint itself
# ---------------------------------------------------------------------------

def test_favorite_videos_success_persists_session(smartgallery_app, nlq_router, seeded_files):
    client = smartgallery_app.app.test_client()
    resp = client.post(_NLQ_URL, json={"query": "favorite videos"})
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "success"
    assert data["backend"] == "heuristic"
    assert data["session_id"]
    assert data["count"] == 2
    assert data["sql"].startswith("SELECT DISTINCT f.id FROM files f")
    assert data["ast"]["result"] == "ids"

    with smartgallery_app.get_db_connection() as conn:
        session_row = conn.execute(
            "SELECT raw_sql FROM omniquery_sessions WHERE session_id = ?", (data["session_id"],)
        ).fetchone()
        result_rows = conn.execute(
            "SELECT file_id FROM omniquery_results WHERE session_id = ?", (data["session_id"],)
        ).fetchall()

    assert session_row is not None
    assert session_row["raw_sql"].startswith("-- OmniQuery local:")
    assert "favorite videos" in session_row["raw_sql"]
    result_ids = {r["file_id"] for r in result_rows}
    assert result_ids == {"nlqtest-vid-fav-1", "nlqtest-vid-fav-2"}


def test_garbage_query_is_unsupported_with_trace(smartgallery_app, nlq_router):
    client = smartgallery_app.app.test_client()
    resp = client.post(_NLQ_URL, json={"query": "qwertyuiop asdf"})
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "unsupported"
    assert data["reasons"]
    assert isinstance(data["trace"], list) and len(data["trace"]) >= 1
    assert data["trace"][0]["backend"] == "heuristic"
    assert "sql" not in data


def test_count_query(smartgallery_app, nlq_router, seeded_files):
    client = smartgallery_app.app.test_client()
    resp = client.post(_NLQ_URL, json={"query": "how many images"})
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["status"] == "success"
    assert data["kind"] == "count"
    assert data["count"] == 2
    assert data["backend"] == "heuristic"
    assert "session_id" not in data


def test_rejects_non_json_body_cleanly(smartgallery_app, nlq_router):
    client = smartgallery_app.app.test_client()
    resp = client.post(_NLQ_URL, data="not json at all", content_type="text/plain")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"


def test_rejects_missing_query_field(smartgallery_app, nlq_router):
    client = smartgallery_app.app.test_client()
    resp = client.post(_NLQ_URL, json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"


def test_never_accepts_raw_sql_field(smartgallery_app, nlq_router, seeded_files):
    """The endpoint must ignore any 'sql' key entirely -- only 'query' (NL
    text) drives it; a would-be injector cannot smuggle raw SQL in."""
    client = smartgallery_app.app.test_client()
    resp = client.post(_NLQ_URL, json={
        "query": "favorite videos",
        "sql": "DROP TABLE files; --",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["count"] == 2
    # the table must still exist and still hold our seeded rows
    with smartgallery_app.get_db_connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM files WHERE id LIKE 'nlqtest-%'").fetchone()[0]
    assert n == 5


def test_legacy_execute_endpoint_still_works(smartgallery_app, seeded_files):
    client = smartgallery_app.app.test_client()
    resp = client.post(_EXEC_URL, json={"sql": "SELECT f.id FROM files f WHERE f.type = 'video' AND f.id LIKE 'nlqtest-%'"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["count"] == 3


# ---------------------------------------------------------------------------
# (C): template smoke -- enable_ai_dam gates the lightbox button + panel include
# ---------------------------------------------------------------------------

def _get_gallery_html(client):
    resp = client.get("/galleryout/", follow_redirects=True)
    assert resp.status_code == 200
    return resp.get_data(as_text=True)


def test_gallery_view_hides_aidam_surfaces_when_disabled(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.AI_CONFIG, "enabled", False)
    client = smartgallery_app.app.test_client()
    html = _get_gallery_html(client)
    assert "lightbox-aidam-btn" not in html
    assert "aidam-overlay" not in html


def test_gallery_view_shows_aidam_surfaces_when_enabled(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.AI_CONFIG, "enabled", True)
    client = smartgallery_app.app.test_client()
    html = _get_gallery_html(client)
    assert "lightbox-aidam-btn" in html
    assert "aidam-overlay" in html
    assert "openAidamPanel()" in html
