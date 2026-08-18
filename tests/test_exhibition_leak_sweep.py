"""Nothing a visitor can ask for may contain the prompt.

Four separate leaks in this branch were the same fault in four places: an
endpoint that returned generation metadata to someone the mode is meant to
keep it from. The album listing, the per-file details, the AI review, and
the thumbnail route when server-side thumbnails are off. Each was found by
reading one endpoint at a time, and each looked fine until it was read.

This asks the question of everything at once. A file is seeded with three
markers -- a prompt, a model filename, and the gallery's own path on disk --
put into a public album, and then every address a visitor can reach is
called with a visitor's session. Any response that contains a marker fails,
naming the endpoint.

It is a canary, not a proof: it covers the endpoints listed below, so a new
one still has to be added here. But it turns "did we remember to redact
this?" from something you notice into something that fails.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "leaksweep_"
_PROMPT = "CANARYPROMPT a brass diving helmet at dusk"
_MODEL = "CANARYMODEL_v3.safetensors"


@pytest.fixture
def canary(smartgallery_app, monkeypatch):
    """A public-album file carrying every marker a visitor must not see."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app.AI_CONFIG, "enabled", True)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}pic.png")
    Image.new("RGB", (24, 24), (12, 34, 56)).save(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}pic.png",)).fetchone()[0]
        conn.execute(
            "UPDATE files SET workflow_prompt = ?, workflow_files = ? WHERE id = ?", (_PROMPT, _MODEL, file_id)
        )
        conn.execute(
            "INSERT INTO collections (name, type, is_public, created_at) VALUES (?, 'user_album', 1, 1.0)",
            (f"{_PREFIX}album",),
        )
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (coll_id, file_id))
        # A stored review, whose alignment elements are slices of the prompt.
        conn.execute(
            "INSERT INTO ai_reviews (file_id, rubric_version, model_id, "
            "model_version, quality_score, prompt_alignment_score, summary, "
            "source_mtime, computed_at) VALUES (?, 'v1', 'critic', '1', 8.0, 0.5, "
            "?, 1.0, 1.0)",
            (file_id, f"missing: {_PROMPT}"),
        )
        review_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ai_review_alignment (review_id, file_id, ordinal, text, "
            "satisfied, confidence) VALUES (?, ?, 0, ?, 0, 0.9)",
            (review_id, file_id, _PROMPT),
        )
        conn.commit()
    finally:
        conn.close()

    yield {"file_id": file_id, "coll_id": coll_id, "base": base}

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.execute("DELETE FROM collections WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
    with contextlib.suppress(OSError):
        os.remove(path)


def _visitor(smartgallery_app):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = "CUSTOMER"
    return client


def _endpoints(state):
    fid, coll = state["file_id"], state["coll_id"]
    return {
        "collection page": f"/galleryout/collection/{coll}",
        "collection json": f"/galleryout/collection/{coll}",
        "all albums": "/galleryout/collection/all",
        "file details": f"/galleryout/api/file_full_details/{fid}",
        "file collections": f"/galleryout/api/file_collections/{fid}",
        "check metadata": f"/galleryout/check_metadata/{fid}",
        "comments": f"/galleryout/api/exhibition/comments?file_id={fid}",
        "album list": "/galleryout/api/collections",
        "search options": "/galleryout/api/search_options",
        "ai review": f"/galleryout/api/aidam/review/{fid}",
        "ai similar": f"/galleryout/api/aidam/similar/{fid}",
        "ai duplicates": f"/galleryout/api/aidam/duplicates/{fid}",
        "ai faces": f"/galleryout/api/aidam/faces/{fid}",
        "the picture": f"/galleryout/file/{fid}",
        "the thumbnail": f"/galleryout/thumbnail/{fid}",
        "the download": f"/galleryout/download/{fid}",
    }


def test_the_canary_is_actually_stored(smartgallery_app, canary):
    """Control: the markers are in the database and in the file, so an
    absence below means redaction and not an empty fixture."""
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute(
            "SELECT workflow_prompt, workflow_files FROM files WHERE id = ?", (canary["file_id"],)
        ).fetchone()
        elements = conn.execute(
            "SELECT text FROM ai_review_alignment WHERE file_id = ?", (canary["file_id"],)
        ).fetchall()
    finally:
        conn.close()

    assert row[0] == _PROMPT
    assert row[1] == _MODEL
    assert elements
    assert elements[0][0] == _PROMPT


def test_staff_can_see_the_canary(smartgallery_app, canary):
    """The other half of the control: these endpoints DO carry the markers
    for someone entitled to them, so the sweep is looking at live data."""
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    body = client.get(f"/galleryout/api/file_full_details/{canary['file_id']}").get_data(as_text=True)

    assert _PROMPT in body
    assert _MODEL in body


@pytest.mark.parametrize("marker", [_PROMPT, _MODEL])
def test_no_visitor_endpoint_returns_the_marker(smartgallery_app, canary, marker):
    """The sweep."""
    client = _visitor(smartgallery_app)
    leaked = []

    for label, url in _endpoints(canary).items():
        headers = {"Accept": "application/json"} if label.endswith("json") else {}
        resp = client.get(url, headers=headers)
        if marker.encode() in resp.get_data():
            leaked.append(f"{label} ({resp.status_code}) -> {url}")

    assert leaked == [], f"{marker!r} was returned to a visitor by: {leaked}"


def test_no_visitor_endpoint_returns_the_server_path(smartgallery_app, canary):
    """The library's location on disk is not a visitor's business either."""
    client = _visitor(smartgallery_app)
    root = canary["base"].replace("\\", "/")
    leaked = []

    for label, url in _endpoints(canary).items():
        headers = {"Accept": "application/json"} if label.endswith("json") else {}
        resp = client.get(url, headers=headers)
        # Bytes with a lenient decode: several of these answer with an image,
        # and a path can just as easily be embedded in one.
        text = resp.get_data().decode("utf-8", "replace")
        flat = text.replace("\\\\", "/").replace("\\", "/")
        if root in flat:
            leaked.append(f"{label} ({resp.status_code}) -> {url}")

    assert leaked == [], f"the gallery path was returned to a visitor by: {leaked}"
