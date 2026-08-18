"""The owner's own journey, on a gallery that requires a login.

Eight or so routes gained the management guard in this branch: the folder
rescan, the sidebar state, the AI status, the zip flow, the storyboard, the
live review runner, the AI search queue and its status. Each has a test
saying staff are not refused it.

None of them checked that the management side still works as a sequence --
that someone can sign in, see their folders, look at a picture, rate it,
tidy it away and get a zip out. A guard that is right in isolation and
breaks the screen it belongs to is still a broken screen, and the
individual tests would all stay green.

This is the counterpart to test_visitor_journey: the same shape of check,
for the person who owns the library rather than the one visiting it.
"""

from __future__ import annotations

import contextlib
import os
import time

import pytest
from inline_executor import InlineExecutor
from PIL import Image

import sg_auth

_PREFIX = "owner_"
_PASSWORD = "correct-horse-battery"


@pytest.fixture
def team_gallery(smartgallery_app, monkeypatch):
    """A login-protected gallery with one admin and two pictures."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    made = []
    for name in (f"{_PREFIX}one.png", f"{_PREFIX}two.png"):
        path = os.path.join(base, name)
        Image.new("RGB", (32, 32), (140, 90, 40)).save(path)
        made.append(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM users WHERE username = 'owner_admin'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {r[0]: r[1] for r in conn.execute(f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()}
        conn.execute(
            "INSERT INTO users (username, password, full_name, role, "
            "is_active) VALUES ('owner_admin', ?, 'Owner', 'ADMIN', 1)",
            (sg_auth.hash_password(_PASSWORD),),
        )
        conn.commit()
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM users WHERE username = 'owner_admin'")
        conn.commit()
    finally:
        conn.close()
    for dirpath, _dirs, names in os.walk(base, topdown=False):
        for name in names:
            if name.startswith(_PREFIX):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(dirpath, name))
        if os.path.basename(dirpath).startswith(_PREFIX):
            with contextlib.suppress(OSError):
                os.rmdir(dirpath)


def _signed_in(smartgallery_app):
    client = smartgallery_app.app.test_client()
    resp = client.post("/galleryout/login", json={"username": "owner_admin", "password": _PASSWORD})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json()["status"] == "success", resp.get_json()
    return client


def test_the_owner_can_run_their_gallery(smartgallery_app, team_gallery):
    """Sign in, look around, work on a picture, tidy up, take a zip."""
    client = _signed_in(smartgallery_app)
    first = team_gallery[f"{_PREFIX}one.png"]

    # The gallery page itself.
    page = client.get("/galleryout/view/_root_", follow_redirects=True)
    assert page.status_code == 200, page.status_code
    assert first in page.get_data(as_text=True), "the library did not render"

    # The panels the page fills in: collections, sidebar, AI status.
    for label, url in [
        ("collections", "/galleryout/api/collections"),
        ("sidebar", "/galleryout/api/sidebar_state"),
        ("ai status", "/galleryout/api/aidam/status"),
        ("indexing status", "/galleryout/ai_indexing/status"),
        ("site settings", "/galleryout/api/site_settings"),
    ]:
        resp = client.get(url)
        assert resp.status_code == 200, f"{label} answered {resp.status_code}"

    # Work on a picture: rate it, comment on it, read the details.
    assert client.post("/galleryout/api/exhibition/rate", json={"file_id": first, "rating": 5}).status_code == 200
    assert (
        client.post("/galleryout/api/exhibition/post_comment", json={"file_id": first, "text": "keep"}).status_code
        == 200
    )
    details = client.get(f"/galleryout/api/file_full_details/{first}")
    assert details.status_code == 200, details.get_data(as_text=True)[:200]

    # Tidy: make a folder, move the picture into it, and check it followed.
    made = client.post("/galleryout/create_folder", json={"folder_name": f"{_PREFIX}kept", "parent_key": "_root_"})
    assert made.status_code == 200, made.get_data(as_text=True)

    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    dest = next((k for k, v in folders.items() if str(v["path"]).replace("\\", "/").endswith(f"{_PREFIX}kept")), None)
    assert dest, "the new folder is not in the folder list"

    moved = client.post("/galleryout/move_batch", json={"file_ids": [first], "destination_folder": dest})
    assert moved.status_code == 200, moved.get_data(as_text=True)
    assert "Failed" not in moved.get_json()["message"], moved.get_json()

    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}one.png",)).fetchone()
        rating = conn.execute("SELECT rating FROM file_ratings WHERE file_id = ?", (row[0],)).fetchone()
    finally:
        conn.close()
    assert rating and rating[0] == 5, "the rating did not follow the move"

    # A zip of the remaining picture, through the whole three-step flow.
    second = team_gallery[f"{_PREFIX}two.png"]
    started = client.post("/galleryout/prepare_batch_zip", json={"file_ids": [second]})
    assert started.status_code == 200, started.get_data(as_text=True)
    job_id = started.get_json()["job_id"]

    for _ in range(50):
        status = client.get(f"/galleryout/check_zip_status/{job_id}")
        assert status.status_code == 200, status.get_data(as_text=True)
        body = status.get_json()
        if body.get("status") in ("ready", "error"):
            break
        time.sleep(0.1)

    assert body.get("status") == "ready", body
    download = client.get(body["download_url"])
    assert download.status_code == 200, download.status_code
    assert download.get_data()[:2] == b"PK", "that is not a zip file"


def test_a_customer_cannot_run_the_gallery(smartgallery_app, team_gallery):
    """The counterpart: the management side is management-only, and the
    journey above passing must not mean it is open to everyone signed in."""
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 41
        session["role"] = "CUSTOMER"

    answered = []
    for label, url in [
        ("sidebar", "/galleryout/api/sidebar_state"),
        ("ai status", "/galleryout/api/aidam/status"),
        ("indexing status", "/galleryout/ai_indexing/status"),
        ("rescan", "/galleryout/sync_status/_root_"),
    ]:
        if client.get(url).status_code == 200:
            answered.append(label)

    assert answered == [], f"a customer reached the management side: {answered}"
