"""Opening an album in exhibition mode.

Exhibition mode exists to show albums to visitors, so a visitor clicking
an album is the mode working as intended. That request answered 500.

`is_privileged` was assigned only inside the branch that handles a
PRIVATE collection, and read later by the query builder whenever a
specific album is rendered -- so the public album, which is the whole
point of the mode, reached the read with the name unbound. Python
evaluates `not is_privileged` as soon as it sees exhibition mode is on,
which means staff hit it too: nobody could open an album.

The two roles are tested separately because they take different paths
through the visibility filtering that follows.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image

from inline_executor import InlineExecutor

_PREFIX = "collview_"


@pytest.fixture
def albums(smartgallery_app, monkeypatch):
    """A public album and a private one, each holding an indexed file."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)

    base = smartgallery_app.BASE_OUTPUT_PATH
    made = []
    for name in (f"{_PREFIX}shown.png", f"{_PREFIX}hidden.png"):
        path = os.path.join(base, name)
        Image.new("RGB", (16, 16), (60, 60, 160)).save(path)
        made.append(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        keys = {}
        for album, is_public, filename in (
            (f"{_PREFIX}public", 1, f"{_PREFIX}shown.png"),
            (f"{_PREFIX}private", 0, f"{_PREFIX}hidden.png"),
        ):
            conn.execute(
                "INSERT INTO collections (name, type, is_public, created_at) VALUES (?, 'user_album', ?, 1.0)",
                (album, is_public),
            )
            coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            file_id = conn.execute("SELECT id FROM files WHERE name = ?", (filename,)).fetchone()[0]
            conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (coll_id, file_id))
            keys[album] = coll_id
        conn.commit()
    finally:
        conn.close()

    yield keys

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute(f"DELETE FROM collections WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    for path in made:
        with contextlib.suppress(OSError):
            os.remove(path)


def _as(smartgallery_app, role):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = role
    return client


_JSON = {"Accept": "application/json"}


@pytest.mark.parametrize("role", ["CUSTOMER", "ADMIN"])
def test_a_public_album_opens(smartgallery_app, albums, role):
    """The regression, for a visitor and for staff: this answered 500,
    both as a page and as the JSON the page actually loads its files from."""
    client = _as(smartgallery_app, role)
    album = albums[f"{_PREFIX}public"]

    page = client.get(f"/galleryout/collection/{album}")
    assert page.status_code == 200, page.get_data(as_text=True)[:300]

    data = client.get(f"/galleryout/collection/{album}", headers=_JSON)
    assert data.status_code == 200, data.get_data(as_text=True)[:300]
    names = [f["name"] for f in (data.get_json() or {}).get("files", [])]
    assert names == [f"{_PREFIX}shown.png"], f"album contents wrong: {names}"


def test_the_album_listing_does_not_hand_a_visitor_the_prompts(smartgallery_app, albums):
    """Exhibition mode strips prompts out of the files themselves, so the
    listing that describes those files must not carry them either."""
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "UPDATE files SET workflow_prompt = ?, workflow_files = ? WHERE name = ?",
            ("SECRETPROMPT_neon_alley", "secretmodel.safetensors", f"{_PREFIX}shown.png"),
        )
        conn.commit()
    finally:
        conn.close()

    client = _as(smartgallery_app, "CUSTOMER")
    body = client.get(f"/galleryout/collection/{albums[f'{_PREFIX}public']}", headers=_JSON).get_data(as_text=True)

    assert f"{_PREFIX}shown.png" in body, "control failed: the listing does not describe the file at all"
    assert "SECRETPROMPT_neon_alley" not in body, "the album listing handed a visitor the prompt"
    assert "secretmodel.safetensors" not in body, "the album listing handed a visitor the model names"


def test_a_visitor_is_still_kept_out_of_a_private_album(smartgallery_app, albums):
    """The counterpart: the crash sat right beside the privacy check, so
    the fix must not open anything that was closed."""
    client = _as(smartgallery_app, "CUSTOMER")

    resp = client.get(f"/galleryout/collection/{albums[f'{_PREFIX}private']}", follow_redirects=False)

    assert resp.status_code in (301, 302), f"a visitor was served a private album ({resp.status_code})"
    assert f"{_PREFIX}hidden.png" not in resp.get_data(as_text=True)


def test_staff_can_open_the_private_album(smartgallery_app, albums):
    client = _as(smartgallery_app, "ADMIN")

    resp = client.get(f"/galleryout/collection/{albums[f'{_PREFIX}private']}", headers=_JSON)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    names = [f["name"] for f in (resp.get_json() or {}).get("files", [])]
    assert names == [f"{_PREFIX}hidden.png"], names
