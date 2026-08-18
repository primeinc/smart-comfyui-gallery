"""A private album stays private under `--force-login`, not just in exhibition mode.

The access check on an album was written as `if IS_EXHIBITION_MODE`, so a
non-staff account under `--force-login` skipped it altogether and could
read any private album's listing -- directly, and through the combined
"all" view, which is the same data by another door.

It is a listing, not the pictures: `is_file_accessible` still refuses that
account the files themselves. What leaked is what exists, under what name,
and how it is rated and commented on.

The same account is bounced out of the management interface with its
session cleared, which is what made this easy to miss -- the interface was
never the way in. Logging in is enough to hold a session, and the API asked
for nothing more.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "cafl_"
_JSON = {"Accept": "application/json"}


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    """A private album shared with user 41, and a public one, each with a file."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    made = []
    for name in (f"{_PREFIX}secret.png", f"{_PREFIX}open.png"):
        path = os.path.join(base, name)
        Image.new("RGB", (16, 16), (90, 30, 30)).save(path)
        made.append(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {}
        for album, is_public, shared, filename in (
            (f"{_PREFIX}private", 0, "41", f"{_PREFIX}secret.png"),
            (f"{_PREFIX}public", 1, "", f"{_PREFIX}open.png"),
        ):
            conn.execute(
                "INSERT INTO collections (name, type, is_public, shared_users, created_at) "
                "VALUES (?, 'user_album', ?, ?, 1.0)",
                (album, is_public, shared),
            )
            coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            file_id = conn.execute("SELECT id FROM files WHERE name = ?", (filename,)).fetchone()[0]
            conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (coll_id, file_id))
            ids[album] = coll_id
        conn.commit()
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.execute("DELETE FROM collections WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
    for path in made:
        with contextlib.suppress(OSError):
            os.remove(path)


def _client(smartgallery_app, role, user_id=9):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["role"] = role
    return client


def _names(resp):
    return sorted(
        f["name"] for f in (resp.get_json() or {}).get("files", []) if str(f.get("name", "")).startswith(_PREFIX)
    )


def test_staff_can_read_the_private_album(smartgallery_app, library):
    """Control: the album has contents, so an empty listing below is the
    filter working rather than an empty fixture."""
    client = _client(smartgallery_app, "ADMIN")

    resp = client.get(f"/galleryout/collection/{library[f'{_PREFIX}private']}", headers=_JSON)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _names(resp) == [f"{_PREFIX}secret.png"]


def test_a_customer_cannot_read_a_private_album(smartgallery_app, library):
    """The regression: this answered 200 with the album's contents."""
    client = _client(smartgallery_app, "CUSTOMER")

    resp = client.get(f"/galleryout/collection/{library[f'{_PREFIX}private']}", headers=_JSON, follow_redirects=False)

    assert resp.status_code in (301, 302), f"a customer read a private album ({resp.status_code})"
    assert f"{_PREFIX}secret.png" not in resp.get_data(as_text=True)


def test_the_all_view_is_not_a_way_around_it(smartgallery_app, library):
    """The combined view is the same data by another door, and it was open."""
    client = _client(smartgallery_app, "CUSTOMER")

    resp = client.get("/galleryout/collection/all", headers=_JSON)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _names(resp) == [f"{_PREFIX}open.png"], (
        f"the all view exposed an album the direct request refuses: {_names(resp)}"
    )


def test_the_account_it_is_shared_with_still_reads_it(smartgallery_app, library):
    """Sharing has to keep working, or this is just a blanket denial."""
    client = _client(smartgallery_app, "CUSTOMER", user_id=41)

    resp = client.get(f"/galleryout/collection/{library[f'{_PREFIX}private']}", headers=_JSON)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _names(resp) == [f"{_PREFIX}secret.png"], "the account the album was shared with lost access to it"


def test_the_default_local_install_reads_everything(smartgallery_app, library, monkeypatch):
    """With no login configured there is one person and the library is
    theirs -- they must not be filtered as though they were a visitor."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = smartgallery_app.app.test_client().get(
        f"/galleryout/collection/{library[f'{_PREFIX}private']}", headers=_JSON
    )

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _names(resp) == [f"{_PREFIX}secret.png"]
