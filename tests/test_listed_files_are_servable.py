"""Anything a visitor is shown has to be something they can be sent.

Two separate decisions govern a picture in the exhibition: the album
listing decides whether it appears, and is_file_accessible decides whether
its bytes may be served. They are written apart, and if they disagree in
one direction the visitor gets a grid of broken thumbnails, and in the
other a file nobody meant to share.

The leak direction has its own sweep already. This holds the other one,
which nothing checked: every file the portal lists must answer 200 for its
thumbnail and for the file itself.

The listing is taken the way the portal takes it -- the same address with
`Accept: application/json` -- because that is what a visitor's browser
receives. Read as a page, an album returns the exhibition shell with no
file in it at all, which is what made an earlier version of this look like
a public album showing nothing.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "servable_"


@pytest.fixture
def exhibition(smartgallery_app, monkeypatch):
    """A public album holding two of three pictures, seen by a guest."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    names = [f"{_PREFIX}shared_one.png", f"{_PREFIX}shared_two.png", f"{_PREFIX}kept_back.png"]
    for name in names:
        Image.new("RGB", (16, 16), (4, 4, 4)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Servable Album'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {
            r["name"]: r["id"]
            for r in conn.execute(f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()
        }
        conn.execute(
            "INSERT INTO collections (name, type, is_public) VALUES (?, ?, 1)", ("Servable Album", "user_album")
        )
        coll_id = conn.execute("SELECT id FROM collections WHERE name = ?", ("Servable Album",)).fetchone()[0]
        for name in names[:2]:
            conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (coll_id, ids[name]))
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "GUEST"
        session["full_name"] = "A Visitor"

    yield client, ids, coll_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Servable Album'")
        conn.commit()
    finally:
        conn.close()
    for name in names:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _listed(client, coll_id):
    response = client.get(f"/galleryout/collection/{coll_id}", headers={"Accept": "application/json"})
    assert response.status_code == 200, response.status_code
    return [f["id"] for f in (response.get_json() or {}).get("files") or []]


def test_the_album_lists_what_was_put_in_it(exhibition):
    """Control. Every check below is about the listed set, so it has to be
    the right one -- and an empty listing would satisfy them all."""
    client, ids, coll_id = exhibition

    listed = _listed(client, coll_id)

    assert set(listed) == {ids[f"{_PREFIX}shared_one.png"], ids[f"{_PREFIX}shared_two.png"]}, listed


def test_everything_listed_can_be_served(exhibition):
    """The direction nothing was holding: a picture on the page whose
    bytes are refused is a broken thumbnail, over and over."""
    client, ids, coll_id = exhibition
    by_id = {file_id: name for name, file_id in ids.items()}

    for file_id in _listed(client, coll_id):
        thumbnail = client.get(f"/galleryout/thumbnail/{file_id}")
        original = client.get(f"/galleryout/file/{file_id}")

        assert thumbnail.status_code == 200, (by_id[file_id], thumbnail.status_code)
        assert original.status_code == 200, (by_id[file_id], original.status_code)


def test_what_was_not_shared_is_still_refused(exhibition):
    """The other direction, kept here beside its opposite so a change that
    widens access to fix a broken thumbnail cannot pass unnoticed."""
    client, ids, coll_id = exhibition
    listed = set(_listed(client, coll_id))
    kept_back = ids[f"{_PREFIX}kept_back.png"]
    assert kept_back not in listed, "the fixture shared the wrong file"

    assert client.get(f"/galleryout/thumbnail/{kept_back}").status_code != 200
    assert client.get(f"/galleryout/file/{kept_back}").status_code != 200
