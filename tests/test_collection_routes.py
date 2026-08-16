"""Virtual collections: curation that must never cost files.

A collection is metadata about files, not a container for them. The
distinction is the whole safety property here: deleting a collection is
supposed to discard the curation and leave every file on disk and in the
library untouched. If that were ever wired the other way round, a user
tidying up their tags would lose their media.

The join table's cleanup is likewise claimed by a comment in the route
("handled automatically by SQLite ON DELETE CASCADE") -- a claim that only
holds while `PRAGMA foreign_keys = ON` is set on the connection, which
SQLite does NOT do by default. That is asserted here rather than trusted.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

_PREFIX = "collroute_"


@pytest.fixture()
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


@pytest.fixture()
def library(smartgallery_app):
    """Two files and a user collection holding both."""
    file_ids = []
    conn = smartgallery_app.get_db_connection()
    try:
        for n in ("x", "y"):
            name = f"{_PREFIX}{n}.png"
            path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name)
            Image.new("RGB", (16, 16), (140, 70, 190)).save(path)
            file_id = f"{_PREFIX}{n}"
            conn.execute(
                "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
                "VALUES (?, ?, ?, ?, 'image', ?)",
                (file_id, path, os.path.getmtime(path), name, os.path.getsize(path)))
            file_ids.append(file_id)
        conn.execute(
            "INSERT INTO collections (name, type, color, is_public, created_at) "
            "VALUES (?, 'user', '#abcdef', 0, 1000.0)", (f"{_PREFIX}album",))
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for file_id in file_ids:
            conn.execute(
                "INSERT INTO collection_files (collection_id, file_id, added_at) "
                "VALUES (?, ?, 1000.0)", (coll_id, file_id))
        conn.commit()
    finally:
        conn.close()

    yield file_ids, coll_id

    conn = smartgallery_app.get_db_connection()
    try:
        for file_id in file_ids:
            row = conn.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
            if row:
                try:
                    os.remove(row[0])
                except OSError:
                    pass
        conn.execute(f"DELETE FROM files WHERE id LIKE '{_PREFIX}%'")
        conn.execute(f"DELETE FROM collections WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()


def _members(smartgallery_app, coll_id):
    conn = smartgallery_app.get_db_connection()
    try:
        return sorted(r[0] for r in conn.execute(
            "SELECT file_id FROM collection_files WHERE collection_id = ?",
            (coll_id,)).fetchall())
    finally:
        conn.close()


def _file_rows(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        return sorted(r[0] for r in conn.execute(
            f"SELECT id FROM files WHERE id LIKE '{_PREFIX}%'").fetchall())
    finally:
        conn.close()


def test_deleting_a_collection_keeps_every_file(smartgallery_app, client, library):
    """The safety property: curation is discarded, media is not."""
    file_ids, coll_id = library
    paths = []
    conn = smartgallery_app.get_db_connection()
    try:
        paths = [r[0] for r in conn.execute(
            f"SELECT path FROM files WHERE id LIKE '{_PREFIX}%'").fetchall()]
    finally:
        conn.close()

    resp = client.post("/galleryout/api/collections/delete", json={"id": coll_id})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _file_rows(smartgallery_app) == sorted(file_ids), "rows were lost"
    for path in paths:
        assert os.path.isfile(path), f"deleting a collection removed {path}"


def test_deleting_a_collection_clears_its_membership_rows(smartgallery_app, client, library):
    """The route relies on ON DELETE CASCADE, which only fires because
    get_db_connection sets PRAGMA foreign_keys=ON -- SQLite defaults it
    off. Orphaned rows would resurface if an id were ever reused."""
    _file_ids, coll_id = library
    assert len(_members(smartgallery_app, coll_id)) == 2

    client.post("/galleryout/api/collections/delete", json={"id": coll_id})

    assert _members(smartgallery_app, coll_id) == [], (
        "membership rows outlived their collection")


def test_deleting_a_file_clears_its_membership(smartgallery_app, client, library):
    """The other direction of the same cascade."""
    file_ids, coll_id = library

    client.post("/galleryout/delete_batch", json={"file_ids": [file_ids[0]]})

    assert _members(smartgallery_app, coll_id) == [file_ids[1]]


def test_system_flags_cannot_be_deleted(smartgallery_app, client):
    """Status tags are structural; removing one would strip that status
    from every file that carries it."""
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute(
            "SELECT id FROM collections WHERE type = 'system_flag' LIMIT 1").fetchone()
    finally:
        conn.close()
    if row is None:
        pytest.skip("no system flags in this database")

    resp = client.post("/galleryout/api/collections/delete", json={"id": row[0]})

    assert resp.status_code == 403
    conn = smartgallery_app.get_db_connection()
    try:
        assert conn.execute("SELECT 1 FROM collections WHERE id = ?",
                            (row[0],)).fetchone() is not None
    finally:
        conn.close()


def test_tag_batch_adds_and_removes_membership(smartgallery_app, client, library):
    file_ids, coll_id = library
    client.post("/galleryout/api/collections/tag_batch",
                json={"file_ids": file_ids, "collection_id": coll_id, "action": "remove"})
    assert _members(smartgallery_app, coll_id) == []

    client.post("/galleryout/api/collections/tag_batch",
                json={"file_ids": file_ids, "collection_id": coll_id, "action": "add"})
    assert _members(smartgallery_app, coll_id) == sorted(file_ids)


def test_tag_batch_adding_twice_does_not_duplicate(smartgallery_app, client, library):
    file_ids, coll_id = library
    for _ in range(3):
        client.post("/galleryout/api/collections/tag_batch",
                    json={"file_ids": file_ids, "collection_id": coll_id,
                          "action": "add"})
    assert _members(smartgallery_app, coll_id) == sorted(file_ids)


def test_tag_batch_needs_files(client):
    resp = client.post("/galleryout/api/collections/tag_batch",
                       json={"file_ids": [], "collection_id": 1, "action": "add"})
    assert resp.status_code == 400
