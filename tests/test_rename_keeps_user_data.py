"""Renaming must carry a file's ratings, comments and albums with it.

A file's id is the md5 of its path, so renaming a file -- or the folder
above it -- gives it a new id. Nine tables reference `files(id)` with
`ON DELETE CASCADE` and no `ON UPDATE` clause, so updating that id on its
own raises `FOREIGN KEY constraint failed`.

The result was that renaming stopped working entirely as soon as anything
inside had been rated or commented on, which is to say as soon as the
gallery was used for the thing it is for. Worse than the error: the file
was renamed on disk before the database was touched, so the rename half
happened. The rows kept pointing at the old path, the next scan saw that
path missing, and the cascade then destroyed the ratings and comments for
good.

These tests rename things that carry user data and check the data is still
attached afterwards.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "rkud_"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


@pytest.fixture
def rated_file(smartgallery_app, monkeypatch):
    """One indexed image in a subfolder, rated, commented on, favourited
    and placed in an album."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    box = os.path.join(base, f"{_PREFIX}box")
    os.makedirs(box, exist_ok=True)
    Image.new("RGB", (16, 16), (30, 140, 90)).save(os.path.join(box, f"{_PREFIX}pic.png"))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}pic.png",)).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO file_ratings "
            "(file_id, client_uuid, rating, created_at) VALUES (?, 'admin', 5, 1.0)",
            (file_id,),
        )
        conn.execute(
            "INSERT INTO file_comments "
            "(file_id, client_uuid, author_name, comment_text, target_audience, created_at) "
            "VALUES (?, 'admin', 'Me', 'worth keeping', 'public', 1.0)",
            (file_id,),
        )
        conn.execute("UPDATE files SET is_favorite = 1 WHERE id = ?", (file_id,))
        conn.execute(
            "INSERT INTO collections (name, type, created_at) VALUES (?, 'user_album', 1.0)", (f"{_PREFIX}album",)
        )
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (coll_id, file_id))
        conn.commit()
    finally:
        conn.close()

    yield file_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.execute("DELETE FROM collections WHERE name LIKE ?", (f"{_PREFIX}%",))
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


def _attached(smartgallery_app, name):
    """(rating, comment_count, favourite, album_count) for the row called `name`."""
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT id, is_favorite FROM files WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        file_id = row[0]
        rating = conn.execute("SELECT rating FROM file_ratings WHERE file_id = ?", (file_id,)).fetchone()
        comments = conn.execute("SELECT COUNT(*) FROM file_comments WHERE file_id = ?", (file_id,)).fetchone()[0]
        albums = conn.execute("SELECT COUNT(*) FROM collection_files WHERE file_id = ?", (file_id,)).fetchone()[0]
        return (rating[0] if rating else None, comments, row[1], albums)
    finally:
        conn.close()


def test_the_fixture_attaches_everything(smartgallery_app, rated_file):
    """Control: without this the assertions below could pass on nothing."""
    assert _attached(smartgallery_app, f"{_PREFIX}pic.png") == (5, 1, 1, 1)


def test_renaming_a_file_keeps_its_rating_and_comments(smartgallery_app, client, rated_file):
    resp = client.post(f"/galleryout/rename_file/{rated_file}", json={"new_name": f"{_PREFIX}renamed.png"})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _attached(smartgallery_app, f"{_PREFIX}renamed.png") == (5, 1, 1, 1), (
        "renaming a file dropped the data attached to it"
    )


def test_renaming_a_folder_keeps_the_ratings_of_the_files_inside(smartgallery_app, client, rated_file):
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    key = next(k for k, v in folders.items() if str(v["path"]).replace("\\", "/").endswith(f"{_PREFIX}box"))

    resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": f"{_PREFIX}renamedbox"})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _attached(smartgallery_app, f"{_PREFIX}pic.png") == (5, 1, 1, 1), (
        "renaming a folder dropped the data attached to the files inside it"
    )


def test_moving_a_file_keeps_its_rating_and_comments(smartgallery_app, client, rated_file):
    """Moving changes the path, so it changes the id too. This one failed
    more quietly than rename: the file was moved before the rows were
    touched, and the error was folded into a "partial success" message."""
    base = smartgallery_app.BASE_OUTPUT_PATH
    dest = os.path.join(base, f"{_PREFIX}dest")
    os.makedirs(dest, exist_ok=True)
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    dest_key = next(k for k, v in folders.items() if str(v["path"]).replace("\\", "/").endswith(f"{_PREFIX}dest"))

    resp = client.post("/galleryout/move_batch", json={"file_ids": [rated_file], "destination_folder": dest_key})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "Failed" not in (body.get("message") or ""), body
    assert os.path.exists(os.path.join(dest, f"{_PREFIX}pic.png")), "the file did not move"
    assert _attached(smartgallery_app, f"{_PREFIX}pic.png") == (5, 1, 1, 1), (
        "moving a file dropped the data attached to it"
    )


def test_a_failed_move_leaves_disk_and_database_agreeing(smartgallery_app, client, rated_file, monkeypatch):
    """The batch reports failures per file and carries on, so a half-applied
    move would be committed with everything else."""
    base = smartgallery_app.BASE_OUTPUT_PATH
    box = os.path.join(base, f"{_PREFIX}box")
    dest = os.path.join(base, f"{_PREFIX}dest")
    os.makedirs(dest, exist_ok=True)
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    dest_key = next(k for k, v in folders.items() if str(v["path"]).replace("\\", "/").endswith(f"{_PREFIX}dest"))

    def _refuse(*_args, **_kwargs):
        raise OSError("simulated: destination volume went away")

    monkeypatch.setattr(smartgallery_app.shutil, "move", _refuse)
    resp = client.post("/galleryout/move_batch", json={"file_ids": [rated_file], "destination_folder": dest_key})
    monkeypatch.undo()

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert sorted(os.listdir(box)) == [f"{_PREFIX}pic.png"], "the file moved anyway"
    assert _attached(smartgallery_app, f"{_PREFIX}pic.png") == (5, 1, 1, 1), "a failed move still altered the database"


def test_a_failed_rename_leaves_disk_and_database_agreeing(smartgallery_app, client, rated_file, monkeypatch):
    """If the file cannot be moved, nothing may be recorded as moved --
    a row pointing at a path that holds no file is deleted by the next
    scan, and the cascade takes the ratings with it."""
    base = smartgallery_app.BASE_OUTPUT_PATH
    box = os.path.join(base, f"{_PREFIX}box")

    def _refuse(*_args, **_kwargs):
        raise OSError("simulated: file is locked by another process")

    monkeypatch.setattr(smartgallery_app.os, "rename", _refuse)
    resp = client.post(f"/galleryout/rename_file/{rated_file}", json={"new_name": f"{_PREFIX}nope.png"})
    monkeypatch.undo()

    assert resp.status_code == 500, resp.status_code
    assert sorted(os.listdir(box)) == [f"{_PREFIX}pic.png"], "the file moved anyway"
    assert _attached(smartgallery_app, f"{_PREFIX}pic.png") == (5, 1, 1, 1), (
        "a failed rename still altered the database"
    )
