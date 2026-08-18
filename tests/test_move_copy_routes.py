"""Moving and copying files between folders.

These routes relocate a user's media and rewrite the rows that point at it,
and neither had a test. The property that matters most is that a name
collision at the destination NEVER overwrites what is already there: a move
that silently replaces an existing file destroys data the user never chose
to delete.

Folder keys come from the live folder config rather than being guessed, so
these tests exercise the same lookup the UI does.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from PIL import Image
import pathlib

_PREFIX = "mvroute_"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


@pytest.fixture
def dest_folder(smartgallery_app):
    """A real subfolder of the gallery, known to the folder config."""
    path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, f"{_PREFIX}dest")
    os.makedirs(path, exist_ok=True)
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    key = next((k for k, v in folders.items() if os.path.normpath(v["path"]) == os.path.normpath(path)), None)
    if key is None:
        pytest.skip("destination folder is not exposed by the folder config")
    return key, path


def _seed(smartgallery_app, name, folder=None, colour=(80, 160, 240), favorite=0):
    root = folder or smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(root, name)
    Image.new("RGB", (24, 24), colour).save(path)
    file_id = f"{_PREFIX}{name}"
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size, is_favorite) "
            "VALUES (?, ?, ?, ?, 'image', ?, ?)",
            (file_id, path, os.path.getmtime(path), name, os.path.getsize(path), favorite),
        )
        conn.commit()
    finally:
        conn.close()
    return file_id, path


def _row(smartgallery_app, where_sql, params):
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute(f"SELECT id, path, name, is_favorite FROM files WHERE {where_sql}", params).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _cleanup(smartgallery_app):
    yield
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()

    # Sweep the disk, not the rows. The collision test writes a file that
    # deliberately has no row of its own, and the destination folder outlives
    # the fixture that made it -- so anything missed here is left in the
    # gallery for the next test that runs a real scan to index as a genuine
    # file, and that test then counts one image too many.
    root = smartgallery_app.BASE_OUTPUT_PATH
    for dirpath, _dirs, names in os.walk(root, topdown=False):
        for name in names:
            if name.startswith(_PREFIX):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(dirpath, name))
        if os.path.basename(dirpath).startswith(_PREFIX):
            with contextlib.suppress(OSError):
                os.rmdir(dirpath)


def test_move_relocates_the_file_and_follows_it_in_the_database(smartgallery_app, client, dest_folder):
    dest_key, dest_path = dest_folder
    file_id, source_path = _seed(smartgallery_app, f"{_PREFIX}move_me.png", favorite=1)

    resp = client.post("/galleryout/move_batch", json={"file_ids": [file_id], "destination_folder": dest_key})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert not os.path.exists(source_path), "the file was left in the source folder"
    assert os.path.exists(os.path.join(dest_path, f"{_PREFIX}move_me.png"))

    moved = _row(smartgallery_app, "name = ?", (f"{_PREFIX}move_me.png",))
    assert moved is not None, "no row points at the moved file"
    assert os.path.normpath(moved["path"]) == os.path.normpath(os.path.join(dest_path, f"{_PREFIX}move_me.png"))
    assert moved["is_favorite"] == 1, "metadata was lost in the move"


def test_move_into_a_name_collision_never_overwrites(smartgallery_app, client, dest_folder):
    """The data-safety property: an existing file at the destination keeps
    its bytes, and the arriving file is renamed beside it."""
    dest_key, dest_path = dest_folder
    name = f"{_PREFIX}clash.png"
    incumbent = os.path.join(dest_path, name)
    with open(incumbent, "wb") as fh:
        fh.write(b"incumbent bytes that must survive")

    file_id, source_path = _seed(smartgallery_app, name)
    original = pathlib.Path(source_path).read_bytes()

    resp = client.post("/galleryout/move_batch", json={"file_ids": [file_id], "destination_folder": dest_key})

    assert resp.status_code == 200
    with open(incumbent, "rb") as fh:
        assert fh.read() == b"incumbent bytes that must survive", "the move overwrote a file already at the destination"
    beside = [f for f in os.listdir(dest_path) if f.startswith(f"{_PREFIX}clash(") and f.endswith(".png")]
    assert beside, f"the moved file was not preserved beside it: {os.listdir(dest_path)}"
    with open(os.path.join(dest_path, beside[0]), "rb") as fh:
        assert fh.read() == original


def test_move_into_the_same_folder_is_skipped(smartgallery_app, client):
    """Both the source and destination are the gallery root here; the file
    must be left exactly where it is rather than churned through a move."""
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    root_key = next(
        (
            k
            for k, v in folders.items()
            if os.path.normpath(v["path"]) == os.path.normpath(smartgallery_app.BASE_OUTPUT_PATH)
        ),
        None,
    )
    if root_key is None:
        pytest.skip("gallery root is not exposed as a folder key")
    file_id, path = _seed(smartgallery_app, f"{_PREFIX}stay.png")

    resp = client.post("/galleryout/move_batch", json={"file_ids": [file_id], "destination_folder": root_key})

    assert resp.status_code == 200
    assert os.path.exists(path), "a same-folder move removed the file"


def test_move_of_a_file_missing_from_disk_clears_its_row(smartgallery_app, client, dest_folder):
    dest_key, _dest_path = dest_folder
    file_id, path = _seed(smartgallery_app, f"{_PREFIX}ghost.png")
    os.remove(path)

    resp = client.post("/galleryout/move_batch", json={"file_ids": [file_id], "destination_folder": dest_key})

    assert resp.status_code == 200
    assert _row(smartgallery_app, "id = ?", (file_id,)) is None, (
        "a row pointing at a nonexistent file survived the move"
    )


def test_copy_leaves_the_original_in_place(smartgallery_app, client, dest_folder):
    dest_key, dest_path = dest_folder
    file_id, source_path = _seed(smartgallery_app, f"{_PREFIX}copy_me.png")

    resp = client.post("/galleryout/copy_batch", json={"file_ids": [file_id], "destination_folder": dest_key})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert os.path.exists(source_path), "copy removed the original"
    assert os.path.exists(os.path.join(dest_path, f"{_PREFIX}copy_me.png"))


def test_move_rejects_an_unknown_destination(smartgallery_app, client):
    file_id, path = _seed(smartgallery_app, f"{_PREFIX}nowhere.png")

    resp = client.post("/galleryout/move_batch", json={"file_ids": [file_id], "destination_folder": "no_such_key"})

    assert resp.status_code >= 400
    assert os.path.exists(path), "the file moved despite an invalid destination"
