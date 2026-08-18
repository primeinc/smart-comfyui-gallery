"""Renaming a file.

Three operations can put a file under a name that is already taken, and
they must all refuse to destroy the incumbent -- by different, deliberate
means:

  move    -> renames the arrival to name(1).ext   (test_move_copy_routes)
  upload  -> renames the arrival to name(1).ext   (test_upload_route)
  rename  -> refuses outright with 409            (here)

Renaming is the one case where silently picking a different name would be
wrong: the user asked for THAT name, so being told it is taken is the
useful answer.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from PIL import Image

_PREFIX = "rnroute_"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


def _seed(smartgallery_app, name, favorite=0):
    path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name)
    Image.new("RGB", (16, 16), (90, 160, 60)).save(path)
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


@pytest.fixture(autouse=True)
def _cleanup(smartgallery_app):
    yield
    root = smartgallery_app.BASE_OUTPUT_PATH
    for entry in os.listdir(root):
        if entry.startswith(_PREFIX):
            with contextlib.suppress(OSError):
                os.remove(os.path.join(root, entry))
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()


def _lookup(smartgallery_app, name):
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT id, path, name, is_favorite FROM files WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def test_rename_moves_the_file_and_updates_its_row(smartgallery_app, client):
    old = f"{_PREFIX}before.png"
    new = f"{_PREFIX}after.png"
    file_id, old_path = _seed(smartgallery_app, old, favorite=1)

    resp = client.post(f"/galleryout/rename_file/{file_id}", json={"new_name": new})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert not os.path.exists(old_path)
    assert os.path.isfile(os.path.join(smartgallery_app.BASE_OUTPUT_PATH, new))
    row = _lookup(smartgallery_app, new)
    assert row is not None, "no row points at the renamed file"
    assert row["is_favorite"] == 1, "metadata was lost in the rename"


def test_rename_onto_an_existing_name_is_refused(smartgallery_app, client):
    """409 rather than a silent overwrite -- and both files keep their own
    bytes."""
    victim = f"{_PREFIX}occupied.png"
    victim_path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, victim)
    with open(victim_path, "wb") as fh:
        fh.write(b"THE FILE ALREADY USING THAT NAME")

    source = f"{_PREFIX}source.png"
    file_id, source_path = _seed(smartgallery_app, source)

    resp = client.post(f"/galleryout/rename_file/{file_id}", json={"new_name": victim})

    assert resp.status_code == 409, "rename should refuse a taken name"
    with open(victim_path, "rb") as fh:
        assert fh.read() == b"THE FILE ALREADY USING THAT NAME"
    assert os.path.exists(source_path), "the source was moved despite the refusal"


def test_rename_keeps_the_extension_when_none_is_given(smartgallery_app, client):
    file_id, _path = _seed(smartgallery_app, f"{_PREFIX}keepext.png")

    resp = client.post(f"/galleryout/rename_file/{file_id}", json={"new_name": f"{_PREFIX}renamed"})

    assert resp.status_code == 200
    assert os.path.isfile(os.path.join(smartgallery_app.BASE_OUTPUT_PATH, f"{_PREFIX}renamed.png"))


@pytest.mark.parametrize("bad", ["", "   ", "a/b.png", "a" + chr(92) + "b.png", "x:y.png", "we*rd.png", 'q"uote.png'])
def test_rename_rejects_unusable_names(smartgallery_app, client, bad):
    file_id, path = _seed(smartgallery_app, f"{_PREFIX}guard.png")

    resp = client.post(f"/galleryout/rename_file/{file_id}", json={"new_name": bad})

    assert resp.status_code == 400, f"{bad!r} was accepted as a filename"
    assert os.path.exists(path), "the file changed despite an invalid name"


def test_rename_to_the_same_name_is_refused(smartgallery_app, client):
    name = f"{_PREFIX}same.png"
    file_id, path = _seed(smartgallery_app, name)

    resp = client.post(f"/galleryout/rename_file/{file_id}", json={"new_name": name})

    assert resp.status_code == 400
    assert os.path.exists(path)
