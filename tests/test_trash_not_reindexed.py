"""A deleted file must not come back because the trash is inside the gallery.

`DELETE_TO` makes deletion recoverable by moving files to a trash folder
instead of destroying them, and pointing it somewhere inside the gallery
is a reasonable thing to do -- it keeps deletions on the same drive, so
the move is instant and the space is accounted for in one place.

The scan excluded a fixed list of folder NAMES (.thumbnails_cache and
friends) and knew nothing about the trash, so it indexed it. The file
disappeared, the next scan put it straight back under its timestamped
trash name, and its ratings and comments did not come with it: the new
path means a new id.

The exclusion is by path, since DELETE_TO can be anywhere.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image

_PREFIX = "trashx_"


class _InlineExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture
def gallery_with_inside_trash(smartgallery_app, monkeypatch):
    """A gallery whose trash folder sits inside it, holding one indexed file."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures,
                        "ProcessPoolExecutor", _InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    delete_to = os.path.join(base, f"{_PREFIX}recycle")
    trash = os.path.join(delete_to, "SmartGallery")
    os.makedirs(trash, exist_ok=True)
    monkeypatch.setattr(smartgallery_app, "DELETE_TO", delete_to)
    monkeypatch.setattr(smartgallery_app, "TRASH_FOLDER", trash)

    name = f"{_PREFIX}doomed.png"
    Image.new("RGB", (16, 16), (6, 6, 6)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '%{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (name,)).fetchone()[0]
    finally:
        conn.close()

    yield {"file_id": file_id, "name": name, "trash": trash, "base": base}

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '%{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    for dirpath, _dirs, files in os.walk(delete_to, topdown=False):
        for f in files:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(dirpath, f))
        with contextlib.suppress(OSError):
            os.rmdir(dirpath)
    with contextlib.suppress(OSError):
        os.remove(os.path.join(base, name))


def _rescan(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
        return [r[0] for r in conn.execute(
            f"SELECT name FROM files WHERE name LIKE '%{_PREFIX}%'").fetchall()]
    finally:
        conn.close()


def test_the_fixture_indexes_the_file(smartgallery_app, gallery_with_inside_trash):
    """Control: the file is in the library before anything is deleted."""
    assert _rescan(smartgallery_app) == [gallery_with_inside_trash["name"]]


def test_a_deleted_file_does_not_come_back(smartgallery_app, gallery_with_inside_trash):
    """The regression: the next scan re-indexed it out of the trash."""
    state = gallery_with_inside_trash
    resp = smartgallery_app.app.test_client().post(
        "/galleryout/delete_batch", json={"file_ids": [state["file_id"]]})
    assert resp.status_code == 200, resp.get_data(as_text=True)

    assert os.listdir(state["trash"]), "the file was not moved to the trash at all"
    assert _rescan(smartgallery_app) == [], (
        "a deleted file was indexed again out of the trash folder")


def test_the_trash_still_receives_the_file(smartgallery_app, gallery_with_inside_trash):
    """Excluding the trash from the scan must not stop deletions reaching
    it -- that is the whole point of DELETE_TO."""
    state = gallery_with_inside_trash
    smartgallery_app.app.test_client().post(
        "/galleryout/delete_batch", json={"file_ids": [state["file_id"]]})

    trashed = os.listdir(state["trash"])
    assert len(trashed) == 1, trashed
    assert trashed[0].endswith(state["name"]), trashed
    assert not os.path.exists(os.path.join(state["base"], state["name"]))


def test_a_normal_folder_is_still_indexed(smartgallery_app, gallery_with_inside_trash):
    """The exclusion has to be the trash and nothing else: a folder that
    merely sits beside it stays in the gallery."""
    base = gallery_with_inside_trash["base"]
    other = os.path.join(base, f"{_PREFIX}keep")
    os.makedirs(other, exist_ok=True)
    kept = f"{_PREFIX}kept.png"
    Image.new("RGB", (16, 16), (9, 9, 9)).save(os.path.join(other, kept))
    try:
        assert kept in _rescan(smartgallery_app), (
            "an ordinary folder was excluded along with the trash")
    finally:
        os.remove(os.path.join(other, kept))
        os.rmdir(other)
