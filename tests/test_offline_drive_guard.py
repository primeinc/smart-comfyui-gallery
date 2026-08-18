"""A gallery folder that is temporarily unreachable must not erase the library.

People keep ComfyUI output on a second drive or a NAS. When that path is
briefly unavailable -- unplugged, asleep, a share that has not remounted
yet -- a scan sees an empty folder and concludes every file was deleted.

That conclusion is unrecoverable. `file_ratings` and `file_comments` are
declared `ON DELETE CASCADE` against `files`, and `PRAGMA foreign_keys`
is on for every connection, so removing the rows takes every star rating
and every comment with them. The images come back when the drive does;
the ratings do not.

The scan already has a guard for exactly this, but it only covers folders
registered in `mounted_folders`. The gallery root itself -- the one path
every install has, and the one most likely to live on the external drive
-- was not covered by it.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image

_PREFIX = "offguard_"


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


def _scan(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
    finally:
        conn.close()


@pytest.fixture
def rated_library(smartgallery_app, monkeypatch):
    """Two indexed images, one of them rated and commented on."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    os.makedirs(base, exist_ok=True)
    made = [os.path.join(base, f"{_PREFIX}one.png"), os.path.join(base, f"{_PREFIX}two.png")]
    for path in made:
        Image.new("RGB", (16, 16), (200, 30, 90)).save(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    _scan(smartgallery_app)

    conn = smartgallery_app.get_db_connection()
    try:
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}one.png",)).fetchone()[0]
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
        conn.commit()
    finally:
        conn.close()

    yield file_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    for path in made:
        with contextlib.suppress(OSError):
            os.remove(path)


def _counts(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        files = conn.execute(f"SELECT COUNT(*) FROM files WHERE name LIKE '{_PREFIX}%'").fetchone()[0]
        ratings = conn.execute("SELECT COUNT(*) FROM file_ratings WHERE file_id = ?", (file_id,)).fetchone()[0]
        comments = conn.execute("SELECT COUNT(*) FROM file_comments WHERE file_id = ?", (file_id,)).fetchone()[0]
        return files, ratings, comments
    finally:
        conn.close()


def test_the_fixture_really_is_indexed_and_rated(smartgallery_app, rated_library):
    """Control: the later assertions are meaningless against an empty DB."""
    assert _counts(smartgallery_app, rated_library) == (2, 1, 1)


def test_an_unreachable_gallery_root_does_not_wipe_the_library(smartgallery_app, rated_library):
    """The drive is moved out from under a running scan, then comes back.

    Nothing about the library should have changed: the files still exist,
    the user simply could not reach them for a moment.
    """
    base = smartgallery_app.BASE_OUTPUT_PATH
    offline = base + "_unplugged"
    os.rename(base, offline)
    try:
        _scan(smartgallery_app)
    finally:
        os.rename(offline, base)
        smartgallery_app.get_dynamic_folder_config(force_refresh=True)

    files, ratings, comments = _counts(smartgallery_app, rated_library)
    assert files == 2, f"a temporarily missing drive emptied the library ({files} rows left)"
    assert ratings == 1, "star ratings were destroyed when the drive went away"
    assert comments == 1, "comments were destroyed when the drive went away"


def test_a_genuinely_deleted_file_is_still_removed(smartgallery_app, rated_library):
    """The counterpart: the guard must not stop ordinary cleanup, or the
    gallery fills with entries for files that really are gone."""
    base = smartgallery_app.BASE_OUTPUT_PATH
    os.remove(os.path.join(base, f"{_PREFIX}two.png"))

    _scan(smartgallery_app)

    files, ratings, _comments = _counts(smartgallery_app, rated_library)
    assert files == 1, f"a deleted file kept its row ({files} rows)"
    assert ratings == 1, "the surviving file lost its rating"
