"""The product's core promise, end to end: point at a folder, see the files.

Everything else in the suite tests a layer. This walks the whole path --
images on disk, a real scan, the database, and the rendered gallery page --
because each layer passing individually still allows the one thing that
matters to be broken.

The process pool is substituted throughout: these tests must not spawn real
worker processes (slow, and on Windows the child re-imports the test runner).
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "e2eprobe_"


def _purge(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    """Two images in the gallery root, scanned in-process."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    root = smartgallery_app.BASE_OUTPUT_PATH
    os.makedirs(root, exist_ok=True)
    made = []
    for name, colour in ((f"{_PREFIX}alpha.png", (210, 40, 40)), (f"{_PREFIX}beta.jpg", (40, 210, 40))):
        path = os.path.join(root, name)
        Image.new("RGB", (80, 60), colour).save(path)
        made.append((name, path))
    _purge(smartgallery_app)

    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
    finally:
        conn.close()

    yield [name for name, _p in made]

    for _name, path in made:
        with contextlib.suppress(OSError):
            os.remove(path)
    _purge(smartgallery_app)


def test_scanned_files_reach_the_database_with_their_metadata(smartgallery_app, library):
    conn = smartgallery_app.get_db_connection()
    try:
        rows = {
            r["name"]: r
            for r in conn.execute(
                "SELECT name, type, dimensions, size FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)
            ).fetchall()
        }
    finally:
        conn.close()

    assert sorted(rows) == sorted(library)
    for name in library:
        assert rows[name]["type"] == "image"
        assert rows[name]["dimensions"] == "80x60", "dimensions were not extracted"
        assert (rows[name]["size"] or 0) > 0


def test_the_gallery_page_actually_lists_them(smartgallery_app, library):
    """A successful scan that the page does not render is still an empty
    gallery as far as the user is concerned."""
    client = smartgallery_app.app.test_client()
    resp = client.get("/galleryout/", follow_redirects=True)

    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    for name in library:
        assert name in html, f"{name} was indexed but does not appear on the page"


def test_rescanning_an_unchanged_library_is_a_no_op(smartgallery_app, library):
    """The second scan must not duplicate rows -- ids are content/path
    derived, and a scan that re-inserted would grow the library forever."""
    conn = smartgallery_app.get_db_connection()
    try:
        before = conn.execute("SELECT COUNT(*) FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)).fetchone()[0]
        smartgallery_app.full_sync_database(conn)
        after = conn.execute("SELECT COUNT(*) FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)).fetchone()[0]
    finally:
        conn.close()
    assert before == after == len(library)


def test_a_deleted_file_leaves_the_library_on_the_next_scan(smartgallery_app, library):
    """Removing a file from disk must remove it from the gallery, or the
    grid fills with entries whose thumbnails 404."""
    root = smartgallery_app.BASE_OUTPUT_PATH
    gone = library[0]
    os.remove(os.path.join(root, gone))

    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
        remaining = sorted(
            r[0] for r in conn.execute("SELECT name FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)).fetchall()
        )
    finally:
        conn.close()

    assert gone not in remaining
    assert remaining == sorted(n for n in library if n != gone)
