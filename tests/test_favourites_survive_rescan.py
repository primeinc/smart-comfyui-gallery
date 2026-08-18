"""A favourite is the person's choice, not something derived from the file.

Three scan statements cleared `is_favorite` whenever a file's mtime moved,
in the same block that clears the AI caption and embedding. Those two are
derived from the content, so a changed file must invalidate them. A
favourite is not derived from anything.

Measured against the shipped code, on one file marked, rated and
commented on:

    before rescan : favourite=1  rating=5  comments=1
    after  rescan : favourite=0  rating=5  comments=1

The rating and the comment survive because they live in tables of their
own. The favourite is a column on `files`, sitting among the derived
fields, and was swept along with them.

What makes it worth fixing rather than shrugging at is why mtime moves. It
is not only "the file was edited". Restoring a backup, copying the library
to another drive, a sync client rewriting files, a tool that rewrites
metadata -- all change mtime on everything at once. Any of those emptied
the whole Favourites view in one scan, without a word.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import time

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_SOURCE = pathlib.Path(__file__).resolve().parent.parent / "smartgallery.py"
_PREFIX = "favkeep_"


@pytest.fixture
def marked_file(smartgallery_app, monkeypatch):
    """One indexed picture the person has marked, rated and commented on."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    name = f"{_PREFIX}treasured.png"
    path = os.path.join(base, name)
    Image.new("RGB", (24, 24), (40, 80, 120)).save(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (name,)).fetchone()[0]
        conn.execute(
            "UPDATE files SET is_favorite = 1, ai_caption = ?, ai_last_scanned = 99 WHERE id = ?",
            ("a caption", file_id),
        )
        conn.execute("INSERT INTO file_ratings (file_id, client_uuid, rating) VALUES (?, ?, ?)", (file_id, "admin", 5))
        conn.execute(
            "INSERT INTO file_comments (file_id, client_uuid, author_name, comment_text) VALUES (?, ?, ?, ?)",
            (file_id, "admin", "me", "love this"),
        )
        conn.commit()
    finally:
        conn.close()

    yield file_id, path

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
    with contextlib.suppress(OSError):
        os.remove(path)


def _rescan_after_touching(smartgallery_app, path):
    """What a restored backup or a sync client does: the bytes may be
    identical, the timestamp is not."""
    later = time.time() + 300
    os.utime(path, (later, later))
    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
    finally:
        conn.close()


def _row(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        files = conn.execute(
            "SELECT is_favorite, ai_caption, ai_last_scanned FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        rating = conn.execute("SELECT rating FROM file_ratings WHERE file_id = ?", (file_id,)).fetchone()
        comments = conn.execute("SELECT COUNT(*) FROM file_comments WHERE file_id = ?", (file_id,)).fetchone()[0]
    finally:
        conn.close()
    return files, rating, comments


def test_it_starts_marked(smartgallery_app, marked_file):
    """Control. Everything below is about surviving a rescan, so the mark
    has to be there to begin with."""
    file_id, _path = marked_file
    files, rating, comments = _row(smartgallery_app, file_id)

    assert files["is_favorite"] == 1
    assert rating["rating"] == 5
    assert comments == 1


def test_the_favourite_survives_a_changed_timestamp(smartgallery_app, marked_file):
    """The bug."""
    file_id, path = marked_file

    _rescan_after_touching(smartgallery_app, path)

    files, _rating, _comments = _row(smartgallery_app, file_id)
    assert files["is_favorite"] == 1, (
        "the favourite was cleared by a rescan; a restored backup would empty the whole Favourites view at once"
    )


def test_the_derived_fields_are_still_cleared(smartgallery_app, marked_file):
    """The other half, and the way to get this wrong: deleting the whole
    conditional block would make the test above pass and leave a caption
    and an embedding describing content that has changed."""
    file_id, path = marked_file

    _rescan_after_touching(smartgallery_app, path)

    files, _rating, _comments = _row(smartgallery_app, file_id)
    assert files["ai_caption"] is None, files["ai_caption"]
    assert files["ai_last_scanned"] == 0, files["ai_last_scanned"]


def test_the_other_user_data_still_survives(smartgallery_app, marked_file):
    """These already did, and are the reason the favourite should: they are
    the same kind of thing, kept in tables of their own."""
    file_id, path = marked_file

    _rescan_after_touching(smartgallery_app, path)

    _files, rating, comments = _row(smartgallery_app, file_id)
    assert rating is not None
    assert rating["rating"] == 5
    assert comments == 1


def test_no_scan_statement_clears_the_favourite(smartgallery_app):
    """Three separate statements did this -- the full scan, the folder
    sync and the background rescan -- so fixing the one a test happens to
    exercise would leave the other two.

    Read as source rather than exercised, because reaching all three
    through their endpoints proves less than the flat statement that none
    of them contains it."""
    source = _SOURCE.read_text(encoding="utf-8")

    assert "ai_embedding = CASE" in source, (
        "the derived-field resets have gone entirely; this check would then pass without meaning anything"
    )

    offenders = re.findall(r"is_favorite\s*=\s*CASE", source)

    assert not offenders, (
        f"{len(offenders)} scan statement(s) still clear is_favorite on an "
        f"mtime change. It is the person's own choice, not derived from the "
        f"file."
    )
