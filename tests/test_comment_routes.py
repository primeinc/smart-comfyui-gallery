"""Comments: irreplaceable human input, and who is allowed to change it.

Ratings and comments are the one kind of data in this app that cannot be
recomputed -- delete a thumbnail and it regenerates, delete a comment and
it is gone.

Two different worlds share these endpoints, and the tests distinguish
them because a check that looks missing in one is correct in the other:

  default install (no --force-login, not exhibition)
      every caller is the local admin. One machine, one operator, nobody
      to protect anything from -- so moderating any comment is right, not
      a hole.

  multi-user install (logins in play)
      ownership is enforced, and keyed on the SESSION user id rather than
      the client_uuid in the request body. That is what stops a caller
      simply claiming to be someone else.

These pin both: an ordinary logged-in user may only touch their own
comment, a refusal leaves the target's words byte-for-byte intact, and
the local admin can still moderate.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from PIL import Image

_PREFIX = "cmtroute_"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


@pytest.fixture
def commented_file(smartgallery_app):
    """A file with two comments from two different anonymous visitors."""
    name = f"{_PREFIX}subject.png"
    path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name)
    Image.new("RGB", (16, 16), (60, 60, 160)).save(path)
    file_id = f"{_PREFIX}file"
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
            "VALUES (?, ?, ?, ?, 'image', ?)",
            (file_id, path, os.path.getmtime(path), name, os.path.getsize(path)))
        conn.execute(
            "INSERT INTO file_comments (file_id, client_uuid, author_name, "
            "comment_text, target_audience, created_at) "
            "VALUES (?, ?, 'Me', 'my words', 'public', 1000.0)",
            (file_id, _MINE))
        mine_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO file_comments (file_id, client_uuid, author_name, "
            "comment_text, target_audience, created_at) "
            "VALUES (?, ?, 'Them', 'their words', 'public', 1001.0)",
            (file_id, _THEIRS))
        theirs_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    yield file_id, mine_id, theirs_id
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM file_comments WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()
    with contextlib.suppress(OSError):
        os.remove(path)


def _comment_text(smartgallery_app, comment_id):
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT comment_text FROM file_comments WHERE id = ?",
                           (comment_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# On a default install (no FORCE_LOGIN, not exhibition) every caller is the
# local admin -- one machine, one operator, nobody to protect anything from.
# Ownership only becomes meaningful once real users exist, so these tests
# sign in as an ordinary logged-in user. The route then keys ownership on
# the SESSION user id, ignoring any client_uuid in the body, which is what
# stops a caller claiming to be someone else.
_MINE = "41"        # comment owner == this user's id
_THEIRS = "99"      # a different user's comment


def _as_ordinary_user(client, user_id=_MINE):
    with client.session_transaction() as session:
        session["user_id"] = int(user_id)
        session["role"] = "CUSTOMER"


def test_a_user_cannot_delete_someone_elses_comment(
        smartgallery_app, client, commented_file):
    _file_id, _mine, theirs = commented_file
    _as_ordinary_user(client)

    resp = client.post("/galleryout/api/exhibition/delete_comment",
                       json={"comment_id": theirs, "client_uuid": _THEIRS})

    assert resp.status_code == 403, "one user deleted another user's comment"
    assert _comment_text(smartgallery_app, theirs) == "their words"


def test_a_user_cannot_edit_someone_elses_comment(
        smartgallery_app, client, commented_file):
    _file_id, _mine, theirs = commented_file
    _as_ordinary_user(client)

    resp = client.post("/galleryout/api/exhibition/edit_comment",
                       json={"comment_id": theirs, "client_uuid": _THEIRS,
                             "new_text": "words I am putting in their mouth"})

    assert resp.status_code == 403, "one user rewrote another user's comment"
    assert _comment_text(smartgallery_app, theirs) == "their words"


def test_a_user_can_edit_and_delete_their_own(
        smartgallery_app, client, commented_file):
    _file_id, mine, _theirs = commented_file
    _as_ordinary_user(client)

    edited = client.post("/galleryout/api/exhibition/edit_comment",
                         json={"comment_id": mine, "client_uuid": _MINE,
                               "new_text": "my revised words"})
    assert edited.status_code == 200, edited.get_data(as_text=True)
    assert _comment_text(smartgallery_app, mine) == "my revised words"

    deleted = client.post("/galleryout/api/exhibition/delete_comment",
                          json={"comment_id": mine, "client_uuid": _MINE})
    assert deleted.status_code == 200
    assert _comment_text(smartgallery_app, mine) is None


def test_edit_requires_text(smartgallery_app, client, commented_file):
    """An empty edit must be refused, not silently blank the comment."""
    _file_id, mine, _theirs = commented_file
    _as_ordinary_user(client)

    resp = client.post("/galleryout/api/exhibition/edit_comment",
                       json={"comment_id": mine, "client_uuid": _MINE, "new_text": "   "})

    assert resp.status_code == 400
    assert _comment_text(smartgallery_app, mine) == "my words"


def test_a_local_admin_can_moderate_any_comment(
        smartgallery_app, client, commented_file):
    """A default install runs without login; that operator is the owner of
    the machine and must be able to remove anything."""
    _file_id, _mine, theirs = commented_file
    if smartgallery_app.FORCE_LOGIN or smartgallery_app.IS_EXHIBITION_MODE:
        pytest.skip("local-admin path only applies to a default install")

    resp = client.post("/galleryout/api/exhibition/delete_comment",
                       json={"comment_id": theirs})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _comment_text(smartgallery_app, theirs) is None


def test_deleting_an_unknown_comment_does_not_touch_the_others(
        smartgallery_app, client, commented_file):
    _file_id, mine, theirs = commented_file

    client.post("/galleryout/api/exhibition/delete_comment",
                json={"comment_id": 99999999})

    assert _comment_text(smartgallery_app, mine) == "my words"
    assert _comment_text(smartgallery_app, theirs) == "their words"
