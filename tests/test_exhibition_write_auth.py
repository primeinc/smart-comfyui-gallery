"""The comment and rating writes must ask who is calling.

`post_comment` and `rate` both begin by refusing a caller with no session
when the server demands logins. Their siblings did not:

  rate_batch      no check at all
  delete_comment  no check; ownership is decided by comparing the row's
                  client_uuid against one taken from the request body
  edit_comment    the same

For an anonymous caller on a --force-login gallery, `is_privileged` is
False and both halves of "is this your comment" arrive in the request:
the comment id and the identity claiming it. Comment ids are AUTOINCREMENT
integers, so they can simply be counted through, and `admin` is the fixed
identity the page uses when there is no login -- so the admin's own
comments could be deleted or rewritten by someone who never signed in.

Ratings went the same way: an unauthenticated caller could write or clear
ratings under any identity it named.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "ewa_"


@pytest.fixture
def seeded(smartgallery_app, monkeypatch):
    """A file with a comment and a rating, both owned by 'admin'."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}pic.png")
    Image.new("RGB", (16, 16), (44, 88, 120)).save(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}pic.png",)).fetchone()[0]
        conn.execute(
            "INSERT INTO file_comments (file_id, client_uuid, author_name, "
            "comment_text, target_audience, created_at) "
            "VALUES (?, 'admin', 'Owner', 'the original words', 'public', 1.0)",
            (file_id,),
        )
        comment_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO file_ratings "
            "(file_id, client_uuid, rating, created_at) VALUES (?, 'admin', 5, 1.0)",
            (file_id,),
        )
        conn.commit()
    finally:
        conn.close()

    yield {"file_id": file_id, "comment_id": comment_id}

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
    with contextlib.suppress(OSError):
        os.remove(path)


def _comment(smartgallery_app, comment_id):
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT comment_text FROM file_comments WHERE id = ?", (comment_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _rating(smartgallery_app, file_id, client_uuid="admin"):
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute(
            "SELECT rating FROM file_ratings WHERE file_id = ? AND client_uuid = ?", (file_id, client_uuid)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_an_anonymous_caller_cannot_delete_someone_elses_comment(smartgallery_app, seeded):
    """The regression: claiming the author's identity was enough."""
    client = smartgallery_app.app.test_client()

    resp = client.post(
        "/galleryout/api/exhibition/delete_comment", json={"comment_id": seeded["comment_id"], "client_uuid": "admin"}
    )

    assert resp.status_code in (401, 403), resp.status_code
    assert _comment(smartgallery_app, seeded["comment_id"]) == "the original words", (
        "an unauthenticated caller deleted the comment"
    )


def test_an_anonymous_caller_cannot_rewrite_someone_elses_comment(smartgallery_app, seeded):
    client = smartgallery_app.app.test_client()

    resp = client.post(
        "/galleryout/api/exhibition/edit_comment",
        json={"comment_id": seeded["comment_id"], "client_uuid": "admin", "new_text": "defaced"},
    )

    assert resp.status_code in (401, 403), resp.status_code
    assert _comment(smartgallery_app, seeded["comment_id"]) == "the original words", (
        "an unauthenticated caller rewrote the comment"
    )


def test_an_anonymous_caller_cannot_write_ratings(smartgallery_app, seeded):
    """rate_batch took an identity from the body and wrote under it."""
    client = smartgallery_app.app.test_client()

    resp = client.post(
        "/galleryout/api/exhibition/rate_batch",
        json={"file_ids": [seeded["file_id"]], "rating": 1, "client_uuid": "admin"},
    )

    assert resp.status_code in (401, 403), resp.status_code
    assert _rating(smartgallery_app, seeded["file_id"]) == 5, "an unauthenticated caller overwrote the rating"


def test_an_anonymous_caller_cannot_clear_ratings(smartgallery_app, seeded):
    """The same route deletes when the rating is 0, which is the more
    destructive half."""
    client = smartgallery_app.app.test_client()

    client.post(
        "/galleryout/api/exhibition/rate_batch",
        json={"file_ids": [seeded["file_id"]], "rating": 0, "client_uuid": "admin"},
    )

    assert _rating(smartgallery_app, seeded["file_id"]) == 5, "an unauthenticated caller cleared the rating"


def test_the_author_can_still_delete_their_own_comment(smartgallery_app, seeded):
    """The counterpart -- refusing everyone would satisfy the tests above
    while breaking comments for the people who own them."""
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = "admin"
        session["role"] = "CUSTOMER"

    resp = client.post("/galleryout/api/exhibition/delete_comment", json={"comment_id": seeded["comment_id"]})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _comment(smartgallery_app, seeded["comment_id"]) is None


def test_staff_can_still_moderate(smartgallery_app, seeded):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    resp = client.post("/galleryout/api/exhibition/delete_comment", json={"comment_id": seeded["comment_id"]})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _comment(smartgallery_app, seeded["comment_id"]) is None


def test_the_default_local_install_still_writes(smartgallery_app, seeded, monkeypatch):
    """No login configured: one person, and the comments are theirs."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()

    resp = client.post(
        "/galleryout/api/exhibition/rate_batch",
        json={"file_ids": [seeded["file_id"]], "rating": 2, "client_uuid": "admin"},
    )

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _rating(smartgallery_app, seeded["file_id"]) == 2
