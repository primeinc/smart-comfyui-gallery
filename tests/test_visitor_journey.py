"""The whole visitor journey through an exhibition, in one pass.

Exhibition mode has been changed a great deal: album listings redacted,
per-file details redacted, reviews redacted, the storyboard and the live
review runner restricted, thumbnails forced through generation, prompts
withheld when they cannot be stripped, comment and rating writes made to
require a caller, expired accounts refused.

Every one of those was tested where it was made. None of them tested that a
visitor can still do the thing the mode exists for. Each change was safe on
its own and the accumulation is what needs watching -- the failure this
guards against is not a bug in any one of them but a gallery that no longer
works for the people it was built to show things to.

So: log in, find the album, list it, open a picture, fetch it, rate it,
comment, and read the comments back. In order, as a person would.
"""

from __future__ import annotations

import concurrent.futures
import os

import pytest
from PIL import Image, PngImagePlugin

import sg_auth

_PREFIX = "journey_"
_PROMPT = "JOURNEYPROMPT a lighthouse in fog"
_PASSWORD = "visitor-password-123"


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


@pytest.fixture()
def exhibition(smartgallery_app, monkeypatch):
    """An exhibition with one public album, one picture, and one visitor."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures,
                        "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}pic.png")
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", _PROMPT)
    Image.new("RGB", (48, 48), (90, 120, 160)).save(path, pnginfo=info)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute(f"DELETE FROM users WHERE username LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?",
                               (f"{_PREFIX}pic.png",)).fetchone()[0]
        conn.execute("UPDATE files SET workflow_prompt = ? WHERE id = ?",
                     (_PROMPT, file_id))
        conn.execute("INSERT INTO collections (name, type, is_public, created_at) "
                     "VALUES (?, 'user_album', 1, 1.0)", (f"{_PREFIX}album",))
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) "
                     "VALUES (?, ?)", (coll_id, file_id))
        conn.execute("INSERT INTO users (username, password, full_name, role, "
                     "is_active) VALUES (?, ?, 'A Visitor', 'CUSTOMER', 1)",
                     (f"{_PREFIX}guest", sg_auth.hash_password(_PASSWORD)))
        conn.commit()
    finally:
        conn.close()

    yield {"file_id": file_id, "coll_id": coll_id, "album": f"{_PREFIX}album"}

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute(f"DELETE FROM collections WHERE name LIKE '{_PREFIX}%'")
        conn.execute(f"DELETE FROM users WHERE username LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    try:
        os.remove(path)
    except OSError:
        pass


def test_a_visitor_can_do_the_whole_journey(smartgallery_app, exhibition):
    """Login, browse, open, view, rate, comment, read back."""
    client = smartgallery_app.app.test_client()
    file_id = exhibition["file_id"]

    # 1. Sign in.
    login = client.post("/galleryout/login",
                        json={"username": f"{_PREFIX}guest", "password": _PASSWORD})
    assert login.status_code == 200, login.get_data(as_text=True)
    assert login.get_json()["status"] == "success"

    # 2. The album list, which is the front page of an exhibition.
    albums = client.get("/galleryout/api/collections")
    assert albums.status_code == 200, albums.get_data(as_text=True)
    assert exhibition["album"] in albums.get_data(as_text=True), (
        "the visitor cannot see the public album at all")

    # 3. Open it and get its contents.
    listing = client.get(f"/galleryout/collection/{exhibition['coll_id']}",
                         headers={"Accept": "application/json"})
    assert listing.status_code == 200, listing.get_data(as_text=True)
    names = [f["name"] for f in listing.get_json()["files"]]
    assert names == [f"{_PREFIX}pic.png"], names

    # 4. The picture itself, and its thumbnail.
    picture = client.get(f"/galleryout/file/{file_id}")
    assert picture.status_code == 200, picture.status_code
    assert picture.get_data()[:8] == b"\x89PNG\r\n\x1a\n", "that is not a PNG"
    assert _PROMPT.encode() not in picture.get_data(), "the prompt came with it"

    thumb = client.get(f"/galleryout/thumbnail/{file_id}")
    assert thumb.status_code == 200, thumb.status_code

    # 5. The details panel behind the picture.
    details = client.get(f"/galleryout/api/file_full_details/{file_id}")
    assert details.status_code == 200, details.get_data(as_text=True)
    assert details.get_json()["file"]["name"] == f"{_PREFIX}pic.png"
    assert _PROMPT not in details.get_data(as_text=True), "the prompt came with it"

    # 6. Rate it.
    rated = client.post("/galleryout/api/exhibition/rate",
                        json={"file_id": file_id, "rating": 4})
    assert rated.status_code == 200, rated.get_data(as_text=True)

    # 7. Say something about it.
    posted = client.post("/galleryout/api/exhibition/post_comment",
                         json={"file_id": file_id, "text": "lovely light"})
    assert posted.status_code == 200, posted.get_data(as_text=True)

    # 8. Read it back, as the page does after posting.
    comments = client.get(f"/galleryout/api/exhibition/comments?file_id={file_id}")
    assert comments.status_code == 200, comments.get_data(as_text=True)
    texts = [c["comment_text"] for c in comments.get_json()["comments"]]
    assert "lovely light" in texts, texts

    # 9. And the rating stuck, under the identity the visitor was given.
    conn = smartgallery_app.get_db_connection()
    try:
        stored = conn.execute("SELECT rating FROM file_ratings WHERE file_id = ?",
                              (file_id,)).fetchone()
    finally:
        conn.close()
    assert stored and stored[0] == 4, "the visitor's rating was not recorded"


def test_browsing_and_writing_need_a_session(smartgallery_app, exhibition):
    """The counterpart: without it, the journey passing proves only that
    something answered, not that anything is protected."""
    client = smartgallery_app.app.test_client()
    file_id = exhibition["file_id"]

    answered = []
    for label, method, url, payload in [
        ("album list", "GET", "/galleryout/api/collections", None),
        ("listing", "GET", f"/galleryout/collection/{exhibition['coll_id']}", None),
        ("rate", "POST", "/galleryout/api/exhibition/rate",
         {"file_id": file_id, "rating": 1}),
        ("comment", "POST", "/galleryout/api/exhibition/post_comment",
         {"file_id": file_id, "text": "no"}),
    ]:
        resp = (client.post(url, json=payload) if method == "POST"
                else client.get(url, headers={"Accept": "application/json"}))
        body = resp.get_data().decode("utf-8", "replace")
        if resp.status_code == 200 and "exhibition_login" not in body:
            answered.append(f"{label} answered 200")

    assert answered == [], f"reachable without signing in: {answered}"


def test_a_public_file_itself_is_served_without_a_session(smartgallery_app,
                                                          exhibition):
    """Not a hole -- the rule, written down.

    is_file_accessible has an explicit branch for a caller with no session:
    in exhibition mode a file may be fetched when it belongs to a PUBLIC
    album. Browsing needs a session and writing needs a session, but the
    bytes of something published do not, which is what makes a direct link
    to an exhibited picture work.

    Pinned here because the difference is easy to mistake for an oversight
    and 'tighten', which would break every direct link to an exhibition."""
    client = smartgallery_app.app.test_client()

    picture = client.get(f"/galleryout/file/{exhibition['file_id']}")
    assert picture.status_code == 200, picture.status_code

    # And the prompt is still withheld from that anonymous caller.
    assert _PROMPT.encode() not in picture.get_data()
    details = client.get(f"/galleryout/api/file_full_details/{exhibition['file_id']}")
    assert _PROMPT not in details.get_data(as_text=True)


def test_a_file_in_no_album_is_not_served_without_a_session(smartgallery_app,
                                                             exhibition):
    """The bound on the rule above: public means in a public album, not
    'any file whose id you can name'."""
    base = smartgallery_app.BASE_OUTPUT_PATH
    private_path = os.path.join(base, f"{_PREFIX}uncurated.png")
    Image.new("RGB", (16, 16), (1, 1, 1)).save(private_path)

    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
        uncurated = conn.execute("SELECT id FROM files WHERE name = ?",
                                 (f"{_PREFIX}uncurated.png",)).fetchone()[0]
    finally:
        conn.close()

    try:
        resp = smartgallery_app.app.test_client().get(f"/galleryout/file/{uncurated}")
        assert resp.status_code == 403, resp.status_code
    finally:
        os.remove(private_path)
