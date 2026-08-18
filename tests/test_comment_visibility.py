"""Who may read a note is decided by the session, never by the request.

The comments endpoint took `client_uuid` from the query string and answered
with that identity's messages. Asking for someone else's id handed over
their private notes, and asking for `admin` handed over the internal staff
ones. The parameter is now ignored in favour of the session.

Exhibition rather than --force-login: the callers below are a CUSTOMER and
a GUEST, and --force-login admits only ADMIN, MANAGER and STAFF to the
interface, so neither could reach a picture there to read comments on.
Exhibition is where those roles exist. Both modes require a session, so the
anonymous case is unchanged.

Each case used to run TWO fresh interpreters -- one to seed with no flags,
because exhibition refuses to start without a database, then one under
--exhibition to assert -- so four tests cost eight process starts and eight
module loads. Nothing about the seeding needed a separate process: it only
needed to happen before the mode flags were read, and those flags are
attributes set per test now rather than argv read once at import.
"""

from __future__ import annotations

import pytest

_PASSWORD = "correct-horse-battery"
_FILE = "comment_visibility_f1"

_COMMENTS = [
    ("41", "Alice", "a public remark", "public"),
    ("admin", "Staff", "internal staff note", "internal"),
    ("admin", "Staff", "private note for user 41", "user:41"),
    ("admin", "Staff", "private note for user 77", "user:77"),
    ("77", "Bob", "bob wrote this", "public"),
]


@pytest.fixture
def exhibited_file(smartgallery_app, monkeypatch):
    """A picture in a public album with the five notes on it, under
    exhibition mode.

    In a public album so a visitor may see the picture at all: reading
    comments refuses a file the caller has no access to, and these tests
    are about which comments they then get, not about that.
    """
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
            "VALUES (?, '/x/a.png', 1.0, 'a.png', 'image', 1)",
            (_FILE,),
        )
        conn.execute("INSERT INTO collections (name, type, is_public) VALUES ('Shown', 'user_album', 1)")
        album = conn.execute("SELECT id FROM collections WHERE name='Shown'").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (album, _FILE))
        for uuid_, author, text, audience in _COMMENTS:
            conn.execute(
                "INSERT INTO file_comments (file_id, client_uuid, author_name, "
                "comment_text, target_audience, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1.0)",
                (_FILE, uuid_, author, text, audience),
            )
        conn.commit()
    finally:
        conn.close()

    force_login, missing, _short = smartgallery_app.derive_login_policy(_PASSWORD, exhibition=True, force_login=False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "ADMIN_PASS_INPUT", _PASSWORD)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", force_login)
    monkeypatch.setattr(smartgallery_app, "ADMIN_CONFIG_MISSING", missing)

    yield smartgallery_app

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM file_comments WHERE file_id = ?", (_FILE,))
        conn.execute("DELETE FROM collection_files WHERE file_id = ?", (_FILE,))
        conn.execute("DELETE FROM collections WHERE name = 'Shown'")
        conn.execute("DELETE FROM files WHERE id = ?", (_FILE,))
        conn.commit()
    finally:
        conn.close()


def _texts(resp):
    body = resp.get_json()
    return sorted(c["comment_text"] for c in (body.get("comments") or []))


def _as(gallery, user_id, role):
    client = gallery.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["role"] = role
    return client


def test_a_user_cannot_read_notes_addressed_to_someone_else(exhibited_file):
    """The parameter is ignored in favour of the session, so asking for
    another user's id does not hand over their messages."""
    client = _as(exhibited_file, 41, "CUSTOMER")

    seen = _texts(client.get(f"/galleryout/api/exhibition/comments?file_id={_FILE}&client_uuid=77"))

    assert "private note for user 77" not in seen, f"a user read another person's private note: {seen}"
    assert "internal staff note" not in seen, f"staff notes leaked: {seen}"
    assert "private note for user 41" in seen, f"own message missing: {seen}"
    assert "a public remark" in seen, seen


def test_staff_see_everything(exhibited_file):
    client = _as(exhibited_file, 1, "ADMIN")

    seen = _texts(client.get(f"/galleryout/api/exhibition/comments?file_id={_FILE}"))

    for expected in ("internal staff note", "private note for user 41", "private note for user 77", "a public remark"):
        assert expected in seen, f"{expected!r} hidden from staff: {seen}"


def test_an_anonymous_caller_is_refused_entirely(exhibited_file):
    """With logins in play the query parameter is never even reached."""
    client = exhibited_file.app.test_client()

    resp = client.get(f"/galleryout/api/exhibition/comments?file_id={_FILE}&client_uuid=41")

    assert resp.status_code == 401, resp.status_code
    body = resp.get_data(as_text=True)
    assert "private note" not in body and "internal staff note" not in body


def test_a_guest_sees_only_public_and_their_own(exhibited_file):
    client = _as(exhibited_file, "guest_deadbeefdeadbeef", "GUEST")

    seen = _texts(client.get(f"/galleryout/api/exhibition/comments?file_id={_FILE}&client_uuid=admin"))

    assert seen == ["a public remark", "bob wrote this"], f"a guest saw more than the public comments: {seen}"
