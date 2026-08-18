"""Who may see which albums exist.

The album list is filtered by what the caller is allowed to see, but the
filtering was written as a property of exhibition mode rather than of the
caller. Under `--force-login` the same non-staff account fell through to
the branch that returns everything: every album name, every file count,
and the shared_users list naming which accounts each private album was
shared with.

Reaching it does not require the interface. A CUSTOMER who opens the
management page has their session cleared and is bounced to the login
screen, but the API only ever asked whether a session exists, and logging
in is enough to get one.

The default single-user install must not be caught by any of this: with
no login configured there is one person, and they are the administrator.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def albums(smartgallery_app):
    """One public album and one private album shared with user 41."""
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM collections WHERE name LIKE 'cla_%'")
        conn.execute("INSERT INTO collections (name, type, is_public, created_at) "
                     "VALUES ('cla_public', 'user_album', 1, 1.0)")
        conn.execute("INSERT INTO collections (name, type, is_public, shared_users, created_at) "
                     "VALUES ('cla_private', 'user_album', 0, '41', 1.0)")
        conn.commit()
    finally:
        conn.close()

    yield

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM collections WHERE name LIKE 'cla_%'")
        conn.commit()
    finally:
        conn.close()


def _names(resp):
    body = resp.get_json() or {}
    entries = body.get("collections") or body.get("albums") or []
    return sorted(c.get("name", "") for c in entries if str(c.get("name", "")).startswith("cla_"))


def _client(smartgallery_app, role, user_id=9):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["role"] = role
    return client


def test_staff_see_every_album(smartgallery_app, albums, monkeypatch):
    """Control: the fixture's albums are visible to someone entitled to
    them, so an empty result below means filtering, not an empty database."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = _client(smartgallery_app, "ADMIN").get("/galleryout/api/collections")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _names(resp) == ["cla_private", "cla_public"]


def test_a_customer_does_not_learn_the_private_albums(smartgallery_app, albums, monkeypatch):
    """The regression: under --force-login this returned everything."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = _client(smartgallery_app, "CUSTOMER").get("/galleryout/api/collections")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert "cla_private" not in _names(resp), (
        "a customer was told a private album exists")
    body = resp.get_data(as_text=True)
    assert '"shared_users": "41"' not in body and "'shared_users': '41'" not in body, (
        "the response named who a private album is shared with")


def test_the_user_it_is_shared_with_still_sees_it(smartgallery_app, albums, monkeypatch):
    """The counterpart: sharing has to keep working, or the fix is just a
    blanket denial."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = _client(smartgallery_app, "CUSTOMER", user_id=41).get("/galleryout/api/collections")

    assert "cla_private" in _names(resp), (
        "the account the album was shared with cannot see it")


def test_the_default_local_install_sees_everything(smartgallery_app, albums, monkeypatch):
    """With no login configured there is one person and they own the
    library -- they must not be filtered as if they were a visitor."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = smartgallery_app.app.test_client().get("/galleryout/api/collections")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _names(resp) == ["cla_private", "cla_public"], (
        "the local administrator lost sight of their own albums")
