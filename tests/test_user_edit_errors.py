"""Editing a user has to fail the way creating one does.

The user routes are one endpoint with a branch per method, and only the
create branch wrapped its write. The database enforces a unique username
and a fixed set of roles, so on the edit branch both came back as an
unhandled exception:

    create alice AGAIN (duplicate)  -> 400 {'message': 'UNIQUE constraint failed...'}
    rename bob to alice (duplicate) -> 500 None
    edit bob to a bad role          -> 500 None

Renaming somebody to a username that already exists is an ordinary slip,
and it answered with a 500 carrying no body at all -- so the screen had
nothing to show and no way to say what was wrong.

Both branches now answer 400, and both say it in words rather than
quoting the constraint: "UNIQUE constraint failed: users.username" is
what the database says, not something to put in front of a person who has
just typed a name twice.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def admin(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username IN ('alice_t', 'bob_t')")
        conn.commit()
    finally:
        conn.close()

    yield client

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username IN ('alice_t', 'bob_t')")
        conn.commit()
    finally:
        conn.close()


def _create(client, username, role="STAFF"):
    return client.post("/galleryout/api/admin/users",
                       json={"username": username, "password": "longenough1",
                             "full_name": username.title(), "role": role})


def _row(smartgallery_app, username):
    conn = smartgallery_app.get_db_connection()
    try:
        return conn.execute("SELECT user_id, username, role FROM users "
                            "WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()


def test_two_people_can_be_created(smartgallery_app, admin):
    """Control: the failures below only mean something while the ordinary
    path works."""
    assert _create(admin, "alice_t").status_code == 200
    assert _create(admin, "bob_t").status_code == 200
    assert _row(smartgallery_app, "alice_t") is not None
    assert _row(smartgallery_app, "bob_t") is not None


def test_a_duplicate_username_is_refused_on_creation(admin):
    """This branch already behaved; it is here as the standard the edit
    branch is being held to."""
    _create(admin, "alice_t")

    again = _create(admin, "alice_t")

    assert again.status_code == 400, again.get_json()
    assert again.get_json()["message"] == "That username is already taken."


def test_a_duplicate_username_is_refused_on_an_edit(smartgallery_app, admin):
    """The bug: a 500 with no body."""
    _create(admin, "alice_t")
    _create(admin, "bob_t")
    bob = _row(smartgallery_app, "bob_t")

    renamed = admin.put("/galleryout/api/admin/users",
                        json={"user_id": bob["user_id"], "username": "alice_t",
                              "full_name": "Bob", "role": "STAFF", "is_active": 1})

    assert renamed.status_code == 400, (renamed.status_code, renamed.get_json())
    assert renamed.get_json()["message"] == "That username is already taken."
    assert _row(smartgallery_app, "bob_t")["username"] == "bob_t", (
        "the rename half-applied")


def test_an_unknown_role_is_refused_on_an_edit(smartgallery_app, admin):
    _create(admin, "bob_t")
    bob = _row(smartgallery_app, "bob_t")

    changed = admin.put("/galleryout/api/admin/users",
                        json={"user_id": bob["user_id"], "username": "bob_t",
                              "full_name": "Bob", "role": "SUPERUSER",
                              "is_active": 1})

    assert changed.status_code == 400, (changed.status_code, changed.get_json())
    assert changed.get_json()["message"] == "That is not a role this gallery knows."
    assert _row(smartgallery_app, "bob_t")["role"] == "STAFF"


@pytest.mark.parametrize("bad", [
    {"username": "alice_t"},
    {"role": "SUPERUSER"},
])
def test_the_message_is_not_the_database_talking(smartgallery_app, admin, bad):
    """A person reading this has just made a typo, and 'CHECK constraint
    failed: role IN (...)' tells them nothing they can act on."""
    _create(admin, "alice_t")
    _create(admin, "bob_t")
    bob = _row(smartgallery_app, "bob_t")
    payload = {"user_id": bob["user_id"], "username": "bob_t", "full_name": "Bob",
               "role": "STAFF", "is_active": 1}
    payload.update(bad)

    message = admin.put("/galleryout/api/admin/users",
                        json=payload).get_json()["message"]

    assert "constraint" not in message.lower(), message
    assert "users." not in message, message


def test_an_ordinary_edit_still_works(smartgallery_app, admin):
    """Over-reach guard: wrapping the write must not swallow the write."""
    _create(admin, "bob_t")
    bob = _row(smartgallery_app, "bob_t")

    changed = admin.put("/galleryout/api/admin/users",
                        json={"user_id": bob["user_id"], "username": "bob_t",
                              "full_name": "Bob Renamed", "role": "MANAGER",
                              "is_active": 1})

    assert changed.status_code == 200, changed.get_json()
    assert _row(smartgallery_app, "bob_t")["role"] == "MANAGER"
