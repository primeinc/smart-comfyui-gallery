"""An admin password has to shut the gallery and never be readable.

Four separate promises, and the file exists because each has been broken
somewhere before:

  * setting a password enforces login on its own -- nobody should have to
    remember --force-login as well
  * the stored row is an Argon2id hash, not the password
  * the right password logs in and a wrong one does not
  * the login route is reachable by anyone who can reach the server, so a
    crafted payload must be refused rather than crash it, and a missing
    account must not be distinguishable from a wrong password

Every case used to start a fresh interpreter with `--admin-pass` on argv,
load the whole gallery and build a client, three seconds apiece, to read
back three booleans and one hash. The booleans are now
`derive_login_policy`, which takes the password and the mode flags and can
simply be called. The hash and the route behaviour need a real admin row,
which `ensure_admin_user` writes -- so the fixture patches what that
function reads and calls it against the session database.
"""

from __future__ import annotations

import ast
import secrets

import pytest

# Generated per run rather than written down: these are throwaway
# credentials for this file's fixtures, and a literal one is something
# somebody eventually pastes into a real config.
_PASSWORD = secrets.token_urlsafe(16)


@pytest.fixture
def gallery_with_admin(smartgallery_app, monkeypatch):
    """The gallery as `--admin-pass <password>` leaves it, admin row and all.

    ensure_admin_user reads ADMIN_PASS_INPUT, FORCE_LOGIN and
    ADMIN_CONFIG_MISSING off the module, so those are set to what
    derive_login_policy would have produced rather than invented here.
    """
    force_login, missing, _short = smartgallery_app.derive_login_policy(_PASSWORD, False, False)
    monkeypatch.setattr(smartgallery_app, "ADMIN_PASS_INPUT", _PASSWORD)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", force_login)
    monkeypatch.setattr(smartgallery_app, "ADMIN_CONFIG_MISSING", missing)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username = 'admin'")
        conn.commit()
        smartgallery_app.ensure_admin_user(conn)
        conn.commit()
    finally:
        conn.close()

    yield smartgallery_app

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username = 'admin'")
        conn.commit()
    finally:
        conn.close()


def test_setting_a_password_switches_login_on_by_itself(smartgallery_app):
    """Nobody should have to remember to add --force-login as well."""
    force_login, missing, _short = smartgallery_app.derive_login_policy(_PASSWORD, False, False)

    assert force_login is True, "an admin password did not enforce login"
    assert missing is False, "a configured password still read as missing"


def test_a_restricted_mode_with_no_password_is_a_lockdown(smartgallery_app):
    """The counterpart: asking for the door to be shut with no key must not
    quietly serve the gallery to anyone."""
    _force, missing, _short = smartgallery_app.derive_login_policy(None, False, True)
    _force_x, missing_x, _short_x = smartgallery_app.derive_login_policy(None, True, False)

    assert missing is True, "--force-login without a password did not lock down"
    assert missing_x is True, "--exhibition without a password did not lock down"


def test_an_ordinary_local_install_is_neither(smartgallery_app):
    """No password, no mode: the single-user gallery stays open, or every
    owner is locked out of their own library."""
    force_login, missing, too_short = smartgallery_app.derive_login_policy(None, False, False)

    assert (force_login, missing, too_short) == (False, False, False)


@pytest.mark.parametrize(
    ("password", "expected"),
    [
        ("shrt", True),
        ("1234567", True),
        ("12345678", False),
        (_PASSWORD, False),
    ],
)
def test_a_short_password_is_flagged(smartgallery_app, password, expected):
    _force, _missing, too_short = smartgallery_app.derive_login_policy(password, False, False)

    assert too_short is expected, (
        f"{password!r} ({len(password)} chars) against a minimum of {smartgallery_app.ADMIN_MIN_PASSWORD_LENGTH}"
    )


def test_the_login_page_is_offered_without_the_library(gallery_with_admin):
    """Properly configured, the gallery answers with its login page rather
    than the lockdown notice -- but still without any of the library."""
    resp = gallery_with_admin.app.test_client().get("/galleryout/view/_root_")
    body = resp.get_data(as_text=True)

    assert resp.status_code == 200, resp.status_code
    assert "password" in body.lower(), "no password field on the login page"
    assert "lightbox-toolbar" not in body, "the gallery interface leaked before login"
    assert "gallery-item" not in body, "file listings leaked before login"


def test_the_password_is_never_stored_readable(gallery_with_admin):
    """The row must hold an Argon2id hash, not the password."""
    conn = gallery_with_admin.get_db_connection()
    try:
        row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
    finally:
        conn.close()

    assert row is not None, "no admin user was created"
    stored = str(row[0])
    assert _PASSWORD not in stored, "the admin password is stored in readable form"
    assert stored.startswith("$argon2"), f"not an argon2 hash: {stored[:24]}"


def test_the_right_password_logs_in_and_a_wrong_one_does_not(gallery_with_admin):
    client = gallery_with_admin.app.test_client()
    ok = client.post("/galleryout/login", json={"username": "admin", "password": _PASSWORD})

    assert ok.status_code == 200, ok.status_code
    assert ok.get_json().get("status") == "success", ok.get_json()

    fresh = gallery_with_admin.app.test_client()
    bad = fresh.post("/galleryout/login", json={"username": "admin", "password": "not-the-password"})

    assert bad.get_json().get("status") != "success", "a wrong password was accepted"
    with fresh.session_transaction() as session:
        assert not session.get("user_id"), "a failed login still created a session"


@pytest.mark.parametrize(
    "value",
    [
        123,
        ["a", "b"],
        {"x": 1},
        None,
        "pässwörd-ünïcode",
        "x" * 5000,
    ],
)
def test_a_crafted_password_cannot_crash_the_login_route(gallery_with_admin, value):
    """The route is reachable by anyone who can reach the server, so a JSON
    int, list or non-ASCII string must be refused, never a 500."""
    resp = gallery_with_admin.app.test_client().post("/galleryout/login", json={"username": "admin", "password": value})

    assert resp.status_code < 500, f"login crashed on {value!r}: {resp.status_code}"


def test_an_unknown_username_is_refused_without_revealing_itself(gallery_with_admin):
    """A missing user still runs one verification against a decoy, so the
    response cannot be told apart from a wrong password."""
    client = gallery_with_admin.app.test_client()
    missing = client.post("/galleryout/login", json={"username": "nobody-here", "password": "whatever"})
    wrong = client.post("/galleryout/login", json={"username": "admin", "password": "whatever"})

    assert missing.status_code == wrong.status_code, f"status differs: {missing.status_code} vs {wrong.status_code}"
    assert missing.get_json().get("status") == wrong.get_json().get("status")
    assert missing.get_json().get("message") == wrong.get_json().get("message"), (
        "the message reveals whether the account exists"
    )


def test_the_environment_variable_works_like_the_flag(smartgallery_app, monkeypatch):
    """CONFIGURATION.md documents ADMIN_PASSWORD as equivalent to the flag,
    and the two meet at env_or before the policy ever sees them."""
    monkeypatch.setenv("ADMIN_PASSWORD", _PASSWORD)

    from_env = smartgallery_app.env_or("ADMIN_PASSWORD", None)
    force_login, missing, _short = smartgallery_app.derive_login_policy(from_env, False, False)

    assert from_env == _PASSWORD, "ADMIN_PASSWORD no longer reaches the setting"
    assert force_login is True, "ADMIN_PASSWORD did not enforce login"
    assert missing is False


def test_startup_still_derives_the_policy(gallery_tree):
    """The three settings were three statements at import once and could be
    again, leaving derive_login_policy correct and unused."""
    called = {
        node.func.id
        for node in ast.walk(gallery_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "derive_login_policy" in called, (
        "startup no longer derives FORCE_LOGIN / ADMIN_CONFIG_MISSING / "
        "ADMIN_PASS_TOO_SHORT through derive_login_policy"
    )
