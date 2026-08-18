"""An expiry date on an account has to mean something.

The users table carries `expiry_date`, the user manager writes it, and that
same screen draws the account in red with an hourglass once the date has
passed. Nothing ever read it back. An operator handing a client access
until Friday saw the account marked expired on Saturday, and it went on
logging in.

That is the worst arrangement of all: the interface asserts a rule the
server does not keep, so the person relying on it has positive confirmation
of something untrue. Verified before the fix -- an account dated
2020-01-01 logged in successfully.

Two decisions worth arguing with, both deliberate:

  * The check runs AFTER the password is verified. Refusing earlier would
    tell a stranger which usernames exist, which the surrounding code takes
    care to avoid; only someone holding the right password learns that the
    account has lapsed, and they are entitled to know why they are out.
  * The admin's env/CLI password is exempt. That credential belongs to
    whoever controls the machine rather than to the account, and it is the
    documented way back in. Without the exemption, an operator who put a
    date on their own admin account would have no route to their gallery
    except editing the database by hand.
"""

from __future__ import annotations

import pytest

import sg_auth

_PASSWORD = "visitor-password-123"


@pytest.fixture
def accounts(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    rows = [
        ("exp_past", "2020-01-01T00:00"),
        ("exp_future", "2099-01-01T00:00"),
        ("exp_none", None),
        ("exp_blank", ""),
        ("exp_dateonly_past", "2019-06-30"),
        ("exp_garbage", "not-a-date"),
    ]
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username LIKE 'exp_%'")
        for username, expiry in rows:
            conn.execute(
                "INSERT INTO users (username, password, full_name, role, is_active, "
                "expiry_date) VALUES (?, ?, 'Visitor', 'CUSTOMER', 1, ?)",
                (username, sg_auth.hash_password(_PASSWORD), expiry))
        conn.commit()
    finally:
        conn.close()

    yield

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username LIKE 'exp_%'")
        conn.commit()
    finally:
        conn.close()


def _login(smartgallery_app, username, password=_PASSWORD):
    return smartgallery_app.app.test_client().post(
        "/galleryout/login", json={"username": username, "password": password})


@pytest.mark.parametrize("username", ["exp_past", "exp_dateonly_past"])
def test_an_expired_account_cannot_log_in(smartgallery_app, accounts, username):
    """The regression: both of these logged in."""
    resp = _login(smartgallery_app, username)

    assert resp.status_code == 403, resp.get_data(as_text=True)
    assert resp.get_json().get("status") != "success"


def test_the_refusal_says_when_and_what_to_do(smartgallery_app, accounts):
    """Someone locked out by a date needs to know it was the date."""
    body = _login(smartgallery_app, "exp_past").get_json()

    assert "expired" in body["message"].lower(), body
    assert "2020-01-01" in body["message"], body
    assert "administrator" in body["message"].lower(), body


@pytest.mark.parametrize("username", ["exp_future", "exp_none", "exp_blank"])
def test_an_unexpired_account_still_logs_in(smartgallery_app, accounts, username):
    """The counterpart -- refusing everyone would satisfy the tests above
    and lock every visitor out of every exhibition."""
    resp = _login(smartgallery_app, username)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert resp.get_json().get("status") == "success"


def test_an_unreadable_date_does_not_lock_anyone_out(smartgallery_app, accounts):
    """A malformed optional field is a configuration slip. Treating it as
    "expired" would turn a typo into a lockout, which is worse than the bug
    being fixed."""
    resp = _login(smartgallery_app, "exp_garbage")

    assert resp.status_code == 200, resp.get_data(as_text=True)


def test_a_wrong_password_is_refused_the_same_way_either_way(smartgallery_app, accounts):
    """The expiry check must not become a way to ask which accounts exist:
    a wrong password gets the same answer for an expired account as for a
    live one."""
    expired = _login(smartgallery_app, "exp_past", "wrong-password").get_json()
    live = _login(smartgallery_app, "exp_future", "wrong-password").get_json()
    unknown = _login(smartgallery_app, "no_such_user", "wrong-password").get_json()

    assert expired == live == unknown, (
        f"the answers differ: expired={expired} live={live} unknown={unknown}")


def test_the_admin_command_line_password_is_not_expirable(smartgallery_app,
                                                          monkeypatch):
    """The documented way back in has to keep working, or an expiry set on
    the only admin account is unrecoverable without a database editor."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "ADMIN_PASS_INPUT", "correct-horse-battery")

    # init_db creates no users; the admin row appears when a password is
    # configured at startup. Make one, expired.
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username = 'admin'")
        conn.execute(
            "INSERT INTO users (username, password, full_name, role, is_active, "
            "expiry_date) VALUES ('admin', ?, 'Admin', 'ADMIN', 1, '2020-01-01T00:00')",
            (sg_auth.hash_password("stored-password-not-used"),))
        conn.commit()
    finally:
        conn.close()
    try:
        resp = _login(smartgallery_app, "admin", "correct-horse-battery")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json().get("role") == "ADMIN", resp.get_json()
    finally:
        conn = smartgallery_app.get_db_connection()
        conn.execute("DELETE FROM users WHERE username = 'admin'")
        conn.commit()
        conn.close()
