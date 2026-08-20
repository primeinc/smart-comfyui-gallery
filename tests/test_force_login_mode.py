"""--force-login has to shut the whole gallery, not just the front page.

The flag exists for an install that is reachable by more than its owner. If
it gated the index and left the APIs open, or gated reads and left writes
open, it would be worse than nothing: the operator believes the door is
shut.

Two separate claims, and both matter:

  * an anonymous caller reaches nothing -- not the pages, not the routes
    that change or destroy media
  * a logged-in CUSTOMER still reaches nothing destructive, because
    authentication is not authorisation

Each case used to start a fresh interpreter with `--force-login` on argv,
load the whole gallery module and build a client: three seconds of process
start and module loading apiece, for a flag that is one attribute.
FORCE_LOGIN is read at request time by the guards, so monkeypatch sets it
on the already-loaded gallery and the same client answers the same way
(pytest doc/en/how-to/monkeypatch.rst:243-247 -- patch the reference the
code actually reads).

test_the_flag_still_reaches_the_setting keeps the argv half honest: without
it, every test here would pass on a build where --force-login parsed to
nothing at all.
"""

from __future__ import annotations

import pytest

# Routes that change or destroy media. An unauthenticated caller must not
# reach any of them while the flag is on.
_DESTRUCTIVE = [
    ("/galleryout/delete_batch", {"file_ids": ["x"]}),
    ("/galleryout/move_batch", {"file_ids": ["x"], "destination_folder": "_root_"}),
    ("/galleryout/copy_batch", {"file_ids": ["x"], "destination_folder": "_root_"}),
    ("/galleryout/delete_folder/_root_", {}),
    ("/galleryout/rename_file/x", {"new_name": "y.png"}),
    ("/galleryout/create_folder", {"folder_name": "x", "parent_key": "_root_"}),
    ("/galleryout/prepare_batch_zip", {"file_ids": ["x"]}),
    ("/galleryout/favorite_batch", {"file_ids": ["x"], "status": True}),
]


@pytest.fixture
def locked_down(smartgallery_app, monkeypatch):
    """The gallery as `--force-login` with no admin password leaves it.

    ADMIN_CONFIG_MISSING is derived from FORCE_LOGIN at startup, not read
    back from it later, so setting only FORCE_LOGIN would leave the pages
    in a state the launcher can never actually produce -- and they answered
    200 while every API refused. Both are set, which is what
    `--force-login` without `--admin-pass` really means.
    """
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "ADMIN_CONFIG_MISSING", True)
    return smartgallery_app.app.test_client()


@pytest.fixture
def wide_open(smartgallery_app, monkeypatch):
    """The default single-user install."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


def test_the_flag_still_reaches_the_setting(smartgallery_app):
    """Control for the whole file. If --force-login stopped setting
    FORCE_LOGIN, every test below would still pass while the flag did
    nothing."""
    parsed, _unknown = smartgallery_app._parser.parse_known_args(["--force-login"])

    assert parsed.force_login is True, "--force-login no longer parses to anything"
    assert hasattr(smartgallery_app, "FORCE_LOGIN"), "FORCE_LOGIN is gone; the guards below have nothing to read"


@pytest.mark.parametrize("path", ["/galleryout/", "/galleryout/view/_root_"])
def test_login_demanded_without_a_password_locks_everything_down(locked_down, path):
    resp = locked_down.get(path, follow_redirects=True)
    body = resp.get_data(as_text=True)

    assert resp.status_code == 403, f"{path} answered {resp.status_code}"
    # The denial has to be legible -- an owner who locked themselves out
    # needs to see that a password is wanted, not a blank page.
    assert "password" in body.lower() or "login" in body.lower(), f"{path} denied without saying why"
    # And it must not carry the gallery itself.
    assert "lightbox-toolbar" not in body, f"{path} leaked the gallery interface"
    assert "gallery-item" not in body, f"{path} leaked file listings"


def test_destructive_apis_refuse_an_anonymous_caller(locked_down):
    bad = [
        (path, r.status_code)
        for path, payload in _DESTRUCTIVE
        for r in [locked_down.post(path, json=payload)]
        if r.status_code not in (401, 403)
    ]

    assert not bad, f"reachable without logging in: {bad}"


def test_a_logged_in_customer_still_cannot_use_the_management_apis(locked_down):
    """Authentication is not authorisation: a CUSTOMER account exists to
    view and rate, and must not be able to delete the library."""
    with locked_down.session_transaction() as session:
        session["user_id"] = 5
        session["role"] = "CUSTOMER"

    bad = [
        (path, r.status_code)
        for path, payload in _DESTRUCTIVE
        for r in [locked_down.post(path, json=payload)]
        if r.status_code != 403
    ]

    assert not bad, f"a CUSTOMER reached management routes: {bad}"


def test_without_the_flag_a_local_install_stays_usable(wide_open):
    """The counterpart: the default single-user install must NOT demand a
    login, or the flag would be meaningless and every owner locked out."""
    resp = wide_open.post("/galleryout/favorite_batch", json={"file_ids": ["nope"], "status": True})

    assert resp.status_code == 200, resp.status_code
