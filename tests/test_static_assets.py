"""/static/ must not hand the management interface to a stranger.

The app is built as `Flask(__name__, static_folder='templates',
static_url_path='/static')`, so every template is also a downloadable file
under /static/. That route is Flask's own, not one of ours, so none of the
per-route decisions applied to it -- and test_every_route_is_classified
could not see it either, because it reads the source for @app.route
functions and this endpoint has none.

Measured on a server started with --force-login, before the fix:

    /galleryout/                            -> 302   (correct)
    /static/index.html                      -> 200   700579 bytes
    /static/modals/user_manager_module.html -> 200    32633 bytes

No gallery data leaves that way: the files are served unrendered, so no
{{ }} is filled in, and every endpoint the markup calls is gated
separately. What it does hand out is the entire shape of the management
side to someone who never signed in, on a server whose whole point is that
they must.

Gating it is safe because of one fact this file pins: the login screen and
the exhibition portal reference no static asset at all. Only index.html
does, for two stylesheets, and index.html is only ever shown to a session
that has already signed in.
"""

from __future__ import annotations

import ast
import io
import pathlib

import pytest

_SOURCE = pathlib.Path(__file__).resolve().parent.parent / "smartgallery.py"

_MANAGEMENT_MARKUP = "/static/index.html"
_USER_MANAGER = "/static/modals/user_manager_module.html"
_STYLESHEET = "/static/css/index.css"


@pytest.fixture()
def local(smartgallery_app, monkeypatch):
    """No login configured -- the common case, and the one that must not
    change."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


@pytest.fixture()
def locked(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


@pytest.fixture()
def exhibition(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    return smartgallery_app.app.test_client()


def test_a_local_gallery_still_serves_its_assets(local):
    """Control, and the no-regression check that matters most: where no
    login is configured nothing is gated, so a blanket refusal -- which
    would satisfy every other test here -- fails this one."""
    for path in (_STYLESHEET, _MANAGEMENT_MARKUP):
        response = local.get(path)
        assert response.status_code == 200, path
        assert len(response.get_data()) > 1000, path


def test_force_login_refuses_the_management_markup(locked):
    """The leak, in the shape it shipped."""
    assert locked.get(_MANAGEMENT_MARKUP).status_code == 403
    assert locked.get(_USER_MANAGER).status_code == 403


def test_exhibition_refuses_it_too(exhibition):
    """A visitor is meant to see the portal, not the management side."""
    assert exhibition.get(_MANAGEMENT_MARKUP).status_code == 403
    assert exhibition.get(_STYLESHEET).status_code == 403


def test_a_signed_in_session_still_gets_the_stylesheets(locked):
    """The interface has to keep working. index.html pulls two stylesheets
    through this same route, and refusing those would leave everyone who
    did sign in looking at unstyled markup."""
    with locked.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    response = locked.get(_STYLESHEET)

    assert response.status_code == 200
    assert len(response.get_data()) > 1000


def test_the_login_screen_needs_no_static_asset(locked):
    """The assumption the whole fix rests on: if the login page pulled a
    stylesheet through /static/, gating it would leave an unstyled login
    screen for everyone."""
    page = locked.get("/galleryout/view/_root_")

    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "password" in body.lower(), "this is meant to be the login screen"
    assert "/static/" not in body, body[:400]


def test_only_the_static_endpoint_escapes_the_route_audit(smartgallery_app):
    """The structural hole behind this bug.

    test_every_route_is_classified reads the source and sorts @app.route
    functions by how they decide who may call them. An endpoint registered
    any other way is invisible to it -- which is exactly how this one went
    unnoticed. Compare what Flask actually serves against what that audit
    can see, so the next add_url_rule has to be accounted for rather than
    silently unguarded."""
    tree = ast.parse(io.open(_SOURCE, encoding="utf-8").read())
    from_source = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for dec in node.decorator_list
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
        and dec.func.attr == "route"
    }
    assert len(from_source) > 50, len(from_source)

    live = {rule.endpoint for rule in smartgallery_app.app.url_map.iter_rules()}
    assert len(live) > len(from_source), (len(live), len(from_source))

    # The AI blueprint registers its own with an explicit policy of its
    # own, stated in create_ai_blueprint and tested separately.
    unaccounted = {e for e in live - from_source if not e.startswith("aidam.")}

    assert unaccounted == {"static"}, (
        f"{sorted(unaccounted)} are served but are not @app.route functions, "
        f"so the route audit never sees them and nothing states who may call "
        f"them. Gate them and list them here.")
