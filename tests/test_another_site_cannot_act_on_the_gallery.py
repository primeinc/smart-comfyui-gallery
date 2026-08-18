"""A page you happen to be visiting must not be able to act on your gallery.

Swept the state-changing routes by what a cross-site request can actually
produce. A form on another page can send urlencoded, multipart or plain
text; it cannot set a JSON content type, which is what happens to protect
the thirty-four routes that read request.json. Six do not need it:

    /galleryout/delete/<file_id>              delete_file
    /galleryout/delete_folder/<folder_key>    delete_folder
    /galleryout/toggle_favorite/<file_id>     toggle_favorite
    /galleryout/api/site_settings             api_site_settings_set
    /galleryout/api/collections/upload_note   upload_collection_note
    /galleryout/login                         exhibition_login

Two of those delete things and neither needs anything guessed: a folder
key is base64 of the relative path, so `videos` is `dmlkZW9z`, and
delete_file takes no body whatsoever.

In the default local mode there is no login at all, so there is no cookie
to withhold. The other page cannot read the answer -- that much the
browser does enforce -- but the request is sent and the folder is gone.

Browsers report where a request came from. The Fetch Metadata spec has a
form submission arriving as same-origin, same-site or cross-site "as
appropriate", and something the person did themselves -- address bar,
bookmark -- as `none`, which servers may "treat as trusted". Anything that
is not a browser sends nothing, so scripts and curl are untouched.

Only cross-site is refused, and nothing is compared here: the browser
worked out the relationship, so a reverse proxy in front changes nothing.

The limit, stated rather than implied: the spec asserts the URL is "a
potentially trustworthy URL" before setting these headers. localhost is;
a plain-http LAN address is not, and the startup banner offers one. The
session cookie is given SameSite=Lax for the modes that have a session,
which browsers apply regardless of scheme, but the default local mode has
no cookie and a LAN address gets no header either. Comparing Origin
against Host would reach that case and would also refuse proxied setups
where the two legitimately differ.
"""

from __future__ import annotations

import ast
import re

import pytest

import smartgallery

# One per shape: a delete needing no body, a delete whose key is
# base64(path), a toggle, and a settings write.
_WRITES = [
    ("POST", "/galleryout/delete/" + "d" * 32),
    ("POST", "/galleryout/delete_folder/dmlkZW9z"),
    ("POST", "/galleryout/toggle_favorite/" + "d" * 32),
    ("POST", "/galleryout/api/site_settings"),
]


@pytest.fixture
def owner(smartgallery_app, monkeypatch):
    """The default local mode, where there is no login to fall back on."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


@pytest.mark.parametrize(("method", "url"), _WRITES,
                         ids=[u.split("/")[2] + ":" + u.split("/")[-1][:12]
                              for _m, u in _WRITES])
def test_a_write_from_another_site_is_refused(owner, method, url):
    """The bug: a form on any page reached these."""
    response = owner.open(url, method=method,
                          headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code == 403, (
        f"{method} {url} was accepted from another site with "
        f"{response.status_code}")
    body = response.get_json()
    assert body is not None, response.get_data(as_text=True)[:160]
    assert "another website" in body["message"], body


@pytest.mark.parametrize("origin", ["same-origin", "same-site", "none"])
def test_the_gallery_can_still_act_on_itself(owner, origin):
    """Over-reach guard, and the whole product. Its own page sends
    same-origin; a proxied setup can produce same-site; a bookmark or the
    address bar sends none, which the spec says may be trusted."""
    response = owner.post("/galleryout/api/site_settings",
                          json={"key": "x", "value": "y"},
                          headers={"Sec-Fetch-Site": origin})

    assert response.status_code != 403, (
        f"a {origin} request was refused as though it came from elsewhere")


def test_something_that_is_not_a_browser_is_unaffected(owner):
    """Over-reach guard: scripts, curl and the container's own health
    checks send no such header, and refusing those would break every
    non-browser caller there is."""
    response = owner.post("/galleryout/api/site_settings",
                          json={"key": "x", "value": "y"})

    assert response.status_code != 403, (
        "a caller with no Sec-Fetch-Site header was refused; nothing but a "
        "browser sends one")


def test_reading_is_not_affected(owner):
    """Over-reach guard: a cross-site GET is an ordinary link. Refusing
    those would break sharing a gallery URL."""
    response = owner.get("/galleryout/api/search_options",
                         headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code != 403


def test_an_unknown_value_is_not_treated_as_hostile(owner):
    """The spec: servers SHOULD ignore the header if it holds a value they
    do not recognise, so a future one cannot lock people out."""
    response = owner.post("/galleryout/api/site_settings",
                          json={"key": "x", "value": "y"},
                          headers={"Sec-Fetch-Site": "some-future-value"})

    assert response.status_code != 403


def test_the_session_cookie_says_how_it_travels():
    """Flask leaves SameSite unset, which leaves it to the browser --
    Chrome treats that as Lax, Firefox does not. The modes that have a
    login should not depend on which browser somebody opened."""
    assert smartgallery.app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert smartgallery.app.config["SESSION_COOKIE_HTTPONLY"] is True


def test_a_form_post_cannot_carry_a_json_body():
    """Control for the whole premise. The thirty-four routes reading
    request.json are said to be out of reach of a cross-site form because
    a form cannot set that content type -- so a urlencoded body must not
    satisfy one."""
    client = smartgallery.app.test_client()

    response = client.post("/galleryout/api/collections/delete",
                           data={"collection_id": "1"})

    assert response.status_code != 200, (
        "a route that reads request.json accepted a plain form body, so "
        "the JSON content type is not the protection it was taken for")


def test_the_write_routes_are_the_ones_that_were_swept(gallery_tree):
    """The sweep, kept. A new route that changes something without needing
    a JSON body is a new one of these, and it would look like the six."""

    tree = gallery_tree
    write_sql = re.compile(r"\b(INSERT|UPDATE|DELETE)\b", re.IGNORECASE)

    def touches_state(fn):
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and write_sql.search(node.value):
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("remove", "rmtree", "rename",
                                           "replace", "unlink", "move"):
                return True
        return False

    def reads(fn, attr):
        return any(isinstance(n, ast.Attribute) and n.attr == attr
                   and isinstance(n.value, ast.Name) and n.value.id == "request"
                   for n in ast.walk(fn))

    formish = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        routes = [d for d in fn.decorator_list
                  if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                  and d.func.attr == "route" and d.args]
        if not routes:
            continue
        methods = []
        for kw in routes[0].keywords:
            if kw.arg == "methods":
                methods = [e.value for e in kw.value.elts]
        if not methods or methods == ["GET"]:
            continue
        if not touches_state(fn):
            continue
        if reads(fn, "json") and not reads(fn, "form"):
            continue
        formish.append(routes[0].args[0].value)

    assert sorted(formish) == sorted([
        "/galleryout/api/collections/upload_note",
        "/galleryout/api/site_settings",
        "/galleryout/delete/<string:file_id>",
        "/galleryout/delete_folder/<string:folder_key>",
        "/galleryout/login",
        "/galleryout/toggle_favorite/<string:file_id>",
    ]), (
        f"the set of state-changing routes a plain form can reach changed: "
        f"{sorted(formish)}. Each one is reachable from any web page a "
        f"person happens to have open.")
