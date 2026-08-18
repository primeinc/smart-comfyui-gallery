r"""A button whose handler does not exist is a button that does nothing.

Every onclick on these pages names a function, and a name that is not
declared anywhere gives a ReferenceError at click time -- in the console,
where nobody is looking. The button appears normal and does nothing at
all. With most of the interface built from partials that are included into
one page, a rename or a removed include is all it takes.

The pages are rendered rather than the templates read, because a handler
written in one file usually calls a function defined in another, and only
the assembled page can say whether the two ever meet.

Measured when this was written: 430 handlers and 190 distinct calls on the
management page, 128 and 58 on the exhibition page, 22 on the dashboard,
and nothing missing on any of them. So this guards rather than fixes --
which is why the control below matters more than the sweep does.
"""

from __future__ import annotations

import re

import pytest

_HANDLER = re.compile(r'\bon[a-z]+\s*=\s*"([^"]*)"')
# A bare call: an identifier with no dot before it, so `x.trim()` and
# CSS's `rgba(...)` are not read as function calls. The first version of
# this counted those and reported thirty-four missing names, all noise.
_CALL = re.compile(r"""(?<![.\w$'"])([A-Za-z_$][\w$]*)\s*\(""")
_STRING = re.compile(r"'[^']*'")

_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "typeof",
    "new",
    "delete",
    "void",
    "function",
    "else",
    "do",
    "in",
    "of",
}
_BUILTINS = {
    "alert",
    "confirm",
    "prompt",
    "setTimeout",
    "setInterval",
    "fetch",
    "parseInt",
    "parseFloat",
    "String",
    "Number",
    "Boolean",
    "Array",
    "Object",
    "JSON",
    "Math",
    "Date",
    "RegExp",
    "Error",
    "Promise",
    "Map",
    "Set",
    "encodeURIComponent",
    "decodeURIComponent",
    "encodeURI",
    "decodeURI",
    "isNaN",
    "console",
    "event",
    "window",
    "document",
    "navigator",
    "location",
    "history",
    "localStorage",
    "sessionStorage",
    "URL",
    "Blob",
    "FormData",
    "Image",
    "FileReader",
    "XMLHttpRequest",
    "EventSource",
    "AbortController",
    "requestAnimationFrame",
    "Intl",
}


def _called(page: str):
    """Function names the page's own handlers invoke, with an example."""
    found = {}
    for body in _HANDLER.findall(page):
        for name in _CALL.findall(_STRING.sub("''", body)):
            if name not in _KEYWORDS:
                found.setdefault(name, body[:80])
    return found


def _defined(page: str):
    return (
        set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", page))
        | set(re.findall(r"\bwindow\.([A-Za-z_$][\w$]*)\s*=", page))
        | set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", page))
        | set(re.findall(r"([A-Za-z_$][\w$]*)\s*[:=]\s*(?:async\s+)?function\b", page))
        | set(re.findall(r"([A-Za-z_$][\w$]*)\s*[:=]\s*(?:async\s*)?\([^)]*\)\s*=>", page))
    )


def _missing(page: str):
    defined = _defined(page)
    return {name: where for name, where in _called(page).items() if name not in defined and name not in _BUILTINS}


@pytest.fixture
def client(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


def test_the_check_notices_a_handler_with_nothing_behind_it():
    """Control, and the reason to believe the three sweeps below.

    They all report an absence, which is also what a matcher that finds no
    handlers produces. This page has one real handler and one dead one."""
    page = """
        <button onclick="realThing()">a</button>
        <button onclick="goneAway(1, 'x')">b</button>
        <script>function realThing() { return 1; }</script>
    """

    missing = _missing(page)

    assert set(missing) == {"goneAway"}, missing


def test_a_method_call_is_not_mistaken_for_a_missing_function():
    """The other half of the control. Reading `x.trim()` as a call to
    `trim` is what made the first version report thirty-four names that
    were all fine."""
    page = """
        <button onclick="this.value.trim(); el.querySelectorAll('a')">a</button>
        <div style="background:rgba(0,0,0,.5)" onmouseover="this.style.filter='brightness(1.2)'"></div>
    """

    assert _missing(page) == {}, _missing(page)


def test_every_handler_on_the_management_page_has_a_function(client):
    page = client.get("/galleryout/view/_root_").get_data(as_text=True)
    assert len(_HANDLER.findall(page)) > 100, "the handlers were not found"

    missing = _missing(page)

    assert not missing, (
        f"{len(missing)} handler(s) call something that is not declared on "
        f"the page, so the button does nothing: {missing}"
    )


def test_every_handler_on_the_exhibition_page_has_a_function(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "GUEST"
        session["full_name"] = "A Visitor"

    page = client.get("/galleryout/view/_root_").get_data(as_text=True)
    assert len(_HANDLER.findall(page)) > 20, "few handlers found; this may be the login screen, not the portal"

    missing = _missing(page)

    assert not missing, missing


def test_every_handler_on_the_ai_dashboard_has_a_function(client):
    response = client.get("/galleryout/aidam")
    assert response.status_code == 200, response.status_code
    page = response.get_data(as_text=True)
    assert len(_HANDLER.findall(page)) > 5, "the handlers were not found"

    missing = _missing(page)

    assert not missing, missing
