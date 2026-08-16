r"""Two functions of the same name in one page: one of them is dead.

Every inline script on a page shares one global scope, and a function
declaration is hoisted, so when a name is declared twice the LAST one wins
silently. Whichever came first is dead code that still looks alive. A fix
applied to it does nothing, and reading it tells you the wrong thing about
what the page does.

This was found the hard way. exhibition.html declares renderMarkdownText
twice -- a full renderer and, further down, a much smaller one -- and the
small one is the one that runs, so the visitor's portal renders notes more
plainly than the owner's does. That is a decision for whoever wrote them,
so it is listed below rather than resolved here.

Then the sweep found jsInAttr three times on the management page: I had
added it to index.html and to two partials that index.html includes, so
the same function stood in one scope three times over with two copies
dead. Removed from the partials, which use index.html's escapeHTML the
same way.

The page is rendered rather than the templates read, because a partial's
functions only collide once it is included somewhere -- which is a fact
about the page, not about the file.
"""

from __future__ import annotations

import collections
import re

import pytest

_TAG = re.compile(r"<script([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)
_TYPE = re.compile(r"""type\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_SRC = re.compile(r"\ssrc\s*=", re.IGNORECASE)
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}
# A declaration at the outer levels of a script block. Deeply indented ones
# are inner helpers, which may legitimately repeat inside their own scopes.
_DECLARATION = re.compile(r"(?m)^[ \t]{0,8}function\s+([A-Za-z_$][\w$]*)\s*\(")

# Known, unresolved, and deliberately not decided here.
_ACCEPTED = {
    "renderMarkdownText": (
        "exhibition.html declares it twice: a full renderer and a smaller "
        "one that wins. Which the visitor's portal should use is a product "
        "decision, not a defect -- see test_note_markdown_links.py."),
}


def _executable_js(html: str) -> str:
    blocks = []
    for attrs, body in _TAG.findall(html):
        if _SRC.search(attrs) or not body.strip():
            continue
        found = _TYPE.search(attrs)
        if (found.group(1).strip().lower() if found else "") in _JS_TYPES:
            blocks.append(body)
    return "\n".join(blocks)


def _duplicates(html: str):
    names = _DECLARATION.findall(_executable_js(html))
    counted = collections.Counter(names)
    return {name: n for name, n in counted.items() if n > 1}, len(names)


@pytest.fixture()
def management_page(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()
    return client.get("/galleryout/view/_root_").get_data(as_text=True)


def test_the_sweep_reaches_the_functions(management_page):
    """Control. An indent limit that was too tight is what hid one of the
    three copies of jsInAttr from the first version of this check, so the
    count is asserted rather than assumed."""
    _dupes, total = _duplicates(management_page)

    assert total > 100, (
        f"only {total} function declarations found in "
        f"{len(_executable_js(management_page))} characters of script")
    assert "function jsInAttr" in management_page, (
        "the page no longer carries the helper this check was written for")


def test_no_function_is_declared_twice_on_the_management_page(management_page):
    dupes, _total = _duplicates(management_page)
    unexpected = {name: n for name, n in dupes.items() if name not in _ACCEPTED}

    assert not unexpected, (
        f"declared more than once, so all but the last are dead: "
        f"{unexpected}. Partials share one scope with the page that "
        f"includes them.")


def test_no_function_is_declared_twice_on_the_exhibition_page(smartgallery_app,
                                                              monkeypatch):
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "GUEST"
        session["full_name"] = "A Visitor"

    page = client.get("/galleryout/view/_root_").get_data(as_text=True)
    dupes, total = _duplicates(page)

    assert total > 20, f"only {total} declarations; this may be the login screen"
    unexpected = {name: n for name, n in dupes.items() if name not in _ACCEPTED}

    assert not unexpected, unexpected


def test_no_function_is_declared_twice_on_the_ai_dashboard(management_page,
                                                           smartgallery_app,
                                                           monkeypatch):
    """The third page. It is a separate template that neither of the other
    two includes, so nothing here is implied by them."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    response = smartgallery_app.app.test_client().get("/galleryout/aidam")
    assert response.status_code == 200, response.status_code

    dupes, total = _duplicates(response.get_data(as_text=True))

    assert total > 5, f"only {total} declarations found on the dashboard"
    unexpected = {name: n for name, n in dupes.items() if name not in _ACCEPTED}
    assert not unexpected, unexpected


def test_the_accepted_list_is_still_describing_something_real():
    """An allowlist that outlives its entry is a lie in a test file. If the
    duplicate is resolved, this fails and the entry goes."""
    from pathlib import Path

    exhibition = (Path(__file__).resolve().parent.parent
                  / "templates" / "exhibition.html").read_text(encoding="utf-8")
    declared = len(re.findall(r"function renderMarkdownText\s*\(", exhibition))

    assert declared >= 2, (
        f"exhibition.html now declares renderMarkdownText {declared} time(s); "
        f"remove it from the accepted list above.")
