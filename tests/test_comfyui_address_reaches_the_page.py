"""If you tell the gallery where ComfyUI is, the page has to be told too.

COMFYUI_SERVER_URL exists, is documented, is printed in the startup banner,
and every server-side call honours it. The page did not: eight places wrote
`http://127.0.0.1:8188` out as a literal, so for anyone running ComfyUI on
another port -- a second instance, a machine on the network, the Docker
setup -- the browser went somewhere else entirely.

Measured before the fix, with COMFYUI_SERVER_URL set to
http://192.168.1.50:8189:

    server read the setting as: http://192.168.1.50:8189
    the configured address appears  0 time(s)
    the built-in default appears    9 time(s)

What that cost: Tools -> Open ComfyUI opened a dead tab; the LoRA Manager
check probed the wrong host, so that menu item never appeared at all and
the feature looked as though it did not exist; LoRA chips in the metadata
panel linked nowhere; and the Remix help text stated the wrong address as
fact.

There was a chain in front of the literal -- localStorage, then
window._remixServerDefaultUrl -- but that middle value is assigned in
exactly one place, inside the Remix window's own data handler. Until you
had opened Remix on a file with a workflow in it, there was nothing there
and every fallback reached the literal.

The address now goes to the page as window.SG_COMFY_URL, once, in the
head. A per-job override typed into Remix still wins; this is the starting
point it starts from.
"""

from __future__ import annotations

import pytest

_CONFIGURED = "http://192.168.1.50:8189"
_BUILTIN = "http://127.0.0.1:8188"

# The literal as it appeared in the code, i.e. quoted. The bare text also
# occurs in a comment explaining why it is gone, which is not a fallback.
_QUOTED = (f"'{_BUILTIN}'", f'"{_BUILTIN}"')


@pytest.fixture()
def viewer(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"
    return client


def _page(client):
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200, response.status_code
    body = response.get_data(as_text=True)
    assert len(body) > 100000, (
        f"the gallery page is only {len(body)} bytes; this is not the page "
        f"these checks are about")
    return body


def test_a_configured_address_reaches_the_page(smartgallery_app, viewer,
                                               monkeypatch):
    """The whole point: a setting the server honours must reach the browser,
    because the browser is what opens ComfyUI."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL", _CONFIGURED)

    body = _page(viewer)

    assert _CONFIGURED in body, (
        f"COMFYUI_SERVER_URL is {_CONFIGURED} and the page never mentions "
        f"it; every ComfyUI link in the browser goes somewhere else")


def test_nothing_on_the_page_still_falls_back_to_the_builtin(smartgallery_app,
                                                             viewer,
                                                             monkeypatch):
    """One reaching the page is not enough -- eight places had their own
    copy, and one left behind is one broken button."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL", _CONFIGURED)

    body = _page(viewer)

    for literal in _QUOTED:
        assert literal not in body, (
            f"the page still carries {literal} as a fallback while "
            f"COMFYUI_SERVER_URL is {_CONFIGURED}")


def test_the_address_is_set_before_the_page_can_ask_for_it(smartgallery_app,
                                                           viewer, monkeypatch):
    """The tools menu is markup with handlers on it, not script, and it sits
    near the top of the body. The value has to be in place by then or the
    first click reads undefined."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL", _CONFIGURED)

    body = _page(viewer)

    assigned = body.find("window.SG_COMFY_URL =")
    assert assigned != -1, "window.SG_COMFY_URL is never assigned"

    used = body.find("window.SG_COMFY_URL", assigned + 1)
    assert used != -1, "nothing on the page reads window.SG_COMFY_URL"

    head_ends = body.find("</head>")
    assert head_ends != -1
    assert assigned < head_ends, (
        "window.SG_COMFY_URL is assigned in the body, after markup that "
        "already refers to it")


def test_the_remix_help_text_names_the_address_in_use(smartgallery_app, viewer,
                                                      monkeypatch):
    """It told people, as a plain statement of fact, an address that was not
    theirs."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL", _CONFIGURED)

    body = _page(viewer)

    assert "Remix sends generation jobs to" in body, "the help text is gone"
    sentence_at = body.find("Remix sends generation jobs to")
    sentence = body[sentence_at:sentence_at + 200]
    assert _CONFIGURED in sentence, f"the help text says: {sentence[:160]}"


def test_the_built_in_default_still_works(smartgallery_app, viewer, monkeypatch):
    """Over-reach guard, and the case almost everyone is in: nothing
    configured, ComfyUI in its usual place, and it must still be found."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL", _BUILTIN)

    body = _page(viewer)

    assert f'window.SG_COMFY_URL = "{_BUILTIN}"' in body, (
        "the ordinary local setup no longer gets the ordinary local address")


def test_an_address_that_is_not_configured_does_not_appear(smartgallery_app,
                                                           viewer, monkeypatch):
    """Control for the checks above. They pass by finding a string in a
    1.4MB page, so the page must not be the sort of thing every string is
    found in."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL", _BUILTIN)

    body = _page(viewer)

    assert _CONFIGURED not in body, (
        "an address nobody configured turned up in the page, so finding one "
        "there proves nothing")


def test_an_awkward_address_cannot_break_the_page(smartgallery_app, viewer,
                                                  monkeypatch):
    """It is a setting an operator types, and it is being written into a
    script tag. A value carrying markup must arrive as a value."""
    monkeypatch.setattr(smartgallery_app, "COMFYUI_SERVER_URL",
                        'http://x/</script><script>window.OWNED=1;//')

    body = _page(viewer)

    assert "http://x/</script>" not in body, (
        "the setting was written into the page as written, so it closed the "
        "script tag and everything after it became markup")
    assert "http://x/\\u003c/script\\u003e" in body, (
        "the value did not arrive at all; this check only means something "
        "while it does")


def test_no_template_keeps_its_own_copy_of_the_address():
    """The sweep that found this. A ninth copy added later is a ninth thing
    that ignores the setting, and it would look exactly like the eight."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "templates"
    offenders = []
    for template in sorted(root.rglob("*.html")):
        text = template.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if any(literal in line for literal in _QUOTED):
                # Permitted: the fallback on comfy_server_url itself, which
                # is what keeps a view that forgot to pass it from emptying
                # the value rather than substituting one. Anything else is
                # a copy that ignores the setting.
                if "comfy_server_url" in line and "default(" in line:
                    continue
                offenders.append(f"{template.relative_to(root)}:{number}")

    assert offenders == [], (
        f"templates writing {_BUILTIN} out for themselves at {offenders}. "
        f"Use window.SG_COMFY_URL, which carries COMFYUI_SERVER_URL.")


def test_both_page_views_pass_the_address():
    """Two views render the gallery template. One passing it and the other
    not would make this depend on how you arrived at the page."""
    import ast
    import io
    import pathlib

    import smartgallery

    source = pathlib.Path(smartgallery.__file__)
    tree = ast.parse(io.open(source, encoding="utf-8").read())

    renders = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "render_template"]
    # The gallery page is the one that is given the file list.
    gallery = [node for node in renders
               if "files" in {kw.arg for kw in node.keywords if kw.arg}]

    assert len(gallery) >= 2, (
        f"expected the two gallery renders, found {len(gallery)}; this check "
        f"is looking at the wrong calls")

    missing = [node.lineno for node in gallery
               if "comfy_server_url" not in {kw.arg for kw in node.keywords
                                             if kw.arg}]
    assert missing == [], (
        f"render_template at lines {missing} builds the gallery page without "
        f"comfy_server_url, so the page falls back to the built-in address")
