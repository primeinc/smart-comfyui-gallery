r"""A note may not smuggle a scheme into a link.

Collection notes are .txt/.md files uploaded through the management side
and read by whoever the collection is shared with, which is the point of
them -- production briefs go to clients. They are turned into HTML in the
browser by renderMarkdownText.

That renderer escapes the note before applying any markdown rule, so a
typed <script> stays text and a quote cannot close an attribute. Both were
confirmed by running it. What it did not check was the scheme of a link:

    [click me](javascript:alert(1))  ->  <a href="javascript:alert(1" ...>
    [x](data:text/html;base64,...)   ->  <a href="data:text/html;base64,...">

so a note could hand the reader a link that runs something when clicked.

Only http, https, ftp and mailto now reach an href or a src; anything else
becomes '#'. Relative paths and #anchors carry no scheme and are left
alone.

Two things worth knowing about this file. exhibition.html declares
renderMarkdownText twice -- a full one and, later, a much smaller one with
no links or images at all. The later declaration is the one that runs, so
the rich renderer there is dead code. Both are fixed anyway, because the
dead one becomes live the moment anyone removes the other. Which of the
two the visitor's portal is meant to use is a question for whoever wrote
them, and is reported rather than decided here.

And the markdown URL pattern is [^)\s]+, so a tab inside a URL ends the
match and never reaches the helper. The helper strips control characters
before reading the scheme regardless, because that is what browsers do,
and it is checked directly below rather than through the markdown path.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from node_runner import run_node

pytestmark = pytest.mark.spawns  # every check here runs another program

_TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "templates"

_BLOCKED = [
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    "data:text/html;base64,PHN2Zz4=",
    "vbscript:msgbox",
    "file:///etc/passwd",
]
_ALLOWED = [
    "https://example.com/a",
    "http://example.com",
    "ftp://files.example.com/x.zip",
    "mailto:someone@example.com",
    "/relative/path.png",
    "#anchor",
    "//cdn.example.com/x.png",
]


def _function(source: str, name: str) -> str:
    """One whole function, matched by counting braces rather than by a
    regex -- these run to two hundred lines and nest."""
    start = source.index(f"function {name}(")
    depth, index = 0, start
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    return source[start : index + 1]


def _renderer(template: str) -> str:
    """The template's own renderer, runnable.

    When markdownSafeUrl is absent a pass-through stands in, so a build
    without the fix still renders and the tests below fail on what it
    produces. Asserting the helper into existence here would make every one
    of them error in the fixture instead -- including the two that are
    supposed to pass on both builds, which are the only reason to trust the
    rest."""
    source = (_TEMPLATES / template).read_text(encoding="utf-8")
    guard = (
        "function markdownSafeUrl(url) { return url; }"
        if "function markdownSafeUrl(" not in source
        else _function(source, "markdownSafeUrl")
    )
    return "\n".join(
        [
            # A stand-in for the page's own escaper, which lives elsewhere.
            (
                "function escapeHTML(v){ return String(v).replace(/[&<>\"']/g,"
                " m => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"
                "\"'\":'&#39;'}[m])); }"
            ),
            guard,
            _function(source, "renderMarkdownText"),
        ]
    )


def _render(template: str, notes):
    script = (
        _renderer(template)
        + r"""
const notes = JSON.parse(process.argv[1]);
console.log(JSON.stringify(notes.map(n => {
    const html = renderMarkdownText(n);
    const found = /(?:href|src)="([^"]*)"/.exec(html);
    return {html, url: found ? found[1] : null};
})));
"""
    )
    return run_node(script, notes)


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_a_dangerous_scheme_never_reaches_a_link(template):
    """The bug, run rather than argued."""
    notes = [f"[click me]({url})" for url in _BLOCKED]

    results = _render(template, notes)

    for url, result in zip(_BLOCKED, results, strict=False):
        assert result["url"] == "#", (url, result["url"])
        assert "javascript:" not in result["html"], (url, result["html"])
        assert "vbscript:" not in result["html"], (url, result["html"])


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_an_ordinary_link_still_works(template):
    """The half that makes this a fix rather than a removal: a brief full
    of reference links has to keep them."""
    notes = [f"[see this]({url})" for url in _ALLOWED]

    results = _render(template, notes)

    for url, result in zip(_ALLOWED, results, strict=False):
        assert result["url"] == url, (url, result["url"])


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_an_image_is_guarded_the_same_way(template):
    """Images take a URL from the same place and were missed the same way."""
    blocked = _render(template, ["![pic](javascript:alert(1))"])[0]
    allowed = _render(template, ["![pic](https://example.com/p.png)"])[0]

    assert blocked["url"] == "#", blocked
    assert allowed["url"] == "https://example.com/p.png", allowed


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_typed_markup_is_still_inert(template):
    """This part already worked -- the note is escaped before any rule runs
    -- and it must keep working, since a scheme check is no substitute."""
    result = _render(template, ["<script>alert(1)</script> and <b>bold</b>"])[0]

    assert "<script>" not in result["html"], result["html"]
    assert "&lt;script&gt;" in result["html"], result["html"]


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_a_scheme_hidden_behind_control_characters_is_caught(template):
    """Browsers strip control characters and spaces before reading a
    scheme, so java<tab>script: is javascript: to them.

    The markdown pattern ends a URL at whitespace, so this cannot arrive
    through a note today. The helper is called directly, because the reason
    it holds should not be an accident of another regex."""
    source = (_TEMPLATES / template).read_text(encoding="utf-8")
    assert "function markdownSafeUrl(" in source, f"{template} has no markdownSafeUrl; there is nothing to call"
    script = (
        _renderer(template)
        + r"""
const values = JSON.parse(process.argv[1]);
console.log(JSON.stringify(values.map(v => markdownSafeUrl(v))));
"""
    )
    blocked_a, blocked_b, blocked_c, allowed = run_node(
        script,
        ["java\tscript:alert(1)", "java\nscript:alert(1)", " javascript:alert(1)", "https://example.com"],
    )

    assert blocked_a == "#", blocked_a
    assert blocked_b == "#", blocked_b
    assert blocked_c == "#", blocked_c
    assert allowed == "https://example.com", allowed


def test_no_markdown_url_reaches_an_attribute_unchecked():
    """The rule behind the six lines, so the next renderer added has to go
    through it too."""
    offenders = []
    for template in sorted(_TEMPLATES.rglob("*.html")):
        for number, line in enumerate(template.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'(?:href|src)="\$\{url\}"', line):
                offenders.append(f"{template.name}:{number}")

    assert not offenders, f"{len(offenders)} markdown link(s) put a URL straight into an attribute: {offenders}"
