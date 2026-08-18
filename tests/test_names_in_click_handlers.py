"""A name in a click handler has to survive two parsers, not one.

The folder menu built its buttons like this:

    <button onclick="deleteFolder(event, '${folderKey}', '${folderDisplayName}')">

with folderDisplayName taken straight off the folder:

    const folderDisplayName = folder.display_name;

Folder names come from the filesystem, so "Bob's renders" is ordinary --
and that apostrophe ends the JavaScript string early, so Delete, Rename
and Unmount all threw instead of doing anything. A name holding a double
quote is worse: it closes the onclick attribute and can add another
attribute of its own, which is a script someone else wrote running in the
gallery owner's browser.

Comment author names had a smaller version of the same hole. They were
escaped for HTML text and then for the JavaScript quote, but not for the
backslash and not for the double quote -- the comment body beside them
escaped both, which is what makes it an oversight rather than a decision.

The value passes through the HTML attribute parser and then the JavaScript
parser, so it needs escaping for both, in that order. These tests run the
real function out of the template with node and put its output back
through both parsers, because reasoning about escaping is exactly the
thing that produced the bug.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from node_runner import run_node

pytestmark = pytest.mark.spawns  # every check here runs another program

_TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "templates"

# Names people really have, and names people really try.
_CASES = [
    "Bob's renders",
    "O'Brien",
    'Anne "Annie" Smith',
    "back\\slash",
    "quote'and\"both",
    "<script>alert(1)</script>",
    '" onmouseover="alert(1)',
    "'); alert(1); ('",
    "ampersand & co",
    "测试 folder",
    "",
]


def _extract(template: str) -> str:
    """The jsInAttr function, exactly as the browser will run it."""
    source = (_TEMPLATES / template).read_text(encoding="utf-8")
    # The closing brace sits at whatever column the enclosing script uses --
    # column 0 in collections.html, four or eight elsewhere. Requiring an
    # indent captured a truncated chunk there and blamed the template.
    match = re.search(r"function jsInAttr\(value\) \{.*?\n {0,8}\}", source, re.DOTALL)
    assert match, f"{template} has no jsInAttr; the escaping has gone"
    extracted = match.group(0)
    assert extracted.count("replace(") == 6, (
        f"{template}: extracted {extracted.count('replace(')} replace calls, "
        f"expected 6 -- the function was cut short:\n{extracted}"
    )
    return extracted


def _round_trip(template: str, values):
    """Escape each value, then undo what the two parsers would do."""
    script = (
        _extract(template)
        + r"""
const values = JSON.parse(process.argv[1]);
const out = values.map(v => {
    const escaped = jsInAttr(v);
    // 1. What the browser does with the attribute: decode entities.
    const decoded = escaped
        .replace(/&quot;/g, '"').replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>').replace(/&amp;/g, '&');
    // 2. What JavaScript then does: read it as a single-quoted string.
    //    eval of a lone string literal, on text this file just produced.
    let parsed = null, threw = null;
    try { parsed = eval("'" + decoded + "'"); } catch (e) { threw = String(e); }
    return {escaped, decoded, parsed, threw};
});
console.log(JSON.stringify(out));
"""
    )
    return run_node(script, values)


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_every_name_survives_both_parsers(template):
    """The whole contract: what goes in is what the handler receives."""
    results = _round_trip(template, _CASES)

    for original, result in zip(_CASES, results, strict=False):
        assert result["threw"] is None, (original, result)
        assert result["parsed"] == original, (original, result)


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_nothing_can_close_the_attribute(template):
    """The security half. A bare double quote ends onclick=" and whatever
    follows becomes attributes on the tag."""
    results = _round_trip(template, _CASES)

    for original, result in zip(_CASES, results, strict=False):
        assert '"' not in result["escaped"], (original, result["escaped"])
        assert "<" not in result["escaped"], (original, result["escaped"])


@pytest.mark.parametrize("template", ["index.html", "exhibition.html"])
def test_the_check_would_notice_an_unescaped_value(template):
    """Control. Without it the two tests above could be passing because the
    round trip is lenient rather than because the escaping works, so the
    same values are put through with no escaping at all and must fail."""
    script = r"""
const values = JSON.parse(process.argv[1]);
const out = values.map(v => {
    const decoded = String(v);
    let parsed = null, threw = null;
    try { parsed = eval("'" + decoded + "'"); } catch (e) { threw = String(e); }
    return {escaped: decoded, parsed, threw};
});
console.log(JSON.stringify(out));
"""
    raw = run_node(script, _CASES)

    broke = [
        c
        for c, r in zip(_CASES, raw, strict=False)
        if r["threw"] is not None or r["parsed"] != c or '"' in r["escaped"]
    ]

    assert len(broke) >= 5, (
        f"only {len(broke)} of the cases misbehave unescaped, so this set is "
        f"too gentle to prove the escaping does anything: {broke}"
    )
    assert "Bob's renders" in broke, broke


@pytest.mark.parametrize("partial", ["collections.html", "modals/remix_modal.html"])
def test_the_partials_use_the_helper_they_do_not_define(partial):
    """These two are included into index.html and share its scope, so they
    call its jsInAttr rather than carrying one of their own.

    They did carry one, briefly -- I added it to all three files, which put
    the same function in a single scope three times over with two copies
    dead. The round trip above therefore runs the two real definitions;
    what has to hold here is that the partials still reach one."""
    source = (_TEMPLATES / partial).read_text(encoding="utf-8")

    assert "jsInAttr(" in source, f"{partial} no longer escapes names at all"
    assert "function jsInAttr" not in source, (
        f"{partial} defines its own jsInAttr again; index.html already "
        f"declares one in the same scope and the later declaration wins, so "
        f"one of the two would be dead"
    )


def test_an_html_entity_escaper_is_never_used_inside_a_javascript_string():
    """The mistake that made the Remix library look protected.

    escapeHTML turns ' into &#39;, which is right for text between tags and
    useless here: the browser decodes the attribute BEFORE JavaScript reads
    it, so the apostrophe comes back and ends the string. Proven with node
    -- "Bob's workflow" became Bob&#39;s workflow, decoded to Bob's
    workflow, and threw Unexpected identifier 's'.

    Deferred in the previous commit on the grounds that those sites "at
    least pass through an escaper". They did; it was the wrong one."""
    offenders = []
    for template in _TEMPLATES.rglob("*.html"):
        for line in template.read_text(encoding="utf-8").splitlines():
            if re.search(r"'\$\{escapeHTML\(|'\$\{escapeHtml\(", line):
                offenders.append(f"{template.name}: {line.strip()[:70]}")

    assert not offenders, (
        f"{len(offenders)} site(s) put an HTML-entity-escaped value inside a "
        f"JavaScript string; use jsInAttr: {offenders[:3]}"
    )


def test_the_folder_menu_uses_the_escaped_name():
    """The site the bug was actually in. Reading the source here rather
    than the browser, because the menu is built from data the server never
    renders, so there is no HTML to assert on from a test."""
    source = (_TEMPLATES / "index.html").read_text(encoding="utf-8")

    handlers = re.findall(r'on\w+="[^"]*\$\{folderDisplayName\}[^"]*"', source)

    assert not handlers, f"{len(handlers)} handler(s) still interpolate the raw folder name: {handlers[:2]}"
    assert "jsInAttr(folderDisplayName)" in source, "the escaped name is not being produced at all"
