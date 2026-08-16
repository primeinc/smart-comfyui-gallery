"""A filename is not markup, and the grid was treating it as both.

The server delivers file data safely -- the page carries it as JSON, so a
file called `<img src=x onerror=alert(1)>.png` arrives as
"\\u003cimg src=x onerror=alert(1)\\u003e.png" and nothing is running yet.
Then the browser builds the grid from it:

    <p title="${file.name}"><strong>${file.name}</strong></p>

and assigns that to innerHTML, at which point the name is parsed as HTML.
The check that missed this looked at the served page, where the raw markup
never appears; the execution happens afterwards, in the browser.

`<`, `>` and `"` are all legal in filenames on Linux, which is what the
Docker image runs, and the gallery indexes whatever is dropped into the
ComfyUI output folder -- a shared machine, a network share, a custom node
that names its own output. The code would run in the owner's browser,
which is the session with every privilege.

Collection names had the same shape, and one line escaped the text while
leaving the title attribute beside it raw, which is what an oversight
looks like rather than a decision.

escapeHTML already existed in index.html and escapes & < > " ' -- correct
for both the text and the attribute. These tests run it under node against
the payloads, because that is the only way to know it does.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest

_TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "templates"

_PAYLOADS = [
    '<img src=x onerror=alert(1)>.png',
    '"><script>alert(1)</script>.png',
    "' onmouseover='alert(1)",
    'a & b.png',
    'plain.png',
    'Ordner-Größe.png',
]


def _node():
    found = shutil.which("node")
    if found is None:
        pytest.skip("node is not on PATH; the shipped escaper cannot be run")
    return found


def _escape_html_source():
    source = (_TEMPLATES / "index.html").read_text(encoding="utf-8")
    match = re.search(r"function escapeHTML\(value\) \{.*?\n {8}\}",
                      source, re.DOTALL)
    assert match, "index.html no longer defines escapeHTML"
    body = match.group(0)
    # Counting `case '` finds only four: the fifth is written `case "'"`,
    # because the value being matched is an apostrophe. Check for the entity
    # each branch produces instead, which is what completeness means here.
    for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert entity in body, (
            f"escapeHTML has no branch producing {entity}; either it was cut "
            f"short by the extraction or it no longer escapes that:\n{body}")
    return body


def _run(values):
    script = _escape_html_source() + r"""
const values = JSON.parse(process.argv[1]);
console.log(JSON.stringify(values.map(v => escapeHTML(v))));
"""
    done = subprocess.run([_node(), "-e", script, json.dumps(values)],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_escaper_removes_everything_that_could_be_markup():
    """Run for real, because an escaper that misses one character is worth
    the same as no escaper at all."""
    escaped = _run(_PAYLOADS)

    for original, result in zip(_PAYLOADS, escaped):
        for char in "<>\"'":
            assert char not in result, (original, result, char)
    # and it must still be the name, not a blank
    assert "Ordner-Größe.png" in escaped[-1], escaped[-1]
    assert escaped[-2] == "plain.png", escaped[-2]


def test_the_escaper_would_notice_a_payload_left_alone():
    """Control. The assertions above are about characters being absent,
    which is also what happens if the payloads are harmless to begin with."""
    for payload in _PAYLOADS[:3]:
        assert any(c in payload for c in "<>\"'"), payload


def test_the_grid_escapes_the_filename_before_it_becomes_html():
    """The sites the bug was in. Read as source, because the grid is built
    in the browser from JSON and never appears in anything the server
    sends -- which is exactly why nothing caught it."""
    offenders = []
    pattern = re.compile(
        r'(?:title|alt)="\$\{(?:file|item|c|w|folder)\.(?:name|display_name)\}"'
        r'|<(?:span|strong|p)[^>]*>\$\{(?:file|item|c)\.name\}')

    for template in sorted(_TEMPLATES.rglob("*.html")):
        for number, line in enumerate(
                template.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{template.name}:{number}")

    assert not offenders, (
        f"{len(offenders)} place(s) put a name into markup unescaped; wrap "
        f"it in escapeHTML: {offenders[:4]}")


def test_the_rule_is_looking_at_the_right_files():
    """Control for the sweep: a glob that found nothing, or a pattern that
    matches nothing anywhere, would report a clean repository."""
    templates = list(_TEMPLATES.rglob("*.html"))
    assert len(templates) > 10, len(templates)

    index = (_TEMPLATES / "index.html").read_text(encoding="utf-8")
    assert "escapeHTML(file.name)" in index, (
        "the grid no longer escapes the filename at all")
    assert index.count("escapeHTML(file.name)") >= 6, (
        f"only {index.count('escapeHTML(file.name)')} escaped uses; the "
        f"grid has several and they should all be covered")
