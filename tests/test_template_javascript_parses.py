r"""The JavaScript the server sends has to parse.

The gallery ships close to a megabyte of inline JavaScript inside its
templates. A single syntax error anywhere in a <script> block stops the
whole block, so one bad character can take out the entire interface --
and nothing in the suite would notice, because every existing test speaks
to the server and never to a browser.

This is not hypothetical. The commit that added this file first wrote

    .replace(/\\/g, '\\\\')

into two templates as

    .replace(/\/g, '\\')

-- an unterminated regular expression, produced by a shell heredoc eating
the backslashes. Both templates would have shipped with their main script
block dead. The escaping tests caught it only because they execute the
function with node; a test that read the source would have passed.

So the page is rendered and every executable inline block is handed to
node --check. Blocks with a src, and blocks whose type is not JavaScript
(the changelog is carried as text/markdown in a script tag), are not run
by browsers and are not checked here either.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

_TAG = re.compile(r"<script([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)
_TYPE = re.compile(r"""type\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_SRC = re.compile(r"\ssrc\s*=", re.IGNORECASE)
# What a browser will actually execute.
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}


def _node():
    found = shutil.which("node")
    if found is None:
        pytest.skip("node is not on PATH; the shipped JavaScript cannot be "
                    "parsed here")
    return found


def _executable_blocks(html: str):
    blocks = []
    for attrs, body in _TAG.findall(html):
        if _SRC.search(attrs) or not body.strip():
            continue
        found = _TYPE.search(attrs)
        kind = (found.group(1).strip().lower() if found else "")
        if kind in _JS_TYPES:
            blocks.append(body)
    return blocks


def _syntax_errors(blocks, tmp_path):
    errors = []
    for index, body in enumerate(blocks):
        path = tmp_path / f"block_{index}.js"
        path.write_text(body, encoding="utf-8")
        done = subprocess.run([_node(), "--check", str(path)],
                              capture_output=True, text=True, timeout=300)
        if done.returncode != 0:
            errors.append((index, done.stderr.strip().splitlines()[:6]))
    return errors


@pytest.fixture()
def client(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


def test_the_checker_reports_a_broken_file(tmp_path):
    """Control. Everything below is an absence of errors, which is also
    what a checker that never runs produces."""
    errors = _syntax_errors(["function ( {", "let x = 1;"], tmp_path)

    assert len(errors) == 1, errors
    assert errors[0][0] == 0, errors


def test_the_management_page_ships_parseable_javascript(client, tmp_path):
    page = client.get("/galleryout/view/_root_").get_data(as_text=True)
    blocks = _executable_blocks(page)

    assert len(blocks) > 10, (
        f"only {len(blocks)} executable script blocks were found in "
        f"{len(page)} bytes; the extraction is not reaching the page")
    assert sum(len(b) for b in blocks) > 100_000, "far less script than expected"

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


def test_the_exhibition_page_ships_parseable_javascript(smartgallery_app,
                                                        monkeypatch, tmp_path):
    """The visitor's page is a different template and was edited by the same
    commit, so it is checked on its own rather than assumed to follow."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "GUEST"
        session["full_name"] = "A Visitor"

    page = client.get("/galleryout/view/_root_").get_data(as_text=True)
    blocks = _executable_blocks(page)

    assert len(blocks) > 3, (
        f"{len(blocks)} blocks found; this may be the login screen rather "
        f"than the exhibition itself")

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


def test_a_script_that_is_not_javascript_is_left_alone():
    """The changelog travels inside a script tag as text/markdown. Browsers
    do not run it and neither does this, or every release note would have
    to be valid JavaScript."""
    html = ('<script id="changelog-data" type="text/markdown">\n'
            '# Changelog\n* a bullet point\n</script>'
            '<script>let ok = 1;</script>'
            '<script src="/x.js"></script>')

    blocks = _executable_blocks(html)

    assert blocks == ["let ok = 1;"], blocks
