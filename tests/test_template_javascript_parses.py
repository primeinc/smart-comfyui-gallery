r"""The JavaScript the server sends has to parse.

The gallery ships close to a megabyte of inline JavaScript inside its
templates. A single syntax error anywhere in a <script> block stops the
whole block, so one bad character can take out the entire interface --
and nothing else in the suite would notice, because every other test
speaks to the server and never to a browser.

This is not hypothetical. The commit that added this file first wrote

    .replace(/\\/g, '\\\\')

into two templates as

    .replace(/\/g, '\\')

-- an unterminated regular expression, produced by a shell heredoc eating
the backslashes. Both templates would have shipped with their main script
block dead. The escaping tests caught it only because they execute the
function with node; a test that read the source would have passed.

So the pages are rendered and every executable inline block is parsed by
node. Blocks with a src, and blocks whose type is not JavaScript (the
changelog is carried as text/markdown in a script tag), are not run by
browsers and are not checked here either.

Every block used to get its own `node --check`: one process per script
block, around sixty process starts to answer one question. They are
parsed in a single node run per page instead -- node parses each block
with `new vm.Script`, the same parse `--check` performs, and reports all
the results at once, so a failure still names the block that broke.
Parsing JavaScript is the one thing here that genuinely needs node;
batching is what removes the cost.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from jinja2 import ChainableUndefined

from node_runner import run_node

pytestmark = pytest.mark.spawns  # every check here runs another program

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_TAG = re.compile(r"<script([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE)
_TYPE = re.compile(r"""type\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_SRC = re.compile(r"\ssrc\s*=", re.IGNORECASE)
# What a browser will actually execute.
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}

# Parses each file the way `node --check` does, and reports on all of them
# rather than exiting at the first failure.
_PARSE_EACH = """
const vm = require('vm');
const fs = require('fs');
const files = JSON.parse(process.argv[1]);
process.stdout.write(JSON.stringify(files.map(function (file) {
    try {
        new vm.Script(fs.readFileSync(file, 'utf8'), {filename: file});
        return null;
    } catch (error) {
        return String(error.message);
    }
})));
"""


def _executable_blocks(html: str):
    blocks = []
    for attrs, body in _TAG.findall(html):
        if _SRC.search(attrs) or not body.strip():
            continue
        found = _TYPE.search(attrs)
        kind = found.group(1).strip().lower() if found else ""
        if kind in _JS_TYPES:
            blocks.append(body)
    return blocks


def _syntax_errors(blocks, tmp_path):
    """[(index, message)] for the blocks that do not parse. One node run."""
    if not blocks:
        return []

    paths = []
    for index, body in enumerate(blocks):
        path = tmp_path / f"block_{index}.js"
        path.write_text(body, encoding="utf-8")
        paths.append(str(path))

    results = run_node(_PARSE_EACH, paths)
    assert len(results) == len(blocks), f"asked about {len(blocks)} blocks, heard about {len(results)}"

    return [(index, message) for index, message in enumerate(results) if message is not None]


@pytest.fixture
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


def test_the_checker_reports_every_broken_block_not_just_the_first(tmp_path):
    """Batching introduces a failure the per-block runs could not have: a
    parser that stopped at the first bad block would hide the rest."""
    errors = _syntax_errors(["function ( {", "let ok = 1;", "let = ;"], tmp_path)

    assert [index for index, _message in errors] == [0, 2], errors


def test_the_management_page_ships_parseable_javascript(client, tmp_path):
    page = client.get("/galleryout/view/_root_").get_data(as_text=True)
    blocks = _executable_blocks(page)

    assert len(blocks) > 10, (
        f"only {len(blocks)} executable script blocks were found in "
        f"{len(page)} bytes; the extraction is not reaching the page"
    )
    assert sum(len(b) for b in blocks) > 100_000, "far less script than expected"

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


def test_the_exhibition_page_ships_parseable_javascript(smartgallery_app, monkeypatch, tmp_path):
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
        f"{len(blocks)} blocks found; this may be the login screen rather than the exhibition itself"
    )

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


def test_the_ai_dashboard_ships_parseable_javascript(client, tmp_path):
    """The third page, and the one this check did not reach for two
    commits. It carries the newest code in the project, which is where a
    syntax error is most likely and where nothing would have reported it."""
    response = client.get("/galleryout/aidam")
    assert response.status_code == 200, response.status_code

    blocks = _executable_blocks(response.get_data(as_text=True))
    assert blocks, "no script found on the dashboard"

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


# The suite runs with ENABLE_AI_DAM=false, so the AI panel never appears in
# any page rendered above. It carries no Jinja, so it is read from disk.
_RAW_TEMPLATES = ["templates/modals/aidam_panel.html"]


@pytest.mark.parametrize("relpath", _RAW_TEMPLATES)
def test_a_template_the_rendered_pages_never_reach_still_parses(relpath, tmp_path):
    html = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert "{{" not in html, (
        f"{relpath} grew Jinja expressions, so it can no longer be read "
        f"from disk; move it to one of the rendered-page checks above"
    )

    blocks = _executable_blocks(html)
    assert blocks, f"{relpath} has no executable script blocks -- probe broken?"

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


# Exhibition mode (`python smartgallery.py --exhibition`) swaps in its own
# top-level templates. The mode is read from CLI args at import time, so the
# test client cannot reach the login screen -- it is rendered directly.
# Undefined values interpolate as JS `null`, so a bare `{{ var }}` inside a
# script stays syntactically valid; `| tojson` values need real data and are
# listed explicitly, so a NEW tojson variable fails loudly with instructions
# rather than quietly dropping coverage.
_EXHIBITION_TEMPLATES = ["exhibition.html", "exhibition_login.html"]
_EXHIBITION_TOJSON_CONTEXT = {"ffmpeg_available": False, "available_extensions": []}


def _render_standalone(app, template_name: str) -> str:

    class _JsNull(ChainableUndefined):
        """Renders as JS `null`, so an unsupplied variable interpolated into
        a script does not produce `const x = ;`. Overriding __str__ is
        enough: ChainableUndefined.__html__ returns str(self)
        (jinja2/runtime.py:988)."""

        def __str__(self):
            return "null"

    env = app.jinja_env.overlay(undefined=_JsNull)
    with app.test_request_context("/galleryout/"):
        return env.get_template(template_name).render(**_EXHIBITION_TOJSON_CONTEXT)


@pytest.mark.parametrize("template_name", _EXHIBITION_TEMPLATES)
def test_the_exhibition_templates_parse_on_their_own(smartgallery_app, template_name, tmp_path):
    try:
        html = _render_standalone(smartgallery_app.app, template_name)
    except TypeError as exc:  # a `| tojson` value that Undefined cannot satisfy
        pytest.fail(
            f"{template_name} could not render for the JS check ({exc}). A new "
            f"`| tojson` variable likely needs a value in "
            f"_EXHIBITION_TOJSON_CONTEXT."
        )

    blocks = _executable_blocks(html)
    assert blocks, f"{template_name} has no executable script blocks -- probe broken?"

    errors = _syntax_errors(blocks, tmp_path)

    assert not errors, errors


def test_a_script_that_is_not_javascript_is_left_alone():
    """The changelog travels inside a script tag as text/markdown. Browsers
    do not run it and neither does this, or every release note would have
    to be valid JavaScript."""
    html = (
        '<script id="changelog-data" type="text/markdown">\n'
        "# Changelog\n* a bullet point\n</script>"
        "<script>let ok = 1;</script>"
        '<script src="/x.js"></script>'
    )

    blocks = _executable_blocks(html)

    assert blocks == ["let ok = 1;"], blocks
