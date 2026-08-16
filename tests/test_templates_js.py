"""Every inline script on the rendered pages must be valid JavaScript.

The app's UI logic lives in Jinja-rendered inline scripts with no build
step and no browser test harness -- a template edit can ship a syntax
error that kills every feature on the page. This renders the real pages
through the Flask test client and runs `node --check` on each executable
script block. Skips (with a reason) when node is not installed.

Non-executable script tags (type="text/markdown" etc.) are data blocks
and are not checked.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import pytest

_NODE = shutil.which("node")

_SCRIPT_RE = re.compile(r"<script([^>]*)>(.*?)</script>", re.S | re.I)
_SRC_RE = re.compile(r"\bsrc\s*=", re.I)
_TYPE_RE = re.compile(r"""\btype\s*=\s*["']([^"']+)["']""", re.I)
_JS_TYPES = {"", "text/javascript", "application/javascript", "module"}


def _executable_blocks(html: str):
    for attrs, body in _SCRIPT_RE.findall(html):
        if _SRC_RE.search(attrs) or not body.strip():
            continue
        m = _TYPE_RE.search(attrs)
        if m and m.group(1).strip().lower() not in _JS_TYPES:
            continue
        yield body


def _node_check(body: str) -> str | None:
    """Return node's stderr on syntax error, None when the block parses."""
    fd, path = tempfile.mkstemp(suffix=".js")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        r = subprocess.run([_NODE, "--check", path],
                           capture_output=True, text=True, timeout=30)
        return None if r.returncode == 0 else r.stderr
    finally:
        os.unlink(path)


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
@pytest.mark.parametrize("url", ["/galleryout/", "/galleryout/aidam"])
def test_rendered_page_scripts_parse(smartgallery_app, url):
    client = smartgallery_app.app.test_client()
    resp = client.get(url, follow_redirects=True)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    blocks = list(_executable_blocks(html))
    assert blocks, f"{url} rendered no executable script blocks -- probe broken?"
    errors = []
    for i, body in enumerate(blocks):
        err = _node_check(body)
        if err:
            errors.append(f"block {i}: {err[:500]}")
    assert not errors, f"{url}: {len(errors)} script block(s) fail to parse:\n" + "\n".join(errors)


# Templates whose scripts contain no Jinja can be checked straight from
# disk -- needed because the suite renders with ENABLE_AI_DAM=false, so
# the AI panel never appears in the rendered pages above.
_RAW_TEMPLATES = ["templates/modals/aidam_panel.html"]


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
@pytest.mark.parametrize("relpath", _RAW_TEMPLATES)
def test_raw_template_scripts_parse(relpath):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, relpath), encoding="utf-8") as fh:
        html = fh.read()
    assert "{{" not in html, (
        f"{relpath} grew Jinja expressions; move it to the rendered-page test")
    blocks = list(_executable_blocks(html))
    assert blocks, f"{relpath} has no executable script blocks -- probe broken?"
    errors = [err for body in blocks if (err := _node_check(body))]
    assert not errors, f"{relpath}: script fails to parse:\n" + "\n".join(e[:500] for e in errors)
