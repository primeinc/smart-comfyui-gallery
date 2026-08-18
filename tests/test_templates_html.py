"""HTML integrity of the templates: no element may carry the same
attribute twice.

A duplicate attribute is silently destructive, not a parse error: HTML5
parsers keep the FIRST occurrence and discard every later one, so a
second `class` (or `style`, or `onclick`) simply never applies and the
element quietly loses behaviour on some code path. Shipped example: the
mobile OmniQuery button carried `class` twice, so its active-state
highlight never rendered on mobile while the desktop copy worked.

Both the rendered pages and every raw template are checked -- rendering
only exercises the Jinja branches that particular render takes, so the
raw pass covers the ones it doesn't.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser

import pytest

# Jinja is stripped before parsing so the remainder is HTML. Only
# attribute NAMES are counted, so losing conditional value fragments
# (`{% if x %}active{% endif %}` inside a class) is irrelevant.
_JINJA_RE = re.compile(r"\{%.*?%\}|\{\{.*?\}\}|\{#.*?#\}", re.DOTALL)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _DupAttrFinder(HTMLParser):
    """Collects (line, tag, duplicated-attribute-names, snippet)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hits: list = []

    def handle_starttag(self, tag, attrs):
        seen, dups = set(), set()
        for name, _value in attrs:
            if name in seen:
                dups.add(name)
            seen.add(name)
        if dups:
            self.hits.append((self.getpos()[0], tag, sorted(dups),
                              (self.get_starttag_text() or "")[:150]))


def _find_duplicates(html: str, strip_jinja: bool = False):
    parser = _DupAttrFinder()
    parser.feed(_JINJA_RE.sub(" ", html) if strip_jinja else html)
    return parser.hits


def _format(hits) -> str:
    return "\n".join(f"  line {ln} <{tag}> duplicated {dups}: {snip}"
                     for ln, tag, dups, snip in hits)


def test_detector_catches_a_planted_duplicate():
    """Positive control: an empty result from the scans below means
    'clean' only because this proves the detector detects."""
    bug = ('<button class="glass-btn mobile-only" type="button" onclick="x()" '
           'class="glass-btn {% if q %}active{% endif %}" title="t">go</button>')
    hits = _find_duplicates(bug, strip_jinja=True)
    assert [h[2] for h in hits] == [["class"]]

    fixed = ('<button type="button" onclick="x()" '
             'class="glass-btn mobile-only {% if q %}active{% endif %}" title="t">go</button>')
    assert _find_duplicates(fixed, strip_jinja=True) == []

    # Not class-specific: any repeated attribute is caught.
    assert _find_duplicates('<div id="a" style="x" id="b" style="y"></div>')[0][2] == ["id", "style"]


def _template_files():
    for dirpath, _dirs, files in os.walk(os.path.join(_ROOT, "templates")):
        for name in sorted(files):
            if name.endswith(".html"):
                path = os.path.join(dirpath, name)
                yield os.path.relpath(path, _ROOT).replace("\\", "/")


@pytest.mark.parametrize("relpath", sorted(_template_files()))
def test_template_has_no_duplicate_attributes(relpath):
    with open(os.path.join(_ROOT, relpath), encoding="utf-8") as fh:
        html = fh.read()
    hits = _find_duplicates(html, strip_jinja=True)
    assert not hits, f"{relpath} has duplicate attributes:\n{_format(hits)}"


@pytest.mark.parametrize("url", ["/galleryout/", "/galleryout/aidam"])
def test_rendered_page_has_no_duplicate_attributes(smartgallery_app, url):
    client = smartgallery_app.app.test_client()
    resp = client.get(url, follow_redirects=True)
    assert resp.status_code == 200
    hits = _find_duplicates(resp.get_data(as_text=True))
    assert not hits, f"{url} rendered duplicate attributes:\n{_format(hits)}"
