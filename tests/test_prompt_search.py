"""Searching the folder view by prompt.

This answered 500 for everyone, in every language, for any prompt search
at all. `import re` sat inside gallery_view's OmniQuery branch, which made
`re` local to the whole function; that branch runs only for an OmniQuery
request, so an ordinary prompt search reached the typed-operator
`re.match` further down with the name unbound.

Nothing about it looks broken from the outside: the same search works in
the collection view, which has no local import, so the feature appears to
work depending on which page you are on.

These tests drive the folder view, since that is the one that crashed.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image

from inline_executor import InlineExecutor

_PREFIX = "psearch_"


@pytest.fixture
def indexed(smartgallery_app, monkeypatch):
    """Two files: one with a prompt and generation parameters, one without."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    names = [f"{_PREFIX}wanted.png", f"{_PREFIX}other.png"]
    for name in names:
        Image.new("RGB", (16, 16), (70, 20, 20)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {r[0]: r[1] for r in conn.execute(f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()}
        conn.execute(
            "UPDATE files SET workflow_prompt = ? WHERE id = ?", ("a neon city at dusk", ids[f"{_PREFIX}wanted.png"])
        )
        conn.execute(
            "INSERT OR REPLACE INTO generation_params "
            "(file_id, tool, detection, model, seed, parsed_at) "
            "VALUES (?, 'comfyui', 'workflow', 'dreamshaper_v8.safetensors', 12345, 1.0)",
            (ids[f"{_PREFIX}wanted.png"],),
        )
        conn.commit()
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    for name in names:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _view(smartgallery_app, query):
    return smartgallery_app.app.test_client().get(f"/galleryout/view/_root_?{query}")


def _matched(resp, ids):
    html = resp.get_data(as_text=True)
    return sorted(name for name, fid in ids.items() if fid and fid in html)


def test_the_fixture_is_visible_without_a_search(smartgallery_app, indexed):
    """Control: both files are on the page when nothing is filtering."""
    assert _matched(_view(smartgallery_app, ""), indexed) == sorted(indexed)


def test_a_prompt_search_does_not_crash(smartgallery_app, indexed):
    """The regression, in its plainest form."""
    resp = _view(smartgallery_app, "workflow_prompt=neon")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]


def test_a_prompt_search_finds_the_file(smartgallery_app, indexed):
    resp = _view(smartgallery_app, "workflow_prompt=neon")

    assert _matched(resp, indexed) == [f"{_PREFIX}wanted.png"]


def test_a_prompt_search_excludes_the_others(smartgallery_app, indexed):
    resp = _view(smartgallery_app, "workflow_prompt=nothingmatchesthis")

    assert resp.status_code == 200
    assert _matched(resp, indexed) == []


@pytest.mark.parametrize("term", ["model:dreamshaper", "seed:12345"])
def test_the_typed_operators_work(smartgallery_app, indexed, term):
    """These are the branch that reads `re` -- the exact line that raised."""
    resp = _view(smartgallery_app, f"workflow_prompt={term}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert _matched(resp, indexed) == [f"{_PREFIX}wanted.png"]


def test_a_negated_term_still_negates(smartgallery_app, indexed):
    resp = _view(smartgallery_app, "workflow_prompt=!neon")

    assert resp.status_code == 200
    assert f"{_PREFIX}wanted.png" not in _matched(resp, indexed)
