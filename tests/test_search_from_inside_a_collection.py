"""A search run from inside an album has to show its results.

The search palette navigates to whichever key the page is showing, and a
collection's key is `collection_<id>`. gallery_view recognises that and
forwards to the collection view, keeping the query string -- which is
careful, and was the problem: the collection view reads neither
omniquery_id nor ai_session_id.

So a search started from inside an album arrived somewhere that could not
show it. The album came back exactly as it was, with every file still in
it, no results, and nothing on screen to say a search had happened. It
looked like the search button did nothing.

Compared as sets of what each view reads, the collection view takes a
strict subset of the folder view's parameters: omniquery_id, ai_session_id
and scope are the three it never looks at, and the first two are how a
search is carried.

A search answers with a list of files from across the library, which is
not something an album can display, so the redirect now sends a search to
the folder view at the root -- the same place the palette reaches from an
ordinary folder, and the only code that knows how to render one. Anything
without a search still goes to the album, unchanged.
"""

from __future__ import annotations

import ast
import contextlib
import os
import uuid

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "collsearch_"


@pytest.fixture
def album(smartgallery_app, monkeypatch):
    """Three files, an album holding one of them, and a stored search that
    answers with a different one."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    names = [f"{_PREFIX}in_album.png", f"{_PREFIX}found_by_search.png", f"{_PREFIX}neither.png"]
    for name in names:
        Image.new("RGB", (8, 8), (3, 3, 3)).save(os.path.join(base, name))

    session_id = f"test-{uuid.uuid4()}"
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.execute("DELETE FROM collections WHERE name = 'Search Test Album'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {
            r["name"]: r["id"]
            for r in conn.execute("SELECT name, id FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)).fetchall()
        }

        conn.execute("INSERT INTO collections (name, type) VALUES (?, ?)", ("Search Test Album", "user_album"))
        coll_id = conn.execute("SELECT id FROM collections WHERE name = ?", ("Search Test Album",)).fetchone()[0]
        conn.execute(
            "INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)",
            (coll_id, ids[f"{_PREFIX}in_album.png"]),
        )

        conn.execute(
            "INSERT INTO omniquery_sessions (session_id, raw_sql) VALUES (?, ?)", (session_id, "SELECT id FROM files")
        )
        conn.execute(
            "INSERT INTO omniquery_results (session_id, file_id) VALUES (?, ?)",
            (session_id, ids[f"{_PREFIX}found_by_search.png"]),
        )
        conn.commit()
    finally:
        conn.close()

    yield ids, coll_id, session_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.execute("DELETE FROM collections WHERE name = 'Search Test Album'")
        conn.execute("DELETE FROM omniquery_results WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM omniquery_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    for name in names:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _shown(smartgallery_app, ids, url):
    """Which of the fixture's files a page ends up displaying."""
    page = smartgallery_app.app.test_client().get(url, follow_redirects=True)
    body = page.get_data(as_text=True)
    return page.status_code, {name for name, fid in ids.items() if fid in body}


def test_the_album_shows_its_own_file(smartgallery_app, album):
    """Control. Everything below is about what a search changes, so the
    album has to be right without one."""
    ids, coll_id, _session = album

    status, shown = _shown(smartgallery_app, ids, f"/galleryout/view/collection_{coll_id}")

    assert status == 200
    assert shown == {f"{_PREFIX}in_album.png"}, shown


def test_a_search_from_inside_the_album_shows_its_results(smartgallery_app, album):
    """The bug: the album came back untouched and the search vanished."""
    ids, coll_id, session_id = album

    status, shown = _shown(smartgallery_app, ids, f"/galleryout/view/collection_{coll_id}?omniquery_id={session_id}")

    assert status == 200
    assert f"{_PREFIX}found_by_search.png" in shown, f"the search result is not on the page; shown: {shown}"
    assert f"{_PREFIX}in_album.png" not in shown, f"the album's own contents came back instead of the search: {shown}"


def test_the_same_holds_for_the_ai_search(smartgallery_app, album):
    """ai_session_id travels the same road and was dropped the same way.

    Asserting on which files come back would be asserting a guess: an
    unknown session filters nothing, so the root gallery answers with
    everything -- including the album's own file, which is also in the
    root. What has to hold is where it arrives, so that is what is
    checked."""
    _ids, coll_id, _session = album
    client = smartgallery_app.app.test_client()

    response = client.get(f"/galleryout/view/collection_{coll_id}?ai_session_id=no-such-session")

    assert response.status_code == 302
    location = response.headers.get("Location", "")
    assert "/galleryout/collection/" not in location, (
        f"the AI search was sent to the album, which cannot show one: {location}"
    )
    assert "ai_session_id=no-such-session" in location, f"the search was dropped on the way: {location}"


def test_going_to_the_album_without_a_search_is_unchanged(smartgallery_app, album):
    """The guard against over-reach: only a search is diverted. Ordinary
    navigation, and the filters that ride along with it, still reach the
    album."""
    ids, coll_id, _session = album

    status, shown = _shown(smartgallery_app, ids, f"/galleryout/view/collection_{coll_id}?sort_by=name&sort_order=ASC")

    assert status == 200
    assert shown == {f"{_PREFIX}in_album.png"}, shown


def test_the_collection_view_still_ignores_these_parameters(gallery_tree, smartgallery_app):
    """Why the fix is a redirect rather than new rendering: the collection
    view has no code for either parameter, so sending a search to it can
    only ever be silent. Stated here so that if someone teaches it to show
    one, this fails and the redirect can go."""

    tree = gallery_tree
    view = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "collection_view")
    read = {
        call.args[0].value
        for call in ast.walk(view)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in ("get", "getlist")
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "args"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    }

    assert read, "no request parameters found; the scan is not reaching the view"
    assert "sort_by" in read, read
    assert "omniquery_id" not in read and "ai_session_id" not in read, (
        "the collection view now reads a search parameter, so the redirect "
        "in gallery_view can send searches here instead"
    )
