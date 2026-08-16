"""The same filter must mean the same thing in a folder and in an album.

The two views were written separately and each carries its own copy of the
filtering. Copies drift, and this pair has now drifted twice: once into a
500 from any prompt search in a folder, and once into this.

Only the folder view learned the typed prompt operators. Searching
`seed:5` there found the picture; the same search inside an album returned
nothing at all -- which reads as "there are none of those in here" rather
than as a missing feature, so there was nothing to report and nothing to
notice.

Found by putting every file into one album and running each filter against
both views. Everything the album holds is everything the folder holds, so
any difference in the answers is a difference in the code. Seventeen of the
eighteen agreed; `workflow_prompt=seed:5` was the one that did not.

Both now build their conditions with one shared function, so the next
operator added arrives in both places or neither.

One thing this comparison has to be careful about: a .txt or .md file in a
collection is a NOTE, and the album page lists it whether or not it matches
the filter. That is the notes feature working, not the filter leaking, so
the fixture uses only pictures -- an earlier run that included a note
reported all eighteen filters as disagreeing.
"""

from __future__ import annotations

import concurrent.futures
import os

import pytest
from PIL import Image, PngImagePlugin

_PREFIX = "bothviews_"
_INFOTEXT = ("a red cat\nNegative prompt: blur\n"
             "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 5, "
             "Size: 8x8, Model: dreamshaper")

_FILTERS = [
    "sort_by=name&sort_order=ASC",
    "sort_by=rating&sort_order=DESC",
    "sort_by=size&sort_order=ASC",
    "search=alpha",
    "search=ALPHA",
    "extension=.png",
    "favorites=true",
    "prefix=" + _PREFIX + "alpha",
    "rating_range=5-5",
    "comment_search=lovely",
    "workflow_prompt=cat",
    "workflow_prompt=seed:5",
    "workflow_prompt=model:dreamshaper",
    "workflow_prompt=steps:20",
    "workflow_prompt=sampler:Euler",
    "workflow_prompt=!cat",
    "workflow_files=dreamshaper",
    "no_workflow=true",
    "no_ai_caption=true",
    "rated_by=admin",
]


class _InlineExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture(scope="module")
def both_views(smartgallery_app):
    """Three pictures, and one album holding all three."""
    original = (smartgallery_app.concurrent.futures.ProcessPoolExecutor,
                smartgallery_app.FORCE_LOGIN, smartgallery_app.IS_EXHIBITION_MODE)
    smartgallery_app.concurrent.futures.ProcessPoolExecutor = _InlineExecutor
    smartgallery_app.FORCE_LOGIN = False
    smartgallery_app.IS_EXHIBITION_MODE = False

    base = smartgallery_app.BASE_OUTPUT_PATH
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", _INFOTEXT)
    names = [f"{_PREFIX}alpha_one.png", f"{_PREFIX}Beta_two.png",
             f"{_PREFIX}gamma_three.png"]
    Image.new("RGB", (8, 8), (1, 1, 1)).save(os.path.join(base, names[0]), pnginfo=info)
    for name in names[1:]:
        Image.new("RGB", (8, 8), (2, 2, 2)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Both Views'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {r["name"]: r["id"] for r in conn.execute(
            f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()}

        conn.execute("INSERT INTO collections (name, type) VALUES (?, ?)",
                     ("Both Views", "user_album"))
        coll_id = conn.execute("SELECT id FROM collections WHERE name = ?",
                               ("Both Views",)).fetchone()[0]
        for file_id in ids.values():
            conn.execute("INSERT INTO collection_files (collection_id, file_id) "
                         "VALUES (?, ?)", (coll_id, file_id))
        conn.execute("UPDATE files SET is_favorite = 1 WHERE name = ?", (names[0],))
        conn.execute("INSERT INTO file_ratings (file_id, client_uuid, rating) "
                     "VALUES (?, 'admin', 5)", (ids[names[1]],))
        conn.execute("INSERT INTO file_comments (file_id, client_uuid, author_name, "
                     "comment_text) VALUES (?, 'admin', 'me', 'lovely picture')",
                     (ids[names[2]],))
        conn.commit()
    finally:
        conn.close()

    yield ids, coll_id

    smartgallery_app.concurrent.futures.ProcessPoolExecutor = original[0]
    smartgallery_app.FORCE_LOGIN, smartgallery_app.IS_EXHIBITION_MODE = original[1:]
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Both Views'")
        conn.commit()
    finally:
        conn.close()
    for name in names:
        try:
            os.remove(os.path.join(base, name))
        except OSError:
            pass


def _shown(smartgallery_app, ids, url):
    response = smartgallery_app.app.test_client().get(url, follow_redirects=True)
    body = response.get_data(as_text=True)
    return response.status_code, frozenset(
        name for name, file_id in ids.items() if file_id in body)


def test_the_album_holds_everything_the_folder_does(smartgallery_app, both_views):
    """Control. The comparison only means something while the two views
    are looking at the same files."""
    ids, coll_id = both_views

    folder = _shown(smartgallery_app, ids, "/galleryout/view/_root_")
    album = _shown(smartgallery_app, ids, f"/galleryout/collection/{coll_id}")

    assert folder == (200, frozenset(ids)), folder
    assert album == (200, frozenset(ids)), album


@pytest.mark.parametrize("query", _FILTERS)
def test_a_filter_answers_the_same_in_both(smartgallery_app, both_views, query):
    """The bug: seed:5 found the picture in the folder and nothing in the
    album."""
    ids, coll_id = both_views

    folder_status, folder_files = _shown(
        smartgallery_app, ids, f"/galleryout/view/_root_?{query}")
    album_status, album_files = _shown(
        smartgallery_app, ids, f"/galleryout/collection/{coll_id}?{query}")

    assert folder_status == album_status == 200, (folder_status, album_status)
    assert folder_files == album_files, (
        f"{query} answers differently: the folder shows "
        f"{sorted(folder_files)}, the album shows {sorted(album_files)}")


def test_the_filters_are_actually_filtering(smartgallery_app, both_views):
    """The control that makes the sweep worth running. If every filter
    returned all three files, the two views would agree about nothing in
    particular and the comparison would pass for ever."""
    ids, coll_id = both_views

    answers = {query: _shown(smartgallery_app, ids,
                             f"/galleryout/collection/{coll_id}?{query}")[1]
               for query in _FILTERS}
    narrowed = {query for query, files in answers.items() if len(files) < len(ids)}

    assert len(narrowed) >= 8, (
        f"only {len(narrowed)} of {len(_FILTERS)} filters narrowed anything: "
        f"{ {q: sorted(a) for q, a in answers.items()} }")
    assert answers["workflow_prompt=seed:5"] == {f"{_PREFIX}alpha_one.png"}, (
        answers["workflow_prompt=seed:5"])
