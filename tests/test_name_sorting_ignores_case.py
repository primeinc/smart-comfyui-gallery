"""Sorting by name put every capital letter first.

The grid ordered with SQL `ORDER BY f.name`, which compares bytes, so
every name beginning with a capital came before every name beginning with
a lowercase letter. Measured before the fix, on eight files:

    Banana  Date  IMG_001  apple  cherry  img_002  zebra  Ähnlich

IMG_001.png and img_002.png are consecutive files from one session, and
five unrelated ones sat between them. This is not a non-English problem:
ComfyUI writes one capitalisation and people type another, so any folder
holding both is shuffled.

The gallery already disagreed with itself about this. The folder list
sorts with Python's str.lower, and so does the OmniQuery result list; only
the grid and the collection lists used the database's byte order. They now
use ulower(), the same case folding registered for search -- which covers
every script rather than ASCII, so ÄHNLICH and ähnlich land together.

What this does NOT do is order accented letters where a reader would
expect them. Ähnlich still sorts after zebra, because folding case does
not reposition a character in the code-point order; that needs real
collation. The last test pins that, so the limit is recorded rather than
implied away.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "sortcase_"
_NAMES = [
    f"{_PREFIX}{part}"
    for part in ("apple.png", "Banana.png", "cherry.png", "Date.png", "IMG_001.png", "img_002.png", "zebra.png")
]
_ACCENTED = f"{_PREFIX}Ähnlich.png"


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    made = [*_NAMES, _ACCENTED]
    for name in made:
        Image.new("RGB", (8, 8), (1, 1, 1)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {
            r["name"]: r["id"]
            for r in conn.execute("SELECT name, id FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)).fetchall()
        }
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
    for name in made:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _order(smartgallery_app, ids, direction="ASC"):
    """The names as the grid lays them out, top to bottom."""
    page = (
        smartgallery_app.app.test_client()
        .get(f"/galleryout/view/_root_?sort_by=name&sort_order={direction}")
        .get_data(as_text=True)
    )
    seen = [(page.find(file_id), name) for name, file_id in ids.items() if page.find(file_id) >= 0]
    return [name for _at, name in sorted(seen)]


def test_the_fixture_put_every_file_on_the_page(smartgallery_app, library):
    """Control: an order can only be judged if everything is there."""
    order = _order(smartgallery_app, library)

    assert len(order) == len(library) == 8, order


def test_capitals_no_longer_come_first(smartgallery_app, library):
    """The bug, on the pair that shows it: two files from one session."""
    order = _order(smartgallery_app, library)

    assert order.index(f"{_PREFIX}IMG_001.png") + 1 == order.index(f"{_PREFIX}img_002.png"), (
        f"IMG_001 and img_002 are not adjacent: {order}"
    )
    assert order[:7] == sorted(_NAMES, key=str.lower), order


def test_the_other_direction_is_the_reverse(smartgallery_app, library):
    """Ordering case-insensitively must not lose the direction."""
    ascending = _order(smartgallery_app, library, "ASC")
    descending = _order(smartgallery_app, library, "DESC")

    assert descending == list(reversed(ascending)), (ascending, descending)


def test_the_grid_agrees_with_the_folder_list(smartgallery_app, library):
    """The gallery disagreed with itself: the folder list has always sorted
    with str.lower while the grid used the database's byte order."""
    order = _order(smartgallery_app, library)

    assert order == sorted(order, key=str.lower), order


def test_another_sort_is_untouched(smartgallery_app, library):
    """Control against over-reach: only the name ordering changed, so a
    different sort must still answer and still list everything."""
    page = smartgallery_app.app.test_client().get("/galleryout/view/_root_?sort_by=date&sort_order=DESC")

    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert all(file_id in body for file_id in library.values())


def test_an_accent_still_sorts_after_z(smartgallery_app, library):
    """Recorded, not fixed. Folding case does not move a character in the
    code-point order, so Ähnlich still follows zebra. Ordering accented
    letters where a reader expects them needs real collation, which is a
    larger change than this one and is not pretended to be done."""
    order = _order(smartgallery_app, library)

    assert order[-1] == _ACCENTED, order
