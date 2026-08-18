"""Your own ratings have to be findable in the default install.

Ratings are stored against a `client_uuid`, and the default single-user
install has no login, so the browser sends the fixed identity `admin`
(templates/index.html mints `clientUUID = 'admin'` whenever the gallery is
not in exhibition or force-login mode).

The gallery page builds `my_rating`, and the per-user rating filters, from
`session['user_id']` instead -- which in that same install is never set.
So the write lands under `admin` and the read looks for the empty string,
and every "my ratings" view comes back empty for the one deployment shape
almost everyone runs.

The symptom is silent: averages and counts are computed globally and stay
correct, so the gallery looks healthy right up until someone turns on
"My Ratings Only" and their library appears unrated.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image

from smartgallery import get_db_connection

from inline_executor import InlineExecutor

_PREFIX = "ratingid_"


def _purge(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM file_ratings WHERE client_uuid IN ('admin', '')")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    """Two images in the gallery root, scanned in-process."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    root = smartgallery_app.BASE_OUTPUT_PATH
    os.makedirs(root, exist_ok=True)
    made = []
    for name, colour in ((f"{_PREFIX}alpha.png", (210, 40, 40)), (f"{_PREFIX}beta.png", (40, 210, 40))):
        path = os.path.join(root, name)
        Image.new("RGB", (80, 60), colour).save(path)
        made.append((name, path))
    _purge(smartgallery_app)

    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
        ids = {
            r["name"]: r["id"]
            for r in conn.execute(f"SELECT id, name FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()
        }
    finally:
        conn.close()

    yield ids

    for _name, path in made:
        with contextlib.suppress(OSError):
            os.remove(path)
    _purge(smartgallery_app)


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


def _rate_as_browser(client, file_id, rating):
    """Exactly what the page sends with no login: the fixed 'admin' identity."""
    resp = client.post(
        "/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": rating, "client_uuid": "admin"}
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp


def test_a_rating_is_stored_under_the_identity_the_page_uses(client, library):
    """Control: the write half works, so a later failure is the read half."""
    alpha = library[f"{_PREFIX}alpha.png"]
    _rate_as_browser(client, alpha, 5)

    conn = get_db_connection()
    try:
        rows = [
            (r[0], r[1])
            for r in conn.execute("SELECT client_uuid, rating FROM file_ratings WHERE file_id = ?", (alpha,)).fetchall()
        ]
    finally:
        conn.close()

    assert rows == [("admin", 5)], f"unexpected stored identity: {rows}"


def test_my_ratings_only_still_shows_the_files_i_rated(client, library):
    """The regression: rate a file, ask for only your own rated files, and
    the file must be there. It was filtered out because the page looked up
    the empty string instead of the identity it had just written."""
    alpha = library[f"{_PREFIX}alpha.png"]
    _rate_as_browser(client, alpha, 5)

    with client.session_transaction() as session:
        session["my_ratings_only"] = True

    resp = client.get("/galleryout/view/_root_?sort_by=rating")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert f"{_PREFIX}alpha.png" in html, "a file the user rated is missing from their own ratings view"
    assert f"{_PREFIX}beta.png" not in html, "an unrated file leaked into the rated-only view"


def test_unrated_means_not_rated_by_me(client, library):
    """The complement, and the more dangerous direction: if the identity is
    wrong here the 'unrated' queue hands back work already done."""
    alpha = library[f"{_PREFIX}alpha.png"]
    _rate_as_browser(client, alpha, 4)

    with client.session_transaction() as session:
        session["my_ratings_only"] = True

    html = client.get("/galleryout/view/_root_?sort_by=unrated").get_data(as_text=True)

    assert f"{_PREFIX}beta.png" in html, "an unrated file is missing from the unrated view"
    assert f"{_PREFIX}alpha.png" not in html, "a file the user already rated came back as unrated"
