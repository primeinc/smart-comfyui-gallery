"""Submitting ratings: identity, one-vote-per-rater, and validation.

Ratings share the "cannot be recomputed" property with comments, and they
carry an extra risk the comments do not: they are aggregated. A rater who
can vote twice, or vote as somebody else, does not just corrupt one row --
they move the average every other decision is made against.

Three properties hold the aggregate honest, and all three are pinned here:
an authenticated caller's identity comes from the SESSION and cannot be
overridden by the request body; a second vote on the same file replaces
the first rather than adding to it; and out-of-range values are refused
instead of being stored and averaged.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from PIL import Image

_PREFIX = "ratroute_"
_ME = 41
_SOMEONE_ELSE = "99"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


@pytest.fixture
def rated_files(smartgallery_app):
    """Two real files with rows, cleaned up afterwards."""
    ids = []
    conn = smartgallery_app.get_db_connection()
    try:
        for n in ("a", "b"):
            name = f"{_PREFIX}{n}.png"
            path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name)
            Image.new("RGB", (16, 16), (30, 90, 30)).save(path)
            file_id = f"{_PREFIX}{n}"
            conn.execute(
                "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) VALUES (?, ?, ?, ?, 'image', ?)",
                (file_id, path, os.path.getmtime(path), name, os.path.getsize(path)),
            )
            ids.append(file_id)
        conn.commit()
    finally:
        conn.close()
    yield ids
    conn = smartgallery_app.get_db_connection()
    try:
        for file_id in ids:
            row = conn.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
            if row:
                with contextlib.suppress(OSError):
                    os.remove(row[0])
        conn.execute("DELETE FROM file_ratings WHERE file_id LIKE ?", (f"{_PREFIX}%",))
        conn.execute("DELETE FROM files WHERE id LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()


def _ratings_for(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        return [
            (r[0], r[1])
            for r in conn.execute(
                "SELECT client_uuid, rating FROM file_ratings WHERE file_id = ? ORDER BY client_uuid", (file_id,)
            ).fetchall()
        ]
    finally:
        conn.close()


def _sign_in(client, user_id=_ME):
    with client.session_transaction() as session:
        session["user_id"] = user_id
        session["role"] = "CUSTOMER"


def test_rating_is_recorded_against_the_session_user(smartgallery_app, client, rated_files):
    file_id = rated_files[0]
    _sign_in(client)

    resp = client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": 4})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert _ratings_for(smartgallery_app, file_id) == [(str(_ME), 4)]


def test_the_body_cannot_forge_a_different_rater(smartgallery_app, client, rated_files):
    """An authenticated caller's identity comes from the session; a
    client_uuid in the body must be ignored, or one account could stuff the
    average under any number of invented identities."""
    file_id = rated_files[0]
    _sign_in(client)

    client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": 5, "client_uuid": _SOMEONE_ELSE})

    recorded = _ratings_for(smartgallery_app, file_id)
    assert recorded == [(str(_ME), 5)], f"the request body chose the rater identity: {recorded}"


def test_rating_again_replaces_rather_than_accumulates(smartgallery_app, client, rated_files):
    file_id = rated_files[0]
    _sign_in(client)

    client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": 1})
    resp = client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": 5})

    assert _ratings_for(smartgallery_app, file_id) == [(str(_ME), 5)], (
        "a second vote was added instead of replacing the first"
    )
    assert resp.get_json()["vote_count"] == 1


def test_rating_zero_withdraws_the_vote(smartgallery_app, client, rated_files):
    file_id = rated_files[0]
    _sign_in(client)
    client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": 3})

    resp = client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": 0})

    assert resp.status_code == 200
    assert _ratings_for(smartgallery_app, file_id) == []
    assert resp.get_json()["vote_count"] == 0


@pytest.mark.parametrize("bad", [6, 99, -1, 1000])
def test_out_of_range_ratings_are_refused(smartgallery_app, client, rated_files, bad):
    """A stored 99 would drag the average of everything it touches."""
    file_id = rated_files[0]
    _sign_in(client)

    resp = client.post("/galleryout/api/exhibition/rate", json={"file_id": file_id, "rating": bad})

    assert resp.status_code == 400, f"rating {bad} was accepted"
    assert _ratings_for(smartgallery_app, file_id) == []


def test_rating_an_unknown_file_is_refused(smartgallery_app, client):
    _sign_in(client)
    resp = client.post("/galleryout/api/exhibition/rate", json={"file_id": f"{_PREFIX}ghost", "rating": 3})
    assert resp.status_code == 404


def test_batch_rating_applies_to_every_file_once(smartgallery_app, client, rated_files):
    _sign_in(client)

    resp = client.post("/galleryout/api/exhibition/rate_batch", json={"file_ids": rated_files, "rating": 2})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    for file_id in rated_files:
        assert _ratings_for(smartgallery_app, file_id) == [(str(_ME), 2)]

    # Re-running must not double up.
    client.post("/galleryout/api/exhibition/rate_batch", json={"file_ids": rated_files, "rating": 4})
    for file_id in rated_files:
        assert _ratings_for(smartgallery_app, file_id) == [(str(_ME), 4)]


def test_batch_rating_cannot_forge_the_rater(smartgallery_app, client, rated_files):
    _sign_in(client)

    client.post(
        "/galleryout/api/exhibition/rate_batch",
        json={"file_ids": rated_files, "rating": 5, "client_uuid": _SOMEONE_ELSE},
    )

    for file_id in rated_files:
        assert _ratings_for(smartgallery_app, file_id) == [(str(_ME), 5)]


@pytest.mark.parametrize(
    "payload",
    [
        {"file_ids": [], "rating": 3},
        {"rating": 3},
    ],
)
def test_batch_rating_needs_files(client, payload):
    _sign_in(client)
    assert client.post("/galleryout/api/exhibition/rate_batch", json=payload).status_code == 400
