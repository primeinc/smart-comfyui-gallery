"""A visitor may only rate what was shared with them.

Both rating routes asked whether the file existed and never whether this
caller was allowed to see it. So in Exhibition a signed-in visitor could
score a picture that is in no album of theirs -- one the gallery refuses
to send them, 403 for the thumbnail and 403 for the file -- and the score
was stored. The owner's average for a private picture was then set by
somebody who had never seen it.

Measured before the fix, as a guest with a picture deliberately left out
of the public album:

    shared with them   single 200   batch 200
    NOT shared         single 200   batch 200
    ratings stored:    shared.png by 42 = 4
                       private.png by 42 = 4

It also answered differently for a file that is hidden and one that does
not exist -- 200 against 404 -- which turns the route into a way of
discovering which ids a library holds. Both now answer 404, so the two
cannot be told apart.

And the batch route handed whatever ids it was given straight to the
insert, so a made-up one came back as a 500 quoting "FOREIGN KEY
constraint failed" at the caller. It filters first now.

None of this touches the management side: is_file_accessible answers True
for ADMIN, MANAGER and STAFF, and True whenever no login is configured at
all, which is most installs. Two tests below hold exactly that.
"""

from __future__ import annotations

import concurrent.futures
import os

import pytest
from PIL import Image

_PREFIX = "ratevis_"
_MISSING_ID = "f" * 32


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


@pytest.fixture()
def library(smartgallery_app, monkeypatch):
    """One picture in a public album, one deliberately left out."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures,
                        "ProcessPoolExecutor", _InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    names = [f"{_PREFIX}shared.png", f"{_PREFIX}private.png"]
    for name in names:
        Image.new("RGB", (8, 8), (5, 5, 5)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Rating Album'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {r["name"]: r["id"] for r in conn.execute(
            f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()}
        conn.execute("INSERT INTO collections (name, type, is_public) VALUES (?, ?, 1)",
                     ("Rating Album", "user_album"))
        coll_id = conn.execute("SELECT id FROM collections WHERE name = ?",
                               ("Rating Album",)).fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)",
                     (coll_id, ids[names[0]]))
        conn.commit()
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Rating Album'")
        conn.commit()
    finally:
        conn.close()
    for name in names:
        try:
            os.remove(os.path.join(base, name))
        except OSError:
            pass


def _visitor(smartgallery_app, monkeypatch, role="GUEST"):
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 42
        session["role"] = role
        session["full_name"] = "A Visitor"
    return client


def _stored(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT rating FROM file_ratings WHERE file_id = ? "
                           "AND client_uuid = '42'", (file_id,)).fetchone()
    finally:
        conn.close()
    return row["rating"] if row else None


def test_a_visitor_can_rate_what_was_shared(smartgallery_app, library, monkeypatch):
    """Control. Every refusal below is only meaningful while the feature
    still works for the pictures a visitor was actually given."""
    client = _visitor(smartgallery_app, monkeypatch)
    shared = library[f"{_PREFIX}shared.png"]

    single = client.post("/galleryout/api/exhibition/rate",
                         json={"file_id": shared, "rating": 5})

    assert single.status_code == 200, single.get_json()
    assert _stored(smartgallery_app, shared) == 5


@pytest.mark.parametrize("route,payload_key", [
    ("/galleryout/api/exhibition/rate", "file_id"),
    ("/galleryout/api/exhibition/rate_batch", "file_ids"),
])
def test_a_visitor_cannot_rate_what_was_not_shared(smartgallery_app, library,
                                                   monkeypatch, route, payload_key):
    """The bug, on both routes."""
    client = _visitor(smartgallery_app, monkeypatch)
    private = library[f"{_PREFIX}private.png"]
    assert client.get(f"/galleryout/file/{private}").status_code == 403, (
        "the fixture shared the picture it was meant to keep back")

    value = private if payload_key == "file_id" else [private]
    response = client.post(route, json={payload_key: value, "rating": 4})

    assert response.status_code == 404, (route, response.get_json())
    assert _stored(smartgallery_app, private) is None, (
        "a score was recorded for a picture the visitor cannot even open")


@pytest.mark.parametrize("route,payload_key", [
    ("/galleryout/api/exhibition/rate", "file_id"),
    ("/galleryout/api/exhibition/rate_batch", "file_ids"),
])
def test_hidden_and_missing_are_indistinguishable(smartgallery_app, library,
                                                  monkeypatch, route, payload_key):
    """Answering 200 for a hidden picture and 404 for one that is not there
    turns the route into a way of finding out which ids exist."""
    client = _visitor(smartgallery_app, monkeypatch)
    private = library[f"{_PREFIX}private.png"]

    def ask(file_id):
        value = file_id if payload_key == "file_id" else [file_id]
        answer = client.post(route, json={payload_key: value, "rating": 3})
        return answer.status_code, answer.get_json()

    assert ask(private) == ask(_MISSING_ID)


def test_the_batch_route_does_not_answer_with_a_database_error(smartgallery_app,
                                                               library, monkeypatch):
    """It used to hand unknown ids to the insert and return the constraint
    failure verbatim."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()

    response = client.post("/galleryout/api/exhibition/rate_batch",
                           json={"file_ids": [_MISSING_ID], "rating": 3,
                                 "client_uuid": "guest_abcdef0123456789"})

    assert response.status_code == 404, response.get_json()
    assert "FOREIGN KEY" not in str(response.get_json()), response.get_json()


def test_a_manager_may_still_rate_anything(smartgallery_app, library, monkeypatch):
    """Over-reach guard: the check must not reach the management side,
    where bulk rating across a whole library is the point of the feature."""
    client = _visitor(smartgallery_app, monkeypatch, role="MANAGER")
    private = library[f"{_PREFIX}private.png"]

    response = client.post("/galleryout/api/exhibition/rate_batch",
                           json={"file_ids": [private], "rating": 2})

    assert response.status_code == 200, response.get_json()
    assert _stored(smartgallery_app, private) == 2


def test_a_local_gallery_with_no_login_is_untouched(smartgallery_app, library,
                                                    monkeypatch):
    """The commonest install of all: no login configured, so every file is
    the owner's own and nothing here applies."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    private = library[f"{_PREFIX}private.png"]

    response = client.post("/galleryout/api/exhibition/rate",
                           json={"file_id": private, "rating": 1,
                                 "client_uuid": "guest_abcdef0123456789"})

    assert response.status_code == 200, response.get_json()
