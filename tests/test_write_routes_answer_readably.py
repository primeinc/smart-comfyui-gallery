"""A write route must answer something the page can read.

Every one of these is called with fetch().then(res => res.json()). When a
route answers with an unhandled exception, Flask sends a 500 carrying an
HTML page, res.json() throws, and the screen has nothing to show -- not
even the reason. That is not hypothetical: the commit before this one
found it on the user-edit branch, where renaming somebody to a username
already taken produced exactly that, a 500 with no body, while the create
branch beside it answered 400 with a message.

So the routes that write are called here with the input that goes wrong in
practice -- nothing, a missing id, a value the database will not accept --
and each must answer under 500, in JSON. What the answer says is each
route's own business; that it can be read at all is this file's.

Swept when written: no route answered 500, and the only replies that were
not JSON were Flask's own 405 for a wrong method, which is correct, and a
404 whose caller has a .catch. Both are excluded below rather than
silently passed over.
"""

from __future__ import annotations

import pytest

_MISSING_FILE = "d" * 32

# (method, url, payload). Chosen for the ways a caller gets it wrong: an
# empty body, an id that is not there, a value out of range.
_CALLS = [
    ("POST", "/galleryout/ai_indexing/add_files", {}),
    ("POST", "/galleryout/ai_indexing/add_files", {"file_ids": [_MISSING_FILE]}),
    ("POST", "/galleryout/ai_indexing/add_folder", {}),
    ("POST", "/galleryout/ai_indexing/add_folder", {"folder_key": "not_a_folder"}),
    ("DELETE", "/galleryout/ai_indexing/watched", {}),
    ("POST", "/galleryout/ai_indexing/control", {}),
    ("POST", "/galleryout/ai_indexing/control", {"action": "not_an_action"}),
    ("POST", "/galleryout/favorite_batch", {}),
    ("POST", "/galleryout/favorite_batch", {"file_ids": [_MISSING_FILE], "status": True}),
    ("POST", "/galleryout/delete/" + _MISSING_FILE, {}),
    ("POST", "/galleryout/api/collections/delete", {}),
    ("POST", "/galleryout/api/collections/delete", {"collection_id": 999999}),
    ("POST", "/galleryout/api/site_settings", {}),
    ("POST", "/galleryout/api/site_settings", {"key": "x" * 300, "value": None}),
    ("POST", "/galleryout/api/exhibition/rate", {}),
    ("POST", "/galleryout/api/exhibition/rate",
     {"file_id": _MISSING_FILE, "rating": 99}),
    ("POST", "/galleryout/api/exhibition/rate_batch", {}),
    ("POST", "/galleryout/api/exhibition/post_comment", {}),
    ("POST", "/galleryout/api/admin/users", {}),
    ("POST", "/galleryout/api/admin/users",
     {"username": "x", "password": "short", "full_name": "X", "role": "STAFF"}),
    ("PUT", "/galleryout/api/admin/users",
     {"user_id": 999999, "username": "y", "full_name": "Y", "role": "NOT_A_ROLE",
      "is_active": 1}),
]


@pytest.fixture()
def owner(smartgallery_app, monkeypatch):
    """The local owner, which is how most galleries run."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "ENABLE_AI_SEARCH", True, raising=False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"
    return client


def _send(client, method, url, payload):
    call = {"POST": client.post, "PUT": client.put,
            "DELETE": client.delete}[method]
    return call(url, json=payload)


@pytest.mark.parametrize("method,url,payload", _CALLS,
                         ids=[f"{m} {u.split('/')[-1]} {sorted(p)}"
                              for m, u, p in _CALLS])
def test_it_answers_below_500_and_in_json(owner, method, url, payload):
    response = _send(owner, method, url, payload)

    assert response.status_code < 500, (
        f"{method} {url} with {payload} answered {response.status_code}; a "
        f"page calling this does res.json() and gets an HTML error instead")
    assert response.get_json() is not None, (
        f"{method} {url} with {payload} answered {response.status_code} with "
        f"a body that is not JSON: {response.get_data(as_text=True)[:200]}")


def test_the_calls_actually_reach_the_routes(owner):
    """Control. Every assertion above is about an answer being acceptable,
    and a 404 from a route that does not exist is acceptable too -- so at
    least some of these have to be doing real work and refusing on their
    own terms."""
    refusals, successes = 0, 0
    for method, url, payload in _CALLS:
        response = _send(owner, method, url, payload)
        if response.status_code == 400:
            refusals += 1
        elif response.status_code == 200:
            successes += 1

    assert refusals >= 5, (
        f"only {refusals} of {len(_CALLS)} calls were refused with a 400; "
        f"these are meant to be inputs the routes reject")
    assert successes >= 3, (
        f"only {successes} answered 200; the calls may not be reaching the "
        f"routes at all")


def test_a_write_that_the_database_refuses_still_answers_readably(smartgallery_app,
                                                                  owner):
    """The case the table above cannot reach, and the one that mattered.

    A bad edit aimed at user 999999 changes no rows, so the constraint
    never fires and the route answers cheerfully -- which is why the
    parametrised calls pass against the build that had the bug. The row has
    to exist for the database to object to it, so this makes one.

    Checked against that build: without a real user this reports nothing,
    with one it reports the 500 and the empty body."""
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM users WHERE username = 'readable_t'")
        conn.commit()
    finally:
        conn.close()

    made = owner.post("/galleryout/api/admin/users",
                      json={"username": "readable_t", "password": "longenough1",
                            "full_name": "Readable", "role": "STAFF"})
    assert made.status_code == 200, made.get_json()

    conn = smartgallery_app.get_db_connection()
    try:
        user_id = conn.execute("SELECT user_id FROM users WHERE username = ?",
                               ("readable_t",)).fetchone()["user_id"]
    finally:
        conn.close()

    try:
        refused = owner.put("/galleryout/api/admin/users",
                            json={"user_id": user_id, "username": "readable_t",
                                  "full_name": "Readable", "role": "NOT_A_ROLE",
                                  "is_active": 1})

        assert refused.status_code < 500, (
            f"the database refused the write and the route answered "
            f"{refused.status_code} with an HTML page")
        assert refused.get_json() is not None, (
            f"body was not JSON: {refused.get_data(as_text=True)[:200]}")
    finally:
        conn = smartgallery_app.get_db_connection()
        try:
            conn.execute("DELETE FROM users WHERE username = 'readable_t'")
            conn.commit()
        finally:
            conn.close()


def test_a_route_that_raises_would_be_caught():
    """Control for the check itself: a 500 has to be visible to it.

    Its own small app rather than a route added to the gallery's -- Flask
    refuses to register one after the first request, and a check for
    unhandled exceptions should not be reaching into the thing it checks
    anyway. What is under test here is the pair of assertions above."""
    from flask import Flask

    app = Flask(__name__)

    @app.route("/explodes", methods=["POST"])
    def _explodes():
        raise RuntimeError("as if a constraint had fired")

    response = app.test_client().post("/explodes", json={})

    assert response.status_code >= 500, response.status_code
    assert response.get_json() is None, (
        "an unhandled exception produced readable JSON, so the checks above "
        "would not notice one")
