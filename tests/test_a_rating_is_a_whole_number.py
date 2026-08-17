"""A rating must be a whole number from 1 to 5, or nothing.

The check was `1 <= rating <= 5` against whatever JSON supplied, on a
route a visitor reaches in Exhibition. That is two faults in one line.

Anything not a number raised, and the caller does res.json():

    "3" (a string) [rate]       500  NO   <!doctype html> <html lang=en>
    "abc"          [rate]       500  NO   <!doctype html> <html lang=en>
    [3] (a list)   [rate]       500  NO   <!doctype html> <html lang=en>
    {} (an object) [rate]       500  NO   <!doctype html> <html lang=en>

    TypeError: '<=' not supported between instances of 'int' and 'str'

And anything that happened to compare was taken. SQLite stores what it is
handed, so a fraction went into an INTEGER column and sailed past
CHECK(rating >= 1 AND rating <= 5):

    alice   stored 3.5   sqlite type real
    bob     stored 2.25  sqlite type real
    carol   stored 1     sqlite type integer   (sent true)
    average everyone sees: 2.25

Both routes now ask parse_rating, because the same line appeared twice --
the single-file route and the batch one -- and one of them being fixed
alone is how the comment limit was walked round two cycles ago.

3.0 is accepted as 3. A client whose JSON has one number type is not
making a mistake, and turning that away would be the same unhelpfulness
as insisting on forward slashes in a path.
"""

from __future__ import annotations

import pytest

import smartgallery

_ROUTES = ["/galleryout/api/exhibition/rate",
           "/galleryout/api/exhibition/rate_batch"]


@pytest.fixture()
def a_picture(smartgallery_app):
    file_id = "r" * 32
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) "
            "VALUES (?,?,?,?,?)",
            (file_id, "/lib/rated.png", 1700000000.0, "rated.png", "image"))
        conn.commit()
    finally:
        conn.close()
    yield file_id
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM file_ratings WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def visitor(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


def _rate(client, route, file_id, value, who="visitor"):
    payload = {"client_uuid": who, "rating": value}
    if route.endswith("rate_batch"):
        payload["file_ids"] = [file_id]
    else:
        payload["file_id"] = file_id
    return client.post(route, json=payload)


def _rows(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        return conn.execute(
            "SELECT rating, typeof(rating) AS kind FROM file_ratings "
            "WHERE file_id = ?", (file_id,)).fetchall()
    finally:
        conn.close()


# --- the rule on its own --------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (1, 1), (3, 3), (5, 5),
    (3.0, 3),                 # a client with one number type
    (0, None), (None, None),  # clearing
])
def test_what_a_rating_may_be(value, expected):
    rating, error = smartgallery.parse_rating(value)

    assert error is None, (value, error)
    assert rating == expected


@pytest.mark.parametrize("value", [
    "3", "abc", "", [3], {}, (3,), True, False,
    3.5, 2.25, -1, 6, 99, float("nan"), float("inf"),
])
def test_what_a_rating_may_not_be(value):
    rating, error = smartgallery.parse_rating(value)

    assert error, f"{value!r} was accepted as a rating"
    assert rating is None


def test_a_boolean_is_not_one_star():
    """True == 1 in Python and isinstance(True, int) is True, so a bool
    passes every obvious check. It quietly became a one-star vote."""
    assert smartgallery.parse_rating(True)[0] is None


# --- through both routes --------------------------------------------------

@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("value", ["3", "abc", [3], {}])
def test_a_rating_of_the_wrong_type_answers_readably(smartgallery_app, visitor,
                                                     a_picture, route, value):
    """The bug: a 500 carrying an HTML page, to a caller doing res.json()."""
    response = _rate(visitor, route, a_picture, value)

    assert response.status_code == 400, response.status_code
    body = response.get_json()
    assert body is not None, response.get_data(as_text=True)[:160]
    assert "1 to 5" in body["message"], body


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("value", [3.5, 2.25, True])
def test_nothing_that_is_not_a_whole_star_is_stored(smartgallery_app, visitor,
                                                    a_picture, route, value):
    """The half that skewed what everyone sees."""
    response = _rate(visitor, route, a_picture, value)

    assert response.status_code == 400, (value, response.get_json())
    assert _rows(smartgallery_app, a_picture) == [], (
        f"{value!r} was stored as a rating")


@pytest.mark.parametrize("route", _ROUTES)
def test_an_ordinary_rating_still_works(smartgallery_app, visitor, a_picture,
                                        route):
    """Over-reach guard, and the whole feature."""
    assert _rate(visitor, route, a_picture, 4).status_code == 200

    rows = _rows(smartgallery_app, a_picture)
    assert len(rows) == 1
    assert rows[0]["rating"] == 4
    assert rows[0]["kind"] == "integer", (
        f"stored as {rows[0]['kind']} rather than a whole number")


@pytest.mark.parametrize("route", _ROUTES)
def test_clearing_a_rating_still_works(smartgallery_app, visitor, a_picture,
                                       route):
    """Over-reach guard: 0 and null mean "I take it back", and the star
    widget sends one of them when you click the star you already chose."""
    assert _rate(visitor, route, a_picture, 4).status_code == 200
    assert _rate(visitor, route, a_picture, 0).status_code == 200
    assert _rows(smartgallery_app, a_picture) == []

    assert _rate(visitor, route, a_picture, 4).status_code == 200
    assert _rate(visitor, route, a_picture, None).status_code == 200
    assert _rows(smartgallery_app, a_picture) == []


def test_both_routes_ask_the_same_rule():
    """The same line appeared in both, and fixing one of a pair is how the
    comment limit was walked round two cycles ago."""
    import ast
    import io
    import pathlib

    source = pathlib.Path(smartgallery.__file__)
    tree = ast.parse(io.open(source, encoding="utf-8").read())

    for name in ("exhibition_rate_file", "exhibition_rate_batch"):
        fn = next((node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == name),
                  None)
        assert fn is not None, f"{name} is gone"
        called = {node.func.id for node in ast.walk(fn)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        assert "parse_rating" in called, (
            f"{name} decides for itself what a rating may be")
