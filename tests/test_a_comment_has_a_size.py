"""A visitor's comment must not be able to fill the database.

Exhibition is the portal built to be handed to family, friends and
clients, optionally with a guest login, and comments are the one thing a
visitor writes into it. Nothing bounded them -- not the route, not the
schema, not the page. Measured, posting to the real route:

    comment size   status   stored?    db bytes
    100 chars      200      yes        100
    100000 chars   200      yes        100100
    5000000 chars  200      yes        5100100
    40000000 chars 200      yes        45100100

Forty million characters accepted with a 200 and stored whole; four posts
put 45 MB into the database. Whatever is sent is then read back to
everyone who opens that picture and rendered in their browser, so one
paste of a large file is enough to make a picture unopenable for
everybody, and it need not be deliberate.

Refused rather than truncated: quietly keeping the first few thousand
characters of what somebody wrote is worse than telling them it was too
long. Long writing about a collection has a place already -- the
collection note, which is a file and is meant for it.
"""

from __future__ import annotations

import pytest

import smartgallery


@pytest.fixture()
def a_picture(smartgallery_app):
    """One file to comment on, cleaned up after."""
    file_id = "c" * 32
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) "
            "VALUES (?,?,?,?,?)",
            (file_id, "/lib/commented.png", 1700000000.0, "commented.png",
             "image"))
        conn.commit()
    finally:
        conn.close()

    yield file_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM file_comments WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def visitor(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app.app.test_client()


def _post(client, file_id, text, author="Someone"):
    return client.post("/galleryout/api/exhibition/post_comment",
                       json={"file_id": file_id, "text": text,
                             "client_uuid": "visitor", "author": author})


def _stored(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(LENGTH(comment_text)), 0) FROM file_comments "
            "WHERE file_id = ?", (file_id,)).fetchone()[0]
    finally:
        conn.close()


def test_an_ordinary_comment_still_posts(smartgallery_app, visitor, a_picture):
    """Over-reach guard, and every comment anybody actually writes. A
    limit that turned away real remarks would be worse than none."""
    response = _post(visitor, a_picture, "Lovely light on this one.")

    assert response.status_code == 200, response.get_json()
    assert _stored(smartgallery_app, a_picture) == 25


def test_a_long_but_reasonable_comment_still_posts(smartgallery_app, visitor,
                                                   a_picture):
    """Over-reach guard: the limit is meant to be generous. Somebody
    writing a couple of paragraphs about a picture must not hit it."""
    text = "x" * (smartgallery.COMMENT_MAX_CHARS - 1)

    response = _post(visitor, a_picture, text)

    assert response.status_code == 200, response.get_json()
    assert _stored(smartgallery_app, a_picture) == len(text)


def test_a_comment_at_the_limit_is_accepted(smartgallery_app, visitor, a_picture):
    """The boundary belongs to the person, not to the check."""
    text = "x" * smartgallery.COMMENT_MAX_CHARS

    assert _post(visitor, a_picture, text).status_code == 200


def test_a_comment_past_the_limit_is_refused(smartgallery_app, visitor,
                                             a_picture):
    """The bug: forty million characters came back 200."""
    text = "x" * (smartgallery.COMMENT_MAX_CHARS + 1)

    response = _post(visitor, a_picture, text)

    assert response.status_code == 400, response.status_code
    assert _stored(smartgallery_app, a_picture) == 0, (
        "the comment was refused and stored anyway")


def test_the_refusal_says_what_the_limit_is(smartgallery_app, visitor,
                                            a_picture):
    """"Too long" without a number leaves somebody deleting words until it
    works."""
    response = _post(visitor, a_picture, "x" * 5_000_000)
    body = response.get_json()

    assert body is not None, response.get_data(as_text=True)[:200]
    assert f"{smartgallery.COMMENT_MAX_CHARS:,}" in body["message"], body
    assert "5,000,000" in body["message"], body


def test_nothing_is_kept_from_a_refused_comment(smartgallery_app, visitor,
                                                a_picture):
    """Refused, not truncated. Keeping the first few thousand characters
    of what somebody wrote is worse than telling them."""
    _post(visitor, a_picture, "y" * 200_000)

    assert _stored(smartgallery_app, a_picture) == 0


def test_a_visitor_name_has_a_size_too(smartgallery_app, visitor, a_picture):
    """Only a real visitor sets this -- a signed-in user gets their
    account name and the local owner gets "System Admin" -- so it is
    exactly the unauthenticated path."""
    response = _post(visitor, a_picture, "hello",
                     author="A" * (smartgallery.COMMENT_AUTHOR_MAX_CHARS + 1))

    assert response.status_code == 400, response.status_code
    assert _stored(smartgallery_app, a_picture) == 0


def test_an_ordinary_name_is_fine(smartgallery_app, visitor, a_picture):
    """Over-reach guard. Names are not short in every language."""
    for name in ["Jo", "Bartholomew Featherstonehaugh", "第一章のゲスト",
                 "Ααααα Βββββ"]:
        response = _post(visitor, a_picture, "hi", author=name)
        assert response.status_code == 200, (name, response.get_json())


def test_the_limit_is_generous_enough_to_be_a_comment():
    """A cap tight enough to annoy people would get raised in a hurry and
    the bug would come back with it."""
    assert smartgallery.COMMENT_MAX_CHARS >= 2000
    assert smartgallery.COMMENT_MAX_CHARS <= 100_000


def test_the_box_says_so_before_anybody_types_past_it():
    """Being told after writing is worse than being stopped while typing.
    The server check is the real one; this is the courtesy."""
    import pathlib

    root = pathlib.Path(smartgallery.__file__).resolve().parent
    for name in ("index.html", "exhibition.html"):
        text = (root / "templates" / name).read_text(encoding="utf-8")
        assert f'maxlength="{smartgallery.COMMENT_MAX_CHARS}"' in text, (
            f"{name} has a comment box with no maxlength, so somebody can "
            f"write past the limit and only find out when it is refused")
