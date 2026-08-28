"""Three things people mean by "save this", and only two had somewhere to go.

An **album** is what somebody deliberately put together. A **smart
collection** is a dynamic grouping that behaves like one: members, an
address, a place on the shelf, things filed under it. A **saved view**
is neither -- it is "that was a useful question, remember it", with no
members, no colour, no parent and nothing filed under it.

Having composed a good question, the only offer was "save view", which
made a smart collection. So remembering five good questions put five
things that are not albums into somebody's album list, and the album
shelf stopped being a list of albums.

They share a GalleryQuery underneath without being one product object.
This stores the canonical SPELLING rather than a typed rule, because the
spelling is entity-aware: a view saved before a rename still answers
afterwards, because the address heals to the live slug as it is
navigated.
"""

from __future__ import annotations

import pytest
from PIL import Image

from db import connect, views
from tests.staging import staged

pytestmark = pytest.mark.slow


def _library(root) -> None:
    for i in range(3):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_good_question_is_worth_remembering", _library) as stage:
        yield stage


@pytest.fixture
def gallery(_stage):
    """One world, restored between tests.

    Every test here REMEMBERS a question, and several count what is
    remembered, so they cannot share a world that keeps its writes. They
    were each building their own -- three pictures, an application, a
    root and a scan -- which is a fifth of a second of setup to answer a
    question that costs a hundredth. The snapshot restores in a
    millisecond and isolates exactly the same.
    """
    _stage.restore()
    return _stage.client


def _remembered(client) -> list[dict]:
    return client.get("/views", headers={"accept": "application/json"}).json()


def test_remembering_a_question_makes_no_collection(gallery):
    """The whole point. A remembered question is a bookmark, not a
    container, and it must not appear where containers are listed."""
    told = gallery.post("/views", json={"name": "portraits from June", "qs": "?kind=image&sort=moment"})
    assert told.status_code in (200, 201), told.text

    assert [one["name"] for one in _remembered(gallery)] == ["portraits from June"]
    shelf = gallery.get("/albums", headers={"accept": "application/json"}).json()
    assert shelf == [], "remembering a question put something on the album shelf"


def test_it_opens_the_question_again(gallery):
    """What it is FOR. The stored spelling is an address, and the address
    is the question."""
    gallery.post("/views", json={"name": "just pictures", "qs": "?kind=image"})
    held = _remembered(gallery)[0]
    assert held["qs"] == "kind=image"

    answered = gallery.get(f"/g?{held['qs']}", headers={"accept": "text/html"})
    assert answered.status_code == 200
    assert "data-grid" in answered.text


def test_the_page_it_opens_at_is_never_page_seven(gallery):
    """A remembered question opens at its beginning. Stored with a page,
    it would open at page 7 of an answer that has since changed length --
    which is a different set of pictures, or none."""
    gallery.post("/views", json={"name": "deep", "qs": "?kind=image&page=7&sort=moment"})
    assert _remembered(gallery)[0]["qs"] == "kind=image&sort=moment"


def test_the_same_name_refines_rather_than_collides(gallery):
    """Somebody typing a name they have used before is refining the
    question, not colliding with themselves -- and a refusal at that
    moment costs them the question they had just composed."""
    gallery.post("/views", json={"name": "June", "qs": "?kind=image"})
    gallery.post("/views", json={"name": "june", "qs": "?kind=video"})

    held = _remembered(gallery)
    assert len(held) == 1, "two names differing only by case became two questions"
    assert held[0]["qs"] == "kind=video"


def test_a_question_with_no_name_is_refused(gallery):
    assert gallery.post("/views", json={"name": "   ", "qs": "?kind=image"}).status_code == 400


def test_the_list_puts_what_somebody_uses_at_the_top(gallery):
    """Used, not created. The order questions were invented in says
    nothing about which one somebody keeps coming back to."""
    first = gallery.post("/views", json={"name": "first", "qs": "?kind=image"}).json()
    gallery.post("/views", json={"name": "second", "qs": "?kind=video"})
    assert [one["name"] for one in _remembered(gallery)] == ["second", "first"]

    gallery.post(f"/views/{first['id']}/opened")
    assert [one["name"] for one in _remembered(gallery)] == ["first", "second"]


def test_forgetting_one_is_a_refusal_the_second_time(gallery):
    """A second press is not a quiet success: it means the thing being
    pressed is gone, and saying so is how somebody finds out."""
    made = gallery.post("/views", json={"name": "briefly", "qs": "?kind=image"}).json()
    assert gallery.post(f"/views/{made['id']}/forget").status_code == 204
    assert _remembered(gallery) == []
    assert gallery.post(f"/views/{made['id']}/forget").status_code == 404


def test_the_gallery_offers_all_three_and_names_them_apart(gallery):
    """The tell this entry was written about: the only offer was "save
    view", and it made a collection."""
    gallery.post("/views", json={"name": "portraits", "qs": "?kind=image"})
    page = gallery.get("/g", headers={"accept": "text/html"}).text

    assert "data-remember-view" in page, "no way to remember a question"
    assert "data-save-smart" in page, "no way to make a smart collection"
    assert ">save view<" not in page, "the button that made a collection still calls itself save view"
    assert "data-remembered-view=" in page, "the remembered questions are not on the page"
    assert "portraits" in page


def test_a_remembered_question_carries_no_membership(gallery):
    """It has no members, and the table it lives in has no room for
    any. That is what makes it a different product object rather than a
    collection with its features unused."""
    conn = connect.connect(gallery.app.state.db_path)
    try:
        views.remember(conn, "anything", "?kind=image", 0.0)
        conn.commit()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(saved_view)")}
    finally:
        connect.close(conn)
    assert columns == {"id", "name", "qs", "created_at", "last_used_at"}
