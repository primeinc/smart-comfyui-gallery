""" "Sarah at the beach" is what somebody types, and it cannot work.

The semantic ranking is over image embeddings, and the text encoder has
never heard of Sarah and never will. No amount of captioning fixes that:
teaching the caption model a name would put the name in prose the
encoder still cannot rank on, and would bake an authored claim into a
derived sentence that goes stale the moment somebody renames her.

What does work is the question splitting into a person -- a filter this
vocabulary already has, over a claim a human signed -- plus the phrase
that is left, which is the part a text search is actually good at.

Offered, never applied. This application says what a question is with
chips somebody can see, and silently rewriting a typed question is how
a person stops trusting the box they typed it into.
"""

from __future__ import annotations

import time as clock

import pytest
from PIL import Image

from db import authored, connect, discovery, naming
from tests.staging import staged

pytestmark = pytest.mark.slow


def _write(root):
    for i in range(2):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with staged(tmp_path_factory, "name-split", _write) as stage:
        yield stage


@pytest.fixture
def library(_world):
    """One served world per module instead of one boot per test: every
    test writes its own people, and the restore takes them back out."""
    _world.restore()
    return _world.client


def _people(client, *names: str | None) -> dict[str, str]:
    conn = connect.connect(client.app.state.db_path)
    try:
        held = {}
        for name in names:
            who = authored.person(conn, name, clock.time())
            slug = naming.entity_slug(conn, who)
            assert slug is not None
            held[name or slug[1]] = slug[1]
        conn.commit()
        return held
    finally:
        connect.close(conn)


def _split(client, phrase: str):
    conn = connect.connect(client.app.state.db_path)
    try:
        return discovery.person_in(conn, phrase)
    finally:
        connect.close(conn)


def test_a_name_in_the_phrase_is_noticed(library):
    """The whole thing. The name comes out, the rest stays a phrase."""
    _people(library, "Sarah")
    assert _split(library, "Sarah at the beach") == ("Sarah", "sarah", "at the beach")


def test_it_is_offered_and_not_applied(library):
    """Rewriting a typed question silently is how somebody stops
    trusting the box. The page says a split is available and the answer
    on screen is still the one that was asked for."""
    _people(library, "Sarah")
    page = library.get("/g", params={"q": "Sarah at the beach"}, headers={"accept": "text/html"}).text

    assert 'data-split-person="sarah"' in page, "the split was not offered"
    assert "data-split-apply" in page
    # and the question itself is unchanged: still the phrase, no person
    assert 'value="Sarah at the beach"' in page or "Sarah at the beach" in page


def test_the_offer_asks_the_split_question(library):
    """One click, and the question becomes a person and a phrase."""
    _people(library, "Sarah")
    conn = connect.connect(library.app.state.db_path)
    try:
        held = discovery.person_in(conn, "Sarah at the beach")
    finally:
        connect.close(conn)
    assert held is not None
    _, slug, rest = held
    answered = library.get("/g", params={"person": slug, "q": rest}, headers={"accept": "text/html"})
    assert answered.status_code == 200


def test_a_name_inside_a_word_is_not_a_name(library):
    """`Ana` must not match `banana`. A suggestion nobody can explain is
    worse than no suggestion."""
    _people(library, "Ana")
    assert _split(library, "banana bread") is None
    assert _split(library, "Ana in the kitchen") is not None


def test_the_longest_name_wins(library):
    """With both `Ana` and `Ana Torres` in the library, "ana torres at
    the beach" means the second -- offering the first would leave
    "torres at the beach" as a phrase, which is a worse answer arrived
    at more confidently."""
    _people(library, "Ana", "Ana Torres")
    held = _split(library, "ana torres at the beach")
    assert held is not None
    assert held[0] == "Ana Torres"
    assert held[2] == "at the beach"


def test_case_does_not_matter(library):
    _people(library, "Sarah")
    assert _split(library, "sarah at the beach") is not None
    assert _split(library, "SARAH AT THE BEACH") is not None


def test_an_unnamed_cluster_is_never_matched(library):
    """A placeholder's slug is `person-<short-id>`, which nobody types.
    Matching one would be matching an accident."""
    held = _people(library, None)
    slug = next(iter(held.values()))
    assert _split(library, slug) is None
    assert _split(library, "person") is None


def test_a_phrase_that_is_only_a_name_offers_their_pictures(library):
    """With nothing left to rank, ordering by similarity to an empty
    phrase is not a thing -- so the offer is simply their pictures."""
    _people(library, "Sarah")
    held = _split(library, "Sarah")
    assert held == ("Sarah", "sarah", "")

    page = library.get("/g", params={"q": "Sarah"}, headers={"accept": "text/html"}).text
    assert "see their pictures" in page
    assert "sort=similarity" not in page.split("data-split-apply")[0][-400:]


def test_a_question_that_already_says_who_is_not_offered_a_person(library):
    """The offer answers "you cannot rank on a name". Somebody who has
    already said the name as a filter is not making that mistake."""
    _people(library, "Sarah")
    page = library.get("/g", params={"person": "sarah", "q": "at the beach"}, headers={"accept": "text/html"}).text
    assert "data-split-person" not in page


def test_a_name_with_punctuation_in_it_does_not_break_the_match(library):
    """A person's name is somebody else's text. It can hold a bracket, a
    dot or a plus, and a version of this built on a regex is one
    forgotten escape from a name that matches everything or raises."""
    _people(library, "J. R. R.")
    assert _split(library, "J. R. R. at the beach") is not None
    _people(library, "C++")
    assert _split(library, "C++ on a laptop") is not None
