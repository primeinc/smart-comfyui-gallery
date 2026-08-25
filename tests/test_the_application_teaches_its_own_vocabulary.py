"""You type what you half-remember; it tells you what it knows.

The filter drawer offered a curated vocabulary and, behind an "advanced"
heading, a text box whose placeholder was `key=value`. That box could
only be used by somebody who already knew the internal spelling -- which
is the one thing the application is supposed to remember for them, and
the reason `db/catalog.py` exists.

One list over two registries: 41 curated dimensions this application
understands well enough to name, and every metadata key any tool
happened to write. The split stays real inside -- a curated dimension
carries operators and a value list, a discovered key is asked through
`param.is` -- and stops being visible.

What the tests below actually pin is the three things that make it
harder than a list, each measured on a real library before it was
written:

    indexed families    `used_wildcards.0..6` is ONE question
    ranking             ~40 of 108 keys are a camera's own plumbing
    observed type       everything is TEXT; `param_key` already knows
"""

from __future__ import annotations

import uuid

import pytest

from db import catalog, resultset
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0


def _file(conn, at: int) -> int:
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, uuid.uuid4().bytes, f"f{at}"))
    conn.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at) VALUES(?,1,?,'image',1,?,0,0)",
        (at, f"f{at}.png", NOW + at),
    )
    return at


@pytest.fixture
def library():
    """Twelve pictures carrying the awkward shapes on purpose.

    Every kind of uselessness a real library has: a key every file
    shares one value of, a key whose every file has its own value, a
    positional family, and a camera's plumbing.
    """
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")

    def param(file_id, source, key, text, num=None):
        conn.execute(
            "INSERT INTO file_param(file_id,source,key,value_text,value_num) VALUES(?,?,?,?,?)",
            (file_id, source, key, text, num),
        )

    for at in range(2, 14):
        _file(conn, at)
        # a camera's plumbing: every file, one value between them all
        param(at, "exif", "YCbCrPositioning", "1", 1.0)
        # discriminating: four values across twelve files
        param(at, "generation", "sampler", f"euler-{at % 4}")
        # a lookup, not a filter: every file its own value
        param(at, "generation", "seed", str(1000 + at), float(1000 + at))
        # text that never parses as a number, on every file
        param(at, "generation", "automaticvae", "True")
        # a positional family on half of them
        if at % 2 == 0:
            for slot in range(3):
                param(at, "generation", f"used_wildcards.{slot}", f"w{slot}-{at % 3}")
    conn.commit()
    yield conn
    conn.close()


def _by_param(found, param: str):
    return next((one for one in found if one.param == param), None)


def _need(found, param: str):
    """The field, or a failure naming what WAS offered -- an assertion
    that dies on `None.value_kind` says nothing about which fields the
    catalog actually held."""
    one = _by_param(found, param)
    assert one is not None, f"{param!r} was not offered; got {sorted(str(each.param) for each in found)}"
    return one


def test_a_positional_family_is_one_question(library):
    """`used_wildcards.0` through `.2` is one concept wearing three
    names. Asked separately, "did this use a wildcard" is three
    questions -- which is what the schema's positional flattening left
    behind, and what the collapse is for."""
    found, _ = catalog.catalog(library, resultset.parse(), most=200)
    family = _need(found, "used_wildcards")
    assert family.repeats == 3, "the family did not collapse"
    assert family.multi == "any", "a family's repeats OR; ALL of them asks for nothing"
    assert not any(one.param == "used_wildcards.0" for one in found), "a member is still offered on its own"

    # coverage is the WIDEST member, never the sum: the same file
    # carries .0 and .1, and adding them would report more files than
    # the answer holds
    assert family.covered == 6, family.covered


def test_the_raw_spelling_still_finds_it(library):
    """The ugly key is a door, not the only door. Somebody who knows it
    types it and arrives; somebody who does not types words."""
    by_ugly, _ = catalog.catalog(library, resultset.parse(), search="used_wildcards.1")
    assert _by_param(by_ugly, "used_wildcards") is not None

    by_words, _ = catalog.catalog(library, resultset.parse(), search="wildcard")
    assert _by_param(by_words, "used_wildcards") is not None


def test_a_field_that_separates_nothing_ranks_below_one_that_does(library):
    """The whole reason this is ranked rather than listed. Every file
    here carries `YCbCrPositioning` and they all agree about it, so it
    cuts the answer into one piece; `sampler` cuts it into four."""
    found, _ = catalog.catalog(library, resultset.parse(), most=200)
    order = [one.param for one in found if one.param]
    assert "sampler" in order, order
    assert "YCbCrPositioning" in order, order
    assert order.index("sampler") < order.index("YCbCrPositioning")

    assert catalog.usefulness(12, 1, 12) == 0.0, "one value between every file separates nothing"
    assert catalog.usefulness(12, 4, 12) > catalog.usefulness(12, 400, 12), "a value per file is a lookup, not a filter"


def test_plumbing_is_ranked_down_and_never_hidden(library):
    """`StripOffsets` is a real fact about a real file and somebody
    debugging a camera import may ask about it. It must simply not be
    offered before the things people search for."""
    found, _ = catalog.catalog(library, resultset.parse(), most=200)
    assert _by_param(found, "YCbCrPositioning") is not None, "an exif key was hidden rather than ranked"
    exif = [one for one in found if one.group in catalog.PLUMBING]
    named = [one for one in found if one.curated]
    assert named, "the curated vocabulary is missing from the catalog"
    assert found.index(exif[0]) > found.index(named[-1]), "plumbing outranked a fact we named"


def test_the_type_is_read_rather_than_guessed(library):
    """`automaticvae` is the string 'True' with a NULL `value_num`, so
    it is honestly text and gets a value list rather than a number
    control. `param_key.value_kind` already knows -- a trigger keeps it
    over a lattice that only widens -- so nothing here infers it."""
    found, _ = catalog.catalog(library, resultset.parse(), most=200)
    assert _need(found, "automaticvae").value_kind == "text"
    assert _need(found, "seed").value_kind == "number"


def test_a_curated_dimension_and_a_discovered_key_are_one_list(library):
    """The split is real inside and invisible outside: both arrive as
    fields with a label, operators and a way to be spelled into a URL."""
    found, _ = catalog.catalog(library, resultset.parse(), search="rating")
    rating = next(one for one in found if one.key == "rating_min")
    assert rating.curated is True
    assert rating.param is None, "a curated dimension's key IS its address"

    found, _ = catalog.catalog(library, resultset.parse(), search="sampler")
    sampler = _need(found, "sampler")
    assert sampler.curated is False
    assert sampler.key == "param.is", "a discovered key is asked through the long-tail door"


def test_a_key_no_file_here_carries_is_not_offered(library):
    """Counted WITH the whole question. The catalog answers "what else
    is true about what I am looking at", so a key that exists in the
    library and in none of this answer is not something to offer."""
    library.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(2,'generation','only_on_two','yes')")
    library.commit()
    whole, _ = catalog.catalog(library, resultset.parse(), most=200)
    assert _by_param(whole, "only_on_two") is not None

    # Every file here is an image, so a question about videos answers
    # with nothing -- and a catalog for it must offer no metadata key at
    # all, however many the library holds.
    narrow, _ = catalog.catalog(library, resultset.parse(kind="video"), most=200)
    assert _by_param(narrow, "only_on_two") is None
    assert not any(one.param for one in narrow), "keys were offered for an answer that holds no files"
    assert any(one.curated for one in narrow), "the curated vocabulary went away with them"


def test_the_list_is_bounded_and_says_how_much_it_cut(library):
    """A truncated list that does not say so reads as a complete one,
    and then a field that IS in the library looks absent."""
    few, more = catalog.catalog(library, resultset.parse(), most=3)
    assert len(few) == 3
    assert more > 0

    everything, none_left = catalog.catalog(library, resultset.parse(), most=500)
    assert none_left == 0
    assert len(everything) > 3
