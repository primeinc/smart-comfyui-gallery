"""Clicking a heading reorders the ANSWER, not the rows on screen.

The distinction is the whole entry. A table showing sixty of four
hundred rows can be shuffled locally in one line of JavaScript, and it
will be wrong: page two is still the old order, the pager still counts
the old order, and the sixty biggest files in the library are not the
sixty on this page sorted by size. So a heading is a LINK carrying the
reordered question, the ResultSet answers it, and reload, Back and a
shared link all land on the same order for the same reason every other
filter does.

Which puts the work in `db/resultset.py`: a sort is a closed vocabulary,
each with one implementation, and each must be TOTAL -- the column then
`f.id` in the same direction. Two files of equal size that swap between
two reads of one answer are an ordinal that moves, which is a filmstrip
that walks somewhere else and a page that shows a picture twice.

And every column here can be NULL, which is the honest part. A sound has
no pixels; a photograph has no length. Those files sort last and say so
by position -- dropping them would misreport what the answer holds, and
calling them zero would invent a fact.
"""

from __future__ import annotations

import urllib.parse
import uuid

import pytest

from db import resultset
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0

#: Deliberately awkward: names that sort differently by case, sizes that
#: tie, one file with no pixels and one with no length.
FILES = [
    # name, kind, size, width, height, duration
    ("beta.png", "image", 300, 40, 10, None),
    ("Alpha.png", "image", 300, 10, 10, None),
    ("gamma.mp4", "video", 100, 20, 20, 4.0),
    ("delta.mp3", "audio", 900, None, None, 90.0),
    ("epsilon.png", "image", 50, 100, 100, None),
]


@pytest.fixture
def library():
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")
    for at, (name, kind, size, width, height, duration) in enumerate(FILES, start=2):
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, uuid.uuid4().bytes, f"f{at}"))
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,width,height,duration,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,?,?,?,?,?,?,0,0)",
            (at, name, kind, size, NOW + at, width, height, duration),
        )
    conn.commit()
    yield conn
    conn.close()


def _names(conn, sort: str) -> list[str]:
    told = resultset.page(conn, "", resultset.parse(sort=sort), 1, NOW)
    return [one["name"] for one in told["items"]]


def test_a_column_orders_the_whole_answer(library):
    assert _names(library, "name") == ["Alpha.png", "beta.png", "delta.mp3", "epsilon.png", "gamma.mp4"]
    assert _names(library, "name-desc") == ["gamma.mp4", "epsilon.png", "delta.mp3", "beta.png", "Alpha.png"]


def test_a_name_sorts_the_way_a_person_reads_it(library):
    """`Alpha` before `beta`, which byte order would not give: upper
    case sorts before every lower-case letter, so a case-sensitive
    column puts every capitalised name in its own block above the rest
    and reads as broken."""
    assert _names(library, "name")[0] == "Alpha.png"


def test_a_number_column_orders_by_the_number(library):
    assert _names(library, "size-desc")[0] == "delta.mp3"
    assert _names(library, "size")[0] == "epsilon.png"


def test_a_file_with_no_value_sorts_last_in_both_directions(library):
    """Last, never dropped and never zero. `delta.mp3` has no pixels and
    the three stills have no length; both facts are true of the answer
    and the position is how the table says so."""
    assert _names(library, "pixels")[-1] == "delta.mp3"
    assert _names(library, "pixels-desc")[-1] == "delta.mp3"

    timed = _names(library, "length")
    assert timed[:2] == ["gamma.mp4", "delta.mp3"], timed
    assert set(timed[2:]) == {"Alpha.png", "beta.png", "epsilon.png"}
    assert len(timed) == len(FILES), "a file with no length was dropped from the answer"


def test_a_tie_is_broken_the_same_way_every_time(library):
    """The ORDERING CONTRACT. `beta.png` and `Alpha.png` are both 300
    bytes, so the column alone leaves them unordered -- and an answer
    whose ordinals move between two reads is a filmstrip that walks
    somewhere else and a page that shows a picture twice."""
    once = _names(library, "size-desc")
    for _ in range(4):
        assert _names(library, "size-desc") == once

    # and the tiebreak runs the SAME WAY as the column, so reversing the
    # sort reverses the tied pair too rather than leaving it stuck
    tied_down = [one for one in _names(library, "size-desc") if one in ("beta.png", "Alpha.png")]
    tied_up = [one for one in _names(library, "size") if one in ("beta.png", "Alpha.png")]
    assert tied_down == list(reversed(tied_up)), (tied_down, tied_up)


def test_every_column_sort_is_a_real_answer(library):
    """The registry and the implementation cannot drift: a sort that
    parses and then falls through to `newest` would be a heading that
    silently does nothing."""
    for sort in resultset.COLUMN_SORTS:
        held = _names(library, sort)
        assert len(held) == len(FILES), sort
    assert set(resultset.COLUMN_SORTS) <= set(resultset.SORTS)


def test_the_sort_rides_the_canonical_spelling(library):
    """It is part of the QUESTION, so it survives a round trip -- which
    is what makes a shared link land on the order it was shared in."""
    asked = resultset.parse(sort="pixels-desc")
    spelled = resultset.canonical(asked)
    assert "sort=pixels-desc" in spelled
    # back through the same door a URL comes in by
    held = urllib.parse.parse_qs(spelled)
    assert resultset.parse(sort=held["sort"][0]).sort == "pixels-desc"


def test_a_sort_nobody_implements_is_refused(library):
    """A closed vocabulary, refused at the door: a typo in a URL must be
    an error where it was made, never a silent fall back to newest."""
    with pytest.raises(ValueError, match="sort must be one of"):
        resultset.parse(sort="pixels-descending")
