"""Files sharing a name is how libraries are ORGANISED, not an edge case.

A cover.png in every album folder; two camera cards both numbering from
DSC00001; a folder.jpg per artist. Every one of those slugifies to the
same seed, and `entity.slug` must still be unique -- so `mint` picks
`cover`, then `cover-2`, then `cover-3`.

It used to find that number by asking "is `cover-2` taken? is `cover-3`
taken?" one SELECT at a time, which costs n reads for the nth file and
n^2 for the library. Measured on this tree before the fix: 4,000 files
all called cover.png held SQLite's one write lane for 16.8 seconds
against 0.6 for 4,000 distinct names, and the total quadrupled with
every doubling -- so twenty thousand albums was minutes of held lane,
and SQLite has exactly one write lane per database. Every write from a
route during that window does not wait, it fails: `busy_timeout` is
5000 ms.

Counted rather than timed, on purpose. A timing assertion on a shape
like this measures the machine as much as the code, and a gate that
fails one run in three teaches people to re-run it. Statements executed
is the thing that actually changed, it is exact, and it is the same
number on a fast machine and a slow one.
"""

from __future__ import annotations

import pytest

from db import scan
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow


def _counted(conn):
    """Statements SQLite was handed, as a list the caller can read."""
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    return seen


def test_the_slugs_are_what_they_have_always_been():
    """The numbering is not being changed, only found faster: the first
    keeps the bare seed and the rest count up from two, with no gaps."""
    conn = fresh_schema()
    try:
        for _ in range(6):
            scan.mint(conn, "file", "cover")
        held = [row[0] for row in conn.execute("SELECT slug FROM entity WHERE kind='file' ORDER BY id")]
    finally:
        conn.close()
    assert held == ["cover", "cover-2", "cover-3", "cover-4", "cover-5", "cover-6"]


def test_a_thousand_files_of_one_name_do_not_cost_a_million_reads():
    """The defect itself. Bounded per file rather than growing with the
    library -- the doubling-and-bisecting search is O(log n) reads, so
    the generous ceiling here still fails by two orders of magnitude on
    anything quadratic."""
    conn = fresh_schema()
    n = 1_000
    try:
        seen = _counted(conn)
        for _ in range(n):
            scan.mint(conn, "file", "cover")
        conn.set_trace_callback(None)
        slugs = conn.execute("SELECT count(DISTINCT slug) FROM entity WHERE kind='file'").fetchone()[0]
    finally:
        conn.close()

    assert slugs == n, "the whole point of the suffix is that every one is different"
    # the old probe cost ~n/2 reads for the nth file: about 500,000 here.
    assert len(seen) < 60 * n, f"{len(seen):,} statements for {n:,} files is not a bounded search"


def test_the_cost_per_file_does_not_grow_with_the_library():
    """Bounded is the claim, so it is measured as a RATIO of counts at
    two sizes rather than as an absolute anybody would have to re-tune.
    Quadratic would double this; O(log n) barely moves it."""
    counts = {}
    for n in (500, 2_000):
        conn = fresh_schema()
        try:
            seen = _counted(conn)
            for _ in range(n):
                scan.mint(conn, "file", "cover")
            conn.set_trace_callback(None)
        finally:
            conn.close()
        counts[n] = len(seen) / n

    grew = counts[2_000] / counts[500]
    assert grew < 1.6, f"reads per file grew {grew:.2f}x from 500 files to 2,000 ({counts})"


def test_a_retired_slug_is_not_handed_to_a_new_picture():
    """`slug_history` answers a retired address on a miss, so reusing one
    would make somebody's saved link resolve to a different picture. A
    gap in the numbering stays a gap."""
    conn = fresh_schema()
    try:
        made = [scan.mint(conn, "file", "cover") for _ in range(4)]
        conn.execute("DELETE FROM entity WHERE id = ?", (made[2],))  # cover-3
        again = scan.mint(conn, "file", "cover")
        held = conn.execute("SELECT slug FROM entity WHERE id = ?", (again,)).fetchone()[0]
    finally:
        conn.close()
    assert held != "cover-3", "a retired address came back on a different picture"
    assert held == "cover-5"


def test_a_neighbour_whose_suffix_is_not_a_number_is_not_confused_for_one():
    """Somebody's file really can be called `cover-x.png`. It takes the
    slug `cover-x`, which shares the prefix and is not a numbering."""
    conn = fresh_schema()
    try:
        scan.mint(conn, "file", "cover")
        scan.mint(conn, "file", "cover-x")
        scan.mint(conn, "file", "cover")
        held = [row[0] for row in conn.execute("SELECT slug FROM entity WHERE kind='file' ORDER BY id")]
    finally:
        conn.close()
    assert held == ["cover", "cover-x", "cover-2"]
    assert len(set(held)) == 3


def test_the_kinds_do_not_share_a_numbering():
    """`UNIQUE (kind, slug)` is per kind, so a folder called cover must
    not push the files along -- and must not collide either."""
    conn = fresh_schema()
    try:
        scan.mint(conn, "folder", "cover")
        scan.mint(conn, "folder", "cover")
        scan.mint(conn, "file", "cover")
        held = {
            kind: [row[1] for row in conn.execute("SELECT kind, slug FROM entity WHERE kind = ? ORDER BY id", (kind,))]
            for kind in ("file", "folder")
        }
    finally:
        conn.close()
    assert held["folder"] == ["cover", "cover-2"]
    assert held["file"] == ["cover"], "a folder's name used up a file's address"
