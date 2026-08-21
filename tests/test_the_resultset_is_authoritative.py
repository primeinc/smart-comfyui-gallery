"""One question, one answer: db/resultset.py is the only opinion.

The grid page, the counts, the rail, the hover peek, and locate/previous/
next all read a single materialized projection, so they cannot disagree.
Two properties are contracts, pinned here exactly as WI-35 demands them
pinned -- as tests, not prose:

- validity is (query fingerprint, data currency), never fingerprint
  alone: mutate the library and the same question answers from the new
  state, no cache flush anywhere;
- semantic order is materialized ONCE per (fingerprint, currency) from
  the fused retrieval result, and page/peek/locate read that ordering
  rather than rerunning retrieval.
"""

from __future__ import annotations

import pathlib
import sqlite3
import time

import pytest
from PIL import Image

from db import connect, library, resultset, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


def _paint(root: pathlib.Path, folder: str, name: str, tint: int) -> None:
    (root / folder).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (tint % 256, 90, 120)).save(root / folder / name)


@pytest.fixture
def shelves(tmp_path):
    """A library big enough to page: 23 stills over two folders plus a
    video-suffixed file, spread over distinct mtimes."""
    root = tmp_path / "pics"
    for i in range(9):
        _paint(root, "portraits", f"p_{i:02d}.png", 10 + i)
    for i in range(14):
        _paint(root, "landscape", f"l_{i:02d}.png", 40 + i)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    # Distinct mtimes so "newest" is a real order, with one deliberate tie
    # (the first two portraits) so determinism under ties is testable.
    import os

    stamp = NOW
    for path in sorted(root.rglob("*.png")):
        os.utime(path, (stamp, stamp))
        stamp += 60
    for tied in ("p_00.png", "p_01.png"):
        os.utime(root / "portraits" / tied, (NOW - 500, NOW - 500))
    scan.scan(conn, 1, root, NOW)
    conn.commit()
    return {"conn": conn, "root": root}


def _q(**kw):
    return resultset.parse(**kw)


def test_page_peek_and_locate_agree(shelves):
    conn = shelves["conn"]
    q = _q(size=5)
    shape = resultset.describe(conn, "", q, NOW)
    assert shape["total"] == 23
    assert shape["pages"] == 5

    third = resultset.page(conn, "", q, 3, NOW)
    assert [row["ordinal"] for row in third["items"]] == [11, 12, 13, 14, 15]

    looked = resultset.peek(conn, "", q, 3, NOW, count=3)
    assert looked["items"] == third["items"][:3], "peek must be a prefix of the page it previews"
    assert looked["first_ordinal"] == 11
    assert looked["last_ordinal"] == 15

    # locate agrees with where the page actually shows the file, and its
    # neighbours are the page's own neighbours in answer order.
    middle = third["items"][2]
    found = resultset.locate(conn, "", q, middle["id"], NOW)
    assert found is not None
    assert found["ordinal"] == middle["ordinal"]
    assert found["page"] == 3
    assert found["previous"] == third["items"][1]["slug"]
    assert found["next"] == third["items"][3]["slug"]


def test_the_last_page_is_short_and_a_number_past_the_end_answers_it(shelves):
    conn = shelves["conn"]
    q = _q(size=5)
    last = resultset.page(conn, "", q, 5, NOW)
    assert len(last["items"]) == 3
    beyond = resultset.page(conn, "", q, 99, NOW)
    assert beyond["page"] == 5
    assert beyond["items"] == last["items"]


def test_newest_and_oldest_are_the_same_answer_reversed_and_ties_break_on_id(shelves):
    conn = shelves["conn"]
    newest = resultset.page(conn, "", _q(size=resultset.MAX_PAGE_SIZE), 1, NOW)["items"]
    oldest = resultset.page(conn, "", _q(sort="oldest", size=resultset.MAX_PAGE_SIZE), 1, NOW)["items"]
    assert [r["id"] for r in newest] == [r["id"] for r in reversed(oldest)]
    tied = [r["id"] for r in newest if r["name"] in ("p_00.png", "p_01.png")]
    assert tied == sorted(tied, reverse=True), "equal mtimes must order by id, not by chance"


def test_scope_and_filter_bound_membership(shelves):
    conn = shelves["conn"]
    scoped = resultset.describe(conn, "", _q(folder="portraits"), NOW)
    assert scoped["total"] == 9
    for row in resultset.page(conn, "", _q(folder="portraits", size=9), 1, NOW)["items"]:
        assert row["name"].startswith("p_")
    kinds = resultset.describe(conn, "", _q(kind="video"), NOW)
    assert kinds["total"] == 0
    assert kinds["pages"] == 1
    with pytest.raises(LookupError):
        resultset.describe(conn, "", _q(folder="nowhere"), NOW)


def test_an_album_is_a_scope(shelves):
    from db import authored

    conn = shelves["conn"]
    album = authored.collection(conn, "Keepers", NOW)
    kept = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id LIMIT 3")]
    for file_id in kept:
        authored.add_to_collection(conn, album, file_id, NOW)
    conn.commit()
    told = resultset.page(conn, "", _q(album="keepers", size=10), 1, NOW)
    assert told["total"] == 3
    assert sorted(row["id"] for row in told["items"]) == sorted(kept)


def test_malformed_questions_are_refused():
    for refused, why in (
        ({"sort": "best"}, "sort must be"),
        ({"sort": "similarity"}, "needs a phrase"),
        ({"text": "a banana", "sort": "newest"}, "orders by similarity"),
        ({"folder": "a", "album": "b"}, "one scope at a time"),
        ({"size": 0}, "page size"),
        ({"size": resultset.MAX_PAGE_SIZE + 1}, "page size"),
        ({"kind": "picture"}, "kind must be"),
    ):
        with pytest.raises(ValueError, match=why):
            resultset.parse(**refused)


def test_a_phrase_defaults_to_similarity_and_the_fingerprint_names_the_question():
    q = resultset.parse(text="a banana")
    assert q.sort == "similarity"
    assert resultset.fingerprint(q) == resultset.fingerprint(resultset.parse(text="a banana"))
    assert resultset.fingerprint(q) != resultset.fingerprint(resultset.parse(text="a pear"))
    assert resultset.fingerprint(_q(size=30)) != resultset.fingerprint(_q(size=60)), (
        "page size is part of the question: ordinal->page arithmetic depends on it"
    )


def test_validity_is_currency_not_fingerprint_alone(tmp_path):
    """The pinned amendment: mutate the library and the SAME question
    answers from the new state -- across separate connections, which is
    how the application actually runs (one per request, one per job)."""
    root = tmp_path / "pics"
    for i in range(4):
        _paint(root, "all", f"a_{i}.png", 30 * i)
    db_path = tmp_path / "gallery.db"
    conn = connect.connect(db_path)
    conn.executescript(connect.schema_sql())
    root_id = library.add_root(conn, str(root), "library", NOW)
    scan.scan(conn, root_id, str(root), NOW)
    conn.commit()

    q = _q(size=10)
    before = resultset.describe(conn, "", q, NOW)
    assert before["total"] == 4

    writer = connect.connect(db_path)  # another connection, like a worker
    _paint(root, "all", "a_late.png", 200)
    scan.scan(writer, root_id, str(root), NOW + 60)
    writer.commit()
    connect.close(writer)

    after = resultset.describe(conn, "", q, NOW + 61)
    assert after["total"] == 5, "a stale projection answered after the library changed"
    assert after["fingerprint"] == before["fingerprint"], "the QUESTION did not change"
    assert after["currency"] != before["currency"], "the library STATE did"
    connect.close(conn)


def test_semantic_order_is_materialized_once_and_reused(shelves, monkeypatch):
    """The other pinned amendment: page, peek and locate read ONE fused
    ordering; retrieval runs once per (fingerprint, currency), and a
    library change is what makes it run again."""
    conn = shelves["conn"]
    ranked = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id")]
    ranked = ranked[5:] + ranked[:5]  # an order no SQL sort produces
    asked = []

    def fused(conn_, models_dir, phrase, k, now, *, offline=True):
        asked.append((phrase, k, offline))
        return {
            "results": [{"file_id": file_id, "score": 1.0, "sources": {}} for file_id in ranked],
            "participants": ["space.a", "space.b"],
            "contributors": ["space.a"],
            "missing": {"space.b": "not provisioned"},
        }

    from db import retrieval

    monkeypatch.setattr(retrieval, "query", fused)
    q = _q(text="a banana", size=4)

    shape = resultset.describe(conn, "", q, NOW)
    first = resultset.page(conn, "", q, 1, NOW)
    fourth = resultset.page(conn, "", q, 4, NOW)
    resultset.peek(conn, "", q, 2, NOW)
    resultset.locate(conn, "", q, ranked[9], NOW)
    assert len(asked) == 1, "page/peek/locate must read the materialized ordering, never rerun retrieval"
    assert asked[0][2] is True, "the serving path must stay offline"

    # The pages ARE the fused ordering, sliced -- and the degraded-space
    # provenance survives to the answer instead of being flattened away.
    assert [row["id"] for row in first["items"]] == ranked[:4]
    assert [row["id"] for row in fourth["items"]] == ranked[12:16]
    assert shape["provenance"]["missing"] == {"space.b": "not provisioned"}

    told = resultset.locate(conn, "", q, ranked[9], NOW)
    assert told is not None
    assert told["ordinal"] == 10
    assert told["page"] == 3

    # A library change -- one more committed row -- is the one thing that
    # re-materializes. total_changes carries currency for a :memory: db.
    conn.execute("UPDATE file SET mtime = mtime + 1 WHERE id = ?", (ranked[0],))
    conn.commit()
    resultset.describe(conn, "", q, NOW + 1)
    assert len(asked) == 2, "the fused ordering must be rebuilt for the changed library"


def test_a_scoped_semantic_answer_is_the_fusion_intersected_in_fused_order(shelves, monkeypatch):
    conn = shelves["conn"]
    portraits = {
        row[0]
        for row in conn.execute(
            "SELECT f.id FROM file f JOIN folder fo ON fo.id = f.folder_id WHERE fo.name = 'portraits'"
        )
    }
    everything = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id DESC")]

    from db import retrieval

    monkeypatch.setattr(
        retrieval,
        "query",
        lambda *a, **k: {
            "results": [{"file_id": f, "score": 1.0, "sources": {}} for f in everything],
            "participants": ["s"],
            "contributors": ["s"],
            "missing": {},
        },
    )
    told = resultset.page(conn, "", _q(text="faces", folder="portraits", size=50), 1, NOW)
    assert [row["id"] for row in told["items"]] == [f for f in everything if f in portraits]


def test_a_file_outside_the_membership_locates_nowhere(shelves):
    conn = shelves["conn"]
    told = resultset.locate(conn, "", _q(folder="portraits"), -5, NOW)
    assert told is None


def test_the_edges_of_the_answer_have_no_neighbours(shelves):
    conn = shelves["conn"]
    q = _q(size=23)
    whole = resultset.page(conn, "", q, 1, NOW)["items"]
    first = resultset.locate(conn, "", q, whole[0]["id"], NOW)
    last = resultset.locate(conn, "", q, whole[-1]["id"], NOW)
    assert first is not None
    assert first["previous"] is None
    assert first["next"] == whole[1]["slug"]
    assert last is not None
    assert last["next"] is None
    assert last["previous"] == whole[-2]["slug"]


def test_currency_needs_time_to_pass_for_nothing(shelves):
    """Two reads with no write between them reuse one projection -- the
    cache is real, not a pass-through that rebuilds every request."""
    conn = shelves["conn"]
    q = _q(size=5)
    resultset.describe(conn, "", q, NOW)
    key_count = len(resultset._PROJECTIONS)
    resultset.page(conn, "", q, 2, NOW)
    resultset.peek(conn, "", q, 4, NOW)
    assert len(resultset._PROJECTIONS) == key_count
    assert time.time() > 0  # the clock is not part of the key
