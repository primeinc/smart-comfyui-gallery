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
from typing import Any

import pytest
from PIL import Image

from db import collections, connect, library, resultset, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


def _paint(root: pathlib.Path, folder: str, name: str, tint: int) -> None:
    (root / folder).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (tint % 256, 90, 120)).save(root / folder / name)


@pytest.fixture(scope="module")
def _shelves(tmp_path_factory):
    """A library big enough to page: 23 stills over two folders plus a
    video-suffixed file, spread over distinct mtimes -- painted and
    scanned once; each test reads a copy of the scanned database."""
    tmp_path = tmp_path_factory.mktemp("shelves")
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
    yield {"conn": conn, "root": root}
    conn.close()


@pytest.fixture
def shelves(_shelves):
    copy = sqlite3.connect(":memory:")
    _shelves["conn"].backup(copy)
    copy.execute("PRAGMA foreign_keys=ON")
    yield {"conn": copy, "root": _shelves["root"]}
    copy.close()


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

    conn = shelves["conn"]
    album = collections.collection(conn, "Keepers", NOW)
    kept = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id LIMIT 3")]
    for file_id in kept:
        collections.set_membership(conn, album, file_id, True, NOW)
    conn.commit()
    told = resultset.page(conn, "", _q(album="keepers", size=10), 1, NOW)
    assert told["total"] == 3
    assert sorted(row["id"] for row in told["items"]) == sorted(kept)


def test_malformed_questions_are_refused():
    refused: dict[str, Any]
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

    def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
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


def test_the_scope_reaches_retrieval_as_the_allowed_set(shelves, monkeypatch):
    """The seam of the ownership split: this module decides WHICH files
    are eligible and hands retrieval the allowed set BEFORE the fusion;
    it never trims a fused answer afterwards -- RRF consumes rank
    positions, and post-fusion filtering keeps global ranks."""
    conn = shelves["conn"]
    portraits = {
        row[0]
        for row in conn.execute(
            "SELECT f.id FROM file f JOIN folder fo ON fo.id = f.folder_id WHERE fo.name = 'portraits'"
        )
    }
    seen: dict = {}

    def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        seen.update({"allowed": allowed, "k": k})
        members = sorted(allowed or (), reverse=True)
        return {
            "results": [{"file_id": f, "score": 1.0, "sources": {}} for f in members],
            "participants": ["s"],
            "contributors": ["s"],
            "missing": {},
        }

    from db import retrieval

    monkeypatch.setattr(retrieval, "query", fused)
    told = resultset.page(conn, "", _q(text="faces", folder="portraits", size=50), 1, NOW)
    assert seen["allowed"] == portraits, "the scope must constrain retrieval, not trim its answer"
    assert seen["k"] == len(portraits), "a scoped ranking is asked at full scope depth"
    assert [row["id"] for row in told["items"]] == sorted(portraits, reverse=True), (
        "the answer is retrieval's constrained ordering, untouched"
    )


def test_a_scope_constrains_each_space_before_the_fusion_not_after(tmp_path, monkeypatch):
    """The hostile geometry: two spaces whose out-of-scope candidates sit
    at different depths. Global-RRF-then-filter keeps global ranks and
    compresses the spaces unequally; constraining each space's ranking
    first renumbers 1..N in scope. The two answers must DISAGREE here --
    space one buries every in-scope file under five outsiders (A,B,C at
    global 6,7,8), space two splits them around its outsiders (C=1, B=2,
    A=8). Fused globally then filtered: C, B, A. Fused in scope:
    A(1/61+1/63) outranks B(1/62+1/62). B before A one way, A before B
    the other -- a refactor that reintroduces post-fusion filtering
    fails this by name."""
    import numpy as np

    from db import connect, derived, retrieval, scan, settings
    from vision import semantic

    root = tmp_path / "pics"
    names = ("x1", "x2", "x3", "x4", "x5", "a", "b", "c")
    for i, name in enumerate(names):
        _paint(root, "all", f"{name}.png", 20 * i)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    ids = {name: file_id for file_id, name in conn.execute("SELECT id, replace(name, '.png', '') FROM file")}
    shas = dict(conn.execute("SELECT id, content_sha256 FROM file"))

    def tilted(cosine: float, axis: int) -> np.ndarray:
        v = np.zeros(16, dtype=np.float32)
        v[0] = cosine
        v[axis] = np.sqrt(1.0 - cosine * cosine)
        return v

    clip = semantic.space("openclip", "ViT-B-32", "laion2b_s34b_b79k", 16)
    qwen = semantic.space("qwen", "Qwen/Qwen3-VL-Embedding-2B", "main", 16)
    by_clip = {"x1": 0.95, "x2": 0.94, "x3": 0.93, "x4": 0.92, "x5": 0.91, "a": 0.80, "b": 0.79, "c": 0.78}
    by_qwen = {"c": 0.95, "b": 0.94, "x1": 0.70, "x2": 0.69, "x3": 0.68, "x4": 0.67, "x5": 0.66, "a": 0.10}
    for spec, cosines in ((clip, by_clip), (qwen, by_qwen)):
        for axis, name in enumerate(names, start=1):
            derived.record_embedding(conn, ids[name], spec, tilted(cosines[name], axis), shas[ids[name]], NOW)
    conn.commit()

    class Asks:
        def encode_query(self, phrase):
            probe = np.zeros(16, dtype=np.float32)
            probe[0] = 1.0
            return probe

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Asks())
    settings.put(conn, "semantic_model", "ViT-B-32/laion2b_s34b_b79k, qwen:Qwen/Qwen3-VL-Embedding-2B")
    conn.commit()

    in_scope = {ids["a"], ids["b"], ids["c"]}
    unscoped = retrieval.query(conn, str(tmp_path), "the probe", 8, NOW, offline=True)
    trimmed = [row["file_id"] for row in unscoped["results"] if row["file_id"] in in_scope]
    scoped = retrieval.query(conn, str(tmp_path), "the probe", 3, NOW, offline=True, allowed=in_scope)
    constrained = [row["file_id"] for row in scoped["results"]]

    assert set(constrained) == in_scope
    assert trimmed.index(ids["b"]) < trimmed.index(ids["a"]), "the hostile geometry lost its teeth"
    assert constrained.index(ids["a"]) < constrained.index(ids["b"]), (
        "in scope, each space ranks A ahead of enough of B's advantage that A must win"
    )
    assert constrained != trimmed, "constraining before fusion must be able to disagree with trimming after it"
    connect.close(conn)


def test_one_response_reads_one_projection(shelves, monkeypatch):
    """One HTTP answer, one projection snapshot: every public operation
    takes `_current` exactly once, so items from one generation can
    never ship under the totals of another when a job commits mid-
    request."""
    conn = shelves["conn"]
    q = _q(size=5)
    anchor = resultset.page(conn, "", q, 1, NOW)["items"][0]["id"]
    takes: list[int] = []
    real = resultset._current

    def counted(*args, **kwargs):
        takes.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(resultset, "_current", counted)
    for ask in (
        lambda: resultset.describe(conn, "", q, NOW),
        lambda: resultset.page(conn, "", q, 2, NOW),
        lambda: resultset.peek(conn, "", q, 3, NOW),
        lambda: resultset.locate(conn, "", q, anchor, NOW),
    ):
        takes.clear()
        ask()
        assert len(takes) == 1, "an operation took the projection twice; two takes can straddle a commit"


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


def test_one_response_reads_one_database_snapshot(tmp_path, monkeypatch):
    """Counting takes proved an operation cannot mix PROJECTIONS; this
    proves construction cannot mix GENERATIONS: membership and item
    hydration are several reads, and a worker committing between them
    must not leak into an answer half-built from the world before it.
    The writer renames a file after membership is read but before
    hydration -- the response must carry the name the snapshot saw."""
    root = tmp_path / "pics"
    for i in range(4):
        _paint(root, "all", f"a_{i}.png", 40 * i)
    db_path = tmp_path / "gallery.db"
    conn = connect.connect(db_path)
    conn.executescript(connect.schema_sql())
    root_id = library.add_root(conn, str(root), "library", NOW)
    scan.scan(conn, root_id, str(root), NOW)
    conn.commit()
    target = conn.execute("SELECT id FROM file ORDER BY id LIMIT 1").fetchone()[0]
    original = conn.execute("SELECT name FROM file WHERE id = ?", (target,)).fetchone()[0]

    real = resultset._timed_ids

    def membership_then_commit(*args, **kwargs):
        ids = real(*args, **kwargs)
        writer = connect.connect(db_path)
        writer.execute("UPDATE file SET name = 'moved-under-us.png' WHERE id = ?", (target,))
        writer.commit()
        connect.close(writer)
        return ids

    monkeypatch.setattr(resultset, "_timed_ids", membership_then_commit)
    told = resultset.page(conn, "", _q(size=10), 1, NOW)
    named = {row["id"]: row["name"] for row in told["items"]}
    assert named[target] == original, (
        "hydration read a newer generation than membership: one response mixed two library states"
    )
    monkeypatch.setattr(resultset, "_timed_ids", real)
    fresh = resultset.page(conn, "", _q(size=10), 1, NOW + 1)
    assert {row["id"]: row["name"] for row in fresh["items"]}[target] == "moved-under-us.png", (
        "the NEXT response must see the commit; the snapshot is per-operation, not a cache"
    )
    connect.close(conn)


def _semantic_shelf(tmp_path, clip_cosines: dict, qwen_cosines: dict):
    """Eight files scanned for real, with embeddings written per space
    only for the names each cosine table mentions."""
    import numpy as np

    from db import derived
    from vision import semantic

    root = tmp_path / "pics"
    names = ("x1", "x2", "x3", "x4", "x5", "a", "b", "c")
    for i, name in enumerate(names):
        _paint(root, "all", f"{name}.png", 20 * i)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    ids = {name: file_id for file_id, name in conn.execute("SELECT id, replace(name, '.png', '') FROM file")}
    shas = dict(conn.execute("SELECT id, content_sha256 FROM file"))

    def tilted(cosine: float, axis: int) -> np.ndarray:
        v = np.zeros(16, dtype=np.float32)
        v[0] = cosine
        v[axis] = np.sqrt(1.0 - cosine * cosine)
        return v

    clip = semantic.space("openclip", "ViT-B-32", "laion2b_s34b_b79k", 16)
    qwen = semantic.space("qwen", "Qwen/Qwen3-VL-Embedding-2B", "main", 16)
    for spec, cosines in ((clip, clip_cosines), (qwen, qwen_cosines)):
        for axis, name in enumerate(names, start=1):
            if name in cosines:
                derived.record_embedding(conn, ids[name], spec, tilted(cosines[name], axis), shas[ids[name]], NOW)
    conn.commit()
    return conn, ids, clip.key, qwen.key


def test_a_space_with_nothing_in_scope_is_missing_not_a_contributor(tmp_path, monkeypatch):
    """`contributors` is the rankings that ENTERED the fusion. A space
    whose current embeddings all sit outside the scope contributed
    nothing -- reporting it as a contributor manufactures agreement the
    page then presents with full confidence."""
    import numpy as np

    from db import retrieval, settings
    from vision import semantic

    everyone = {"x1": 0.9, "x2": 0.8, "x3": 0.7, "x4": 0.6, "x5": 0.5, "a": 0.4, "b": 0.3, "c": 0.2}
    outsiders_only = {"x1": 0.9, "x2": 0.8, "x3": 0.7, "x4": 0.6, "x5": 0.5}
    conn, ids, clip_key, qwen_key = _semantic_shelf(tmp_path, everyone, outsiders_only)

    class Asks:
        def encode_query(self, phrase):
            probe = np.zeros(16, dtype=np.float32)
            probe[0] = 1.0
            return probe

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Asks())
    settings.put(conn, "semantic_model", "ViT-B-32/laion2b_s34b_b79k, qwen:Qwen/Qwen3-VL-Embedding-2B")
    conn.commit()

    in_scope = {ids["a"], ids["b"], ids["c"]}
    found = retrieval.query(conn, str(tmp_path), "the probe", 3, NOW, offline=True, allowed=in_scope)
    assert found["contributors"] == [clip_key]
    assert found["missing"] == {qwen_key: "no current embeddings in this scope"}
    assert [row["file_id"] for row in found["results"]] == [ids["a"], ids["b"], ids["c"]]
    assert len(found["participants"]) == 2


def test_a_scope_no_space_can_answer_is_empty_not_fake_agreement(tmp_path, monkeypatch):
    import numpy as np

    from db import retrieval, settings
    from vision import semantic

    outsiders_only = {"x1": 0.9, "x2": 0.8, "x3": 0.7, "x4": 0.6, "x5": 0.5}
    conn, ids, clip_key, qwen_key = _semantic_shelf(tmp_path, outsiders_only, dict(outsiders_only))

    class Asks:
        def encode_query(self, phrase):
            probe = np.zeros(16, dtype=np.float32)
            probe[0] = 1.0
            return probe

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Asks())
    settings.put(conn, "semantic_model", "ViT-B-32/laion2b_s34b_b79k, qwen:Qwen/Qwen3-VL-Embedding-2B")
    conn.commit()

    found = retrieval.query(conn, str(tmp_path), "the probe", 3, NOW, offline=True, allowed={ids["a"], ids["b"]})
    assert found["results"] == []
    assert found["contributors"] == []
    assert set(found["missing"]) == {clip_key, qwen_key}
