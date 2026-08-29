"""A precache that owns the only worker for an hour should not.

The runner works one item at a time, and must keep doing so: an item is
started, committed, worked and settled on its own, which is what makes a
job resumable, cancellable at a boundary, and able to fail one picture
without losing the rest. What moves here is only WHEN the pixels are
computed -- an item renders its own thumbnails and the next few of its
job's pending items beside them, so those are already on disk when their
turn comes and they take the `already-cached` return.

The same bargain `_Ahead` makes for vectors, and a cheaper one: a
vector has to be held in memory because writing a row ahead would not be
safe, while a thumbnail's result IS a file in a content-addressed cache.
Rendering one early is exactly what the job would have produced, so
nothing is held and a cancel undoes nothing.

Measured end to end through `run_next`, 32 pictures at 4000x3000:

    1 in flight     4.64 files/sec
    2               9.55
    4              16.95
    8              23.55

libvips already uses every core to calculate ONE image
(../refs/libvips/libvips/doc/using-threads.md), which reads like an
argument that this cannot help. One thumbnail is not enough work to
fill sixteen cores; the win is across files, and libvips is documented
thread-safe for exactly this -- images are immutable and shareable, and
only the drawing operators and Regions are not.

What must not change is everything else, which is what is tested here.
"""

from __future__ import annotations

import pathlib

import pytest
from PIL import Image

from db import build, connect, jobs, ledger, runner, scan
from vision import thumbs

pytestmark = pytest.mark.slow


def _library(tmp_path: pathlib.Path, pictures: int, *, broken: int | None = None):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(pictures):
        target = root / f"p{i:03d}.png"
        if i == broken:
            # a real file, a real hash, and nothing a decoder will take
            target.write_bytes(b"not a picture, whatever the suffix says")
        else:
            Image.new("RGB", (120, 90), (20 + 30 * i % 200, 90, 140)).save(target)

    db = tmp_path / "gallery.db"
    build.build(db)
    conn = connect.connect(str(db))
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    conn.commit()
    scan.scan(conn, 1, str(root), 1.0)
    conn.commit()
    cache = tmp_path / "thumbs"
    cache.mkdir()
    return conn, cache


def _drain(conn, since: float = 2.0, **kw):
    """Turns until the job settles.

    The clock ADVANCES. `jobs.pause` expires the lease on the spot so the
    next turn resumes the job, and a claim wants a lease strictly in the
    past -- so a drain that kept asking at the same instant as the pause
    got no turn at all and looked like a job that had stopped early.
    """
    turns = []
    while True:
        since += 1.0
        told = runner.run_next(conn, "w1", since, **kw)
        if told is None:
            return turns
        turns.append(told)
        if told["state"] in ("done", "failed", "cancelled"):
            return turns


def _cached(conn, cache) -> int:
    held = conn.execute("SELECT content_sha256 FROM file WHERE content_sha256 IS NOT NULL").fetchall()
    return sum(1 for (sha,) in held if all(thumbs.path_for(cache, sha, v).exists() for v in thumbs.EDGES))


def test_every_picture_still_gets_its_thumbnails(tmp_path):
    """The whole contract. Rendering ahead must cover the job, not race
    through part of it -- more than one group's worth of pictures, so
    the second group is formed from what the first left."""
    conn, cache = _library(tmp_path, 20)
    try:
        job = runner.submit_thumbs(conn, 1.0, thumbs_dir=str(cache))
        assert job is not None
        _drain(conn)
        state = conn.execute("SELECT state FROM job WHERE id = ?", (job,)).fetchone()[0]
        assert state == "done", state
        assert list(jobs.pending(conn, job)) == []
        assert _cached(conn, cache) == 20
    finally:
        connect.close(conn)


def test_the_first_item_renders_more_than_its_own_picture(tmp_path):
    """That a group is formed at all.

    `budget=1` performs exactly ONE item, so serially exactly one
    picture would be on disk afterwards. The others are there because
    the item that was worked rendered them beside its own -- which is
    the whole feature, stated as a count rather than as a stopwatch.
    """
    conn, cache = _library(tmp_path, 12)
    try:
        job = runner.submit_thumbs(conn, 1.0, thumbs_dir=str(cache))
        assert job is not None
        told = runner.run_next(conn, "w1", 2.0, budget=1)
        assert told is not None
        assert told["did"] == 1, told
        assert len(list(jobs.pending(conn, job))) == 11, "more than one item was performed"
        # both halves: more than its own, and exactly the group size. The first
        # alone is satisfied by any lookahead at all, the second alone by a
        # group size of one comparing the knob against itself.
        assert _cached(conn, cache) > 1, "one item performed cached only its own picture"
        assert _cached(conn, cache) == runner.thumbs_in_flight(), (
            f"one item performed cached {_cached(conn, cache)} pictures; "
            f"{runner.thumbs_in_flight()} were meant to render together"
        )
    finally:
        connect.close(conn)


def test_each_item_is_still_started_and_settled_on_its_own(tmp_path):
    """Resumability, cancellability and per-item failure all rest on the
    item boundary. Rendering several at once must not collapse twenty
    items into one."""
    conn, cache = _library(tmp_path, 12)
    try:
        job = runner.submit_thumbs(conn, 1.0, thumbs_dir=str(cache))
        _drain(conn)
        started = {
            row["item_id"]
            for row in ledger.since(conn, 0)
            if row["job_id"] == job and row["type"] == "item.started" and row["item_id"] is not None
        }
    finally:
        connect.close(conn)
    assert len(started) == 12, f"{len(started)} items were started; there are 12"


def test_a_picture_that_cannot_be_rendered_fails_only_its_own_item(tmp_path):
    """The safety property that makes rendering ahead allowed at all.

    A picture rendered SPECULATIVELY is not the item being worked. If it
    raises, that says nothing about the file whose turn it is, and
    reporting it there would blame the wrong one. It meets its own
    failure, attributed to itself, when its turn comes -- so the job ends
    with exactly one failed item and every other picture cached.
    """
    conn, cache = _library(tmp_path, 10, broken=4)
    try:
        job = runner.submit_thumbs(conn, 1.0, thumbs_dir=str(cache))
        _drain(conn)
        (state,) = conn.execute("SELECT state FROM job WHERE id = ?", (job,)).fetchone()
        failures = [row for row in ledger.since(conn, 0) if row["job_id"] == job and row["type"] == "item.failed"]
        assert len(failures) == 1, f"{len(failures)} items failed; exactly one picture is undecodable"
        # and it is the undecodable one, not whichever item happened to
        # be leading the group it was rendered in
        broken_id = conn.execute("SELECT id FROM file WHERE name = 'p004.png'").fetchone()[0]
        assert failures[0]["item_id"] == broken_id, "the failure was charged to the wrong picture"
        assert _cached(conn, cache) == 9, "one bad picture cost the others their thumbnails"
        assert state == "done", state
    finally:
        connect.close(conn)


def test_a_budget_still_stops_the_turn_where_it_says(tmp_path):
    """`budget` bounds ITEMS PERFORMED, and the resumption contract rests
    on it. Rendering ahead is not performing: the pixels exist early, the
    items are still worked one at a time and the turn still stops where
    it was told to."""
    conn, cache = _library(tmp_path, 12)
    try:
        job = runner.submit_thumbs(conn, 1.0, thumbs_dir=str(cache))
        assert job is not None
        told = runner.run_next(conn, "w1", 2.0, budget=3)
        assert told is not None
        assert told["did"] == 3, told
        assert told["state"] == "running", told
        assert len(list(jobs.pending(conn, job))) == 9, "the turn performed more items than its budget"
        _drain(conn)
        assert _cached(conn, cache) == 12
    finally:
        connect.close(conn)


def test_the_number_in_flight_never_turns_the_feature_off(tmp_path):
    """One would silently mean serial, and a machine reporting a single
    core is the case that would do it."""
    assert runner.thumbs_in_flight() >= 2
    assert runner.thumbs_in_flight() <= 8, "past the measured knee the two thread pools fight"


def test_a_warm_cache_still_costs_nothing(tmp_path, monkeypatch):
    """The `already-cached` return is what the pictures rendered ahead
    take when their turn comes, so it has to keep working -- and a second
    run over a warm cache must decode nothing at all."""
    from vision import derive

    conn, cache = _library(tmp_path, 8)
    try:
        runner.submit_thumbs(conn, 1.0, thumbs_dir=str(cache))
        _drain(conn)
        assert _cached(conn, cache) == 8

        rendered: list[str] = []
        real = derive.put_all

        def counted(cache_dir, sha, path, kind, orientation):
            rendered.append(sha)
            return real(cache_dir, sha, path, kind, orientation)

        # `monkeypatch` rather than assigning the module attribute back in
        # a finally: it restores on failure too, and rebinding a module's
        # `def` is not an assignment a type checker will take.
        monkeypatch.setattr(derive, "put_all", counted)
        again = runner.submit_thumbs(conn, 3.0, thumbs_dir=str(cache))
        assert again is None, "a warm cache queued a job"
        assert rendered == []
    finally:
        connect.close(conn)
