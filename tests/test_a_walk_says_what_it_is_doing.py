"""Scanning is visible while it happens.

Every expensive sweep in this application is a job: a row, a state, a
count that moves, and a live feed the operations console draws. Hashing,
thumbnails, embeddings, faces -- all of them say where they are.

The directory walk did not. It was the ONE expensive thing done inline
by a route, and it is the most expensive of them: it reads every byte of
every changed file. So a person who asked to scan a large root watched a
request hang with nothing to look at, while the cheap work that followed
reported itself in detail.

It is a `walk` job now, run by the request itself through the same
submit / claim / checkpoint / settle the runner uses -- so the row obeys
the same invariants rather than being a second kind of job nothing else
understands. The answer is still the counts, because the request is
still synchronous; what is new is that somebody watching has something
to watch.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from db import connect
from sg_web.app import WALK_EVERY
from tests.staging import hosting

#: Enough files that the walk must report more than once, whatever the
#: cadence: the point is a count that MOVES, and a single report at the
#: end is the silence this replaced.
MANY = WALK_EVERY * 3 + 40


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    """One library and one scan for the whole module: writing several
    hundred pictures and reading every byte of them is what this file is
    about, and doing it seven times says nothing seven times.

    Snapshotted AFTER the scan, so the restore between tests puts back
    the walked world rather than the empty one -- the second-scan test is
    the only one that writes, and it gets its own first rescan."""
    with hosting(tmp_path_factory, "test_a_walk_says_what_it_is_doing") as stage:
        (stage.root / "deeper").mkdir(parents=True, exist_ok=True)
        for i in range(MANY):
            # a nested folder as well as a flat one: reporting per DIRECTORY
            # looks thriftier and says nothing, because a library is often
            # one flat folder
            where = stage.root if i % 3 else stage.root / "deeper"
            # Bytes, not an encoded picture. A walk stats and HASHES; it
            # never decodes, and nothing in this module looks at a pixel
            # -- so `Image.new(...).save(...)` was paying an encoder per
            # file, MANY times, to produce input no test reads as an
            # image. Distinct per file, because the walk's whole subject
            # is telling them apart by content.
            (where / f"p{i:05d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + f"{i:05d}".encode() * 8)
        made = stage.client.post("/roots", json={"path": str(stage.root)}).json()
        answer = stage.client.post(f"/roots/{made['id']}/scan")
        assert answer.status_code == 201, answer.text
        stage.held["told"] = answer.json()
        stage.snapshot()
        yield stage


@pytest.fixture
def scanned(_world):
    _world.restore()
    return _world.client, _world.held["told"], _world.db


def _events(db: pathlib.Path, kind: str = "walk") -> list[dict]:
    conn = connect.connect(str(db), read_only=True)
    try:
        job = conn.execute("SELECT id FROM job WHERE kind = ?", (kind,)).fetchone()
        assert job is not None, f"no {kind} job was ever created"
        return [
            {"type": row[0], "message": row[1], "data": json.loads(row[2]) if row[2] else {}}
            for row in conn.execute("SELECT type, message, data FROM job_event WHERE job_id = ? ORDER BY id", (job[0],))
        ]
    finally:
        connect.close(conn)


def test_the_answer_is_still_the_counts(scanned):
    """The route stayed synchronous, so nothing that reads it changed."""
    _client, told, _db = scanned
    assert told["added"] == MANY, told
    assert told["hashed"] == MANY
    assert set(told) >= {"root", "added", "matched", "replaced", "ambiguous", "missing", "hashed", "precache"}


def test_the_walk_is_a_job_like_every_other_sweep(scanned):
    _client, _told, db = scanned
    conn = connect.connect(str(db), read_only=True)
    try:
        row = conn.execute("SELECT kind, state, done_count, payload FROM job WHERE kind = 'walk'").fetchone()
    finally:
        connect.close(conn)
    assert row is not None, "the walk left no job row"
    kind, state, done, payload = row
    assert (kind, state) == ("walk", "done")
    assert done == MANY, f"the walk finished having counted {done} of {MANY}"
    # the root is in the PAYLOAD: `target_id` references `entity`, and a
    # root is not one -- its folder is, and on a first scan that folder
    # does not exist until this walk makes it
    assert json.loads(payload)["root"] == 1


def test_it_says_where_it_is_while_it_goes(scanned):
    """The whole point. One report at the end is what this replaced."""
    _client, _told, db = scanned
    moved = [one["data"]["done"] for one in _events(db) if one["type"] == "checkpoint.changed"]
    assert len(moved) >= 3, f"a walk of {MANY} files reported {len(moved)} time(s): {moved}"
    assert moved == sorted(moved), f"the count went backwards: {moved}"
    assert moved[0] < moved[-1] < MANY + 1


def test_the_transitions_are_there_too(scanned):
    """`jobs.settle` does not write the ledger -- the RUNNER does -- so a
    job run outside the runner has to speak the end itself, or the
    console shows a row that appeared and never finished.

    There is no `job.claimed`, and there should not be: `jobs.begin`
    inserts the row already running and owned, precisely so there is no
    queued moment for the background worker to take it in. The one
    submitted event names the owner instead.
    """
    _client, _told, db = scanned
    events = _events(db)
    said = [one["type"] for one in events]
    assert said[0] == "job.submitted", said
    assert "job.claimed" not in said, "a claim means the row was queued, which is the race this avoids"
    assert events[0]["data"]["owner"].startswith("scan-"), events[0]
    assert said[-1] == "job.done", said


def test_the_background_worker_never_takes_the_walk(scanned):
    """The defect that made this shape necessary. Submitted and then
    claimed, the row sat QUEUED for a moment; the worker polls for any
    runnable kind, took it, had no handler for it and failed it -- while
    the request that created it was still reaching for its own work."""
    _client, _told, db = scanned
    conn = connect.connect(str(db), read_only=True)
    try:
        states = [one[0] for one in conn.execute("SELECT state FROM job WHERE kind = 'walk'")]
    finally:
        connect.close(conn)
    assert states == ["done"], f"a walk ended {states}; a failed one is the worker having taken it"
    # `job.owner` is cleared when a job settles, so who ran it is read
    # from the ledger, which keeps what the row lets go of
    said = _events(db)[0]
    assert said["data"]["owner"].startswith("scan-"), said


def test_the_done_event_says_what_the_walk_found(scanned):
    _client, _told, db = scanned
    done = next(one for one in _events(db) if one["type"] == "job.done")
    assert f"{MANY} file(s)" in done["message"], done["message"]
    assert done["data"]["added"] == MANY


def test_a_second_scan_walks_again_and_hashes_nothing(scanned):
    """A rescan is the cheap case -- every hash is reused -- and it must
    still be visible, because "nothing is happening" and "it finished
    instantly" look identical from outside."""
    client, _told, db = scanned
    again = client.post("/roots/1/scan").json()
    assert again["hashed"] == 0, "a rescan re-read the bytes"
    assert again["matched"] == MANY

    conn = connect.connect(str(db), read_only=True)
    try:
        walks = conn.execute("SELECT count(*), sum(done_count) FROM job WHERE kind = 'walk'").fetchone()
    finally:
        connect.close(conn)
    assert walks[0] == 2, "the second scan left no job row"
    assert walks[1] == MANY * 2, "the second walk counted nothing"
