"""The order stops being knowledge a person carries.

Adding a root meant pressing eight buttons in a sequence only the
application knew: scan, ingest, context, events, embed, detect_faces,
cluster_faces, annotate. The order is REAL and the failure it causes is
quiet -- `cluster_faces` over an unembedded library settles `done`
having clustered nothing -- so pressing them out of order does not look
like a mistake. It looks like a library with no people in it.

A step now records what it comes after, and `jobs.claim` will not take
it until that one has settled `done`. Grouping alone would not have been
worth building: if a collection were only a label over rows, the console
would be quieter and a person would still not know what to press. The
edge is the part that earns it.

What a failed step does to its collection was the product decision, and
the answer here is deliberate: it stops exactly what depended on it. A
partial catch-up is normal and useful -- one unreadable file must not
abandon the other four thousand -- so everything unrelated in the same
collection runs to completion.
"""

from __future__ import annotations

import itertools

import pytest

from db import jobs
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0
OWNER = "test-worker"


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


def _chain(conn, *kinds: str) -> list[int]:
    """Each step gated on the one before, all queued at once."""
    made: list[int] = []
    after = None
    for kind in kinds:
        after = jobs.submit(conn, kind, NOW, collection="catch up", after_id=after)
        made.append(after)
    conn.commit()
    return made


def _claimed(conn) -> int | None:
    held = jobs.claim(conn, OWNER, NOW)
    return None if held is None else held[0]


def test_a_step_is_not_claimable_until_the_one_before_it_is_done(db):
    """The whole feature. Both are queued; only the first can be taken."""
    first, second = _chain(db, "scan", "embed")

    assert _claimed(db) == first
    assert _claimed(db) is None, "the second step was claimable while the first was still running"

    jobs.settle(db, first, _fence(db, first), "done", NOW)
    db.commit()
    assert _claimed(db) == second


def _fence(conn, job_id: int) -> int:
    return int(conn.execute("SELECT fence FROM job WHERE id = ?", (job_id,)).fetchone()[0])


def test_an_ungated_job_is_claimable_exactly_as_before(db):
    """The control, and the compatibility claim: every job that existed
    before steps did has a NULL edge and is unchanged by all of this."""
    alone = jobs.submit(db, "scan", NOW)
    db.commit()
    assert _claimed(db) == alone


def test_a_failed_step_stops_what_waited_on_it_and_nothing_else(db):
    """The product decision, pinned.

    A partial catch-up is normal: one unreadable file must not abandon
    the other four thousand. So the failure travels down the edges it
    owns, and an unrelated step in the same collection is untouched.
    """
    first, second, third = _chain(db, "scan", "embed", "cluster_faces")
    unrelated = jobs.submit(db, "annotate", NOW, collection="catch up")
    db.commit()

    jobs.claim(db, OWNER, NOW)
    jobs.settle(db, first, _fence(db, first), "failed", NOW, error="the disk went away")
    db.commit()

    states = dict(db.execute("SELECT id, state FROM job").fetchall())
    assert states[second] == "cancelled", "a step whose predecessor failed can never run and was left queued"
    assert states[third] == "cancelled", "the cascade stopped one link short"
    assert states[unrelated] == "queued", "a failure abandoned work that did not depend on it"


def test_the_cancelled_step_says_why(db):
    """A row that stops has to say what stopped it, or the console shows
    a collection one step short of finished with nothing to explain it."""
    first, second = _chain(db, "scan", "embed")
    jobs.claim(db, OWNER, NOW)
    jobs.settle(db, first, _fence(db, first), "failed", NOW, error="the disk went away")
    db.commit()

    said = db.execute("SELECT error FROM job WHERE id = ?", (second,)).fetchone()[0]
    assert said == "the step before it did not finish"


def test_a_cancelled_step_stops_its_dependents_too(db):
    """Cancelling is settling, and a step after a cancelled one is as
    unrunnable as one after a failure -- `claim` gates on `done`."""
    first, second = _chain(db, "scan", "embed")
    jobs.claim(db, OWNER, NOW)
    jobs.settle(db, first, _fence(db, first), "cancelled", NOW)
    db.commit()
    assert db.execute("SELECT state FROM job WHERE id = ?", (second,)).fetchone()[0] == "cancelled"


def test_a_step_already_running_is_not_cancelled_from_under_the_worker(db):
    """Bookkeeping does not kill work in flight. A dependent that was
    claimed before its predecessor settled is the runner's business --
    stopping it here would mark a row terminal while a worker is still
    writing under it."""
    first, second = _chain(db, "scan", "embed")
    # take the second by hand, the way an expired lease would let it be
    db.execute("UPDATE job SET state = 'running', owner = ?, fence = fence + 1 WHERE id = ?", (OWNER, second))
    db.commit()

    jobs.claim(db, OWNER, NOW)
    jobs.settle(db, first, _fence(db, first), "failed", NOW, error="gone")
    db.commit()
    assert db.execute("SELECT state FROM job WHERE id = ?", (second,)).fetchone()[0] == "running"


def test_the_collection_is_a_name_the_steps_share(db):
    """What a schedule points at. "Every night, catch up" names this;
    naming individual kinds would mean re-deriving the order at 3am."""
    made = _chain(db, "scan", "embed", "cluster_faces")
    held = db.execute(
        "SELECT count(*) FROM job WHERE collection = 'catch up' AND id IN (?, ?, ?)", tuple(made)
    ).fetchone()[0]
    assert held == 3


def test_the_catch_up_recipe_queues_the_chain_in_order(tmp_path):
    """End to end, through the route: one ask, and every step gated on
    the one before it."""
    from litestar.testing import TestClient
    from PIL import Image

    from db import connect
    from sg_web.app import build_app

    root = tmp_path / "lib"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")

    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")

        told = client.post("/jobs/catch-up")
        assert told.status_code in (200, 201), told.text
        body = told.json()
        assert body["collection"] == "catch up"
        steps = body["steps"]
        assert len(steps) >= 2, f"a catch-up over a fresh library queued {steps}"

        conn = connect.connect(client.app.state.db_path)
        try:
            rows = {
                one: (kind, after)
                for one, kind, after in conn.execute(
                    "SELECT id, kind, after_id FROM job WHERE collection = 'catch up' ORDER BY id"
                ).fetchall()
            }
            assert set(rows) == set(steps)
            # every step but the first names the one before it
            assert rows[steps[0]][1] is None, "the first step is gated on something"
            for before, this in itertools.pairwise(steps):
                assert rows[this][1] == before, f"{rows[this][0]} is not gated on {rows[before][0]}"
        finally:
            connect.close(conn)


def test_a_step_that_knows_it_has_nothing_to_do_is_simply_absent(tmp_path):
    """The chain closes over the hole rather than gating on it.

    A submitter that can tell in advance it has nothing to do returns no
    job -- ingest over a library with nothing unread, embed with nothing
    unembedded. The next step is then gated on the last one that DID
    queue, never on a job that does not exist.

    Not every step can tell: `events` and `cluster_faces` reach "nothing
    to do" by running, so they queue over an empty library and settle
    `done` having done nothing. That is why `steps` is rarely empty and
    why this asserts the SHAPE of the chain rather than its length.
    """
    from litestar.testing import TestClient

    from db import connect
    from sg_web.app import build_app

    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        told = client.post("/jobs/catch-up")
        assert told.status_code in (200, 201), told.text
        body = told.json()
        assert body["collection"] == "catch up"
        steps = body["steps"]

        conn = connect.connect(client.app.state.db_path)
        try:
            kinds = dict(conn.execute("SELECT id, kind FROM job WHERE collection = 'catch up'").fetchall())
            after = dict(conn.execute("SELECT id, after_id FROM job WHERE collection = 'catch up'").fetchall())
        finally:
            connect.close(conn)

        assert "ingest" not in kinds.values(), "ingest queued over a library with nothing to read"
        assert after[steps[0]] is None
        for before, this in itertools.pairwise(steps):
            assert after[this] == before, "the chain gated a step on a job that was never queued"


# --- and the console stops showing eight rows for one act --------------------


def _console(tmp_path):
    from litestar.testing import TestClient

    from sg_web.app import build_app

    return TestClient(app=build_app(str(tmp_path / "run"), worker=False))


def test_the_console_folds_a_collection_into_one_row(tmp_path):
    """The half a person sees.

    Eight rows where somebody asked for one thing. Each honest, none the
    answer -- "is the catch-up going well" was a sum they had to do
    themselves while the rows moved.
    """
    from db import connect

    with _console(tmp_path) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            made = _chain(conn, "scan", "embed", "cluster_faces")
            alone = jobs.submit(conn, "annotate", NOW)
            conn.commit()
        finally:
            connect.close(conn)

        told = client.get("/operations/overview").json()
        folded = told["collections"]
        assert len(folded) == 1
        assert folded[0]["name"] == "catch up"
        assert folded[0]["steps"] == made
        assert folded[0]["state"] == "queued"

        # the steps are still in the matrix -- this is a fold over them,
        # not a replacement, and a client that wants one still has it
        seen = {row["id"] for row in told["matrix"]}
        assert set(made) <= seen
        assert alone in seen


def test_a_failed_step_makes_the_whole_collection_read_failed(tmp_path):
    """One failed step means the collection did not do what it was asked,
    whatever the others managed. Saying "running" while the rest of the
    chain drains would be the console agreeing with the queue instead of
    with the person."""
    from db import connect

    with _console(tmp_path) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            first, _second = _chain(conn, "scan", "embed")
            jobs.claim(conn, OWNER, NOW)
            jobs.settle(conn, first, _fence(conn, first), "failed", NOW, error="gone")
            conn.commit()
        finally:
            connect.close(conn)

        folded = client.get("/operations/overview").json()["collections"][0]
        assert folded["state"] == "failed"
        assert folded["failed"] == first, "the row somebody needs is not named"


def test_the_fold_shows_the_steps_rather_than_hiding_them(tmp_path):
    """Collapsing is not the same as hiding. The page carries every step
    under the fold, and the one that failed is reachable without going
    anywhere else."""
    from db import connect

    with _console(tmp_path) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            first, second = _chain(conn, "scan", "embed")
            conn.commit()
        finally:
            connect.close(conn)

        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert 'data-matrix-collection="catch up"' in page
        for one in (first, second):
            assert f'data-matrix-job="{one}"' in page, "a step vanished when its collection folded"
        assert page.count('data-matrix-job="') == 2, "a step was rendered twice, folded and loose"


def test_a_job_that_is_nobodys_step_is_untouched(tmp_path):
    """The control. Every job submitted on its own renders exactly as it
    did before collections existed."""
    from db import connect

    with _console(tmp_path) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            alone = jobs.submit(conn, "annotate", NOW)
            conn.commit()
        finally:
            connect.close(conn)

        told = client.get("/operations/overview").json()
        assert told["collections"] == []
        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert f'data-matrix-job="{alone}"' in page


def test_the_console_offers_it_first_and_pressing_it_queues_the_chain(tmp_path):
    """The button, and where it is.

    Twelve sweeps, each honest, each needing somebody to already know
    which to press and when. This one is the answer to that question, so
    it is not the thirteenth in the row -- it is the first.
    """
    from db import connect

    with _console(tmp_path) as client:
        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert 'data-launch="catch_up"' in page, "the console cannot start a catch-up"
        assert "data-launch-primary" in page

        told = client.post("/operations/jobs/catch_up", headers={"accept": "text/html"})
        assert told.status_code in (200, 201), told.text
        assert "nothing to do" not in told.text

        conn = connect.connect(client.app.state.db_path)
        try:
            steps = [
                (one, after)
                for one, after in conn.execute(
                    "SELECT id, after_id FROM job WHERE collection = 'catch up' ORDER BY id"
                ).fetchall()
            ]
        finally:
            connect.close(conn)
        assert steps, "the button queued nothing"
        assert steps[0][1] is None
        for (before, _), (_, after) in itertools.pairwise(steps):
            assert after == before, "the button queued the steps ungated"


def test_an_unknown_sweep_name_is_still_refused(tmp_path):
    """The control: adding a launcher did not turn the name into
    something the route accepts anything for."""
    with _console(tmp_path) as client:
        assert client.post("/operations/jobs/catch-up").status_code == 404, "the underscore name is the one that works"


# --- a step decides what to do when it RUNS ----------------------------------


def test_a_later_step_sees_what_an_earlier_one_produced(tmp_path):
    """The flaw putting the steps in order does not fix by itself.

    `cluster_faces` used to enumerate embedding spaces at SUBMIT time.
    Queued behind `detect_faces`, there were none yet -- so it queued
    zero items and settled `done` having clustered nothing, which is the
    exact failure the ordering exists to prevent: a library with no
    people in it and no row that looks wrong.

    Ordering is necessary and it is not sufficient. A step whose work
    depends on an earlier step has to decide what that work IS when it
    runs.
    """
    import numpy as np

    from db import connect, derived, runner, scan

    with _console(tmp_path) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            # a library with a file and no faces at all, then the chain
            made = conn.execute("INSERT INTO root(path, kind, created_at) VALUES('Z:/s','library',0)")
            root = int(made.lastrowid or 0)
            folder = scan.mint(conn, "folder", "s")
            conn.execute(
                "INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?,?,NULL,'s',0)", (folder, root)
            )
            file_id = scan.mint(conn, "file", "one")
            conn.execute(
                "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
                " VALUES(?, ?, 'one.png', 'image', 1, 0, ?, 0, 0)",
                (file_id, folder, "a" * 64),
            )
            conn.commit()

            job = runner.submit_cluster(conn, NOW)
            conn.commit()

            # and NOW the faces arrive, the way detect_faces would have
            # produced them after this job was already queued
            derived.record_faces(
                conn,
                file_id,
                "opencv/yunet+sface",
                "1",
                "a" * 64,
                NOW,
                [
                    {
                        "region": derived.region(conn, 0.1, 0.1, 0.2, 0.2),
                        "embedding": np.ones(4, np.float32).tobytes(),
                    }
                ],
            )
            conn.commit()

            while runner.run_next(conn, OWNER, NOW) is not None:
                conn.commit()
            conn.commit()

            state = conn.execute("SELECT state FROM job WHERE id = ?", (job,)).fetchone()[0]
            runs = conn.execute("SELECT count(*) FROM derived_face_run").fetchone()[0]
        finally:
            connect.close(conn)

    assert state == "done"
    assert runs == 1, "the step settled done having clustered a space that existed by the time it ran"


def test_a_walk_can_be_queued_and_a_worker_claims_it(tmp_path):
    """The walk becomes a job somebody does not have to be present for.

    `POST /roots/{id}/scan` walks inline and is its own worker, which is
    right for a person who just pressed it and is watching. Nothing
    unattended could ask for one -- and a scheduled catch-up that cannot
    walk derives forever over a library it never notices growing, which
    is the most useless kind of scheduled job: busy, and blind.
    """
    from PIL import Image

    from db import connect, runner

    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (16, 12), (30, 90, 140)).save(root / "one.png")

    with _console(tmp_path) as client:
        client.post("/roots", json={"path": str(root)})
        conn = connect.connect(client.app.state.db_path)
        try:
            before = conn.execute("SELECT count(*) FROM file").fetchone()[0]
            job = runner.submit_walk(conn, NOW)
            assert job is not None, "a registered root is something to walk"
            conn.commit()

            # claimed by an ordinary worker turn, not by the request
            turn = runner.run_next(conn, OWNER, NOW)
            conn.commit()
            assert turn is not None, "nothing claimed the walk"

            state = conn.execute("SELECT state FROM job WHERE id = ?", (job,)).fetchone()[0]
            after = conn.execute("SELECT count(*) FROM file").fetchone()[0]
        finally:
            connect.close(conn)

    assert state == "done", "the walk did not settle"
    assert before == 0
    assert after == 1, "the walk found nothing it was queued to find"


def test_walking_a_library_with_no_roots_is_nothing_to_do(tmp_path):
    """Not an error. A library nobody has pointed at a folder yet is a
    normal state, and the chain has to be able to say so."""
    from db import connect, runner

    with _console(tmp_path) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            assert runner.submit_walk(conn, NOW) is None
        finally:
            connect.close(conn)
