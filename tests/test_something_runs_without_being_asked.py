"""A schedule names a collection, and two never run at once.

"Every night, catch up" is one row. It could not have been written
before collections, because naming individual kinds would mean
re-deriving the order at 3am -- which of scan, ingest, embed,
detect_faces, cluster_faces, and in which sequence -- and that order is
exactly what `job.after_id` exists so nobody has to carry.

It would also have been worthless before the walk led the chain. A
catch-up that cannot walk derives forever over a library it never
notices growing, so a nightly one would have been busy and blind: the
most useless kind of scheduled job.

The two refusals below matter more than the scheduling. A collection
already going is never started a second time, and the clock runs from
when a collection STARTED rather than when it finished.
"""

from __future__ import annotations

import contextlib

import pytest

from db import jobs, scheduling
from tests.staging import HOUR, NOW, fresh_schema, hosting

pytestmark = pytest.mark.slow

NIGHTLY = 24.0


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with hosting(tmp_path_factory, "test_something_runs_without_being_asked") as stage:
        yield stage


@pytest.fixture
def served(_world):
    """One application for the three served claims here."""
    _world.restore()
    return _world.client


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


def _named(conn) -> list[str]:
    return [row["collection"] for row in scheduling.due(conn, NOW)]


def test_a_schedule_that_has_never_run_is_due_now(db):
    """Turning one on should prove it works tonight, not next week. A
    schedule that waited a full interval before its first run would be
    indistinguishable from one that is broken."""
    scheduling.put(db, "catch up", NIGHTLY, NOW)
    db.commit()
    assert _named(db) == ["catch up"]


def test_it_is_not_due_again_until_its_interval_has_passed(db):
    """And the clock runs from the START.

    A three-hour catch-up on a nightly schedule runs once a night.
    Measured from the finish it would slip three hours later every day
    and be running at noon by the end of the week.
    """
    scheduling.put(db, "catch up", NIGHTLY, NOW)
    scheduling.started(db, "catch up", NOW)
    db.commit()

    assert scheduling.due(db, NOW + 3 * HOUR) == []
    assert scheduling.due(db, NOW + 23 * HOUR) == []
    assert _named_at(db, NOW + 24 * HOUR) == ["catch up"]


def _named_at(conn, when: float) -> list[str]:
    return [row["collection"] for row in scheduling.due(conn, when)]


def test_a_collection_already_going_is_never_started_again(db):
    """The guard that matters. A nightly job over a library that takes
    thirty hours to catch up would otherwise be seven overlapping
    catch-ups by Sunday, each one slowing the others."""
    scheduling.put(db, "catch up", NIGHTLY, NOW)
    db.commit()
    assert _named(db) == ["catch up"], "the control: it is due"

    jobs.submit(db, "scan", NOW, collection="catch up")
    db.commit()
    assert scheduling.due(db, NOW) == [], "it started a second catch-up over a running one"


def test_a_step_still_queued_behind_another_counts_as_going(db):
    """A chain is not finished until its last step is, and the step
    somebody can see working is not the only one outstanding."""
    first = jobs.submit(db, "scan", NOW, collection="catch up")
    jobs.submit(db, "embed", NOW, collection="catch up", after_id=first)
    jobs.claim(db, "test-worker", NOW)
    fence = int(db.execute("SELECT fence FROM job WHERE id = ?", (first,)).fetchone()[0])
    jobs.settle(db, first, fence, "done", NOW)
    scheduling.put(db, "catch up", NIGHTLY, NOW)
    db.commit()

    # the first step is done; the second is queued and has not run
    assert scheduling.running(db, "catch up") is True
    assert scheduling.due(db, NOW) == []


def test_a_settled_collection_stops_holding_the_schedule_back(db):
    """The other half of that guard: once every step has settled, the
    next night is free to start one."""
    made = jobs.submit(db, "scan", NOW, collection="catch up")
    jobs.claim(db, "test-worker", NOW)
    fence = int(db.execute("SELECT fence FROM job WHERE id = ?", (made,)).fetchone()[0])
    jobs.settle(db, made, fence, "done", NOW)
    scheduling.put(db, "catch up", NIGHTLY, NOW)
    db.commit()

    assert scheduling.running(db, "catch up") is False
    assert _named(db) == ["catch up"]


def test_a_disabled_schedule_never_runs(db):
    scheduling.put(db, "catch up", NIGHTLY, NOW, enabled=False)
    db.commit()
    assert scheduling.due(db, NOW + 400 * HOUR) == []
    assert scheduling.next_due(scheduling.all_of(db)[0]) is None


def test_a_schedule_for_something_nothing_runs_is_refused(db):
    """Stored, it would be a row that looks like it works and never
    does: nothing would ever start it, and the page would show it as
    scheduled."""
    with pytest.raises(ValueError, match="catch up"):
        scheduling.put(db, "embed everything nightly", NIGHTLY, NOW)
    with pytest.raises(ValueError, match="repeats every"):
        scheduling.put(db, "catch up", 0, NOW)


def test_one_row_per_collection(db):
    """Two schedules for one act would disagree about when it last ran,
    and the guard against starting it twice is per-collection."""
    scheduling.put(db, "catch up", NIGHTLY, NOW)
    scheduling.put(db, "catch up", 6.0, NOW)
    db.commit()
    held = scheduling.all_of(db)
    assert len(held) == 1
    assert held[0]["every_hours"] == 6.0


# --- and the runner is the only thing that starts one ------------------------


def test_the_runner_starts_what_is_due_and_stamps_it(tmp_path, served):
    """`run_schedules` is called on the worker's own turn rather than by
    a timer of its own: a second scheduler is a second thing that can be
    running while nobody thinks anything is."""
    from db import connect, runner

    with contextlib.nullcontext(served) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            scheduling.put(conn, "catch up", NIGHTLY, NOW)
            conn.commit()

            started = runner.run_schedules(conn, NOW, models_dir=str(tmp_path / "models"))
            conn.commit()
            assert started == ["catch up"]

            steps = conn.execute("SELECT count(*) FROM job WHERE collection = 'catch up'").fetchone()[0]
            stamped = conn.execute("SELECT last_started_at FROM schedule").fetchone()[0]

            # and asking again immediately starts nothing: the one it
            # just started is still going
            assert runner.run_schedules(conn, NOW, models_dir=str(tmp_path / "models")) == []
        finally:
            connect.close(conn)

    assert steps > 0, "the schedule started a collection with no steps in it"
    assert stamped == NOW


def test_the_console_shows_it_and_can_set_it(served):
    """Set where the sweeps are, in hours. A cron expression is a small
    language, and a small language wants a parser, a validator and a way
    to say what it will do next."""
    from db import connect

    with contextlib.nullcontext(served) as client:
        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert 'data-schedule="catch up"' in page
        assert "never set" in page, "an unscheduled collection does not say so"

        told = client.post("/operations/schedules/catch up", data={"every_hours": "12", "enabled": "true"})
        assert told.status_code in (200, 201), told.text

        conn = connect.connect(client.app.state.db_path)
        try:
            held = scheduling.all_of(conn)
        finally:
            connect.close(conn)
        assert len(held) == 1
        assert held[0]["every_hours"] == 12.0
        assert held[0]["enabled"] == 1

        # an unchecked box sends nothing at all, and that absence is the
        # answer -- a default of True would read it as "on"
        client.post("/operations/schedules/catch up", data={"every_hours": "12"})
        conn = connect.connect(client.app.state.db_path)
        try:
            assert scheduling.all_of(conn)[0]["enabled"] == 0, "unchecking the box left it on"
        finally:
            connect.close(conn)


def test_the_console_refuses_a_collection_it_cannot_run(served):
    with contextlib.nullcontext(served) as client:
        told = client.post("/operations/schedules/nonsense", data={"every_hours": "12", "enabled": "true"})
        assert told.status_code == 400, told.text
