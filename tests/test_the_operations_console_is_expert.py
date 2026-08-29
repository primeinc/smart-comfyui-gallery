"""The Operations Console is a data contract, held at every seam.

The job row is current truth (db/jobs.py); the ledger is historical
observation (db/ledger.py); the channel is transport (/ws/events). The
console (db/inspecting.py, sg_web/console.py, /operations) exposes what
the backend knows, renders every event type in words, and never
samples. Each test here is one clause of the acceptance contract, at the
narrowest seam that proves it; the browser proof lives beside this file.
"""

from __future__ import annotations

import json
import re
import time
import typing

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, inspecting, jobs, ledger, runner
from sg_web import console
from sg_web.app import build_app
from tests.staging import NOW, fresh_schema, hosting


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


class Spoken:
    """What the runner said on each seam, in order."""

    def __init__(self) -> None:
        self.deltas: list[dict] = []
        self.events: list[dict] = []

    def progress(self, delta: dict) -> None:
        self.deltas.append(delta)

    def event(self, event: dict) -> None:
        self.events.append(event)


class Reporting:
    """A handler that reports phases, then succeeds, fails, or crashes."""

    def __init__(self, fails_on=None, crashes_on=None):
        self.fails_on = fails_on
        self.crashes_on = crashes_on
        self.saw_started: list[int] = []

    def __call__(self, conn, item_id, payload, now):
        # the start is already durable when the handler runs
        kinds = [row[0] for row in conn.execute("SELECT type FROM job_event WHERE item_id = ? ORDER BY id", (item_id,))]
        if "item.started" in kinds:
            self.saw_started.append(item_id)
        told = runner.report()
        told.phase("decoding", file_id=item_id)
        told.progress("frames", 48, 220)
        told.observe("faces-found", count=7)
        told.phase("embedding")
        if item_id == self.fails_on:
            raise ValueError(f"item {item_id} is corrupt")
        if item_id == self.crashes_on:
            raise TypeError("the encoder grew a bug")


def _types(db, job_id):
    return [row[0] for row in db.execute("SELECT type FROM job_event WHERE job_id = ? ORDER BY id", (job_id,))]


# --- the vocabulary is closed, and every word has a rendering ----------------


@pytest.mark.parametrize("type_", ledger.TYPES)
def test_every_event_type_renders_to_words(type_):
    event = {"id": 1, "job_id": 7, "at": NOW, "type": type_, "item_id": 3, "phase": "decoding", "severity": "info"}
    event["message"] = "m"
    event["data"] = {"owner": "w", "attempt": 2, "fence": 3, "error": "boom", "did": 4, "failed": 1, "seconds": 2.5}
    words = console.describe(event)
    assert words.strip(), type_
    told = console.envelope(event)
    assert told.text == words
    assert told.type == type_


def test_an_item_failure_and_a_worker_defect_are_distinct_conditions():
    assert console.CONDITIONS["item.failed"] != console.CONDITIONS["worker.turn_failed"]
    assert "continues" in console.describe({"type": "item.failed", "item_id": 1, "data": {"error": "x"}})
    crashed = console.describe(
        {
            "type": "worker.turn_failed",
            "item_id": 1,
            "data": {"exception": "TypeError", "error": "x", "lease_until": 1.0},
        }
    )
    assert "CRASHED" in crashed
    assert "reclaimable" in crashed


def test_the_ledger_refuses_what_it_cannot_render(db):
    job_id = jobs.submit(db, "embed", NOW, items=[1])
    with pytest.raises(ValueError, match="not an event type"):
        ledger.record(db, job_id, "job.exploded", NOW)
    with pytest.raises(ValueError, match="not a severity"):
        ledger.record(db, job_id, "job.done", NOW, severity="loud")


# --- the runner persists every transition and speaks it after commit ---------


PER_ITEM = [
    "item.started",
    "phase.started",
    "phase.progress",
    "item.observed",
    "phase.finished",
    "phase.started",
    "phase.finished",
]


def test_a_turn_is_a_typed_append_only_history_with_item_starts_and_phases(db):
    job_id = jobs.submit(db, "embed", NOW, items=[1, 2, 3])
    handler = Reporting(fails_on=2)
    said = Spoken()
    turn = runner.run_next(
        db, "w1", NOW + 1, handlers={"embed": handler}, on_progress=said.progress, on_event=said.event
    )
    assert turn == {"job": job_id, "state": "done", "did": 3, "failed": 1}
    assert handler.saw_started == [1, 2, 3], "item.started is committed before the handler runs"
    types = _types(db, job_id)
    assert types[:2] == ["job.submitted", "job.claimed"]
    assert types[-1] == "job.done"
    assert types[2 : 2 + len(PER_ITEM) + 1] == [*PER_ITEM, "item.done"]
    failed_at = types.index("item.failed")
    assert types[failed_at - len(PER_ITEM) : failed_at] == PER_ITEM, "a failed item's phases survive its rollback"
    # every persisted row was spoken, with its id, after its commit; the
    # pending reports were spoken too, without ids
    persisted = [e for e in said.events if not e.get("pending")]
    rows = db.execute(
        "SELECT id FROM job_event WHERE job_id = ? AND type <> 'job.submitted' ORDER BY id", (job_id,)
    ).fetchall()
    assert [e["id"] for e in persisted] == [row[0] for row in rows]
    pending = [e for e in said.events if e.get("pending")]
    assert len(pending) == 3 * 6
    assert all("id" not in e for e in pending)
    assert {e["type"] for e in pending} == {"phase.started", "phase.progress", "item.observed", "phase.finished"}
    failed = db.execute("SELECT message, severity, data FROM job_event WHERE type = 'item.failed'").fetchone()
    assert (failed[0], failed[1]) == ("item 2 is corrupt", "warning")
    assert json.loads(failed[2])["job_continues"] is True


def test_a_finished_phase_says_how_long_it_took(db):
    """`phase.finished` carries its own duration.

    Without it, anything that wanted to know where a job's time went would
    have to pair the started/finished events itself and subtract their `at`
    stamps, which makes "which phase is slow" a question only a program can
    answer.

    The duration is `perf_counter`, not the ledger's clock. This test
    pins `at` to NOW for every row, so a clock-derived elapsed would be
    exactly 0.0 for every phase -- a plausible-looking zero rather than a
    visible absence, which is the failure this asserts against.
    """
    slept = 0.02

    def slow(conn, item_id, payload, now):
        told = runner.report()
        told.phase("dawdling")
        time.sleep(slept)
        told.phase("brisk")

    job_id = jobs.submit(db, "embed", NOW, items=[1])
    runner.run_next(db, "w1", NOW + 1, handlers={"embed": slow})

    finished = db.execute(
        "SELECT phase, at, message, data FROM job_event WHERE job_id = ? AND type = 'phase.finished' ORDER BY id",
        (job_id,),
    ).fetchall()
    took = {}
    for phase, at, message, data in finished:
        assert at == NOW + 1, "the ledger stamp is still the turn's clock"
        held = json.loads(data)
        assert "elapsed_ms" in held, f"{phase} finished without saying how long it took"
        assert "ms" in message, "the message a person reads carries it too"
        took[phase] = held["elapsed_ms"]

    assert set(took) == {"dawdling", "brisk"}
    # The phase that slept must report having slept. A duration derived
    # from `at` could not: this turn stamps every row NOW + 1, so it would
    # read 0.0 here and look plausible while measuring nothing.
    assert took["dawdling"] >= slept * 1000 * 0.5, f"the slow phase reported {took['dawdling']} ms"
    assert took["brisk"] < took["dawdling"], "and the phase that did nothing is the shorter of the two"


def test_a_worker_defect_is_recorded_with_its_traceback_and_the_job_stays_running(db):
    job_id = jobs.submit(db, "embed", NOW, items=[1, 2])
    said = Spoken()
    with pytest.raises(TypeError):
        runner.run_next(db, "w1", NOW + 1, handlers={"embed": Reporting(crashes_on=2)}, on_event=said.event)
    assert jobs.snapshot(db, job_id)["state"] == "running"
    row = db.execute("SELECT item_id, severity, data FROM job_event WHERE type = 'worker.turn_failed'").fetchone()
    assert (row[0], row[1]) == (2, "error")
    data = json.loads(row[2])
    assert data["exception"] == "TypeError"
    assert "the encoder grew a bug" in data["traceback"]
    assert (data["reclaimable"], data["fence"], data["attempt"]) == (True, 1, 1)
    assert said.events[-1]["type"] == "worker.turn_failed", "the crash is spoken before the exception propagates"
    assert _types(db, job_id).count("item.done") == 1, "item 1's history survived the crash on item 2"


def test_a_lapsed_lease_is_a_reclaim_and_a_pause_is_a_resume(db):
    job_id = jobs.submit(db, "embed", NOW, items=[1, 2, 3])
    runner.run_next(db, "w1", NOW + 1, handlers={"embed": Reporting()}, budget=1)
    assert _types(db, job_id)[-1] == "job.paused"
    runner.run_next(db, "w2", NOW + 2, handlers={"embed": Reporting()}, budget=1)
    claims = db.execute(
        "SELECT type, data FROM job_event WHERE job_id = ? AND type IN ('job.claimed','job.reclaimed') ORDER BY id",
        (job_id,),
    ).fetchall()
    assert [c[0] for c in claims] == ["job.claimed", "job.claimed"]
    assert json.loads(claims[1][1])["resumed"] is True
    # now the lease lapses for real: a crash, then a claim after LEASE_SECONDS
    with pytest.raises(TypeError):
        runner.run_next(db, "w3", NOW + 3, handlers={"embed": Reporting(crashes_on=3)})
    runner.run_next(db, "w4", NOW + 3 + jobs.LEASE_SECONDS + 1, handlers={"embed": Reporting()})
    assert "job.reclaimed" in _types(db, job_id)
    assert jobs.snapshot(db, job_id)["state"] == "done"


def test_cancellation_progresses_through_request_cooperative_stop_and_terminal(db):
    job_id = jobs.submit(db, "embed", NOW, items=[1, 2])
    jobs.cancel(db, job_id, NOW + 0.5)
    jobs.cancel(db, job_id, NOW + 0.6)  # a second press records nothing new
    assert inspecting.job_detail(db, job_id, NOW + 1)["derived"]["cancellation"] == "requested"
    runner.run_next(db, "w1", NOW + 1, handlers={"embed": Reporting()})
    types = _types(db, job_id)
    assert types.count("job.cancel_requested") == 1
    assert types[-1] == "job.cancelled"
    assert types.index("job.cancel_requested") < types.index("job.claimed") < types.index("job.cancelled")
    assert inspecting.job_detail(db, job_id, NOW + 2)["derived"]["cancellation"] == "cancelled"


def test_a_checkpoint_move_is_an_event(db):
    job_id = jobs.submit(db, "embed", NOW, payload={"x": 1})
    claimed = jobs.claim(db, "w1", NOW)
    assert claimed is not None
    _, fence = claimed
    jobs.checkpoint(db, job_id, fence, {"page": 3}, 30, at=NOW + 1)
    row = db.execute("SELECT data FROM job_event WHERE type = 'checkpoint.changed'").fetchone()
    assert json.loads(row[0]) == {"checkpoint": {"page": 3}, "done": 30, "fence": fence}


# --- the read model exposes what the row knows -------------------------------


CONTRACT_FIELDS = {
    "id",
    "kind",
    "target",
    "state",
    "payload",
    "created_at",
    "started_at",
    "finished_at",
    "total",
    "done_count",
    "failed_count",
    "pending_count",
    "attempt",
    "owner",
    "fence",
    "heartbeat_at",
    "lease_until",
    "checkpoint",
    "error",
    "current",
    "failures",
    "defects",
    "attempts",
    "derived",
    "recent_events",
    "event_count",
}


def test_job_detail_exposes_every_contract_field_and_redacts_secrets(db):
    payload = {"models_dir": "D:/m", "api_key": "hunter2", "nested": {"token": "t"}}
    job_id = jobs.submit(db, "embed", NOW, payload=payload, items=[1, 2, 3])
    handler = Reporting(fails_on=2)
    runner.run_next(db, "w1", NOW + 1, handlers={"embed": handler}, clock=lambda: NOW + 5)
    told = inspecting.job_detail(db, job_id, NOW + 10)
    assert set(told) >= CONTRACT_FIELDS, sorted(CONTRACT_FIELDS - set(told))
    assert told["payload"] == {"models_dir": "D:/m", "api_key": ledger.REDACTED, "nested": {"token": ledger.REDACTED}}
    assert told["failures"] == [{"id": 2, "name": None, "href": None, "error": "item 2 is corrupt"}]
    assert (told["failed_count"], told["succeeded_count"], told["pending_count"]) == (1, 2, 0)
    d = told["derived"]
    assert (d["elapsed"], d["queue_wait"], d["fraction"]) == (4.0, 1.0, 1.0)
    assert d["cancellation"] == "not_requested"
    assert d["rate"] == pytest.approx(3 / 4)
    assert d["eta"] is None, "a settled job has no ETA"
    assert (told["attempts"][0]["type"], told["attempts"][0]["fence"]) == ("job.claimed", 1)
    assert told["event_count"] == len(_types(db, job_id))
    assert told["current"]["item"] is None
    assert told["current"]["last_settled_phase"]["phase"] == "embedding"


def test_job_detail_names_the_lease_and_the_defect_while_running(db):
    job_id = jobs.submit(db, "embed", NOW, items=[1, 2])
    with pytest.raises(TypeError):
        runner.run_next(db, "w1", NOW + 1, handlers={"embed": Reporting(crashes_on=2)}, clock=lambda: NOW + 2)
    told = inspecting.job_detail(db, job_id, NOW + 3)
    assert (told["state"], told["owner"], told["fence"]) == ("running", "w1", 1)
    assert told["derived"]["heartbeat_age"] == pytest.approx(1.0)
    assert told["derived"]["lease_remaining"] == pytest.approx(jobs.LEASE_SECONDS - 1.0)
    assert told["current"]["item"] is None, "the crash settled nothing; the ledger's last word is the defect"
    assert len(told["defects"]) == 1
    assert told["defects"][0]["item"]["id"] == 2
    assert "traceback" in told["defects"][0]


def test_the_overview_reads_the_worker_the_queue_and_the_ledger_head(db):
    jobs.submit(db, "embed", NOW, items=[1])
    jobs.submit(db, "hash", NOW + 1, items=[1])
    jobs.claim(db, "worker-1", NOW + 2)
    told = inspecting.overview(db, NOW + 3)
    assert told["queue"] == {
        "queued": 1,
        "running": 1,
        "oldest_queued_age": 2.0,
        "oldest_running_age": 3.0,
        "settled_24h": {},
    }
    assert told["worker"]["owners"] == ["worker-1"]
    assert told["worker"]["working"] is True
    assert told["worker"]["heartbeat_age"] == pytest.approx(1.0)
    assert told["ledger"] == {"last_id": 2, "events": 2}, "two submits; a bare claim records nothing"


# --- nothing is sampled: pages sum to the whole -------------------------------


def test_every_event_of_a_large_job_is_reachable_by_paging(db):
    n = 300
    job_id = jobs.submit(db, "embed", NOW, items=list(range(1, n + 1)))
    runner.run_next(db, "w1", NOW + 1, handlers={"embed": lambda *a: None})
    produced = 1 + 1 + 2 * n + 1  # submitted, claimed, started+done per item, done
    assert ledger.count_for_job(db, job_id) == produced
    seen: list[int] = []
    after = 0
    while True:
        page = inspecting.events(db, job_id=job_id, after=after, limit=100)
        seen.extend(e["id"] for e in page["events"])
        if page["next_after"] is None:
            break
        after = page["next_after"]
    assert len(seen) == produced, "paging dropped an event"
    assert seen == sorted(set(seen)), "paging repeated an event"
    earlier = inspecting.events_before(db, seen[50], job_id=job_id, limit=20)
    assert [e["id"] for e in earlier] == seen[30:50]
    done = inspecting.items(db, job_id, state="done", after=0, limit=100)
    assert (len(done["items"]), done["next_after"]) == (100, 100)


# --- the application: feed, routes, shell -----------------------------------


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("console")
    root = tmp / "lib"
    root.mkdir()
    for i in range(6):
        Image.new("RGB", (8, 8), (20 * i, 80, 100)).save(root / f"c_{i}.png")
    with TestClient(app=build_app(str(tmp / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        yield client


@pytest.fixture(scope="module")
def _bare_stage(tmp_path_factory):
    """One application over an EMPTY home, for the tests that bring their
    own library. Each was building its own -- an interpreter's worth of
    imports and a migration -- to register a root and read it back."""
    with hosting(tmp_path_factory, "console_bare") as stage:
        yield stage


@pytest.fixture
def bare(_bare_stage):
    """That application with nothing in it: the snapshot is restored, so
    `/roots` numbers from 1 again and no test inherits another's library."""
    _bare_stage.restore()
    return _bare_stage.client


def _turn(client, job_id: int) -> dict:
    """Worker turns, speaking on the app's channels, until `job_id` is
    settled -- the queue holds other jobs (the scan's thumbnail job), and
    a turn claims the oldest first."""
    conn = connect.connect(client.app.state.db_path)
    try:
        while True:
            told = runner.run_next(conn, "test-worker", NOW, on_event=client.app.state.publish_event)
            conn.commit()
            assert told is not None, f"nothing runnable, job {job_id} never settled"
            if told["job"] == job_id:
                return told
    finally:
        connect.close(conn)


def test_the_feed_sends_the_backlog_then_live_rows_with_contiguous_ids(served):
    client = served
    job_id = client.post("/jobs/verify").json()["id"]
    with client.websocket_connect("/ws/events?after=0") as feed:
        backlog = feed.receive_json(timeout=10)
        assert backlog["frame"] == "backlog"
        held = [e["id"] for e in backlog["events"]]
        assert held
        assert held[-1] == backlog["last_id"]
        assert any(e["type"] == "job.submitted" and e["job_id"] == job_id for e in backlog["events"])
        assert all("text" in e for e in backlog["events"]), "every frame carries its words"
        turn = _turn(client, job_id)
        assert turn["state"] == "done"
        live = []
        while not any(e["type"] == "job.done" and e["job_id"] == job_id for e in live):
            frame = feed.receive_json(timeout=10)
            assert frame["frame"] in ("event", "pending")
            if frame["frame"] == "pending":
                assert "id" not in frame, "a pending report carries no id"
                continue
            live.append(frame)
        ids = held + [e["id"] for e in live]
        assert ids == list(range(ids[0], ids[0] + len(ids))), "the ids skipped or repeated"
        mine = [e["type"] for e in live if e["job_id"] == job_id]
        assert mine[:2] == ["job.claimed", "item.started"]
        assert mine[-1] == "job.done"
    # a reconnect resumes from the last id held: nothing repeated, nothing lost
    with client.websocket_connect(f"/ws/events?after={ids[-1]}") as again:
        told = again.receive_json(timeout=10)
        assert (told["frame"], told["events"], told["last_id"]) == ("backlog", [], ids[-1])
    with client.websocket_connect(f"/ws/events?after={ids[-3]}") as again:
        told = again.receive_json(timeout=10)
        assert [e["id"] for e in told["events"]] == ids[-2:]


def test_the_console_routes_answer_the_rows(served):
    client = served
    job_id = client.post("/jobs/verify").json()["id"]
    page = client.get("/operations", headers={"accept": "text/html"})
    assert page.status_code == 200
    for marker in (
        "data-console",
        "data-health-transport",
        "data-matrix-rows",
        "data-tape-scroll",
        "data-inspector-body",
    ):
        assert marker in page.text, marker
    assert f'data-matrix-job="{job_id}"' in page.text
    detail = client.get(f"/operations/job/{job_id}", headers={"accept": "application/json"}).json()
    assert (detail["state"], detail["derived"]["cancellation"]) == ("queued", "not_requested")
    fragment = client.get(f"/operations/job/{job_id}", headers={"accept": "text/html"})
    assert "<html" not in fragment.text
    assert 'data-block="execution"' in fragment.text
    assert client.get("/operations/job/999999", headers={"accept": "application/json"}).status_code == 404
    overview = client.get("/operations/overview").json()
    assert overview["overview"]["queue"]["queued"] >= 1
    events = client.get(f"/operations/events?job={job_id}").json()
    assert [e["type"] for e in events["events"]] == ["job.submitted"]
    items = client.get(f"/operations/job/{job_id}/items?state_filter=pending").json()
    assert len(items["items"]) == 6
    assert items["items"][0]["href"].startswith("/i/")
    assert client.get(f"/operations/job/{job_id}/items?state_filter=sideways").status_code == 400
    fragment = client.get(
        f"/operations/job/{job_id}/items?state_filter=pending&limit=4", headers={"accept": "text/html"}
    )
    assert fragment.status_code == 200
    assert fragment.text.count("data-item=") == 4
    assert "data-items-more" in fragment.text, "a fuller page offers the next one"
    assert "<html" not in fragment.text
    cursor = client.get(f"/operations/job/{job_id}/items?state_filter=pending&limit=4").json()["next_after"]
    rest = client.get(
        f"/operations/job/{job_id}/items?state_filter=pending&limit=4&after={cursor}", headers={"accept": "text/html"}
    )
    assert rest.text.count("data-item=") == 2
    assert "data-items-more" not in rest.text


def test_a_cancel_is_spoken_on_both_feeds_and_settles_cooperatively(served):
    client = served
    job_id = client.post("/jobs/verify").json()["id"]
    with client.websocket_connect("/ws/events?after=0") as feed:
        feed.receive_json(timeout=10)
        assert client.post(f"/jobs/{job_id}/cancel").json()["cancel_requested"] == 1
        asked = feed.receive_json(timeout=10)
        assert (asked["frame"], asked["type"], asked["job_id"]) == ("event", "job.cancel_requested", job_id)
        detail = client.get(f"/operations/job/{job_id}", headers={"accept": "application/json"}).json()
        assert detail["derived"]["cancellation"] == "requested"
        # the worker reaches the job: the cooperative stop is its own event
        turn = _turn(client, job_id)
        assert turn["state"] == "cancelled"
        mine = []
        while not mine or mine[-1]["type"] != "job.cancelled":
            frame = feed.receive_json(timeout=10)
            if frame["job_id"] == job_id:
                mine.append(frame)
        assert [e["type"] for e in mine] == ["job.claimed", "job.cancelled"]
        assert "boundary" in mine[-1]["text"]
    detail = client.get(f"/operations/job/{job_id}", headers={"accept": "application/json"}).json()
    assert detail["derived"]["cancellation"] == "cancelled"


def test_the_phase_inside_a_running_item_survives_a_reconnect(served):
    """A handler's report lands in the ledger only when its item settles;
    between, the inspector answers it from the process's live memory, so
    a console that reconnects mid-item sees the phase instead of
    "waiting". A committed row that settles the item clears it."""
    client = served
    job_id = client.post("/jobs/verify").json()["id"]
    conn = connect.connect(client.app.state.db_path)
    try:
        # claim THIS job so it is running; other queued jobs may precede it
        while jobs.snapshot(conn, job_id)["state"] != "running":
            assert jobs.claim(conn, "w-live", NOW) is not None
        conn.commit()
    finally:
        connect.close(conn)
    publish_event = client.app.state.publish_event
    publish_event(
        {
            "pending": True,
            "job_id": job_id,
            "at": NOW + 1,
            "type": "phase.progress",
            "item_id": 3,
            "phase": "decoding",
            "severity": "info",
            "message": "48 / 220 frames",
            "data": {"unit": "frames", "done": 48, "total": 220},
        }
    )
    told = client.get(f"/operations/job/{job_id}", headers={"accept": "application/json"}).json()
    assert told["current"]["phase"]["phase"] == "decoding"
    matrix = client.get("/operations/overview").json()
    row = next(j for j in matrix["matrix"] if j["id"] == job_id)
    assert row["live"]["phase"] == "decoding", "the matrix row says what its job is inside"
    assert matrix["overview"]["worker"]["thread_alive"] is False, "this app was built without a worker thread"
    assert "data-matrix-live" in client.get("/operations", headers={"accept": "text/html"}).text
    assert told["current"]["phase"]["live"] is True
    assert "48 / 220 frames" in told["current"]["phase"]["text"]
    page = client.get(f"/operations/job/{job_id}", headers={"accept": "text/html"}).text
    assert "data-current-phase data-live" in page
    assert "decoding" in page
    publish_event(
        {
            "id": 10**9,
            "job_id": job_id,
            "at": NOW + 2,
            "type": "item.done",
            "item_id": 3,
            "severity": "info",
            "message": "item 3 done",
            "data": None,
        }
    )
    told = client.get(f"/operations/job/{job_id}", headers={"accept": "application/json"}).json()
    assert told["current"]["phase"] is None, "a settled item clears its live report"


def test_the_shell_counts_what_is_running(served):
    client = served
    job_id = client.post("/jobs/verify").json()["id"]
    page = client.get("/g", headers={"accept": "text/html"}).text
    assert re.search(r"activity · \d+ running", page), "the shell summary names the live count"
    assert _turn(client, job_id)["state"] == "done"


def test_every_job_kind_has_words_beside_its_raw_name(bare, tmp_path):
    """The console shows what a job does AND the schema's name for it.
    A kind the schema admits but the console cannot word is a row that
    reads as its identifier. The vocabulary is read from db/jobs.py, the
    one place it is spelled -- sglint SG709 holds that equal to the
    schema's CHECK, so this file never parses the DDL to learn it."""
    assert set(console.KINDS) == set(typing.get_args(jobs.JobKind))
    assert console.describe_kind("hash") == "verify every file's bytes"
    assert console.describe_kind("hash", "groups") == "group perceptual copies"
    assert console.describe_kind("annotate") == "caption every picture"
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "one.png")
    # No worker: every claim below is about how the console WORDS these
    # two jobs, and a running one fingerprints and groups the library to
    # tell it nothing it did not already know from the queued rows.
    client = bare
    client.post("/roots", json={"path": str(root)})
    client.post("/roots/1/scan")
    fingerprint = client.post("/operations/jobs/phash").text
    dupes = client.post("/operations/jobs/dupes").text
    assert "queued #" in fingerprint
    assert "queued #" in dupes
    matrix = client.get("/operations/overview").json()["matrix"]
    told = {(row["kind"], row.get("derive")): row["what"] for row in matrix}
    # The claim is that the MODE is told apart -- these are two acts behind one
    # kind. The count comes from the row and depends on the library, so this
    # asserts the shape rather than a sentence.
    assert told[("hash", "perceptual")].startswith("fingerprint ")
    assert "group perceptual copies" in told[("hash", "groups")]
    assert told[("hash", "perceptual")] != told[("hash", "groups")]
    page = client.get("/operations", headers={"accept": "text/html"}).text
    assert "fingerprint " in page
    assert '<code class="raw">hash</code>' in page
    one = next(row["id"] for row in matrix if row.get("derive") == "perceptual")
    detail = client.get(f"/operations/job/{one}", headers={"accept": "application/json"}).json()
    # The inspector reads the same line as the matrix -- one library,
    # one job, one description, whichever surface asks.
    assert detail["what"] == told[("hash", "perceptual")]
    assert detail["what"].startswith("fingerprint ")
    inspector = client.get(f"/operations/job/{one}", headers={"accept": "text/html"}).text
    assert detail["what"] in inspector


def test_the_console_says_what_each_sweep_still_has_to_do(bare, tmp_path):
    """Coverage beside the buttons: present files, and per missing-only
    sweep how many it would still queue -- counted the way the sweep
    counts, so the number beside the button and the job it queues agree.
    A sweep that ran takes its count to zero."""
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (1, 2, 3 + i)).save(root / f"p{i}.png")
    client = bare
    client.post("/roots", json={"path": str(root)})
    client.post("/roots/1/scan")
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", time.time()) is not None:  # the scan's precache job
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)
    told = client.get("/operations/overview").json()["overview"]["coverage"]
    assert told["files"] == 2
    assert told["missing"] == {"annotate": 2, "context": 2, "embed": 2, "faces": 2, "ingest": 2, "phash": 2}
    assert list(told["embed_spaces"].values()) == [2], "one configured space, nothing minted: every picture"
    page = client.get("/operations", headers={"accept": "text/html"}).text
    assert re.search(r'data-missing="phash"[^>]*>2 missing', page)
    assert re.search(r'data-missing="ingest"[^>]*>2 missing', page)
    assert 'data-missing="verify"' not in page, "verify reads everything by design; no count"

    assert client.post("/jobs/phash").status_code == 201
    assert client.post("/jobs/ingest").status_code == 201
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", time.time()) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)
    after = client.get("/operations/overview").json()["overview"]["coverage"]["missing"]
    assert (after["phash"], after["ingest"]) == (0, 0)
    assert after["context"] == 2, "untouched sweeps keep their count"
    assert client.post("/jobs/phash").status_code == 204, "the count beside the button and the job agree"


def test_the_activity_surface_words_the_hash_kinds_mode_too():
    """The drawer on every page reads the same words as the console: a
    perceptual job is not "verify every file's bytes" there."""
    from sg_web import activity

    row = {"id": 1, "kind": "hash", "state": "queued", "done_count": 0, "total": 3, "cancel_requested": 0}
    # And it reads THIS job's numbers, not its kind's: the row says 3.
    assert activity.row_view({**row, "derive": "perceptual"})["what"] == "fingerprint 3 pictures"
    assert activity.row_view({**row, "derive": None})["what"] == "verify the bytes of 3 files"
    delta = {"job": 1, "kind": "hash", "state": "running", "done": 1, "total": 3, "derive": "groups"}
    assert activity.delta_view(delta)["what"] == "group perceptual copies across 3 pictures"


def test_a_phase_speaks_on_the_delta_feed_while_it_is_true(db):
    """The activity row's live line: a beginning or progressing phase
    rides the DELTA feed as `doing`, not only the events channel -- a
    one-item clustering held at "running, 0 / 1" for its whole life
    because its phases were audible only to the expert console."""
    jobs.submit(db, "embed", NOW, items=[1])
    said = Spoken()
    runner.run_next(db, "w1", NOW + 1, handlers={"embed": Reporting()}, on_progress=said.progress)
    doing = [(d["state"], d["doing"]) for d in said.deltas]
    assert doing[0] == ("running", None), "the claim is a boundary, not a phase"
    assert ("running", "decoding") in doing
    assert ("running", "decoding: 48 / 220 frames") in doing, "progress carries its phase and its own words"
    assert ("running", "embedding") in doing
    assert doing[-1] == ("done", None), "a terminal delta clears the line"


def test_the_activity_row_renders_the_doing_line_only_while_there_is_one(served):
    """On the module's application, not one of its own: this asks what a
    delta RENDERS to, so all it needs is the template engine. Building a
    whole application for that cost more than the assertion."""
    from sg_web import activity

    engine = served.app.template_engine
    delta = {"job": 5, "kind": "embed", "state": "running", "done": 0, "total": 1, "doing": "clustering"}
    row = activity.render_delta(engine, delta, {5})
    assert "job-doing" in row
    assert "clustering" in row
    settled = activity.render_delta(engine, {**delta, "state": "done", "doing": None}, {5})
    assert "job-doing" not in settled


def test_no_count_sits_beside_a_sweep_that_would_be_refused(tmp_path):
    conn = fresh_schema()
    from db import settings

    settings.put(conn, "caption_model", "blip")
    told = inspecting.coverage(conn)
    assert "annotate" not in told["missing"], "the sweep refuses that setting; a count beside it would lie"
    settings.put(conn, "caption_model", "Salesforce/blip-image-captioning-base")
    assert "annotate" in inspecting.coverage(conn)["missing"]


def test_the_tape_pages_backwards_through_the_route_without_a_gap_or_a_repeat(bare, tmp_path):
    """`/operations/events/before` is the tape's "earlier" button: paging
    down from the newest id reaches every persisted event once, and
    meets `/operations/events?after=` coming up -- never sampled."""
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), (1, 2, 3 + i)).save(root / f"p{i}.png")
    client = bare
    client.post("/roots", json={"path": str(root)})
    client.post("/roots/1/scan")
    client.post("/jobs/ingest")
    client.post("/jobs/context")
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", time.time()) is not None:
            conn.commit()
        conn.commit()
        produced = ledger.count(conn)
    finally:
        connect.close(conn)
    assert produced > 6
    forward: list[int] = []
    after = 0
    while True:
        page = client.get("/operations/events", params={"after": after, "limit": 3}).json()
        forward.extend(e["id"] for e in page["events"])
        if page.get("next_after") is None or not page["events"]:
            break
        after = page["next_after"]
    assert len(forward) == produced
    backward: list[int] = []
    before = forward[-1] + 1
    while True:
        page = client.get("/operations/events/before", params={"before": before, "limit": 3}).json()
        if not page["events"]:
            break
        ids = [e["id"] for e in page["events"]]
        assert ids == sorted(ids), "a page is ascending"
        backward = ids + backward
        before = ids[0]
    assert backward == forward, "backwards reaches every event once and meets the forward walk"


# --- saying what THIS job is, not what its kind is --------------------------


def test_a_job_says_its_own_numbers_not_its_kind(db):
    """The complaint, exactly: every description was the same all the
    time. `job.total`, the hash mode and a walk's own root path were all
    on the row and none of them was read, so "read every file's
    metadata" was the line for four files and for eighty thousand."""
    from sg_web import console

    assert console.describe_kind("scan", None, 412) == "read metadata for 412 files"
    assert console.describe_kind("scan", None, 1) == "read metadata for 1 file"
    assert console.describe_kind("embed", None, 400) == "embed 400 pictures for search"
    # The LEAF, never the path: the activity strip carries this line onto
    # /folders and /f/<slug>, whose rule is that a place is entered by entity.
    # `root.path` is where a library sits, not what it is (schema.sql root.uuid).
    assert console.describe_kind("walk", None, None, "D:/Photos/2019") == "look for files under 2019"
    assert console.describe_kind("walk", None, None, "D:/") == "look for files under D:/", (
        "a root with no leaf still says something"
    )


def test_four_acts_behind_one_kind_read_as_four(db):
    """`hash` is verify, fingerprint, render thumbnails and group copies,
    told apart only by the payload's `derive` -- and one of them is the
    one somebody would cancel."""
    from sg_web import console

    assert console.describe_kind("hash", "thumbs", 1204) == "render 1,204 missing thumbnails"
    assert console.describe_kind("hash", None, 1204) == "verify the bytes of 1,204 files"
    assert console.describe_kind("hash", "perceptual", 9) != console.describe_kind("hash", "groups", 9)


def test_a_job_with_no_count_still_says_something_true(db):
    """`every` survives where the items were never enumerable. Inventing
    a number for a job that has none would be worse than the constant it
    replaced."""
    from sg_web import console

    assert console.describe_kind("scan") == "read every file's metadata"
    assert console.describe_kind("scan", None, 0) == "read every file's metadata"
    assert console.describe_kind("walk") == "look for files on disk"


def test_an_item_is_named_where_the_runner_knows_the_name(db):
    """ "item 41 started" is a hundred thousand lines naming an integer
    nobody can resolve. The file's name was one indexed lookup away."""
    from sg_web import console

    told = console.describe(
        {"type": "item.started", "item_id": 41, "data": {"item_name": "DSC_0042.NEF"}, "message": "item 41 started"}
    )
    assert "41" in told, "the number stays: it is what job_item is keyed on"
    assert "DSC_0042.NEF" in told


def test_an_item_the_runner_cannot_name_still_reads(db):
    """A `cluster_faces` item is an index into a payload, not a file --
    so there is no name, and inventing one by looking up the integer as
    a file id would put a real picture's name beside item 2 of a
    clustering run."""
    from sg_web import console

    told = console.describe({"type": "item.started", "item_id": 2, "data": {}, "message": "item 2 started"})
    assert told == "item 2 started"


def test_the_observation_name_and_the_file_name_stay_apart(db):
    """Two facts, two keys. Read from one, an observed event rendered
    its own name twice and the file's not at all."""
    from sg_web import console

    told = console.describe(
        {
            "type": "item.observed",
            "item_id": 7,
            "data": {"name": "captioned", "item_name": "beach.png", "words": 12},
            "message": "",
        }
    )
    assert "beach.png" in told
    assert told.count("captioned") == 1


# --- the operating point is reachable ----------------------------------------


def test_the_face_threshold_is_a_setting_and_auto_means_measured(db):
    """The knob, and its default.

    Per-embedder operating points were constants in vision/faces.py:
    changing one was an edit and a restart. "auto" keeps the measured
    point (db/derived.py SAME_PERSON), which is what it should stay
    unless somebody is deliberately experimenting -- the spaces are not
    comparable and one number is wrong for all but one of them.
    """
    from db import runner, settings

    assert settings.value(db, "face_cluster_threshold") == "auto"
    assert runner.chosen_threshold(db) is None, "auto must not pin a number over the measured one"

    settings.put(db, "face_cluster_threshold", "0.62")
    assert runner.chosen_threshold(db) == pytest.approx(0.62)


def test_a_threshold_that_could_not_mean_anything_is_refused_at_submit(db):
    """Validated where `dupe_threshold` is, and for the same reason: a
    bad value must be a refused submit, never a job that fails on its
    third item.

    The bounds are wide on purpose -- this is somebody's own library and
    the point of the knob is finding out what a different point does.
    What is refused is where the answer is not interesting but broken."""
    from db import runner, settings

    for bad in ("0", "1", "1.5", "-0.2", "tight"):
        settings.put(db, "face_cluster_threshold", bad)
        with pytest.raises(ValueError, match="face_cluster_threshold"):
            runner.chosen_threshold(db)


def test_the_operating_point_is_pinned_at_submit_not_read_per_item(db):
    """One run, one threshold. Read per item, a setting changed while the
    job runs gives two embedding spaces two different answers inside one
    run -- and the run row records a single number for both."""
    import numpy as np

    from db import derived, runner, scan, settings

    root = int(db.execute("INSERT INTO root(path, kind, created_at) VALUES('Z:/x','library',0)").lastrowid or 0)
    folder = scan.mint(db, "folder", "x")
    db.execute("INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?,?,NULL,'x',0)", (folder, root))
    file_id = scan.mint(db, "file", "one")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'one.png', 'image', 1, 0, ?, 0, 0)",
        (file_id, folder, "a" * 64),
    )
    derived.record_faces(
        db,
        file_id,
        "opencv/yunet+sface",
        "1",
        "a" * 64,
        0.0,
        [{"region": derived.region(db, 0.1, 0.1, 0.2, 0.2), "embedding": np.ones(4, np.float32).tobytes()}],
    )
    settings.put(db, "face_cluster_threshold", "0.71")
    db.commit()

    job = runner.submit_cluster(db, 0.0)
    payload = json.loads(db.execute("SELECT payload FROM job WHERE id = ?", (job,)).fetchone()[0])
    assert payload["threshold"] == pytest.approx(0.71)

    # and changing it now cannot reach the job already queued
    settings.put(db, "face_cluster_threshold", "auto")
    db.commit()
    payload = json.loads(db.execute("SELECT payload FROM job WHERE id = ?", (job,)).fetchone()[0])
    assert payload["threshold"] == pytest.approx(0.71)


def test_a_run_queued_before_the_setting_existed_still_clusters(db):
    """`.get`, not `[]`. A job sitting in the queue from an older build
    carries no threshold key, and it must cluster at the measured point
    rather than fail on a KeyError nobody can act on."""
    from db import derived

    assert derived.threshold_for("opencv/yunet+arcface") == pytest.approx(0.48)
    payload: dict = {"spaces": [["opencv/yunet+arcface", "1"]]}
    asked = payload.get("threshold")
    assert asked is None


# --- and two runs can be put side by side ------------------------------------


def _two_runs(conn):
    """One file both runs name, one they disagree about, one only one names."""
    import numpy as np

    from db import derived, scan

    root = int(conn.execute("INSERT INTO root(path, kind, created_at) VALUES('Z:/c','library',0)").lastrowid or 0)
    folder = scan.mint(conn, "folder", "c")
    conn.execute("INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?,?,NULL,'c',0)", (folder, root))
    files = []
    for i in range(3):
        file_id = scan.mint(conn, "file", f"c{i}")
        conn.execute(
            "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
            " VALUES(?, ?, ?, 'image', 1, 0, ?, 0, 0)",
            (file_id, folder, f"c{i}.png", f"{i:064d}"),
        )
        derived.record_faces(
            conn,
            file_id,
            "opencv/yunet+sface",
            "1",
            f"{i:064d}",
            0.0,
            [{"region": derived.region(conn, 0.1, 0.1, 0.2, 0.2), "embedding": np.ones(4, np.float32).tobytes()}],
        )
        files.append(file_id)
    hannah = authored_module().person(conn, "Hannah", 0.0)
    ivan = authored_module().person(conn, "Ivan", 0.0)
    left = derived.run_for(conn, "opencv/yunet+sface", "1", derived.DEFAULT_METHOD, 0.55, 0.0)
    right = derived.run_for(conn, "opencv/yunet+sface", "1", derived.DEFAULT_METHOD, 0.40, 0.0)
    # agree about the first, disagree about the second, only left names the third
    derived.attribute(conn, files[0], hannah, left, "opencv/yunet+sface", "1", face_count=1)
    derived.attribute(conn, files[0], hannah, right, "opencv/yunet+sface", "1", face_count=1)
    derived.attribute(conn, files[1], hannah, left, "opencv/yunet+sface", "1", face_count=1)
    derived.attribute(conn, files[1], ivan, right, "opencv/yunet+sface", "1", face_count=1)
    derived.attribute(conn, files[2], ivan, left, "opencv/yunet+sface", "1", face_count=1)
    conn.commit()
    return left, right, hannah, ivan, files


def authored_module():
    from db import authored

    return authored


def test_two_runs_are_compared_by_what_they_say_about_the_same_picture(db):
    """The other half of a reachable threshold.

    Trying one is safe because a new threshold writes a new run beside
    the old. That left somebody with two runs, two numbers, and no way
    to see what moved -- and the counts cannot tell them, because "more
    groups" is both what a threshold that split one person in four does
    and what one that stopped welding strangers does.
    """
    from db import pages

    left, right, _hannah, _ivan, files = _two_runs(db)
    held = pages.disagreements(db, left, right)

    assert held["total"] == 2, "the picture both name the same way is not a disagreement"
    named = {one["name"]: (one["left_says"], one["right_says"]) for one in held["pictures"]}
    assert named["c1.png"] == ("Hannah", "Ivan")
    assert named["c2.png"] == ("Ivan", None), "a run naming nobody is an answer, not a missing value"
    assert "c0.png" not in named
    assert {one["id"] for one in held["pictures"]} == {files[1], files[2]}


def test_both_columns_spell_their_people_the_same_readable_way(db):
    """Side by side, so spelled alike -- and by NAME.

    Without an ORDER BY, `group_concat` concatenates in scan order,
    which for this WITHOUT ROWID table is person_id: the order the
    people were CREATED in. That is consistent between the two columns,
    so the comparison is sound either way -- this is not a correctness
    fix and should not be mistaken for one. It is that "Ivan, Hannah" is
    the order somebody happened to be added to the library in, and a
    reader scanning two columns for a name is looking alphabetically.

    Measured on 3.47.1: unordered both sides read "Ivan,Hannah".
    """
    from db import authored, derived, pages

    left, right, hannah, _ivan, files = _two_runs(db)
    # Created LAST and sorting FIRST, which is the only arrangement that
    # can tell the two orders apart. Named alike and the test passes
    # whether or not the query orders anything.
    aaron = authored.person(db, "Aaron", 0.0)
    assert aaron > hannah, "the fixture must create Aaron after Hannah for this to discriminate"
    for who in (hannah, aaron):
        derived.attribute(db, files[0], who, left, "opencv/yunet+sface", "1", face_count=1)
    derived.attribute(db, files[0], hannah, right, "opencv/yunet+sface", "1", face_count=1)
    db.commit()

    held = pages.disagreements(db, left, right)
    said = {one["name"]: one["left_says"] for one in held["pictures"]}
    assert said["c0.png"] == "Aaron,Hannah", said["c0.png"]


def test_the_comparison_says_how_many_it_did_not_show(db):
    """A bounded list alone cannot tell "these are the only twelve" from
    "the first fifty of nine thousand", and those are opposite answers to
    the question being asked."""
    from db import pages

    left, right, _hannah, _ivan, _files = _two_runs(db)
    held = pages.disagreements(db, left, right, limit=1)
    assert held["shown"] == 1
    assert held["total"] == 2, "the total must survive the limit"


def test_the_console_offers_the_comparison_and_renders_it(bare):
    """Reachable from the panel where the threshold is changed, against
    the run the site is actually showing."""
    from db import connect, derived

    client = bare
    conn = connect.connect(client.app.state.db_path)
    try:
        left, right, _hannah, _ivan, _files = _two_runs(conn)
        derived.make_primary(conn, left)
        conn.commit()
    finally:
        connect.close(conn)

    page = client.get("/operations", headers={"accept": "text/html"}).text
    assert f'data-compare-run="{right}"' in page, "no way to compare from the panel"
    assert f'data-compare-run="{left}"' not in page, "the primary has nothing to be compared against"

    told = client.get(f"/operations/clusterings/{left}/against/{right}", headers={"accept": "text/html"})
    assert told.status_code == 200, told.text
    assert 'data-compare-total="2"' in told.text
    assert "Hannah" in told.text
    assert "Ivan" in told.text

    assert client.get(f"/operations/clusterings/{left}/against/424242").status_code == 404


def test_every_setting_the_registry_models_is_on_the_page():
    """A setting nobody can reach is a setting that does not ship.

    `db/settings.py REGISTRY` is the vocabulary -- what this application
    can be configured to do. `sg_web/operations.py SETTING_GROUPS` is
    what the page draws, grouped by what each row is about. They are two
    lists of the same thing, which is two chances to disagree, so they
    are held against each other here.

    Both directions matter and they fail differently. A registry key
    missing from the groups is a knob the application HAS and nobody can
    turn -- it works, it is tested, it has a default, and there is no
    entry point, which is the exact shape of an unshipped capability. A
    grouped key missing from the registry is a control that renders and
    then refuses every write, because `settings.put` validates against
    the registry and raises KeyError for anything else.

    Enumeration comes from the application; this test states no list of
    its own. Adding a setting therefore fails here until it is given a
    group, rather than shipping invisible.
    """
    from db import settings
    from sg_web import operations

    grouped: list[str] = [key for _, _, keys in operations.SETTING_GROUPS for key in keys]
    registry = set(settings.REGISTRY)

    unreachable = registry - set(grouped)
    assert not unreachable, (
        f"{sorted(unreachable)} are in the registry and on no group, so the page cannot reach them; "
        "give each one a group in sg_web/operations.py SETTING_GROUPS"
    )
    invented = set(grouped) - registry
    assert not invented, (
        f"{sorted(invented)} are grouped but are not settings; `settings.put` refuses them, "
        "so the control would render and then fail every write"
    )
    # One group each: a key in two groups draws two controls over one
    # value, and whichever the reader does not use silently goes stale.
    assert len(grouped) == len(set(grouped)), (
        f"a setting is in more than one group: {sorted({k for k in grouped if grouped.count(k) > 1})}"
    )
