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
import pathlib
import re
import time

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, inspecting, jobs, ledger, runner
from sg_web import console
from sg_web.app import build_app
from tests.staging import fresh_schema

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


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


def test_the_vocabulary_is_one_list_in_three_places():
    """db/ledger.py TYPES, the schema's CHECK, and the console's
    renderings name exactly the same set: a type the ledger can write
    without words, or words for a type that cannot be written, fails."""
    ddl = SCHEMA.read_text(encoding="utf-8")
    found = re.search(r"CREATE TABLE job_event \((.*?)\) STRICT;", ddl, re.DOTALL)
    assert found is not None, "job_event left the schema"
    block = found.group(1)
    checked = set(re.findall(r"'([a-z_]+\.[a-z_]+)'", block.split("type ", 1)[1].split("item_id", 1)[0]))
    assert checked == set(ledger.TYPES), "the schema CHECK and ledger.TYPES disagree"
    assert set(console.RENDERINGS) == set(ledger.TYPES), "an event type has no console rendering"


@pytest.mark.parametrize("type_", ledger.TYPES)
def test_every_event_type_renders_to_words(type_):
    event = {"id": 1, "job_id": 7, "at": NOW, "type": type_, "item_id": 3, "phase": "decoding", "severity": "info"}
    event["message"] = "m"
    event["data"] = {"owner": "w", "attempt": 2, "fence": 3, "error": "boom", "did": 4, "failed": 1, "seconds": 2.5}
    words = console.describe(event)
    assert words.strip(), type_
    told = console.envelope(event)
    assert told["text"] == words
    assert told["type"] == type_


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


def test_every_shipped_handler_reports_from_inside_its_item():
    """Contract 5: a long handler that says nothing between item.started
    and item.done is a frozen bar. Every handler the runner ships reaches
    the reporting seam; a new kind that does not fails here."""
    import inspect as inspecting_source

    silent = [
        kind
        for kind, handler in runner.HANDLERS.items()
        if kind != "hash" and "report()" not in inspecting_source.getsource(handler)
    ]
    assert silent == [], f"these handlers never report a phase: {silent}"
    # the hash kind dispatches to four modes; each of those reports too
    for mode in (runner._verify_item, runner._perceptual_item, runner._thumbs_item, runner._dupe_groups_item):
        assert "report()" in inspecting_source.getsource(mode), mode.__name__


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


def _schema_job_kinds() -> set[str]:
    table = SCHEMA.read_text(encoding="utf-8").split("CREATE TABLE job (", 1)[1]
    check = re.search(r"kind\s+TEXT NOT NULL CHECK \(kind IN\s*\(([^)]*)\)", table)
    assert check is not None
    return set(re.findall(r"'([a-z_]+)'", check.group(1)))


def test_every_job_kind_has_words_beside_its_raw_name(tmp_path):
    """The console shows what a job does AND the schema's name for it.
    A kind the schema admits but the console cannot word is a row that
    reads as its identifier -- the contract holds the two vocabularies
    equal, and the hash kind's modes are told apart by the payload."""
    assert set(console.KINDS) == _schema_job_kinds()
    assert console.describe_kind("hash") == "verify every file's bytes"
    assert console.describe_kind("hash", "groups") == "group perceptual copies"
    assert console.describe_kind("annotate") == "caption every picture"
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "one.png")
    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        fingerprint = client.post("/operations/jobs/phash").text
        dupes = client.post("/operations/jobs/dupes").text
        assert "queued #" in fingerprint
        assert "queued #" in dupes
        matrix = client.get("/operations/overview").json()["matrix"]
        told = {(row["kind"], row.get("derive")): row["what"] for row in matrix}
        assert told[("hash", "perceptual")] == "fingerprint every picture"
        assert told[("hash", "groups")] == "group perceptual copies"
        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert "fingerprint every picture" in page
        assert '<code class="raw">hash</code>' in page
        one = next(row["id"] for row in matrix if row.get("derive") == "perceptual")
        detail = client.get(f"/operations/job/{one}", headers={"accept": "application/json"}).json()
        assert detail["what"] == "fingerprint every picture"
        inspector = client.get(f"/operations/job/{one}", headers={"accept": "text/html"}).text
        assert "fingerprint every picture" in inspector


def test_the_console_says_what_each_sweep_still_has_to_do(tmp_path):
    """Coverage beside the buttons: present files, and per missing-only
    sweep how many it would still queue -- counted the way the sweep
    counts, so the number beside the button and the job it queues agree.
    A sweep that ran takes its count to zero."""
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (1, 2, 3 + i)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
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
    assert activity.row_view({**row, "derive": "perceptual"})["what"] == "fingerprint every picture"
    assert activity.row_view({**row, "derive": None})["what"] == "verify every file's bytes"
    delta = {"job": 1, "kind": "hash", "state": "running", "done": 1, "total": 3, "derive": "groups"}
    assert activity.delta_view(delta)["what"] == "group perceptual copies"


def test_no_count_sits_beside_a_sweep_that_would_be_refused(tmp_path):
    conn = fresh_schema()
    from db import settings

    settings.put(conn, "caption_model", "blip")
    told = inspecting.coverage(conn)
    assert "annotate" not in told["missing"], "the sweep refuses that setting; a count beside it would lie"
    settings.put(conn, "caption_model", "Salesforce/blip-image-captioning-base")
    assert "annotate" in inspecting.coverage(conn)["missing"]
