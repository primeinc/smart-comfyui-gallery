"""Events are grouping. Groupers consume the one interpretation.

The Seam has two real adapters with different semantics behind one
interface; a session splits on TEMPORAL separation only -- the prompt
evolution inside it is its story, never its boundary; membership is
hashed so a changed membership is visibly a different event; regroup
keeps the latest hypothesis; and the durable work is two explicit jobs,
never a side effect of anything.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, events, ingest, runner
from sg_web.app import build_app

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0


def _ctx(file_id, origin="generated", instant=None, local=None, prompt=None, workflow=None):
    return context.MediaContext(
        file_id=file_id,
        uuid=f"{file_id:032x}",
        origin=origin,
        local_at=local,
        instant_at=instant,
        tz_offset_min=None,
        prompt_id=prompt,
        workflow_id=workflow,
    )


# --- the seam, unit-level: proposals from contexts alone --------------------


def test_prompt_evolution_stays_inside_one_session():
    """The hostile case the first draft got wrong: an afternoon of
    refining -- prompt tweaks, a new LoRA workflow, parameter changes --
    is ONE session whose changes are its history. Splitting on prompt
    identity would summarize every revision as its own event."""
    held = [
        _ctx(1, instant=NOW + 0 * MIN, prompt=10, workflow=50),  # astronaut
        _ctx(2, instant=NOW + 3 * MIN, prompt=11, workflow=50),  # + cinematic
        _ctx(3, instant=NOW + 7 * MIN, prompt=12, workflow=50),  # + orange suit
        _ctx(4, instant=NOW + 10 * MIN, prompt=12, workflow=51),  # LoRA experiment
        _ctx(5, instant=NOW + 13 * MIN, prompt=12, workflow=51),  # parameter change
    ]
    made = events.GenerationSessionGrouper().groups(held)
    assert len(made) == 1, "prompt/workflow changes are the story INSIDE the session, never its boundary"
    assert made[0].file_ids == (1, 2, 3, 4, 5)
    assert (made[0].start_at, made[0].end_at) == (NOW, NOW + 13 * MIN)


def test_sessions_split_where_the_maker_walked_away():
    held = [
        _ctx(1, instant=NOW),
        _ctx(2, instant=NOW + 5 * MIN),
        _ctx(3, instant=NOW + 2 * HOUR),  # the gap
        _ctx(4, instant=NOW + 2 * HOUR + 4 * MIN),
        _ctx(5, instant=NOW + 9 * HOUR),  # a singleton is not a session
    ]
    made = events.GenerationSessionGrouper().groups(held)
    assert [one.file_ids for one in made] == [(1, 2), (3, 4)]


def test_the_two_adapters_share_the_interface_not_the_semantics():
    """Captured media cluster over a wider gap, and each grouper only
    sees its own origin -- one interface, different implementations."""
    held = [
        _ctx(1, origin="captured", instant=NOW),
        _ctx(2, origin="captured", instant=NOW + 2 * HOUR),  # within a capture moment, beyond a generation gap
        _ctx(3, instant=NOW + 1 * MIN),  # generated: invisible to the capture adapter
        _ctx(4, origin="captured", instant=NOW + 9 * HOUR),
    ]
    moments = events.CaptureSessionGrouper().groups(held)
    assert [one.file_ids for one in moments] == [(1, 2)]
    assert all(one.kind == "capture_session" for one in moments)
    assert events.GenerationSessionGrouper().groups(held) == [], "one generated file is no session"


def test_an_unzoned_wall_clock_still_places_media_in_a_session():
    """The moment axis is the instant when knowable, the wall clock
    otherwise -- an old camera without an offset still has an afternoon."""
    held = [
        _ctx(1, origin="captured", local=NOW),
        _ctx(2, origin="captured", local=NOW + 20 * MIN),
    ]
    assert [one.file_ids for one in events.CaptureSessionGrouper().groups(held)] == [(1, 2)]


def test_groupers_consume_the_metadata_interface_not_source_tables():
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "db" / "events.py").read_text(encoding="utf-8")
    for named in ("FROM file", "FROM capture", "FROM generation", "JOIN entity", "FROM entity"):
        assert named not in source, f"a grouper read {named!r}; groupers consume MediaContext"
    assert "context.contexts(" in source, "the grouping input is the Metadata interface"


# --- persistence and the jobs, against a real library ------------------------


def _library(tmp) -> tuple:
    root = tmp / "lib"
    root.mkdir()
    for i in range(4):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"a tin lighthouse\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / f"gen_{i}.png", pnginfo=info)
    return tmp / "run", root


@pytest.fixture
def grouped(tmp_path):
    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            names = dict(conn.execute("SELECT name, id FROM file").fetchall())
            for name, file_id in names.items():
                ingest.one(conn, file_id, root / name, NOW)
            # The filesystem's birth times are this machine's real clock;
            # the SOURCE CLAIMS are set deliberately so grouping is a
            # function of the fixture, not of when the test ran: three
            # tight, one far away.
            for i in range(4):
                moment = NOW + i * 3 * MIN if i < 3 else NOW + 8 * HOUR
                conn.execute("UPDATE file SET btime = ?, mtime = ? WHERE name = ?", (moment, moment, f"gen_{i}.png"))
            conn.commit()
        finally:
            connect.close(conn)
        yield client


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def test_the_two_jobs_are_the_only_writers_and_hashes_track_membership(grouped):
    """POST /jobs/context rebuilds per file; POST /jobs/events proposes
    per grouper; a membership change is a hash change; regrouping keeps
    one run per grouper, not a museum."""
    told = grouped.post("/jobs/context").json()
    assert (told["kind"], told["total"]) == ("context", 4)
    told = grouped.post("/jobs/events").json()
    assert (told["kind"], told["total"]) == ("events", 2)
    _drain(grouped)

    conn = connect.connect(grouped.app.state.db_path)
    try:

        def session():
            return conn.execute(
                "SELECT e.id, e.member_hash, e.run_id FROM derived_event e"
                " JOIN derived_event_run r ON r.id = e.run_id WHERE r.grouper = 'generation_session'"
            ).fetchall()

        held = session()
        assert len(held) == 1
        _, first_hash, first_run = held[0]
        members = conn.execute(
            "SELECT f.name FROM derived_event_file ef JOIN file f ON f.id = ef.file_id"
            " WHERE ef.event_id = ? ORDER BY ef.ordinal",
            (held[0][0],),
        ).fetchall()
        assert [row[0] for row in members] == ["gen_0.png", "gen_1.png", "gen_2.png"]

        # The straggler's SOURCE CLAIM moves into the window -- through
        # the writer seam, which also stales the old hypothesis.
        conn.execute("UPDATE file SET btime = ?, mtime = ? WHERE name = 'gen_3.png'", (NOW + 9 * MIN, NOW + 9 * MIN))
        context.stale(conn, conn.execute("SELECT id FROM file WHERE name = 'gen_3.png'").fetchone()[0])
        conn.commit()
    finally:
        connect.close(conn)

    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        held = conn.execute(
            "SELECT e.member_hash, e.run_id FROM derived_event e"
            " JOIN derived_event_run r ON r.id = e.run_id WHERE r.grouper = 'generation_session'"
        ).fetchall()
        assert len(held) == 1
        second_hash, second_run = held[0]
        assert second_hash != first_hash, "membership changed; the hash must say so"
        assert second_run != first_run
        assert (
            conn.execute("SELECT count(*) FROM derived_event_run WHERE grouper = 'generation_session'").fetchone()[0]
            == 1
        ), "regrouping keeps the latest hypothesis, not a museum"
        assert (
            conn.execute(
                "SELECT count(*) FROM derived_event_file ef"
                " WHERE NOT EXISTS (SELECT 1 FROM derived_event e WHERE e.id = ef.event_id)"
            ).fetchone()[0]
            == 0
        )
    finally:
        connect.close(conn)


def test_the_timeline_is_a_view_with_a_door_into_the_gallery(grouped):
    """GET /timeline writes nothing and renders whatever the jobs last
    produced; every day links into /g through the registered day facet,
    so the ResultSet answers the media and the timeline never grows a
    second membership engine."""
    from db import resultset

    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)

    conn = connect.connect(grouped.app.state.db_path)
    try:
        before = resultset.currency(conn)
    finally:
        connect.close(conn)
    body = grouped.get("/timeline", headers={"accept": "application/json"}).json()
    assert sum(row["pictures"] for row in body["months"]) == 4
    assert any(row["kind"] == "generation_session" for row in body["events"])
    day = body["days"][-1]
    walked = grouped.get(f"/g?{day['qs']}")
    assert walked.status_code == 200
    assert f'data-total="{day["pictures"]}"' in walked.text, "the day door answers exactly the day's media"
    page = grouped.get("/timeline", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert "data-timeline-day=" in page.text
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET interpreted something; only the jobs may"
    finally:
        connect.close(conn)
