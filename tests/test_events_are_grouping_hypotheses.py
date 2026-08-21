"""Events are grouping. Groupers consume the one interpretation.

The Seam has two real adapters behind one interface; a session splits
on TEMPORAL separation only -- prompt evolution is its story, never its
boundary; time has a DOMAIN, and unlike domains are never subtracted;
precision is orthogonal to certainty, and a claim too coarse for the
gap never enters the arithmetic -- "insufficient temporal precision"
is an answer. Currentness is PROVEN: every run names the context
generation it was computed over, a changed outsider makes every
hypothesis stale, and a race between proposal and persistence refuses
rather than publishing a run the contexts no longer support.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, events, ingest, pages, runner
from sg_web.app import build_app

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0


def _ctx(
    file_id,
    *,
    generated=False,
    captured=False,
    instant=None,
    local=None,
    precision="second",
    prompt=None,
    workflow=None,
):
    origin = "mixed" if generated and captured else "generated" if generated else "captured" if captured else "imported"
    return context.MediaContext(
        file_id=file_id,
        uuid=f"{file_id:032x}",
        origin=origin,
        has_capture=captured,
        has_generation=generated,
        local_at=local,
        instant_at=instant,
        tz_offset_min=None,
        time_precision=precision if (instant is not None or local is not None) else None,
        prompt_id=prompt,
        workflow_id=workflow,
    )


# --- the seam, unit-level: proposals from contexts alone --------------------


def test_prompt_evolution_stays_inside_one_session():
    """An afternoon of refining -- prompt tweaks, a new LoRA workflow,
    parameter changes -- is ONE session whose changes are its history.
    Splitting on prompt identity would summarize every revision as its
    own event."""
    held = [
        _ctx(1, generated=True, instant=NOW + 0 * MIN, prompt=10, workflow=50),
        _ctx(2, generated=True, instant=NOW + 3 * MIN, prompt=11, workflow=50),
        _ctx(3, generated=True, instant=NOW + 7 * MIN, prompt=12, workflow=50),
        _ctx(4, generated=True, instant=NOW + 10 * MIN, prompt=12, workflow=51),
        _ctx(5, generated=True, instant=NOW + 13 * MIN, prompt=12, workflow=51),
    ]
    made = events.GenerationSessionGrouper().groups(held)
    assert len(made) == 1, "prompt/workflow changes are the story INSIDE the session, never its boundary"
    assert made[0].file_ids == (1, 2, 3, 4, 5)
    assert (made[0].instant_start, made[0].instant_end) == (NOW, NOW + 13 * MIN)


def test_sessions_split_where_the_maker_walked_away():
    held = [
        _ctx(1, generated=True, instant=NOW),
        _ctx(2, generated=True, instant=NOW + 5 * MIN),
        _ctx(3, generated=True, instant=NOW + 2 * HOUR),  # the gap
        _ctx(4, generated=True, instant=NOW + 2 * HOUR + 4 * MIN),
        _ctx(5, generated=True, instant=NOW + 9 * HOUR),  # a singleton is not a session
    ]
    made = events.GenerationSessionGrouper().groups(held)
    assert [one.file_ids for one in made] == [(1, 2), (3, 4)]


def test_a_day_resolution_claim_never_becomes_a_session_boundary():
    """The hostile case the embedded-date fix exposed: files sharing a
    day-resolution generation.date all land on the same synthetic
    midnight, and grouping them by a 30-minute gap would manufacture a
    contiguous sitting out of evidence that only names a DAY. The
    honest answer is no session -- while the timeline day still counts
    all three."""
    held = [
        _ctx(1, generated=True, local=NOW, precision="day"),
        _ctx(2, generated=True, local=NOW, precision="day"),
        _ctx(3, generated=True, local=NOW, precision="day"),
    ]
    assert events.GenerationSessionGrouper().groups(held) == [], (
        "insufficient temporal precision is an answer, never a session"
    )
    fine = [
        _ctx(1, generated=True, local=NOW, precision="second"),
        _ctx(2, generated=True, local=NOW + 8 * MIN, precision="second"),
        _ctx(3, generated=True, local=NOW + 19 * MIN, precision="second"),
    ]
    assert [one.file_ids for one in events.GenerationSessionGrouper().groups(fine)] == [(1, 2, 3)], (
        "the same shape WITH minutes really is one session"
    )
    split = [
        _ctx(1, generated=True, local=NOW, precision="second"),
        _ctx(2, generated=True, local=NOW + 8 * MIN, precision="second"),
        _ctx(3, generated=True, local=NOW + 5 * HOUR, precision="second"),
    ]
    assert [one.file_ids for one in events.GenerationSessionGrouper().groups(split)] == [(1, 2)]


def test_unlike_time_domains_are_never_subtracted():
    """An unzoned afternoon and a UTC instant are not seconds apart --
    they are incomparable. Media cluster within their own domain, and a
    cross-domain pair that would have been 'adjacent' as bare numbers
    stays two singletons."""
    held = [
        _ctx(1, generated=True, instant=NOW),
        _ctx(2, generated=True, local=NOW + 5 * MIN),  # numerically adjacent, in a DIFFERENT domain
    ]
    assert events.GenerationSessionGrouper().groups(held) == [], "cross-domain adjacency is numeric coincidence"

    both_local = [
        _ctx(1, generated=True, local=NOW),
        _ctx(2, generated=True, local=NOW + 5 * MIN),
    ]
    made = events.GenerationSessionGrouper().groups(both_local)
    assert [one.file_ids for one in made] == [(1, 2)]
    assert (made[0].local_start, made[0].instant_start) == (NOW, None), (
        "a wall-clock session carries a wall-clock interval, and no invented instants"
    )


def test_the_two_adapters_share_the_interface_not_the_semantics():
    """Captured media cluster over a wider gap, participation is the
    explicit has_* fact -- and a mixed file belongs to both stories."""
    held = [
        _ctx(1, captured=True, instant=NOW),
        _ctx(2, captured=True, generated=True, instant=NOW + 2 * HOUR),  # mixed: in BOTH stories
        _ctx(3, generated=True, instant=NOW + 2 * HOUR + 5 * MIN),
        _ctx(4, captured=True, instant=NOW + 9 * HOUR),
    ]
    moments = events.CaptureSessionGrouper().groups(held)
    assert [one.file_ids for one in moments] == [(1, 2)], "the mixed file is capture-story too"
    sessions = events.GenerationSessionGrouper().groups(held)
    assert [one.file_ids for one in sessions] == [(2, 3)], "and generation-story too"


def test_groupers_consume_the_metadata_interface_not_source_tables():
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "db" / "events.py").read_text(encoding="utf-8")
    for named in ("FROM file", "FROM capture", "FROM generation", "JOIN entity", "FROM entity"):
        assert named not in source, f"a grouper read {named!r}; groupers consume MediaContext"
    assert "context.contexts(" in source, "the grouping input is the Metadata interface"


# --- currentness, persistence and the jobs -----------------------------------


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
            # function of the fixture: three tight, one far away.
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


def test_events_refuse_to_run_before_any_interpretation_exists(grouped):
    """No context state means nothing to prove a hypothesis against:
    the events job fails its items honestly and publishes no run."""
    told = grouped.post("/jobs/events").json()
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        settled = grouped.get(f"/jobs/{told['id']}").json()
        assert (settled["state"], settled["failed_count"]) == ("done", 2)
        assert conn.execute("SELECT count(*) FROM derived_event_run").fetchone()[0] == 0
    finally:
        connect.close(conn)


def test_a_changed_outsider_makes_every_hypothesis_stale(grouped):
    """The straggler was never a member of the session -- but once its
    claims change, the session's ABSENCE of it is itself stale: nothing
    stays current until regrouping proves itself over the new
    generation."""
    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)

    conn = connect.connect(grouped.app.state.db_path)
    try:
        session = conn.execute(
            "SELECT e.member_hash, e.run_id FROM derived_event e"
            " JOIN derived_event_run r ON r.id = e.run_id WHERE r.grouper = 'generation_session'"
        ).fetchall()
        assert len(session) == 1
        first_hash, first_run = session[0]
        assert len(pages.timeline_events(conn)) >= 1, "the hypothesis is current before the change"

        # The OUTSIDER's claim changes, through the writer seam.
        outsider = conn.execute("SELECT id FROM file WHERE name = 'gen_3.png'").fetchone()[0]
        conn.execute("UPDATE file SET btime = ?, mtime = ? WHERE id = ?", (NOW + 9 * MIN, NOW + 9 * MIN, outsider))
        context.stale(conn, outsider)
        conn.commit()

        assert pages.timeline_events(conn) == [], (
            "a hypothesis over an old generation must stop being current IMMEDIATELY -- "
            "even when the changed file was never one of its members"
        )
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
        assert len(pages.timeline_events(conn)) >= 1, "the re-proved hypothesis is current again"
    finally:
        connect.close(conn)


def test_a_grouping_proof_cannot_race_a_context_mutation(grouped, monkeypatch):
    """The WI-45 shape at the grouping seam: proposals computed over one
    generation must never persist under another. A mutation in the
    handoff triggers ONE recompute; a persistent race refuses with
    nothing written."""
    grouped.post("/jobs/context")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        grouper = events.GenerationSessionGrouper()
        real_groups = grouper.groups
        raced: list[int] = []

        def racing(held):
            raced.append(1)
            if len(raced) == 1:
                context._advance(conn)  # a context mutation lands mid-proposal...
                conn.commit()  # ...and COMMITS, as a real writer would
            return real_groups(held)

        monkeypatch.setattr(grouper, "groups", racing, raising=False)
        run_id = events.regroup_one(conn, grouper, NOW + 25 * HOUR)
        conn.commit()
        assert len(raced) == 2, "the stale proposal must be recomputed, not trusted"
        tagged = conn.execute("SELECT context_generation FROM derived_event_run WHERE id = ?", (run_id,)).fetchone()[0]
        assert tagged == context.state(conn)[0], "the run proves the generation it was computed over"

        def always_racing(held):
            context._advance(conn)
            conn.commit()
            return real_groups(held)

        monkeypatch.setattr(grouper, "groups", always_racing, raising=False)
        with pytest.raises(ValueError, match="kept moving"):
            events.regroup_one(conn, grouper, NOW + 26 * HOUR)
        conn.commit()
        runs = conn.execute("SELECT count(*) FROM derived_event_run WHERE grouper = 'generation_session'").fetchone()[0]
        assert runs == 1, "a refused race persists nothing"
    finally:
        connect.close(conn)


def test_the_timeline_is_a_view_with_a_door_into_the_gallery(grouped):
    """GET /timeline writes nothing and renders whatever the jobs last
    produced; every day links into /g through the Facet Interface's own
    spelling, so the ResultSet answers the media and the timeline never
    grows a second membership engine."""
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
    told = next(row for row in body["events"] if row["kind"] == "generation_session")
    assert told["instant_start"] is not None, "the interval names its domain"
    assert told["local_start"] is None, "and invents no wall clock nothing claimed"
    day = body["days"][-1]
    assert day["qs"] == f"f=context.local_day%3Aeq%3A{day['day']}", "the door is the Facet Interface's own spelling"
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
