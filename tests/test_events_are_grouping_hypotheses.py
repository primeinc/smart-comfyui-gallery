"""Events are grouping. Groupers consume per-claim occurrences.

The Seam has two real adapters behind one interface; each consumes the
occurrence rows of its OWN claim, so a mixed file tells the capture
story at the camera's time and the generation story at the generator's
claimed time. A session splits on TEMPORAL separation only; time has a
DOMAIN, and unlike domains are never subtracted; precision is
orthogonal to certainty, and a claim too coarse for the gap never
enters the arithmetic -- "insufficient temporal precision" is an
answer. Currentness is PROVEN and stable is not complete: a run must
show the interpretation covers every present file, names the
generation it read, and survives a revalidation race -- and an
upgraded policy blinds every reader until the context job runs.
"""

from __future__ import annotations

import datetime
import pathlib

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, events, ingest, pages, runner
from tests.staging import Stage, staged

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0
Y2023 = 1_685_577_600.0  # 2023-06-01
Y2026 = 1_787_308_800.0  # 2026-08-19


def _spelled(moment: float) -> str:
    """A second-resolution embedded date claim, as a generator writes it."""
    return datetime.datetime.fromtimestamp(moment, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


def _occ(file_id, *, kind="generation", instant=None, local=None, precision="second"):
    return context.Occurrence(
        file_id=file_id,
        uuid=f"{file_id:032x}",
        kind=kind,
        local_at=local,
        instant_at=instant,
        time_precision=precision,
    )


# --- the seam, unit-level: proposals from occurrences alone -----------------


def test_a_sitting_is_one_session_whatever_changed_inside_it():
    """An afternoon of refining is ONE session: the grouper cannot even
    SEE prompts or workflows -- an occurrence is a time and a claim --
    so splitting on prompt identity is structurally impossible, not
    merely avoided."""
    held = [
        _occ(1, instant=NOW + 0 * MIN),
        _occ(2, instant=NOW + 3 * MIN),
        _occ(3, instant=NOW + 7 * MIN),
        _occ(4, instant=NOW + 10 * MIN),
        _occ(5, instant=NOW + 13 * MIN),
    ]
    made = events.GenerationSessionGrouper().groups(held)
    assert len(made) == 1
    assert made[0].file_ids == (1, 2, 3, 4, 5)
    assert (made[0].instant_start, made[0].instant_end) == (NOW, NOW + 13 * MIN)


def test_sessions_split_where_the_maker_walked_away():
    held = [
        _occ(1, instant=NOW),
        _occ(2, instant=NOW + 5 * MIN),
        _occ(3, instant=NOW + 2 * HOUR),  # the gap
        _occ(4, instant=NOW + 2 * HOUR + 4 * MIN),
        _occ(5, instant=NOW + 9 * HOUR),  # a singleton is not a session
    ]
    made = events.GenerationSessionGrouper().groups(held)
    assert [one.file_ids for one in made] == [(1, 2), (3, 4)]


def test_a_day_resolution_claim_never_becomes_a_session_boundary():
    """Files sharing a day-resolution generation.date all land on the
    same synthetic midnight, and grouping them by a 30-minute gap would
    manufacture a contiguous sitting out of evidence that only names a
    DAY. The honest answer is no session -- while the timeline day
    still counts all three."""
    held = [
        _occ(1, local=NOW, precision="day"),
        _occ(2, local=NOW, precision="day"),
        _occ(3, local=NOW, precision="day"),
    ]
    assert events.GenerationSessionGrouper().groups(held) == [], (
        "insufficient temporal precision is an answer, never a session"
    )
    fine = [
        _occ(1, local=NOW),
        _occ(2, local=NOW + 8 * MIN),
        _occ(3, local=NOW + 19 * MIN),
    ]
    assert [one.file_ids for one in events.GenerationSessionGrouper().groups(fine)] == [(1, 2, 3)], (
        "the same shape WITH minutes really is one session"
    )
    split = [
        _occ(1, local=NOW),
        _occ(2, local=NOW + 8 * MIN),
        _occ(3, local=NOW + 5 * HOUR),
    ]
    assert [one.file_ids for one in events.GenerationSessionGrouper().groups(split)] == [(1, 2)]


def test_unlike_time_domains_are_never_subtracted():
    """An unzoned afternoon and a UTC instant are not seconds apart --
    they are incomparable. Media cluster within their own domain, and a
    cross-domain pair that would have been 'adjacent' as bare numbers
    stays two singletons."""
    held = [
        _occ(1, instant=NOW),
        _occ(2, local=NOW + 5 * MIN),  # numerically adjacent, in a DIFFERENT domain
    ]
    assert events.GenerationSessionGrouper().groups(held) == [], "cross-domain adjacency is numeric coincidence"

    both_local = [
        _occ(1, local=NOW),
        _occ(2, local=NOW + 5 * MIN),
    ]
    made = events.GenerationSessionGrouper().groups(both_local)
    assert [one.file_ids for one in made] == [(1, 2)]
    assert (made[0].local_start, made[0].instant_start) == (NOW, None), (
        "a wall-clock session carries a wall-clock interval, and no invented instants"
    )


def test_a_mixed_file_tells_each_story_at_its_own_time():
    """One media identity, two historical acts: the photograph was
    CAPTURED in 2023 and run through a generator in 2026. Each grouper
    consumes its own claim's occurrence, so the capture story happens
    at the camera's time and the generation story at the generator's --
    never one timestamp impersonating both acts."""
    captures = [
        _occ(1, kind="capture", instant=Y2023),
        _occ(2, kind="capture", instant=Y2023 + 9 * MIN),
    ]
    generations = [
        _occ(1, kind="generation", local=Y2026),  # file 1 is the MIXED one
        _occ(3, kind="generation", local=Y2026 + 6 * MIN),
    ]
    moments = events.CaptureSessionGrouper().groups(captures)
    assert [one.file_ids for one in moments] == [(1, 2)]
    assert moments[0].instant_start == Y2023, "the capture story happens at capture time"
    sessions = events.GenerationSessionGrouper().groups(generations)
    assert [one.file_ids for one in sessions] == [(1, 3)]
    assert sessions[0].local_start == Y2026, "the generation story happens at the generator's claimed time"
    assert sessions[0].instant_start is None, "an unzoned generator claim invents no instant"


# --- currentness, persistence and the jobs -----------------------------------


def _library(root: pathlib.Path) -> None:
    for i in range(4):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"a tin lighthouse\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / f"gen_{i}.png", pnginfo=info)


def _claims(stage: Stage) -> None:
    conn = stage.conn()
    try:
        names = dict(conn.execute("SELECT name, id FROM file").fetchall())
        for name, file_id in names.items():
            ingest.one(conn, file_id, stage.root / name, NOW)
        # The GENERATOR's own second-resolution claims are the source
        # facts grouping is a function of: three tight, one far away.
        conn.executemany(
            "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', 'date', ?)",
            [(names[f"gen_{i}.png"], _spelled(NOW + i * 3 * MIN if i < 3 else NOW + 8 * HOUR)) for i in range(4)],
        )
        conn.commit()
    finally:
        connect.close(conn)


@pytest.fixture(scope="module")
def _grouped(tmp_path_factory):
    with staged(tmp_path_factory, "events", _library, _claims) as stage:
        yield stage


@pytest.fixture
def grouped(_grouped):
    _grouped.restore()
    return _grouped.client


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
        assert (settled["state"], settled["failed_count"]) == ("done", 3)
        assert conn.execute("SELECT count(*) FROM derived_event_run").fetchone()[0] == 0
    finally:
        connect.close(conn)


def test_events_refuse_a_stable_but_incomplete_interpretation(grouped):
    """STABLE IS NOT COMPLETE: a paused context job holds the generation
    still while most of the library is uninterpreted, and a hypothesis
    proven over that silence would stay current indefinitely. Grouping
    demands coverage -- every present file, a current-policy context --
    and refuses with nothing written until the context job finishes."""
    conn = connect.connect(grouped.app.state.db_path)
    try:
        held = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id").fetchall()]
        for file_id in held[:2]:  # the job pauses after two of four items
            context.rebuild_one(conn, file_id, NOW + 24 * HOUR)
        conn.commit()
        with pytest.raises(ValueError, match="incomplete"):
            events.regroup_one(conn, events.GenerationSessionGrouper(), NOW + 25 * HOUR)
        conn.commit()
        assert conn.execute("SELECT count(*) FROM derived_event_run").fetchone()[0] == 0, (
            "a refused hypothesis persists nothing"
        )
        for file_id in held[2:]:  # the job resumes and finishes
            context.rebuild_one(conn, file_id, NOW + 24 * HOUR)
        conn.commit()
        events.regroup_one(conn, events.GenerationSessionGrouper(), NOW + 26 * HOUR)
        conn.commit()
        assert conn.execute("SELECT count(*) FROM derived_event_run").fetchone()[0] == 1, (
            "the complete interpretation may publish"
        )
    finally:
        connect.close(conn)


def test_an_upgraded_policy_blinds_every_reader_until_rebuild(grouped, monkeypatch):
    """The software upgrades its ladder; the database still holds
    yesterday's interpretation. EVERY reader -- timeline shelves, the
    facet link, the groupers -- binds the RUNNING policy, so the old
    rows are honestly invisible everywhere at once until the context
    job re-interprets. Serving them as current would be two definitions
    of 'current metadata' in one library."""
    from db import facets

    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        day = pages.timeline_days(conn)[0][0]
        assert pages.timeline_months(conn) != []
        assert len(pages.timeline_events(conn)) >= 1
        # `predicate` returns the values a template binds, as a LIST:
        # most keys bind one, and an advanced `key=value` binds two,
        # because the long tail is rows rather than columns.
        link_sql, link_values = facets.predicate(facets.facet("context.local_day", "eq", day))
        opened = conn.execute(f"SELECT count(*) FROM file f WHERE {link_sql}", link_values).fetchone()[0]
        assert opened >= 1, "the link answers while the interpretation is current"

        monkeypatch.setattr(context, "POLICY_VERSION", context.POLICY_VERSION + 1)
        assert pages.timeline_months(conn) == [], "an upgraded build shows honest absence, not yesterday's ladder"
        assert pages.timeline_days(conn) == []
        assert pages.timeline_events(conn) == []
        link_sql, link_values = facets.predicate(facets.facet("context.local_day", "eq", day))
        assert conn.execute(f"SELECT count(*) FROM file f WHERE {link_sql}", link_values).fetchone()[0] == 0, (
            "the facet link and the timeline must agree on what 'current' means"
        )
        assert context.occurrences(conn, "generation") == [], "the occurrence reader goes dark with every other reader"
        with pytest.raises(ValueError, match="older policy"):
            events.regroup_one(conn, events.GenerationSessionGrouper(), NOW + 25 * HOUR)
        conn.commit()
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

        # The OUTSIDER's claim changes, through the writer seam: the
        # generator's date turns out to be nine minutes after the burst.
        outsider = conn.execute("SELECT id FROM file WHERE name = 'gen_3.png'").fetchone()[0]
        conn.execute(
            "UPDATE file_param SET value_text = ? WHERE file_id = ? AND source = 'generation' AND key = 'date'",
            (_spelled(NOW + 9 * MIN), outsider),
        )
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


def test_a_departed_file_makes_the_hypothesis_stale_in_the_same_scan(grouped):
    """Yesterday's complete session is not current over today's library.
    A file leaving is a POPULATION change: the scan itself advances the
    proof identity -- no context or events job is needed to make the old
    hypothesis stop rendering, and an event that claimed the departed
    picture is deleted outright rather than surviving to count it."""
    import os

    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert len(pages.timeline_events(conn)) >= 1
        root = conn.execute("SELECT path FROM root").fetchone()[0]
    finally:
        connect.close(conn)

    # The OUTSIDER departs: never a member, but the absence proof is stale.
    os.remove(str(pathlib.Path(root) / "gen_3.png"))
    grouped.post("/roots/1/scan")
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert pages.timeline_events(conn) == [], (
            "the scan that shrank the population must stale the hypothesis by itself"
        )
    finally:
        connect.close(conn)

    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert len(pages.timeline_events(conn)) >= 1, "re-proved over the smaller library"
        member = conn.execute("SELECT id FROM file WHERE name = 'gen_0.png'").fetchone()[0]
        root = conn.execute("SELECT path FROM root").fetchone()[0]
    finally:
        connect.close(conn)

    # A MEMBER departs: the event claiming it dies in the same transaction.
    os.remove(str(pathlib.Path(root) / "gen_0.png"))
    grouped.post("/roots/1/scan")
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert pages.timeline_events(conn) == []
        assert (
            conn.execute("SELECT count(*) FROM derived_event_file ef WHERE ef.file_id = ?", (member,)).fetchone()[0]
            == 0
        ), "no persisted event may keep claiming a missing picture"
    finally:
        connect.close(conn)


def test_an_arrived_file_makes_the_hypothesis_stale_and_regroup_refuses(grouped):
    """The inverse ordering: complete, published -- then the library
    GROWS. The scan advances the proof identity, the old event stops
    rendering immediately, and regroup refuses the now-incomplete
    coverage until the context job interprets the newcomer."""
    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert len(pages.timeline_events(conn)) >= 1
        root = conn.execute("SELECT path FROM root").fetchone()[0]
    finally:
        connect.close(conn)

    Image.new("RGB", (12, 12), (10, 200, 40)).save(str(pathlib.Path(root) / "arrival.png"))
    grouped.post("/roots/1/scan")
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert pages.timeline_events(conn) == [], "an event proven over four files is not current over five"
        assert context.coverage(conn) == (4, 5), "the newcomer is present and uninterpreted"
        with pytest.raises(ValueError, match="incomplete"):
            events.regroup_one(conn, events.GenerationSessionGrouper(), NOW + 30 * HOUR)
        conn.commit()
    finally:
        connect.close(conn)

    grouped.post("/jobs/context")
    grouped.post("/jobs/events")
    _drain(grouped)
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert context.coverage(conn) == (5, 5)
        assert len(pages.timeline_events(conn)) >= 1, "interpreted, the grown library may publish again"
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
        state = context.state(conn)
        assert state is not None
        assert tagged == state[0], "the run proves the generation it was computed over"

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


def test_the_timeline_is_a_view_with_a_link_into_the_gallery(grouped):
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
    body = grouped.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    assert body["extent"]["pictures"] == 4
    told = next(row for row in body["sessions"] if row["kind"] == "generation_session")
    assert told["domain"] == "wall", "the interval names its domain: the generator claimed a wall clock"
    bar = body["bins"][-1]
    assert bar["qs"].startswith("f=context.granule%3Alte%3A86400&f=context.moment%3Agte%3A"), (
        "the link is the Facet Interface's own spelling"
    )
    assert bar["qs"].endswith("&sort=moment"), "ordered by the moment it opened on"
    walked = grouped.get(f"/g?{bar['qs']}")
    assert walked.status_code == 200
    assert f'data-total="{bar["pictures"]}"' in walked.text, "the day link answers exactly the day's media"
    page = grouped.get("/timeline", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert "data-bin-at=" in page.text
    conn = connect.connect(grouped.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET interpreted something; only the jobs may"
    finally:
        connect.close(conn)
