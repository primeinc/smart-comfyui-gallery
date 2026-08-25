"""A file session has a story: what the files said about themselves.

No camera, no generator -- a screenshot run, a download batch. The
FileHistoryPlanner phases the session on pauses, finds bursts, names the
basis every date rests on, says when the filesystem disputed a name,
and notes a mix of media. The renderer words it as saving, never as
shooting. The plan is v5 of the frozen grammar; v4 refuses it; the
timeline offers the story for every file session.
"""

from __future__ import annotations

import pytest

from db import connect, derived, planning, rendering, stories
from tests.staging import staged
from tests.test_the_timeline_is_a_surface import HOUR, NOW, _drain, _interpreted, _library


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "file-story", _library, _interpreted) as stage:
        yield stage


@pytest.fixture
def told(_stage):
    _stage.restore()
    return _stage.client


def _frozen(client, domain: str):
    conn = connect.connect(client.app.state.db_path)
    try:
        column = "local_start" if domain == "wall" else "instant_start"
        event_id = conn.execute(
            f"SELECT id FROM derived_event WHERE kind = 'file_session' AND {column} IS NOT NULL"
        ).fetchone()[0]
        snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
        return conn, snap, stories.load_snapshot(conn, snap.id)
    except Exception:
        connect.close(conn)
        raise


def test_the_screenshots_are_two_phases_with_a_pause_and_named_bases(told):
    """Five screenshots named to the second: three inside 14:00-14:15,
    two at 16:30 -- one pause of over two hours, every date from the
    file name, and the renderer says so in saving words."""
    conn, snap, document = _frozen(told, "wall")
    try:
        plan = planning.FileHistoryPlanner(None, {"pause_minutes": 30, "burst_seconds": 5.0}).plan(
            document, snap.sha256
        )
        assert planning.validate_current_plan(plan, document, snap.sha256) == []
        assert plan["v"] == 7
        assert plan["subject"]["sequenced"] is True
        assert plan["subject"]["label_hint"] == "stamped-name session · 5 files · 2 phases"
        assert [p["label_hint"] for p in plan["phases"]] == ["Phase 1", "Phase 2 · after a pause"]
        assert [list(p["member_refs"]) for p in plan["phases"]] == [
            ["member-001", "member-002", "member-003"],
            ["member-004", "member-005"],
        ]
        kinds = [c["kind"] for c in plan["claims"]]
        assert kinds[:3] == ["time_basis", "pause", "time_basis"]
        assert set(kinds) <= {"time_basis", "pause", "disputed_time"}, "nothing a file session cannot know"
        assert plan["claims"][0]["facts"] == {"filename": 3}
        assert plan["claims"][1]["facts"]["gap_seconds"] == pytest.approx(138 * 60, abs=1)
        render = rendering.TemplateStoryRenderer("memory").render(
            document, plan, snap.sha256, planning.identity(plan)[1]
        )
        assert rendering.violations(render, plan, document, snap.sha256, planning.identity(plan)[1]) == []
        assert render["title"] == "5 files from June 10, 2023"
        assert render["summary"] == "These 5 files were saved on June 10, 2023 and fall into 2 phases."
        texts = [b["text"] for s in render["sections"] for b in s["blocks"]]
        assert "Dated by 3 files by a stamp in the file name." in texts
        assert "Nothing was saved for 2 h 18 min before this phase." in texts
        assert not any("camera" in t for t in texts), "nobody held a camera"
        assert render["renderer"]["version"] == 6
        assert render["renderer"]["reads"]["plan"] == 7
        with pytest.raises(ValueError, match="file sessions only"):
            planning.FileHistoryPlanner().plan(
                {**document, "subject": {**document["subject"], "event_kind": "capture_session"}}, snap.sha256
            )
        with pytest.raises(ValueError, match="capture sessions only"):
            planning.CaptureHistoryPlanner().plan(document, snap.sha256)
    finally:
        connect.close(conn)


def test_the_downloads_are_one_phase_on_the_filesystem_alone(told):
    """Two claimless downloads an hour apart, dated by mtime only: one
    phase, the basis named honestly, sequenced on instants."""
    conn, snap, document = _frozen(told, "instant")
    try:
        plan = planning.FileHistoryPlanner().plan(document, snap.sha256)
        assert planning.validate_current_plan(plan, document, snap.sha256) == []
        assert plan["subject"]["sequenced"] is True
        assert len(plan["phases"]) == 2, "an hour apart is longer than the default pause"
        assert plan["subject"]["label_hint"].startswith("filesystem session · 2 files")
        assert {c["kind"] for c in plan["claims"]} == {"time_basis", "pause"}
        assert all(c["facts"] == {"mtime": 1} for c in plan["claims"] if c["kind"] == "time_basis")
        wide = planning.FileHistoryPlanner(None, {"pause_minutes": 120, "burst_seconds": 5.0}).plan(
            document, snap.sha256
        )
        assert len(wide["phases"]) == 1
        render = rendering.TemplateStoryRenderer("technical").render(
            document, wide, snap.sha256, planning.identity(wide)[1]
        )
        assert "Dated by 2 files by the filesystem's modified time." in [
            b["text"] for s in render["sections"] for b in s["blocks"]
        ]
        assert render["title"].endswith("UTC"), "an instant session's day is UTC and says so"
    finally:
        connect.close(conn)


def test_a_file_story_is_told_through_the_routes_the_timeline_uses(told):
    """The timeline's button: freeze, plan (durable work, no weights),
    render, read. The session then carries its story link."""
    client = told
    whole = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    session = next(s for s in whole["sessions"] if s["domain"] == "wall")
    assert (session["tellable"], session["planner"], session["story"]) == (True, "file_history", None)
    frozen = client.post("/stories/snapshots", json={"event_id": session["id"]})
    assert frozen.status_code in (200, 201), frozen.text
    asked = client.post("/stories/plans", json={"snapshot_id": frozen.json()["id"], "planner": "file_history"})
    assert asked.status_code == 202, asked.text
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        job = conn.execute("SELECT state, error FROM job WHERE kind = 'story_plan'").fetchone()
        assert job == ("done", None), job
        plan_id, planner, similarity = conn.execute("SELECT id, planner, similarity FROM story_plan").fetchone()
        assert (planner, similarity) == ("file_history", "none")
    finally:
        connect.close(conn)
    again = client.post("/stories/plans", json={"snapshot_id": frozen.json()["id"], "planner": "file_history"})
    assert again.status_code == 200, "the same request finds the plan its job made; it never queues again"
    assert again.json()["plan_id"] == plan_id
    made = client.post("/stories/renders", json={"plan_id": plan_id})
    assert made.status_code == 201, made.text
    page = client.get(f"/stories/renders/{made.json()['id']}", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert "5 files from June 10, 2023" in page.text
    again = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    held = next(s for s in again["sessions"] if s["id"] == session["id"])
    assert held["story"]["href"] == f"/stories/renders/{made.json()['id']}", "the story rides its session's card"
    assert held["story"]["title"]
    assert held["story"]["heroes"], "with its heroes"
    # the shelf lists it, newest first, with its words and links
    shelf = client.get("/stories", headers={"accept": "application/json"}).json()
    assert [s["id"] for s in shelf] == [made.json()["id"]]
    assert shelf[0]["title"] == "5 files from June 10, 2023"
    assert (shelf[0]["kind"], shelf[0]["profile"], shelf[0]["members"]) == ("file_session", "memory", 5)
    assert shelf[0]["heroes"], "the shelf shows the story's heroes"
    # `/thumb` prefixes both: the content-addressed
    # `/thumbs/<shard>/<sha>.webp` for a hashed file, and the
    # `/thumb/<slug>` route for one ingest has not reached. A card's
    # heroes address a picture the same way the story page does, which is
    # what makes the two share a cache entry rather than fetch it twice.
    assert all(h["thumbnail"].startswith("/thumb") for h in shelf[0]["heroes"])
    page = client.get("/stories", headers={"accept": "text/html"}).text
    assert f'data-story="{made.json()["id"]}"' in page
    # the shelf filters by session kind, and says how many of each
    assert 'data-stories-kind="file_session"' in page
    assert (
        client.get("/stories", params={"kind": "file_session"}, headers={"accept": "application/json"}).json() == shelf
    )
    assert (
        client.get("/stories", params={"kind": "capture_session"}, headers={"accept": "application/json"}).json() == []
    )
    assert client.get("/stories", params={"kind": "vibe_session"}).status_code == 400
    assert f'data-story-heroes="{len(shelf[0]["heroes"])}"' in page
    # a hero's caption, once a model has said one, is shown beside the
    # frozen name -- live, by address, labelled as today's
    story_page = client.get(f"/stories/renders/{made.json()['id']}", headers={"accept": "text/html"}).text
    assert "data-story-hero-said" not in story_page, "nothing captioned yet"
    conn = connect.connect(client.app.state.db_path)
    try:
        # By the sha the address carries, not by a slug parsed out of it:
        # a content-addressed thumbnail deliberately holds no slug, which
        # is the lookup it exists to save.
        #
        # And it names BYTES rather than a file. Every picture in this
        # fixture is the same 12x12 image, so one sha is eight files --
        # which is the honest shape of content addressing and not an
        # artifact of the fixture. A caption about what the picture SHOWS
        # is true of every file that is that picture, so it is recorded
        # against all of them.
        sha = shelf[0]["heroes"][0]["thumbnail"].rsplit("/", 1)[1].removesuffix(".webp")
        holders = [one for (one,) in conn.execute("SELECT id FROM file WHERE content_sha256 = ?", (sha,))]
        assert holders, f"the hero addresses {sha}, which no file in the library has"
        for one in holders:
            derived.annotate(conn, one, "caption", "a window full of icons", "m", "1", sha, NOW)
        conn.commit()
    finally:
        connect.close(conn)
    story_page = client.get(f"/stories/renders/{made.json()['id']}", headers={"accept": "text/html"}).text
    assert "a window full of icons" in story_page
    assert story_page.count("data-story-hero-said") >= 1
    assert f'<img src="{shelf[0]["heroes"][0]["thumbnail"]}"' in page
    assert "5 files from June 10, 2023" in page
    # and the story's crumb opens this session's window on the timeline
    story_page = client.get(
        made.json()["id"] and f"/stories/renders/{made.json()['id']}", headers={"accept": "text/html"}
    ).text
    assert "data-story-session" in story_page
    assert "/timeline?bin=hour&amp;start=" in story_page


def test_the_v5_grammar_holds_the_file_vocabulary_and_v4_refuses_it():
    plan = {
        "v": 5,
        "snapshot_sha256": "a" * 64,
        "planner": {
            "kind": "file_history",
            "version": 1,
            "settings": {"pause_minutes": 30, "burst_seconds": 5.0},
            "similarity": {"name": "none", "version": "1"},
        },
        "subject": {"kind": "file_session", "sequenced": True, "label_hint": "x"},
        "phases": [
            {
                "id": "phase-001",
                "member_refs": ["member-001", "member-002"],
                "representative_refs": ["member-001"],
                "label_hint": "Phase 1",
                "claim_refs": ["claim-001", "claim-002", "claim-003"],
            }
        ],
        "claims": [
            {
                "id": "claim-001",
                "kind": "time_basis",
                "confidence": 1.0,
                "evidence_refs": ["member-001:occurrence.basis"],
                "facts": {"filename": 1, "mtime": 1},
            },
            {
                "id": "claim-002",
                "kind": "disputed_time",
                "confidence": 1.0,
                "evidence_refs": ["member-001:occurrence.conflicts"],
                "facts": {"members": 1},
            },
            {
                "id": "claim-003",
                "kind": "media_mix",
                "confidence": 1.0,
                "evidence_refs": ["member-001:media_kind", "member-002:media_kind"],
                "facts": {"image": 1, "video": 1},
            },
        ],
        "unsupported": [],
    }
    assert planning.validate_story_plan(plan) == []
    assert any("not a v4" in why for why in planning.validate_story_plan({**plan, "v": 4})), (
        "the file vocabulary is v5's"
    )
    one_kind = {**plan, "claims": [{**plan["claims"][2], "facts": {"image": 2}}]}
    assert any("do not fit media_mix" in why for why in planning.validate_story_plan(one_kind)), "a mix needs two kinds"
    bent = {**plan, "claims": [{**plan["claims"][0], "facts": {"astrology": 1}}]}
    assert any("do not fit time_basis" in why for why in planning.validate_story_plan(bent))
    capture = {
        **plan,
        "planner": {
            **plan["planner"],
            "kind": "capture_history",
            "settings": {"pause_minutes": 10, "burst_seconds": 2.0},
        },
        "subject": {**plan["subject"], "kind": "capture_session"},
        "claims": [],
        "phases": [{**plan["phases"][0], "claim_refs": []}],
    }
    assert planning.validate_story_plan(capture) == [], "v4's plans read under v5"
    assert planning.validate_story_plan({**capture, "v": 4}) == [], "and still under v4"


def test_default_settings_spell_like_given_ones():
    """A request without settings and a request naming the defaults are
    one request: the identity is hashed from canonical JSON, where 30
    and 30.0 differ -- the first file story ever asked for queued its
    plan job again on every click because of exactly that."""
    defaults = planning.FileHistoryPlanner.defaults
    assert planning.validated_settings(None, defaults) == planning.validated_settings(dict(defaults), defaults)
    assert planning.canonical(planning.validated_settings(None, defaults)) == planning.canonical(
        planning.validated_settings({"pause_minutes": 30, "burst_seconds": 5}, defaults)
    )
    assert all(isinstance(v, float) for v in planning.validated_settings(None, defaults).values())


def test_a_caption_frozen_after_the_first_story_makes_a_new_story_say_it(told):
    """The whole chain on the routes: caption a member, tell again. The
    snapshot is new (its evidence changed), the v6 plan carries `seen`
    citing the frozen annotation, and the render quotes the sentence."""
    client = told
    whole = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    session = next(s for s in whole["sessions"] if s["domain"] == "wall")
    first = client.post("/stories/snapshots", json={"event_id": session["id"]}).json()["id"]
    conn = connect.connect(client.app.state.db_path)
    try:
        member = conn.execute(
            "SELECT file_id FROM derived_event_file WHERE event_id = ? ORDER BY file_id LIMIT 1", (session["id"],)
        ).fetchone()[0]
        sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (member,)).fetchone()[0]
        derived.annotate(conn, member, "caption", "a desktop with many windows open", "m", "1", sha, NOW)
        conn.commit()
    finally:
        connect.close(conn)
    frozen = client.post("/stories/snapshots", json={"event_id": session["id"]}).json()
    assert frozen["id"] != first, "new evidence is a new snapshot, never a rewritten one"
    asked = client.post("/stories/plans", json={"snapshot_id": frozen["id"], "planner": "file_history"})
    assert asked.status_code == 202, asked.text
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        plan_id, document = conn.execute(
            "SELECT id, document_json FROM story_plan WHERE snapshot_id = ?", (frozen["id"],)
        ).fetchone()
    finally:
        connect.close(conn)
    import json

    plan = json.loads(document)
    assert plan["v"] == 7
    seen = [claim for claim in plan["claims"] if claim["kind"] == "seen"]
    assert len(seen) == 1
    assert seen[0]["facts"] == {"members": 1, "models": ["m"]}
    made = client.post("/stories/renders", json={"plan_id": plan_id})
    assert made.status_code == 201, made.text
    story = client.get(f"/stories/renders/{made.json()['id']}", headers={"accept": "application/json"}).json()
    words = " ".join(block["text"] for section in story["sections"] for block in section["blocks"])
    assert 'of one it said "a desktop with many windows open"' in words
