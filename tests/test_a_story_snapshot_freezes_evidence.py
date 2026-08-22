"""A StorySnapshot freezes evidence, not prose.

It is an immutable, self-contained record of exactly what the
application knew about ONE current event at ONE instant: file identity
AND the bytes actually observed, the occurrence that placed each member
in this event, the source facts by value -- never a foreign key into
today's rebuildable hypotheses. Identity is the canonical document's
hash, so identical evidence is one row; freezing proves currentness
the way regrouping does; and after the library, the policy and the
event have all moved on, the old snapshot loads byte-identical while a
fresh one is visibly different.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import sqlite3

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, events, ingest, runner, scan, stories
from tests.staging import Stage, staged

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0


def _spelled(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


def _library(root: pathlib.Path) -> None:
    for i in range(3):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"a tin lighthouse, variant {i}\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {100 + i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / f"gen_{i}.png", pnginfo=info)


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _prepare(stage: Stage) -> None:
    """Three generated stills forming ONE current generation session,
    with the generator's own parameter bag present to be frozen."""
    client, root = stage.client, stage.root
    conn = connect.connect(client.app.state.db_path)
    try:
        names = dict(conn.execute("SELECT name, id FROM file").fetchall())
        for name, file_id in names.items():
            ingest.one(conn, file_id, root / name, NOW)
        conn.executemany(
            "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', ?, ?)",
            [
                *[(names[f"gen_{i}.png"], "date", _spelled(NOW + i * 4 * MIN)) for i in range(3)],
                *[(names[f"gen_{i}.png"], "original_prompt", "a __material__ lighthouse") for i in range(3)],
            ],
        )
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_story_snapshot_freezes_evidence", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def storied(_stage):
    _stage.restore()
    return _stage.client, _stage.root


def _event(conn) -> int:
    return conn.execute(
        "SELECT e.id FROM derived_event e JOIN derived_event_run r ON r.id = e.run_id"
        " WHERE e.kind = 'generation_session'"
    ).fetchone()[0]


def test_a_snapshot_freezes_identity_bytes_occurrence_and_facts(storied):
    """The document carries what a later story needs and nothing a later
    story could re-derive: uuids AND observed content hashes, the claim
    that placed each member here, prompts, the generator's own bag."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        made = stories.snapshot_event(conn, _event(conn), NOW + 30 * HOUR)
        conn.commit()
        assert made.reused is False
        told = stories.load_snapshot(conn, made.id)
        assert told["v"] == stories.FORMAT_VERSION
        subject = told["subject"]
        assert (subject["event_kind"], subject["claim"], subject["grouper"]) == (
            "generation_session",
            "generation",
            "generation_session",
        )
        assert (subject["context_generation"], subject["context_policy_version"]) == context.state(conn)
        assert subject["time"]["local"] == [NOW, NOW + 8 * MIN], "the interval in the domain the event knows"
        assert subject["time"]["instant"] is None, "and no invented instant"
        members = told["members"]
        assert [one["ordinal"] for one in members] == [0, 1, 2]
        shas = dict(conn.execute("SELECT name, content_sha256 FROM file").fetchall())
        uuids = {
            name: uuid.hex()
            for name, uuid in conn.execute("SELECT f.name, e.uuid FROM file f JOIN entity e ON e.id = f.id")
        }
        for one in members:
            assert one["content_sha256"] == shas[one["name"]], "the bytes actually observed, frozen"
            assert one["file_uuid"] == uuids[one["name"]]
            assert one["occurrence"]["kind"] == "generation"
            assert (one["occurrence"]["basis"], one["occurrence"]["precision"]) == ("embedded", "second")
            assert one["generation"]["prompt"].startswith("a tin lighthouse, variant")
            assert one["generation"]["params"]["original_prompt"] == "a __material__ lighthouse"
            assert one["generation"]["seed"] in (100, 101, 102)
            assert one["capture"] is None, "a generated still makes no camera claim"
            assert one["people"] is None
            assert one["lineage"] is None
            assert one["annotations"] is None
        assert stories.verify(told, made.sha256), "the stored document hashes to the identity it was stored under"
    finally:
        connect.close(conn)


def test_identical_evidence_is_one_snapshot(storied):
    """The identity is the evidence's hash: freezing the same current
    event twice -- at different wall-clock instants -- is one row."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        first = stories.snapshot_event(conn, _event(conn), NOW + 30 * HOUR)
        conn.commit()
        again = stories.snapshot_event(conn, _event(conn), NOW + 31 * HOUR)
        conn.commit()
        assert (again.id, again.sha256, again.reused) == (first.id, first.sha256, True)
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 1
    finally:
        connect.close(conn)


def test_the_snapshot_outlives_everything_it_was_made_from(storied, monkeypatch):
    """The hostile acceptance test, modelled on every mistake already
    paid for. After the snapshot: a member is renamed, a member's bytes
    are replaced in place under the same address, a member's prompt and
    seed change, the interpretation policy moves, the library regroups
    so the original event disappears and its run is deleted. The old
    snapshot loads byte-identical -- original uuids, original content
    hashes, original prompt, original occurrence times, original
    ordering and member hash, original provenance -- and a fresh
    snapshot of the new current world has a different identity."""
    client, root = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = _event(conn)
        frozen = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
        before = stories.load_snapshot(conn, frozen.id)
        original_run = conn.execute("SELECT run_id FROM derived_event WHERE id = ?", (event_id,)).fetchone()[0]
        ids = dict(conn.execute("SELECT name, id FROM file").fetchall())
    finally:
        connect.close(conn)

    # rename A on disk; replace B's bytes in place; change C's prompt and seed
    os.rename(root / "gen_0.png", root / "renamed_a.png")
    info = PngInfo()
    info.add_text("parameters", "something else entirely\nSteps: 5, Seed: 7")
    Image.new("RGB", (12, 12), (250, 250, 250)).save(root / "gen_1.png", pnginfo=info)
    client.post("/roots/1/scan")
    conn = connect.connect(client.app.state.db_path)
    try:
        prompt_id = scan.mint(conn, "prompt", "a brass diving helmet")
        conn.execute(
            "INSERT INTO prompt(id, text, text_hash, created_at) VALUES(?, 'a brass diving helmet', 'h-helmet', ?)",
            (prompt_id, NOW),
        )
        conn.execute("UPDATE generation SET seed = 999 WHERE file_id = ?", (ids["gen_2.png"],))
        conn.execute(
            "UPDATE generation_prompt SET prompt_id = ? WHERE file_id = ? AND role = 'effective'",
            (prompt_id, ids["gen_2.png"]),
        )
        conn.execute(
            "UPDATE file_param SET value_text = ? WHERE file_id = ? AND source = 'generation' AND key = 'date'",
            (_spelled(NOW + 3 * HOUR), ids["gen_2.png"]),
        )
        context.stale(conn, ids["gen_2.png"])
        conn.commit()
    finally:
        connect.close(conn)

    # the interpretation policy moves on (a real upgrade: the running
    # constant changes and the jobs re-interpret at the new meaning),
    # and the library regroups
    monkeypatch.setattr(context, "POLICY_VERSION", context.POLICY_VERSION + 1)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)

    conn = connect.connect(client.app.state.db_path)
    try:
        gone = conn.execute("SELECT count(*) FROM derived_event_run WHERE id = ?", (original_run,)).fetchone()[0]
        assert gone == 0, "the original hypothesis is gone; regrouping keeps no museum"
        # SQLite reuses the rowid: whatever now sits at the old event id is
        # a DIFFERENT hypothesis -- which is why the snapshot keeps the id
        # as provenance only, never as the thing it survives by.
        reused = conn.execute("SELECT run_id FROM derived_event WHERE id = ?", (event_id,)).fetchone()
        assert reused is None or reused[0] != original_run

        after = stories.load_snapshot(conn, frozen.id)
        assert after == before, "the snapshot is BYTE-IDENTICAL after everything it was made from changed"
        assert stories.verify(after, frozen.sha256)
        names = [one["name"] for one in after["members"]]
        assert names == ["gen_0.png", "gen_1.png", "gen_2.png"], "original names and ordering"
        live_sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["gen_1.png"],)).fetchone()[0]
        frozen_sha = next(one["content_sha256"] for one in after["members"] if one["name"] == "gen_1.png")
        assert frozen_sha != live_sha, "the entity survived a replacement; the snapshot kept yesterday's bytes"
        frozen_c = next(one for one in after["members"] if one["name"] == "gen_2.png")
        assert frozen_c["generation"]["prompt"] == "a tin lighthouse, variant 2"
        assert frozen_c["generation"]["seed"] == 102
        assert frozen_c["occurrence"]["local_at"] == NOW + 8 * MIN, "the original occurrence time"
        assert after["subject"]["observed_event_id"] == event_id, "provenance by value, not by survival"

        # the new current world is a DIFFERENT snapshot
        fresh_event = conn.execute(
            "SELECT e.id FROM derived_event e WHERE e.kind = 'generation_session' ORDER BY e.id DESC LIMIT 1"
        ).fetchone()
        assert fresh_event is not None, "two of the three still sit within the gap"
        fresh = stories.snapshot_event(conn, fresh_event[0], NOW + 40 * HOUR)
        conn.commit()
        assert fresh.sha256 != frozen.sha256
        assert fresh.id != frozen.id
        told = stories.load_snapshot(conn, fresh.id)
        assert told["subject"]["member_hash"] != before["subject"]["member_hash"]
        assert told["subject"]["context_policy_version"] == before["subject"]["context_policy_version"] + 1, (
            "the fresh snapshot says which policy it was frozen under"
        )
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 2
    finally:
        connect.close(conn)


def test_a_snapshot_is_insert_only(storied):
    """Immutability is enforced by the schema, not admired in a docstring."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        made = stories.snapshot_event(conn, _event(conn), NOW + 30 * HOUR)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE story_snapshot SET document_json = '{}' WHERE id = ?", (made.id,))
        conn.rollback()
        assert stories.verify(stories.load_snapshot(conn, made.id), made.sha256)
    finally:
        connect.close(conn)


def test_freezing_refuses_anything_but_a_current_complete_event(storied):
    """A stale hypothesis is not a subject. An unknown id, a run whose
    generation moved on, an incomplete interpretation: each refuses
    with nothing written."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        with pytest.raises(LookupError, match="no event"):
            stories.snapshot_event(conn, 99_999, NOW + 30 * HOUR)
        conn.rollback()
        event_id = _event(conn)
        context.repopulated(conn)  # the population moved; the run is no longer current
        conn.commit()
        with pytest.raises(ValueError, match="not current"):
            stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.rollback()
        member = conn.execute("SELECT id FROM file WHERE name = 'gen_2.png'").fetchone()[0]
        context.stale(conn, member)  # a member's claim changes: the hypothesis itself is deleted
        conn.commit()
        with pytest.raises(LookupError, match="no event"):
            stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 0
    finally:
        connect.close(conn)


def test_freezing_cannot_race_the_library(storied, monkeypatch):
    """The WI-45 shape at the story seam: evidence collected over one
    generation must never land as a snapshot of another. A mutation in
    the handoff triggers ONE recollect; a persistent race refuses."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = _event(conn)
        real = stories._document
        raced: list[int] = []

        def racing(conn_, subject, now):
            raced.append(1)
            document = real(conn_, subject, now)
            if len(raced) == 1:
                context._advance(conn_)
                conn_.commit()
                # the run is no longer current after that advance: prove it
                # by re-proving the world (the events job would) so the
                # retry finds a CURRENT subject again
                events.regroup(conn_, now)
            return document

        monkeypatch.setattr(stories, "_document", racing)
        with pytest.raises((ValueError, LookupError)):
            # the original event id died with its run in the recompute;
            # the honest outcome of a race that changed the subject is a
            # refusal, never a snapshot of a different world
            stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 0
        assert len(raced) >= 1
    finally:
        connect.close(conn)


def test_the_http_adapters_freeze_and_read_only(storied):
    """POST freezes synchronously and says whether it reused; GET reads
    history and writes nothing."""
    from db import resultset

    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = _event(conn)
    finally:
        connect.close(conn)
    made = client.post("/stories/snapshots", json={"event_id": event_id})
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["reused"] is False
    assert len(body["sha256"]) == 64
    again = client.post("/stories/snapshots", json={"event_id": event_id})
    assert (again.status_code, again.json()["reused"], again.json()["id"]) == (200, True, body["id"])
    assert client.post("/stories/snapshots", json={"event_id": 99_999}).status_code == 404
    assert client.post("/stories/snapshots", json={}).status_code == 400, "a request without an event is a 400"

    conn = connect.connect(client.app.state.db_path)
    try:
        before = resultset.currency(conn)
    finally:
        connect.close(conn)
    read = client.get(f"/stories/snapshots/{body['id']}")
    assert read.status_code == 200
    assert read.json()["subject"]["member_hash"]
    assert client.get("/stories/snapshots/424242").status_code == 404
    conn = connect.connect(client.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET of history wrote something"
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 1
    finally:
        connect.close(conn)
