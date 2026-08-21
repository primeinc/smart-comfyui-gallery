"""Metadata is evidence. MediaContext is interpretation.

Source facts stay per-source claims and raw evidence stays reparseable.
derived_media_context is the ONE fallback ladder: two time concepts --
the human clock and the knowable instant -- with the basis always
recorded, unknown timezones explicitly uncertain, and a known wall
clock never replaced by a filesystem time. Stale interpretations are
deleted at the source-fact writer seams, never served. The typed facet
registry is the one vocabulary every gallery filter speaks, riding the
canonical spelling, the projection identity and the pre-RRF gate --
and a faceted view refuses to save until a rule version can carry it.
"""

from __future__ import annotations

import os
import re
import sqlite3

import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import collection_rules, connect, context, ingest, resultset
from sg_web.app import build_app

NOW = 1_700_000_000.0
HOUR = 3600.0


def _library(tmp) -> tuple:
    """Four generated stills and three plain files that will carry
    camera claims (with offset, without offset, and none at all)."""
    root = tmp / "lib"
    root.mkdir()
    for i in range(4):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"a tin lighthouse\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {i}, "
            f"Size: 512x512, Model: alpha",
        )
        path = root / f"gen_{i}.png"
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(path, pnginfo=info)
        os.utime(path, (NOW + i * 60, NOW + i * 60))
    for name in ("photo_a.png", "photo_b.png", "photo_c.png"):
        path = root / name
        Image.new("RGB", (12, 12), (200, 90, 140)).save(path)
        os.utime(path, (NOW + 9 * HOUR, NOW + 9 * HOUR))
    return tmp / "run", root


@pytest.fixture
def interpreted(tmp_path):
    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            names = dict(conn.execute("SELECT name, id FROM file").fetchall())
            for name, file_id in names.items():
                if name.startswith("gen_"):
                    ingest.one(conn, file_id, root / name, NOW)
            # Camera CLAIMS, as source facts: a wall clock with its
            # offset (a knowable instant), a wall clock without one
            # (honest uncertainty), and no claim at all.
            conn.execute(
                "INSERT INTO capture(file_id, captured_at, tz_offset_min, iso, gps_lat, gps_lon, parsed_at)"
                " VALUES(?, ?, ?, ?, ?, ?, ?)",
                (names["photo_a.png"], NOW + 12 * HOUR, -600, 1600, 21.27, -157.82, NOW),
            )
            conn.execute(
                "INSERT INTO capture(file_id, captured_at, tz_offset_min, iso, parsed_at) VALUES(?, ?, NULL, 100, ?)",
                (names["photo_b.png"], NOW + 13 * HOUR, NOW),
            )
            # The GENERATOR's embedded claims: a real date on gen_0 (and a
            # decoy on photo_a, which the camera outranks), garbage on
            # gen_1 -- a claim that does not parse is no claim.
            conn.executemany(
                "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text)"
                " VALUES(?, 'generation', 'date', ?)",
                [
                    (names["gen_0.png"], "2023-06-01"),
                    (names["gen_1.png"], "last tuesday"),
                    (names["photo_a.png"], "2023-06-01"),
                ],
            )
            context.rebuild(conn, NOW + 24 * HOUR)
            conn.commit()
        finally:
            connect.close(conn)
        yield client, root


def _raw(client) -> sqlite3.Connection:
    return connect.connect(client.app.state.db_path)


# --- the ladder -------------------------------------------------------------


def test_two_time_concepts_and_every_date_names_its_basis(interpreted):
    """A camera claim with an offset yields both the wall clock and the
    instant at full certainty; without the offset the wall clock STANDS
    and the instant stays honestly absent -- a known human clock is
    never replaced by a filesystem time. Only claimless media fall to
    the filesystem, instants with no local story."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        held = {
            name: row
            for name, *row in conn.execute(
                "SELECT f.name, mc.origin, mc.local_at, mc.instant_at, mc.tz_offset_min,"
                " mc.time_basis, mc.time_certainty"
                " FROM derived_media_context mc JOIN file f ON f.id = mc.file_id"
            )
        }
        origin, local, instant, offset, basis, certainty = held["photo_a.png"]
        assert (origin, basis, certainty, offset) == ("captured", "capture", 1.0, -600)
        assert local == NOW + 12 * HOUR, "the wall clock is the local story"
        assert instant == (NOW + 12 * HOUR) - (-600 * 60), "the offset makes the instant knowable"

        origin, local, instant, offset, basis, certainty = held["photo_b.png"]
        assert (origin, basis, certainty) == ("captured", "capture", 0.8)
        assert local == NOW + 13 * HOUR, "the known wall clock STANDS"
        assert instant is None, "an unzoned claim has no instant -- uncertainty is explicit, never fabricated"
        assert offset is None

        origin, local, instant, offset, basis, _certainty = held["photo_c.png"]
        assert origin == "imported"
        assert basis in ("btime", "mtime"), "the filesystem's claims are the fallback, named as themselves"
        assert local is None, "a filesystem instant has no local story to tell"
        assert instant is not None

        origin, local, instant, offset, basis, certainty = held["gen_0.png"]
        assert (origin, basis, certainty) == ("generated", "embedded", 0.6), (
            "the generator's own date claim outranks every filesystem time"
        )
        assert local == 1_685_577_600.0, "2023-06-01 as a wall claim -- the day the media HAPPENED"
        assert (instant, offset) == (None, None), "a date without a zone has no instant"

        assert held["gen_1.png"][4] in ("btime", "mtime"), (
            "a claim that does not parse is no claim; the ladder falls through, never invents"
        )
        assert all(row[4] is not None for row in held.values()), "no unexplained dates: every time names its basis"
    finally:
        connect.close(conn)


def test_rebuilding_touches_no_evidence_and_no_source_facts(interpreted):
    """The interpretation reads claims and writes understanding -- raw
    evidence and source facts are byte-identical across a rebuild, and
    the rebuild is idempotent."""
    client, _ = interpreted
    conn = _raw(client)
    try:

        def sources():
            return [
                conn.execute("SELECT hash, byte_len FROM blob ORDER BY hash").fetchall(),
                conn.execute("SELECT file_id, carrier, slot, blob_hash FROM file_blob ORDER BY 1,2,3").fetchall(),
                conn.execute("SELECT file_id, source, key, value_text FROM file_param ORDER BY 1,2,3").fetchall(),
                conn.execute("SELECT * FROM capture ORDER BY file_id").fetchall(),
                conn.execute("SELECT file_id, seed, sampler, prompt_id FROM generation ORDER BY 1").fetchall(),
            ]

        before = sources()
        first = conn.execute(
            "SELECT file_id, origin, local_at, instant_at, time_basis FROM derived_media_context ORDER BY file_id"
        ).fetchall()
        context.rebuild(conn, NOW + 48 * HOUR)
        conn.commit()
        assert sources() == before, "a rebuild replaced evidence or source facts"
        again = conn.execute(
            "SELECT file_id, origin, local_at, instant_at, time_basis FROM derived_media_context ORDER BY file_id"
        ).fetchall()
        assert again == first, "the same claims must interpret identically"
    finally:
        connect.close(conn)


def test_a_changed_source_claim_deletes_the_interpretation(interpreted):
    """Invalidation at the writer seams: a reparse or a filesystem time
    change makes the stale context VANISH -- deleted, never served --
    until the explicit job rebuilds it. Event hypotheses over the file
    go with it."""
    client, root = interpreted
    conn = _raw(client)
    try:
        gen0 = conn.execute("SELECT id FROM file WHERE name = 'gen_0.png'").fetchone()[0]
        assert conn.execute("SELECT count(*) FROM derived_media_context WHERE file_id = ?", (gen0,)).fetchone()[0] == 1
        # a fake event hypothesis over the file, to watch it invalidate
        run = conn.execute(
            "INSERT INTO derived_event_run(grouper, grouper_version, settings_hash, created_at)"
            " VALUES('generation_session', '1', 'x', ?)",
            (NOW,),
        ).lastrowid
        event = conn.execute(
            "INSERT INTO derived_event(run_id, kind, start_at, end_at, member_hash)"
            " VALUES(?, 'generation_session', ?, ?, 'h')",
            (run, NOW, NOW + 1),
        ).lastrowid
        conn.execute(
            "INSERT INTO derived_event_file(event_id, file_id, ordinal, score) VALUES(?, ?, 0, NULL)", (event, gen0)
        )
        # The REPARSE seam: ingest rewrites the file's claims.
        ingest.one(conn, gen0, root / "gen_0.png", NOW + 50 * HOUR)
        conn.commit()
        assert conn.execute("SELECT count(*) FROM derived_media_context WHERE file_id = ?", (gen0,)).fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM derived_event_run WHERE id = ?", (run,)).fetchone()[0] == 0, (
            "an event over a reinterpreted file is a stale hypothesis"
        )
        context.rebuild(conn, NOW + 51 * HOUR)
        conn.commit()
    finally:
        connect.close(conn)

    # The FILESYSTEM seam: the file's times change on disk; the rescan
    # that records the new claim deletes the old interpretation.
    os.utime(root / "gen_1.png", (NOW + 60 * HOUR, NOW + 60 * HOUR))
    client.post("/roots/1/scan")
    conn = _raw(client)
    try:
        gone = conn.execute(
            "SELECT count(*) FROM derived_media_context mc JOIN file f ON f.id = mc.file_id WHERE f.name = 'gen_1.png'"
        ).fetchone()[0]
        assert gone == 0, "a moved clock left its old interpretation standing"
    finally:
        connect.close(conn)


# --- the typed facet registry -----------------------------------------------


def test_the_registry_is_the_one_vocabulary(interpreted):
    """Registered keys answer through the same eligibility everything
    rides; unregistered keys, wrong operators and wrong value shapes
    refuse loudly at parse time."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        sampled_truth = conn.execute("SELECT count(*) FROM generation WHERE sampler = 'Euler a'").fetchone()[0]
        assert sampled_truth >= 1, "the fixture must really hold the sampler it filters by"
        fast = resultset.page(conn, "", resultset.parse(facets=["capture.iso:gte:800"]), 1, NOW)
        assert [row["name"] for row in fast["items"]] == ["photo_a.png"]
        sampled = resultset.page(conn, "", resultset.parse(facets=["generation.sampler:eq:Euler a"]), 1, NOW)
        assert sampled["total"] == sampled_truth
        origin = resultset.page(conn, "", resultset.parse(facets=["context.origin:eq:captured"]), 1, NOW)
        assert origin["total"] == 2
        seeded = resultset.page(
            conn, "", resultset.parse(facets=["generation.seed:eq:2", "capture.iso:lte:99"]), 1, NOW
        )
        assert seeded["total"] == 0, "facets are a conjunction"
        day = resultset.page(conn, "", resultset.parse(facets=["context.local_day:eq:2023-11-15"]), 1, NOW)
        assert day["total"] >= 1, "the timeline's day door answers through the same machinery"

        for hostile, why in (
            ("capture.moon:eq:1", "nothing is registered"),
            ("capture.iso:like:800", "allows"),
            ("capture.iso:eq:eight", "integer"),
            ("context.origin:eq:imagined", "one of"),
            ("context.local_day:eq:yesterday", "YYYY-MM-DD"),
            ("capture.iso", "spelled"),
        ):
            with pytest.raises(ValueError, match=re.escape(why)):
                resultset.parse(facets=[hostile])
    finally:
        connect.close(conn)


def test_a_facet_rides_the_spelling_the_identity_and_the_semantic_gate(interpreted, monkeypatch):
    """One facet is part of the question everywhere: the canonical qs,
    the projection fingerprint (two orders of one conjunction are ONE
    question), and semantic retrieval's pre-RRF allowed set."""
    from db import retrieval

    client, _ = interpreted
    conn = _raw(client)
    try:
        asked = resultset.parse(facets=["capture.iso:gte:800"])
        told = resultset.describe(conn, "", asked, NOW)
        assert "f=capture.iso%3Agte%3A800" in told["qs"]
        plain = resultset.describe(conn, "", resultset.parse(), NOW)
        assert told["fingerprint"] != plain["fingerprint"], "a facet is a different question"
        both = ["capture.iso:gte:800", "context.origin:eq:captured"]
        assert resultset.parse(facets=both).facets == resultset.parse(facets=list(reversed(both))).facets
        assert resultset.parse(facets=both + both).facets == resultset.parse(facets=both).facets

        members = {row[0] for row in conn.execute("SELECT file_id FROM capture WHERE iso >= 800")}
        witnessed: dict = {}

        def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
            witnessed["allowed"] = None if allowed is None else set(allowed)
            return {
                "results": [{"file_id": i, "score": 1.0, "sources": {}} for i in sorted(allowed)],
                "participants": ["fake"],
                "contributors": ["fake"],
                "missing": {},
            }

        monkeypatch.setattr(retrieval, "query", fused)
        resultset.page(conn, "", resultset.parse(facets=["capture.iso:gte:800"], text="beach"), 1, NOW)
        assert witnessed["allowed"] == members, "the facet must constrain retrieval BEFORE fusion"
    finally:
        connect.close(conn)


def test_a_faceted_view_refuses_to_save_until_a_rule_can_carry_it(interpreted):
    """The save-view landmine, defused the fail-closed way: silently
    dropping the facets would save a smart collection whose membership
    differs from the answer on screen."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        with pytest.raises(ValueError, match="later rule version"):
            collection_rules.from_gallery_query(
                conn, resultset.parse(facets=["capture.iso:gte:800"]), actor_id=None, take=None
            )
    finally:
        connect.close(conn)
    landed = client.post("/albums/smart", json={"name": "Fast Film"})
    assert landed.status_code == 201, "control: an unfaceted save still lands"
    dropped = client.post("/albums/smart", json={"name": "Doomed", "f": "capture.iso:gte:800"})
    assert dropped.status_code == 400, "the HTTP adapter must carry the facet INTO the refusal, never drop it"
    assert "later rule version" in dropped.json()["detail"]


def test_take_is_an_exact_integer_before_any_coercion(interpreted):
    """int(True) is 1, and a boolean quietly becoming a one-item cutoff
    is the truthiness species the rules module exists to refuse."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        for hostile in (True, "5", 1.0):
            with pytest.raises(ValueError, match="exact integer"):
                collection_rules.from_gallery_query(conn, resultset.parse(kind="image"), actor_id=None, take=hostile)
    finally:
        connect.close(conn)
