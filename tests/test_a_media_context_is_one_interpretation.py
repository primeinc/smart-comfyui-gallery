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
import pathlib
import re
import sqlite3
import typing

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import collection_rules, connect, context, facets, ingest, naming, resultset
from tests.staging import Stage, staged

NOW = 1_700_000_000.0
HOUR = 3600.0


def _library(root: pathlib.Path) -> None:
    """Four generated stills and three plain files that will carry
    camera claims (with offset, without offset, and none at all)."""
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
    # a camera writes the file as the shutter closes: photo_a's claim is
    # NOW+12h at -10:00 (the instant NOW+22h), photo_b's NOW+13h with no
    # zone (read on the host's clock); photo_c claims nothing
    written = {
        "photo_a.png": NOW + 22 * HOUR + 1,
        "photo_b.png": _instant(NOW + 13 * HOUR) + 1,
        "photo_c.png": NOW + 9 * HOUR,
    }
    for name, at in written.items():
        path = root / name
        Image.new("RGB", (12, 12), (200, 90, 140)).save(path)
        os.utime(path, (at, at))


def _instant(wall: float) -> float:
    """The UTC instant whose HOST wall-clock reading is `wall`."""
    import datetime

    naive = datetime.datetime.fromtimestamp(wall, datetime.UTC).replace(tzinfo=None)
    return naive.astimezone().timestamp()


def _prepare(stage: Stage) -> None:
    client, root = stage.client, stage.root
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
        # ...and photo_a was ALSO run through a generator: coexistence
        # is fact, and precedence must not erase either claim.
        conn.execute(
            "INSERT INTO generation(file_id, tool, detection, parser, parsed_at)"
            " VALUES(?, 'test', 'marker', 'test', ?)",
            (names["photo_a.png"], NOW),
        )
        conn.execute(
            "INSERT INTO capture(file_id, captured_at, tz_offset_min, iso, parsed_at) VALUES(?, ?, NULL, 100, ?)",
            (names["photo_b.png"], NOW + 13 * HOUR, NOW),
        )
        # The GENERATOR's embedded claims: a real date on gen_0 (and a
        # decoy on photo_a, which the camera outranks), garbage on
        # gen_1 -- a claim that does not parse is no claim.
        conn.executemany(
            "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', 'date', ?)",
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


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_media_context_is_one_interpretation", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def interpreted(_stage):
    _stage.restore()
    return _stage.client, _stage.root


def _raw(client) -> sqlite3.Connection:
    return connect.connect(client.app.state.db_path)


# --- the ladder -------------------------------------------------------------


def test_two_time_concepts_and_every_date_names_its_basis(interpreted):
    """A camera claim with an offset yields both the wall clock and the
    instant, corroborated by the file's write; without the offset the wall clock STANDS
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
                " mc.time_basis, mc.time_certainty, mc.time_precision, mc.has_capture, mc.has_generation"
                " FROM derived_media_context mc JOIN file f ON f.id = mc.file_id"
            )
        }
        origin, local, instant, offset, basis, certainty, precision, has_c, has_g = held["photo_a.png"]
        assert (origin, has_c, has_g) == ("mixed", 1, 1), "coexisting claims erase nothing"
        assert (basis, certainty, offset, precision) == ("capture", 0.9, -600, "second")
        assert local == NOW + 12 * HOUR, "the wall clock is the local story"
        assert instant == (NOW + 12 * HOUR) - (-600 * 60), "the offset makes the instant knowable"

        origin, local, instant, offset, basis, certainty, precision, _c, _g = held["photo_b.png"]
        assert (origin, basis, certainty, precision) == ("captured", "capture", 0.9, "second")
        assert local == NOW + 13 * HOUR, "the known wall clock STANDS"
        assert instant is None, "an unzoned claim has no instant -- uncertainty is explicit, never fabricated"
        assert offset is None

        origin, local, instant, offset, basis, _certainty, precision, _c, _g = held["photo_c.png"]
        assert origin == "imported"
        assert precision == "subsecond", "a distrusted filesystem time is still a FINE time"
        assert basis in ("btime", "mtime"), "the filesystem's claims are the fallback, named as themselves"
        assert local is None, "a filesystem instant has no local story to tell"
        assert instant is not None

        origin, local, instant, offset, basis, certainty, precision, _c, _g = held["gen_0.png"]
        assert (origin, basis) == ("generated", "embedded"), "the generator's own day is the claim"
        assert certainty == 0.4, "a 2023 day on a file written today: the mtime DISAGREES and says so"
        assert precision == "day", "a bare date is DAY-fine, whatever its certainty -- never minute evidence"

        assert local == 1_685_577_600.0, "2023-06-01 as a wall claim -- the day the media HAPPENED"
        assert (instant, offset) == (None, None), "a date without a zone has no instant"

        assert held["gen_1.png"][4] in ("btime", "mtime"), (
            "a claim that does not parse is no claim; the ladder falls through, never invents"
        )
        assert all(row[4] is not None for row in held.values()), "no unexplained dates: every time names its basis"
    finally:
        connect.close(conn)


def test_each_claim_is_its_own_occurrence(interpreted):
    """The context keeps ONE primary human-timeline interpretation; the
    occurrence rows keep each CLAIM at its own time. photo_a's camera
    outranks its decoy generator date on the primary ladder -- but the
    generator's claim is not erased: it stands as the generation
    occurrence, day-fine, exactly as claimed. A claim that does not
    parse produces no occurrence at all."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        held = {
            (name, kind): (local, instant, basis, certainty, precision)
            for name, kind, local, instant, basis, certainty, precision in conn.execute(
                "SELECT f.name, o.kind, o.local_at, o.instant_at, o.basis, o.certainty, o.time_precision"
                " FROM derived_media_occurrence o JOIN file f ON f.id = o.file_id"
            )
        }
        local, instant, basis, certainty, precision = held[("photo_a.png", "capture")]
        assert (basis, certainty, precision) == ("capture", 0.9, "second")
        assert (local, instant) == (NOW + 12 * HOUR, (NOW + 12 * HOUR) - (-600 * 60))
        local, instant, basis, certainty, precision = held[("photo_a.png", "generation")]
        assert (basis, certainty, precision) == ("embedded", 0.4, "day"), (
            "the decoy date the camera outranks on the primary ladder STANDS as the generation act's own claim"
        )
        assert (local, instant) == (1_685_577_600.0, None)
        local, instant, basis, certainty, precision = held[("photo_b.png", "capture")]
        assert (certainty, instant) == (0.9, None), "an unzoned capture occurrence invents no instant"
        assert held[("gen_0.png", "generation")][4] == "day"
        assert ("gen_1.png", "generation") not in held, "a claim that does not parse is no occurrence"
        assert ("photo_c.png", "capture") not in held, "no claim, no occurrence -- the filesystem is not an act"
        assert all(kind != "capture" or name.startswith("photo") for name, kind in held), (
            "capture occurrences exist only where a camera spoke"
        )
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
                conn.execute(
                    "SELECT g.file_id, g.seed, g.sampler, gp.prompt_id FROM generation g"
                    " LEFT JOIN generation_prompt gp ON gp.file_id = g.file_id AND gp.role = 'effective' ORDER BY 1"
                ).fetchall(),
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
        state = context.state(conn)
        assert state is not None
        generation_now = state[0]
        run = conn.execute(
            "INSERT INTO derived_event_run(grouper, grouper_version, settings_hash,"
            " context_generation, context_policy_version, created_at)"
            " VALUES('generation_session', '1', 'x', ?, ?, ?)",
            (generation_now, context.POLICY_VERSION, NOW),
        ).lastrowid
        event = conn.execute(
            "INSERT INTO derived_event(run_id, kind, instant_start, instant_end, member_hash)"
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
        assert origin["total"] == 1, "photo_a is MIXED now -- captured means captured-only"
        mixed = resultset.page(conn, "", resultset.parse(facets=["context.origin:eq:mixed"]), 1, NOW)
        assert [row["name"] for row in mixed["items"]] == ["photo_a.png"]
        seeded = resultset.page(
            conn, "", resultset.parse(facets=["generation.seed:eq:2", "capture.iso:lte:99"]), 1, NOW
        )
        assert seeded["total"] == 0, "facets are a conjunction"
        day = resultset.page(conn, "", resultset.parse(facets=["context.local_day:eq:2023-11-15"]), 1, NOW)
        assert day["total"] >= 1, "the timeline's day link answers through the same machinery"

        for hostile, why in (
            ("capture.moon:eq:1", "no filter named"),
            ("capture.iso:like:800", "allows"),
            ("capture.iso:eq:eight", "integer"),
            ("context.origin:eq:imagined", "one of"),
            ("context.local_day:eq:yesterday", "YYYY-MM-DD"),
            ("capture.iso", "written"),
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
                "results": [{"file_id": i, "score": 1.0, "sources": {}} for i in sorted(allowed or ())],
                "participants": ["fake"],
                "contributors": ["fake"],
                "missing": {},
            }

        monkeypatch.setattr(retrieval, "query", fused)
        resultset.page(conn, "", resultset.parse(facets=["capture.iso:gte:800"], text="beach"), 1, NOW)
        assert witnessed["allowed"] == members, "the facet must constrain retrieval BEFORE fusion"
    finally:
        connect.close(conn)


def test_a_faceted_view_saves_whole_as_a_v3_rule(interpreted):
    """The save-view landmine, closed: a v3 rule carries the facets, so
    the smart collection's membership IS the answer on screen -- today
    and after the library grows. A session's link is the one facet a
    rule refuses: a hypothesis is not a durable membership."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        rule = collection_rules.from_gallery_query(
            conn, resultset.parse(facets=["capture.iso:gte:800"]), actor_id=None, take=None
        )
        assert rule.version == 3
        assert [facets.spell(one) for one in rule.facets] == ["capture.iso:gte:800"]
        with pytest.raises(ValueError, match="hypothesis"):
            collection_rules.from_gallery_query(
                conn, resultset.parse(facets=["event.id:eq:1"]), actor_id=None, take=None
            )
        on_screen = resultset.page(conn, "", resultset.parse(facets=["capture.iso:gte:800"]), 1, NOW)
    finally:
        connect.close(conn)
    saved = client.post("/albums/smart", json={"name": "Fast Film", "f": "capture.iso:gte:800"})
    assert saved.status_code == 201, saved.text
    slug = saved.json()["slug"]
    conn = _raw(client)
    try:
        resolved = naming.resolve(conn, "collection", slug)
        assert resolved is not None
        stored = collection_rules.load(conn, resolved[0])
        assert stored is not None
        assert stored.version == 3
        assert [facets.spell(one) for one in stored.facets] == ["capture.iso:gte:800"]
        inside = resultset.page(conn, "", resultset.parse(album=slug), 1, NOW)
        assert [row["id"] for row in inside["items"]] == [row["id"] for row in on_screen["items"]]
        assert inside["total"] == on_screen["total"] > 0
    finally:
        connect.close(conn)
    # the collection's page opens the same question in the gallery
    page = client.get(f"/t/{slug}", headers={"accept": "text/html"}).text
    assert 'data-rule-gallery href="/g?f=capture.iso%3Agte%3A800"' in page
    refused = client.post("/albums/smart", json={"name": "One Session", "f": "event.id:eq:1"})
    assert refused.status_code == 400
    assert "hypothesis" in refused.json()["detail"]
    # two facets arrive as a list and both are saved -- the browser's
    # button sends every `f`, and the route must take every one
    two = client.post("/albums/smart", json={"name": "Two", "f": ["capture.iso:gte:800", "context.origin:eq:captured"]})
    assert two.status_code == 201, two.text
    conn = _raw(client)
    try:
        resolved = naming.resolve(conn, "collection", two.json()["slug"])
        assert resolved is not None
        held = collection_rules.load(conn, resolved[0])
        assert held is not None
        assert [facets.spell(one) for one in held.facets] == ["capture.iso:gte:800", "context.origin:eq:captured"]
    finally:
        connect.close(conn)


def test_take_is_an_exact_integer_before_any_coercion(interpreted):
    """int(True) is 1, and a boolean quietly becoming a one-item cutoff
    is the truthiness species the rules module exists to refuse."""
    client, _ = interpreted
    conn = _raw(client)
    try:
        hostiles: list[object] = [True, "5", 1.0]
        for hostile in hostiles:
            with pytest.raises(ValueError, match="exact integer"):
                # the wrong type on purpose: the rules module refuses truthiness species at the seam
                collection_rules.from_gallery_query(
                    conn, resultset.parse(kind="image"), actor_id=None, take=typing.cast("int", hostile)
                )
    finally:
        connect.close(conn)


def test_a_rule_over_the_interpretation_refuses_while_files_are_uninterpreted(interpreted, monkeypatch):
    """After a policy bump every context facet answers for nobody until
    the context job runs: a smart rule over one would evaluate to an
    empty set wearing an answer's clothes, so it refuses with the count
    and the remedy instead."""
    from db import collection_rules, resultset

    client, _ = interpreted
    saved = client.post("/albums/smart", json={"name": "Captured", "f": "context.origin:eq:captured"})
    assert saved.status_code == 201, saved.text
    # the bump comes with a deploy, i.e. a fresh process: no projection of
    # this question is cached when the rule is first evaluated under it
    monkeypatch.setattr(context, "POLICY_VERSION", context.POLICY_VERSION + 1)
    conn = _raw(client)
    try:
        with pytest.raises(collection_rules.UnavailableCollectionRule, match="run the context job"):
            resultset.page(conn, "", resultset.parse(album=saved.json()["slug"]), 1, NOW)
    finally:
        connect.close(conn)
    monkeypatch.undo()
    assert client.get("/g", params={"album": saved.json()["slug"]}).status_code == 200, "interpreted: it answers"
