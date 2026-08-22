"""When a file happened is judged from EVERY claim, not the first one.

The generator's day, SwarmUI's request minute (or stamped second) in
the file name, the file's mtime and btime and the generation time are
judged together: the occurrence is the generator's own finest claim,
every other source supports it, satisfies a constraint, or conflicts
-- named and persisted, never compressed into a score; quality is an
ordinal. mtime is the finish and never the occurrence: the request it
implies is an ESTIMATE kept beside the claim. The table of cases here
IS the policy; the live library proves a real SwarmUI run forms a
session at minute precision and freezes who agreed.
"""

from __future__ import annotations

import datetime
import json
import os

import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, ingest, planning, runner, stories, when
from sg_web.app import build_app

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0
DAY = 86400.0
JULY_18 = 1_784_332_800.0  # 2026-07-18 00:00 as a wall clock


def _instant(wall: float) -> float:
    """The UTC instant whose HOST wall-clock reading is `wall` -- what a
    file written at that wall time carries as mtime."""
    naive = datetime.datetime.fromtimestamp(wall, datetime.UTC).replace(tzinfo=None)
    return naive.astimezone().timestamp()


def _judge(**over):
    base = {
        "date_text": "2026-07-18",
        "name": "0947001-c5afa607-qwnImageEdit.png",
        "tool": "SwarmUI",
        "mtime": _instant(JULY_18 + 9 * HOUR + 48 * MIN + 32),
        "btime": _instant(JULY_18 + 33 * DAY),  # copied over a month later
        "generation_time": 64.33,
    }
    return when.judge_generation(**{**base, **over})


def test_the_claim_is_the_generators_minute_and_the_finish_is_evidence_beside_it():
    """The real i2i run: day from the generator, minute from the file
    name; the finish (mtime) minus generation time lands in the request
    minute, so it supports the claim and yields an ESTIMATE to the
    second -- while the claim itself stays the minute."""
    told = _judge()
    assert told is not None
    assert (told.precision, told.basis) == ("minute", "filename")
    assert told.local_at == JULY_18 + 9 * HOUR + 47 * MIN, "the claim is the request minute, untouched"
    assert told.instant_at is None, "a wall-clock claim with no zone has no instant"
    assert told.supports == ("embedded_day", "mtime_finish_consistent", "btime_after_generation", "host_zone_assumed")
    assert told.conflicts == ()
    assert told.source_order == 1, "the request counter is order inside the minute"
    assert told.usable is True
    assert (told.quality, told.certainty) == ("corroborated", 0.9)
    assert told.finished_at == _instant(JULY_18 + 9 * HOUR + 48 * MIN + 32)
    assert told.estimated_at == pytest.approx(JULY_18 + 9 * HOUR + 47 * MIN + 27.67, abs=0.01), (
        "finish minus generation time: the request to the second, as an estimate"
    )


def test_an_mtime_outside_the_window_is_a_named_conflict_not_a_lost_vote():
    """Re-saved in August: the minute stands, the mtime is named as the
    dissenter with how far off it is, no estimate, quality contested."""
    told = _judge(mtime=_instant(JULY_18 + 33 * DAY + 5 * HOUR))
    assert (told.precision, told.basis, told.local_at) == ("minute", "filename", JULY_18 + 9 * HOUR + 47 * MIN)
    assert told.supports == ("embedded_day", "btime_after_generation", "host_zone_assumed")
    assert len(told.conflicts) == 1
    assert told.conflicts[0].startswith("filesystem: mtime 2026-08-20 05:00:00 is")
    assert "days after" in told.conflicts[0]
    assert (told.quality, told.certainty) == ("contested", 0.4)
    assert told.usable is True, "a filesystem dissent, read through the host's zone, cannot demote the claim"
    assert told.estimated_at is None
    assert told.finished_at is None


def test_an_mtime_before_the_request_is_the_loud_case():
    told = _judge(mtime=_instant(JULY_18 + 9 * HOUR + 40 * MIN))
    assert "is before the claimed 2026-07-18 09:47:00" in told.conflicts[0]
    assert told.quality == "contested"


def test_a_default_name_carries_no_day_so_only_the_filesystem_can_dissent():
    """The default Swarm name is a minute without a date: it is always
    inside whatever day the generator claims. When the generator claims
    the NEXT day, the mtime is the dissenter."""
    told = _judge(date_text="2026-07-19", mtime=_instant(JULY_18 + 9 * HOUR + 48 * MIN + 32))
    assert (told.precision, told.basis) == ("minute", "filename")
    assert told.local_at == JULY_18 + DAY + 9 * HOUR + 47 * MIN
    assert len(told.conflicts) == 1
    assert "before" in told.conflicts[0]


def test_without_a_filename_minute_the_day_is_the_claim_and_mtime_only_supports():
    a1111 = {"name": "00012-123456.png", "tool": "A1111 / Forge"}
    inside = _judge(**a1111, mtime=_instant(JULY_18 + 15 * HOUR))
    assert (inside.precision, inside.basis) == ("day", "embedded"), "no generator minute, no minute"
    assert "mtime_finish_consistent" in inside.supports
    assert inside.estimated_at == pytest.approx(JULY_18 + 15 * HOUR - 64.33, abs=0.01), "the estimate is still there"
    outside = _judge(**a1111, mtime=_instant(JULY_18 + 3 * DAY))
    assert (outside.precision, outside.basis, outside.quality) == ("day", "embedded", "contested")
    assert "after the claimed window" in outside.conflicts[0]
    plain = _judge(**a1111, mtime=_instant(JULY_18 + 15 * HOUR), generation_time=None)
    assert "mtime_consistent" in plain.supports
    assert plain.estimated_at is None, "no duration, no estimate of the request"
    assert plain.finished_at is not None


def test_btime_is_a_constraint_and_never_an_instant():
    edited = _judge(btime=_instant(JULY_18 - 10 * DAY))
    assert "bytes born before generated" in edited.conflicts[0]
    assert edited.quality == "contested"
    no_mtime = _judge(mtime=None)
    assert (no_mtime.precision, no_mtime.basis, no_mtime.instant_at) == ("minute", "filename", None)
    assert no_mtime.supports == ("embedded_day", "btime_after_generation", "host_zone_assumed")
    alone = _judge(mtime=None, btime=None)
    assert alone.supports == ("embedded_day",), "no filesystem time, no zone assumed"
    assert when.judge_filesystem(None, _instant(NOW)).basis == "btime"
    fallback = when.judge_filesystem(_instant(NOW), _instant(NOW + 3 * HOUR))
    assert (fallback.basis, fallback.precision, fallback.supports, fallback.quality) == (
        "mtime",
        "subsecond",
        ("btime_consistent",),
        "corroborated",
    )
    assert fallback.local_at is None, "a filesystem instant has no local story"
    assert when.judge_filesystem(None, None) is None


def test_a_full_generator_stamp_is_its_own_finest_claim():
    told = _judge(date_text="2026-07-18 09:47:12", mtime=_instant(JULY_18 + 9 * HOUR + 48 * MIN + 32))
    assert (told.precision, told.basis) == ("second", "embedded")
    assert told.local_at == JULY_18 + 9 * HOUR + 47 * MIN + 12
    assert "mtime_finish_consistent" in told.supports, "finish minus 64 s lands within slack of the stamp"
    assert _judge(date_text="yesterday-ish") is None, "a claim that does not parse is no claim"
    assert when.swarm_minute("2547001-x.png") is None, "hour 25 is not a Swarm name"
    assert when.swarm_minute("0947001-c5afa607-m.png") == (9, 47, 1)


def test_a_stamped_name_is_the_generators_own_second():
    """A stamped name -- `[year][month][day]T[hour][minute][second]` or
    `[year][month][day]_[hour]h[minute]m[second]s[millisecond]ms` --
    is the generator's own second; the T or the h..m..s..ms is the
    marker. It stands on its own: with no embedded date the claim
    survives; an embedded day that agrees corroborates; one that
    contradicts is a GENERATOR conflict, the stamp still stands, and
    the claim is unfit for chronology."""
    stamped = _judge(name="20260718T094712001-c5afa607-qwnImageEdit.png")
    assert (stamped.precision, stamped.basis) == ("second", "filename")
    assert stamped.local_at == JULY_18 + 9 * HOUR + 47 * MIN + 12
    assert stamped.quality == "corroborated"
    assert stamped.source_order == 1
    assert "embedded_day" in stamped.supports
    bare = _judge(name="20260718T094712001-c5afa607-m.png", date_text=None)
    assert (bare.precision, bare.basis, bare.local_at) == ("second", "filename", stamped.local_at), (
        "a name is not optional metadata: the stamp stands without the embedded date"
    )
    assert "embedded_day" not in bare.supports
    wrong_day = _judge(name="20260719T094712001-c5afa607-m.png")
    assert (wrong_day.precision, wrong_day.basis, wrong_day.quality) == ("second", "filename", "contested")
    assert wrong_day.local_at == JULY_18 + DAY + 9 * HOUR + 47 * MIN + 12, "the stamp stands"
    assert wrong_day.conflicts[0].startswith("generator: ")
    assert "is not inside the embedded day" in wrong_day.conflicts[0]
    assert wrong_day.usable is False, "the generator disagrees with itself: recorded, never sequenced"
    agreed = _judge(name="20260718T094712001-c5afa607-m.png", date_text="2026-07-18 09:47:12")
    assert "embedded_stamp" in agreed.supports
    swarm_mixed = _judge(
        name="20260718_09h47m12s313ms_flux2Klein9Merged_v10.png",
        mtime=_instant(JULY_18 + 9 * HOUR + 48 * MIN + 17),
    )
    assert (swarm_mixed.precision, swarm_mixed.basis) == ("second", "filename")
    assert swarm_mixed.local_at == pytest.approx(JULY_18 + 9 * HOUR + 47 * MIN + 12.313)
    assert swarm_mixed.source_order is None, "that grammar carries no counter"
    assert swarm_mixed.quality == "corroborated"
    assert when.swarm_stamp("20260718T094712-x.png") == (JULY_18 + 9 * HOUR + 47 * MIN + 12, None)
    assert when.swarm_stamp("0947001-x.png") is None
    assert when.swarm_minute("20260718T094712001-x.png") is None, "the grammars never collide"
    assert when.swarm_minute("20260718_09h47m12s313ms_m.png") is None
    assert _judge(name="0947001-x.png", date_text=None) is None, "a default name without a day is no claim"


def test_a_verdict_depends_on_its_file_alone():
    """No sibling reaches the judge: the same claims give the same
    verdict whatever else is in the folder (a folder-level profile is
    a later, separately invalidated interpretation)."""
    import inspect

    assert "collapsed" not in inspect.signature(when.judge_generation).parameters
    assert "folder_id" not in inspect.getsource(when)


# --- the live library: a real SwarmUI run forms a session --------------------------


def _swarm(
    path,
    hhmm: str,
    counter: int,
    prompt: str,
    finish_wall: float,
    generation_time: float,
    *,
    name=None,
    day="2026-07-18",
):
    payload = {
        "sui_image_params": {"prompt": prompt, "model": "qwen", "seed": counter, "steps": 20, "cfgscale": 7},
        "sui_extra_data": {"date": day, "generation_time": f"{generation_time:.2f} sec"},
    }
    info = PngInfo()
    info.add_text("parameters", json.dumps(payload))
    file = path / (name or f"{hhmm}{counter:03d}-{abs(hash(prompt)) % 10**8:08x}-qwen.png")
    Image.new("RGB", (12, 12), (counter * 40 % 255, 90, 140)).save(file, pnginfo=info)
    os.utime(file, (_instant(finish_wall), _instant(finish_wall)))
    return file


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def test_a_real_swarm_run_becomes_a_minute_precision_session_with_estimates_to_the_second(tmp_path):
    """Five Swarm stills named 0947001.., finishing 65 s apart, whose
    metadata says only the DAY: the occurrences are minute-fine claims
    with second-fine estimates beside them, the session grouper accepts
    them, and the snapshot freezes the claim, the supports and the
    estimate."""
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(5):
        start = JULY_18 + 9 * HOUR + (47 + i) * MIN
        _swarm(root, f"09{47 + i:02d}", 1, f"a lighthouse variant {i}", start + 65, 64.3)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            for name, file_id in conn.execute("SELECT name, id FROM file").fetchall():
                ingest.one(conn, file_id, root / name, NOW)
            conn.commit()
        finally:
            connect.close(conn)
        client.post("/jobs/context")
        client.post("/jobs/events")
        _drain(client)
        conn = connect.connect(client.app.state.db_path)
        try:
            rows = conn.execute(
                "SELECT basis, time_precision, supports, conflicts, certainty, estimated_at - local_at"
                " FROM derived_media_occurrence WHERE kind = 'generation' ORDER BY local_at"
            ).fetchall()
            agreed = '["embedded_day", "mtime_finish_consistent", "btime_after_generation", "host_zone_assumed"]'
            assert [r[:5] for r in rows] == [("filename", "minute", agreed, None, 0.9)] * 5
            assert all(abs(r[5] - 0.7) < 0.01 for r in rows), "each request estimated 0.7 s into its minute"
            timeline = conn.execute(
                "SELECT mc.time_precision, mc.local_at - o.local_at, mc.time_supports FROM derived_media_context mc"
                " JOIN derived_media_occurrence o ON o.file_id = mc.file_id AND o.kind = 'generation'"
            ).fetchall()
            assert all(
                p == "second" and abs(d - 0.7) < 0.01 and '"estimate_inside_claim"' in s for p, d, s in timeline
            ), "the human timeline reads the refined second; the occurrence keeps the minute claim"
            sessions = conn.execute(
                "SELECT (SELECT count(*) FROM derived_event_file ef WHERE ef.event_id = e.id) FROM derived_event e"
                " WHERE e.kind = 'generation_session'"
            ).fetchall()
            assert sessions == [(5,)], "one session of five, at minute precision"
            assert context.state(conn)[1] == context.POLICY_VERSION
            event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'generation_session'").fetchone()[0]
            snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
            conn.commit()
            document = stories.load_snapshot(conn, snap.id)
            member = document["members"][0]["occurrence"]
            assert member["precision"] == "minute"
            assert member["supports"] == [
                "embedded_day",
                "mtime_finish_consistent",
                "btime_after_generation",
                "host_zone_assumed",
            ]
            assert member["conflicts"] == []
            assert member["source_order"] == 1
            assert member["estimated_at"] is not None
            assert member["finished_at"] is not None
            plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, snap.sha256)
            assert plan["subject"]["sequenced"] is True, "minute precision sequences"
        finally:
            connect.close(conn)


def _library(tmp_path, client, root) -> None:
    client.post("/roots", json={"path": str(root)})
    client.post("/roots/1/scan")
    conn = connect.connect(client.app.state.db_path)
    try:
        for name, file_id in conn.execute("SELECT name, id FROM file").fetchall():
            ingest.one(conn, file_id, root / name, NOW)
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)


def test_the_generators_own_order_inside_a_minute_survives_shuffled_file_ids(tmp_path):
    """Three requests inside 09:47, counters 001/002/003, deliberately
    scanned so that file ids run the other way: the session's members
    stay in the generator's order, and the counter never changes the
    claimed minute -- it is order, not seconds."""
    root = tmp_path / "lib"
    root.mkdir()
    start = JULY_18 + 9 * HOUR + 47 * MIN
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        # the files arrive one scan at a time, last request first, so
        # that file ids run against the generator's counter
        for counter in (3, 2, 1):
            _swarm(root, "0947", counter, f"variant {counter}", start + 20 * counter, 18.0)
            client.post("/roots/1/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            rows = conn.execute("SELECT id, name FROM file ORDER BY id").fetchall()
            assert [r[1][:7] for r in rows] == ["0947003", "0947002", "0947001"], "the control: ids run the other way"
            for file_id, name in rows:
                ingest.one(conn, file_id, root / name, NOW)
            conn.commit()
            ids = {name[:7]: file_id for file_id, name in rows}
        finally:
            connect.close(conn)
        client.post("/jobs/context")
        client.post("/jobs/events")
        _drain(client)
        conn = connect.connect(client.app.state.db_path)
        try:
            occurrences = conn.execute(
                "SELECT f.name, o.local_at, o.source_order FROM derived_media_occurrence o"
                " JOIN file f ON f.id = o.file_id WHERE o.kind = 'generation' ORDER BY o.source_order"
            ).fetchall()
            assert [(r[0][:7], r[1], r[2]) for r in occurrences] == [
                ("0947001", start, 1),
                ("0947002", start, 2),
                ("0947003", start, 3),
            ], "one claimed minute, three orders"
            members = conn.execute(
                "SELECT f.name FROM derived_event_file ef JOIN file f ON f.id = ef.file_id"
                " JOIN derived_event e ON e.id = ef.event_id WHERE e.kind = 'generation_session' ORDER BY ef.ordinal"
            ).fetchall()
            assert [m[0][:7] for m in members] == ["0947001", "0947002", "0947003"]
            assert ids["0947001"] > ids["0947003"], "the control: file ids ran the other way"
        finally:
            connect.close(conn)


def test_a_claim_the_generator_disputes_with_itself_is_recorded_and_never_sequenced(tmp_path):
    """A stamped name whose embedded day contradicts it: the conflict
    is frozen on the occurrence, the stamp stands as the claim, and
    the session grouper leaves the file out rather than lending a
    contested second full chronological authority."""
    root = tmp_path / "lib"
    root.mkdir()
    start = JULY_18 + 9 * HOUR + 47 * MIN
    for i in range(3):
        _swarm(root, "0947", i + 1, f"clean {i}", start + 20 * (i + 1), 18.0)
    _swarm(root, "", 9, "disputed", start + 90, 18.0, name="20260718T094730009-aaaaaaaa-qwen.png", day="2026-07-19")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        _library(tmp_path, client, root)
        conn = connect.connect(client.app.state.db_path)
        try:
            disputed = conn.execute(
                "SELECT o.time_precision, o.local_at, o.conflicts FROM derived_media_occurrence o"
                " JOIN file f ON f.id = o.file_id WHERE f.name LIKE '20260718T%' AND o.kind = 'generation'"
            ).fetchone()
            assert disputed[0] == "second"
            assert disputed[1] == start + 30, "the stamp stands"
            assert json.loads(disputed[2])[0].startswith("generator: ")
            members = conn.execute(
                "SELECT f.name FROM derived_event_file ef JOIN file f ON f.id = ef.file_id"
                " JOIN derived_event e ON e.id = ef.event_id WHERE e.kind = 'generation_session'"
            ).fetchall()
            assert len(members) == 3
            assert not any(m[0].startswith("20260718T") for m in members), "recorded, not sequenced"
        finally:
            connect.close(conn)
