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
    assert told.supports == ("embedded_day", "mtime_finish_consistent", "btime_after_generation")
    assert told.conflicts == ()
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
    assert told.supports == ("embedded_day", "btime_after_generation")
    assert len(told.conflicts) == 1
    assert told.conflicts[0].startswith("mtime 2026-08-20 05:00:00 is")
    assert "days after" in told.conflicts[0]
    assert (told.quality, told.certainty) == ("contested", 0.4)
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
    assert no_mtime.supports == ("embedded_day", "btime_after_generation")
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
    """The opt-in `[year][month][day]T[hour][minute][second]` name: the
    T is the marker, the second is the generator's; a stamp outside the
    claimed day is a named conflict and the day stands."""
    stamped = _judge(name="20260718T094712001-c5afa607-qwnImageEdit.png")
    assert (stamped.precision, stamped.basis) == ("second", "filename")
    assert stamped.local_at == JULY_18 + 9 * HOUR + 47 * MIN + 12
    assert stamped.quality == "corroborated"
    wrong_day = _judge(name="20260719T094712001-c5afa607-m.png")
    assert (wrong_day.precision, wrong_day.basis, wrong_day.quality) == ("day", "embedded", "contested")
    assert "is not inside the claimed day" in wrong_day.conflicts[0]
    assert when.swarm_stamp("20260718T094712-x.png") == (JULY_18 + 9 * HOUR + 47 * MIN + 12, 0)
    assert when.swarm_stamp("0947001-x.png") is None
    assert when.swarm_minute("20260718T094712001-x.png") is None, "the grammars never collide"


def test_a_verdict_depends_on_its_file_alone():
    """No sibling reaches the judge: the same claims give the same
    verdict whatever else is in the folder (a folder-level profile is
    a later, separately invalidated interpretation)."""
    import inspect

    assert "collapsed" not in inspect.signature(when.judge_generation).parameters
    assert "folder_id" not in inspect.getsource(when)


# --- the live library: a real SwarmUI run forms a session --------------------------


def _swarm(path, hhmm: str, counter: int, prompt: str, finish_wall: float, generation_time: float):
    payload = {
        "sui_image_params": {"prompt": prompt, "model": "qwen", "seed": counter, "steps": 20, "cfgscale": 7},
        "sui_extra_data": {"date": "2026-07-18", "generation_time": f"{generation_time:.2f} sec"},
    }
    info = PngInfo()
    info.add_text("parameters", json.dumps(payload))
    file = path / f"{hhmm}{counter:03d}-{abs(hash(prompt)) % 10**8:08x}-qwen.png"
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
            agreed = '["embedded_day", "mtime_finish_consistent", "btime_after_generation"]'
            assert [r[:5] for r in rows] == [("filename", "minute", agreed, None, 0.9)] * 5
            assert all(abs(r[5] - 0.7) < 0.01 for r in rows), "each request estimated 0.7 s into its minute"
            sessions = conn.execute(
                "SELECT (SELECT count(*) FROM derived_event_file ef WHERE ef.event_id = e.id) FROM derived_event e"
                " WHERE e.kind = 'generation_session'"
            ).fetchall()
            assert sessions == [(5,)], "one session of five, at minute precision"
            assert context.state(conn)[1] == 5
            event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'generation_session'").fetchone()[0]
            snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
            conn.commit()
            document = stories.load_snapshot(conn, snap.id)
            member = document["members"][0]["occurrence"]
            assert member["precision"] == "minute"
            assert member["supports"] == ["embedded_day", "mtime_finish_consistent", "btime_after_generation"]
            assert member["conflicts"] == []
            assert member["estimated_at"] is not None
            assert member["finished_at"] is not None
            plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, snap.sha256)
            assert plan["subject"]["sequenced"] is True, "minute precision sequences"
        finally:
            connect.close(conn)
