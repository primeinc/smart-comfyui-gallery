"""A camera's day is a story told in ACTS from the camera's own clock.

The camera writes more than a second: SubSecTimeOriginal, its body
serial, and -- on a Canon body -- the zone its clock was set to, in the
maker note. A video carries the same facts in its container and its
CNDA thumbnail. The judge (db/when.py) settles the capture act from all
of it with mtime and btime as evidence beside it; a RAW and its JPEG are
one act (one act key), never two members; the session grouper counts
acts; the capture planner phases a session by pauses and lens changes
and claims bursts, exposure ranges, equipment, renditions and clips; the
renderer words exactly those claims. The real RAW library proves it on
638 files when present.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import struct

import pytest
from litestar.testing import TestClient
from PIL import ExifTags, Image

from db import capture, connect, ingest, planning, rendering, runner, stories, when
from sg_web.app import build_app

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0
DAY = 86400.0
FEB_10 = 1_360_454_400.0  # 2013-02-10 00:00 as a wall clock
RAW = pathlib.Path("C:/ComfyUI/output/sample-datasets/RAW")


def _instant(wall: float) -> float:
    naive = datetime.datetime.fromtimestamp(wall, datetime.UTC).replace(tzinfo=None)
    return naive.astimezone().timestamp()


def _spelled(wall: float) -> str:
    return datetime.datetime.fromtimestamp(wall, datetime.UTC).strftime("%Y:%m:%d %H:%M:%S")


# --- the reader: the camera's finer clock ---------------------------------------------------


def _tiff_with_canon_time_info(zone_minutes: int, endian: str = "<") -> bytes:
    """A TIFF whose Exif IFD holds a Canon maker note with TimeInfo (tag
    0x35) -- the bare-IFD layout whose value offsets are TIFF-relative."""
    e = endian
    header = (b"II" if e == "<" else b"MM") + struct.pack(e + "HI", 42, 8)
    ifd0 = struct.pack(e + "H", 1) + struct.pack(e + "HHII", 0x8769, 4, 1, 26) + struct.pack(e + "I", 0)
    maker_len = 18 + 16
    exif_ifd = struct.pack(e + "H", 1) + struct.pack(e + "HHII", 0x927C, 7, maker_len, 44) + struct.pack(e + "I", 0)
    maker = struct.pack(e + "H", 1) + struct.pack(e + "HHII", 0x35, 4, 4, 62) + struct.pack(e + "I", 0)
    info = struct.pack(e + "iiii", 16, zone_minutes, 12, 0)
    blob = header + ifd0 + exif_ifd + maker + info
    assert (len(header + ifd0), len(header + ifd0 + exif_ifd), len(blob)) == (26, 44, 62 + 16)
    return blob


def test_the_canon_clock_zone_is_read_from_the_maker_note():
    assert capture.canon_time_zone(_tiff_with_canon_time_info(-300)) == -300
    assert capture.canon_time_zone(_tiff_with_canon_time_info(330, ">")) == 330
    assert capture.canon_time_zone(_tiff_with_canon_time_info(-300)[:60]) is None, "truncated is absence"
    assert capture.canon_time_zone(b"II*\x00\x08\x00\x00\x00\x00\x00") is None
    assert capture.canon_time_zone(None) is None
    assert capture.canon_time_zone(_tiff_with_canon_time_info(20_000)) is None, "no clock is 333 hours off"


def _photograph(path, *, when_text: str, subsec: str = "17", serial: str = "182029002226", lens="EF24-105mm", iso=400):
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "Canon"
    exif[ExifTags.Base.Model] = "Canon EOS 5D Mark III"
    photo = exif.get_ifd(ExifTags.IFD.Exif)
    photo[ExifTags.Base.DateTimeOriginal] = when_text
    photo[ExifTags.Base.SubsecTimeOriginal] = subsec
    photo[ExifTags.Base.BodySerialNumber] = serial
    photo[ExifTags.Base.LensModel] = lens
    photo[ExifTags.Base.ISOSpeedRatings] = iso
    photo[ExifTags.Base.FNumber] = 4.0
    photo[ExifTags.Base.ExposureTime] = 1 / 60
    photo[ExifTags.Base.FocalLength] = 50.0
    Image.new("RGB", (16, 16), (90, 110, 130)).save(path, exif=exif)
    return path


def test_the_reader_keeps_the_subsecond_and_the_body(tmp_path):
    found = capture.read(_photograph(tmp_path / "a.jpg", when_text="2013:02:10 08:29:58", subsec="17"))
    assert (found.subsec_ms, found.body_serial) == (170, "182029002226")
    assert found.maker_tz_offset_min is None, "no maker note, no zone -- absence, not a default"
    assert capture.read(_photograph(tmp_path / "b.jpg", when_text="2013:02:10 08:29:58", subsec="7")).subsec_ms == 700
    assert capture.read(_photograph(tmp_path / "c.jpg", when_text="2013:02:10 08:29:58", subsec="")).subsec_ms is None
    zero = capture.read(_photograph(tmp_path / "d.jpg", when_text="2013:02:10 08:29:58", iso=0))
    assert zero.iso is None, "ISO 0 is not recorded (a clip's thumbnail writes it), never a sensitivity"


@pytest.mark.skipif(not RAW.exists(), reason="the RAW sample library is not on this machine")
@pytest.mark.slow
def test_a_real_canon_body_is_read_whole():
    """5D Mark III files: the CR2 through Pillow's TIFF path, the JPEG,
    and the MOV through its CNDA thumbnail -- subsecond, serial, and the
    maker note's zone on every one of them."""
    cr2 = capture.read(RAW / "2013-02-11/666A1072.CR2")
    assert (cr2.subsec_ms, cr2.body_serial, cr2.maker_tz_offset_min) == (170, "182029002226", -300)
    assert cr2.camera == "Canon EOS 5D Mark III"
    assert cr2.lens == "EF24-105mm f/4L IS USM"
    jpg = capture.read(RAW / "2013-02-10/666A0200.JPG")
    assert (jpg.body_serial, jpg.maker_tz_offset_min, jpg.captured_at) == (
        "182029002226",
        -300,
        FEB_10 + 8 * HOUR + 29 * MIN + 58,
    )
    mov = capture.read_video(RAW / "2013-02-10/666A0209.MOV")
    assert mov.captured_at == FEB_10 + 8 * HOUR + 30 * MIN + 50
    assert mov.iso is None, "the clip's thumbnail says ISO 0: not recorded"
    assert (mov.body_serial, mov.maker_tz_offset_min, mov.camera) == ("182029002226", -300, "Canon EOS 5D Mark III")
    assert mov.lens == "EF24-105mm f/4L IS USM", "the CNDA thumbnail carries the still's EXIF"


# --- the judge: the capture act ---------------------------------------------------------------


def _judge(**over) -> when.Verdict:
    base = {
        "captured_at": FEB_10 + 8 * HOUR + 29 * MIN + 58,
        "subsec_ms": 170,
        "tz_offset_min": None,
        "maker_tz_offset_min": -300,
        "mtime": FEB_10 + 8 * HOUR + 29 * MIN + 58 + 5 * HOUR,  # the instant: 08:29:58 at -05:00
        "btime": FEB_10 + 200 * DAY,
        "duration": None,
    }
    told = when.judge_capture(**{**base, **over})
    assert told is not None
    return told


def test_the_maker_zone_makes_the_camera_clock_an_instant_and_mtime_is_the_write():
    told = _judge()
    assert (told.precision, told.basis) == ("subsecond", "capture")
    assert told.local_at is not None
    assert told.local_at == pytest.approx(FEB_10 + 8 * HOUR + 29 * MIN + 58.17)
    assert told.instant_at == pytest.approx(told.local_at + 5 * HOUR)
    assert told.tz_offset_min == -300
    assert told.supports == ("exif_subsecond", "maker_timezone", "mtime_write_consistent", "btime_after_capture")
    assert "host_zone_assumed" not in told.supports, "instants compare with instants; the host's zone is not consulted"
    assert told.conflicts == ()
    assert (told.quality, told.usable) == ("corroborated", True)
    assert told.finished_at == _judge().finished_at is not None


def test_an_mtime_an_hour_off_is_a_named_filesystem_dissent_that_cannot_demote_the_claim():
    told = _judge(mtime=FEB_10 + 8 * HOUR + 29 * MIN + 58 + 4 * HOUR)
    assert len(told.conflicts) == 1
    assert told.conflicts[0].startswith("filesystem: mtime")
    assert "1.0 h before the capture" in told.conflicts[0]
    assert (told.quality, told.usable) == ("contested", True)
    assert told.finished_at is None
    late = _judge(mtime=FEB_10 + 8 * HOUR + 29 * MIN + 58 + 5 * HOUR + 3 * DAY)
    assert "days after the capture" in late.conflicts[0]


def test_without_any_zone_the_host_clock_is_assumed_and_said_so():
    told = _judge(maker_tz_offset_min=None, mtime=_instant(FEB_10 + 8 * HOUR + 30 * MIN), btime=_instant(FEB_10 + DAY))
    assert told.instant_at is None
    assert told.tz_offset_min is None
    assert told.supports == ("exif_subsecond", "host_zone_assumed", "mtime_write_consistent", "btime_after_capture")


def test_the_two_zone_claims_must_agree():
    agreed = _judge(tz_offset_min=-300)
    assert {"exif_offset", "maker_timezone"} <= set(agreed.supports)
    assert agreed.local_at == pytest.approx(FEB_10 + 8 * HOUR + 29 * MIN + 58.17)
    split = _judge(tz_offset_min=60)
    assert split.conflicts[0].startswith("camera: OffsetTimeOriginal says +01:00 but the maker note")
    assert split.usable is True, "a zone dispute is recorded; the wall clock itself is not in doubt"


def test_btime_before_the_capture_is_bytes_born_before_taken_and_a_clip_writes_at_its_end():
    born = _judge(btime=FEB_10)
    assert "bytes born before taken" in born.conflicts[0]
    clip = _judge(duration=61.0, mtime=FEB_10 + 8 * HOUR + 29 * MIN + 58 + 5 * HOUR + 70)
    assert "mtime_write_consistent" in clip.supports, "a 61 s clip is written 61 s after it started"
    assert (
        when.judge_capture(
            captured_at=None, subsec_ms=None, tz_offset_min=None, maker_tz_offset_min=None, mtime=1.0, btime=None
        )
        is None
    )


# --- the live library: acts, sessions, the plan, the story --------------------------------


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _library(client, root) -> None:
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


def _pair(root, stem: str, wall: float, *, subsec="00", lens="EF24-105mm", iso=400):
    """One shutter press kept twice: a RAW-standing PNG and its JPEG,
    the same EXIF in both, mtime the camera's own write."""
    for suffix in ("png", "jpg"):
        file = _photograph(root / f"{stem}.{suffix}", when_text=_spelled(wall), subsec=subsec, lens=lens, iso=iso)
        os.utime(file, (_instant(wall), _instant(wall)))


def test_a_raw_and_its_jpeg_are_one_act_and_a_session_counts_acts(tmp_path):
    """Four shutter presses kept as eight files, three of them a burst,
    the last after a lens change; then one lone press kept twice. The
    pairs share an act key; the session has eight members but four
    acts, members ordered act by act; the lone pair is one act and no
    session."""
    root = tmp_path / "lib"
    root.mkdir()
    start = FEB_10 + 8 * HOUR
    _pair(root, "IMG_0001", start, subsec="00")
    _pair(root, "IMG_0002", start + 1, subsec="50")
    _pair(root, "IMG_0003", start + 2, subsec="10")
    _pair(root, "IMG_0004", start + 15 * MIN, subsec="00", lens="EF70-200mm", iso=1600)
    _pair(root, "IMG_0009", start + 9 * HOUR, subsec="00")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        _library(client, root)
        conn = connect.connect(client.app.state.db_path)
        try:
            keys = conn.execute(
                "SELECT f.name, o.act_key, o.time_precision, o.supports FROM derived_media_occurrence o"
                " JOIN file f ON f.id = o.file_id WHERE o.kind = 'capture' ORDER BY f.name"
            ).fetchall()
            by_stem: dict[str, set] = {}
            for name, key, precision, supports in keys:
                by_stem.setdefault(name.rsplit(".", 1)[0], set()).add(key)
                assert precision == "subsecond"
                assert '"exif_subsecond"' in supports
                assert '"mtime_write_consistent"' in supports
            assert all(len(held) == 1 for held in by_stem.values()), "both renditions of one press share the key"
            assert len({next(iter(v)) for v in by_stem.values()}) == 5, "five presses, five keys"
            sessions = conn.execute(
                "SELECT e.id, (SELECT count(*) FROM derived_event_file ef WHERE ef.event_id = e.id)"
                " FROM derived_event e WHERE e.kind = 'capture_session'"
            ).fetchall()
            assert [s[1] for s in sessions] == [8], "one session of eight files; the lone pair is one act, not two"
            order = conn.execute(
                "SELECT f.name FROM derived_event_file ef JOIN file f ON f.id = ef.file_id"
                " WHERE ef.event_id = ? ORDER BY ef.ordinal",
                (sessions[0][0],),
            ).fetchall()
            assert [o[0] for o in order] == [
                "IMG_0001.jpg",
                "IMG_0001.png",
                "IMG_0002.jpg",
                "IMG_0002.png",
                "IMG_0003.jpg",
                "IMG_0003.png",
                "IMG_0004.jpg",
                "IMG_0004.png",
            ], "act by act; neither is RAW, so by name inside each (RAW-first is proven on the real CR2s)"
            snap = stories.snapshot_event(conn, sessions[0][0], NOW + 30 * HOUR)
            conn.commit()
            document = stories.load_snapshot(conn, snap.id)
            first = document["members"][0]
            assert first["occurrence"]["act_key"] == document["members"][1]["occurrence"]["act_key"]
            assert first["capture"]["subsec_ms"] == 0
            assert first["capture"]["body_serial"] == "182029002226"
            assert first["capture"]["maker_tz_offset_min"] is None
            assert document["members"][2]["capture"]["subsec_ms"] == 500

            planner = planning.CaptureHistoryPlanner(None, {"pause_minutes": 10, "burst_seconds": 2.0})
            plan = planner.plan(document, snap.sha256)
            assert planning.validate_current_plan(plan, document, snap.sha256) == []
            assert plan["v"] == 7
            assert plan["subject"]["sequenced"] is True
            assert plan["subject"]["label_hint"] == "Canon EOS 5D Mark III session · 4 frames · 2 phases"
            assert [p["member_refs"] for p in plan["phases"]] == [
                ["member-001", "member-002", "member-003", "member-004", "member-005", "member-006"],
                ["member-007", "member-008"],
            ]
            kinds = {claim["id"]: claim for claim in plan["claims"]}
            first_phase = [kinds[c]["kind"] for c in plan["phases"][0]["claim_refs"]]
            second_phase = [kinds[c]["kind"] for c in plan["phases"][1]["claim_refs"]]
            assert first_phase == ["burst", "exposure_range", "equipment", "renditions"]
            assert second_phase == ["pause", "lens_change", "exposure_range", "equipment", "renditions"]
            burst = next(c for c in plan["claims"] if c["kind"] == "burst")
            assert burst["facts"] == {"frames": 3, "span_seconds": 2.1, "frames_per_second": 0.95}
            assert burst["evidence_refs"] == ["member-001:occurrence", "member-003:occurrence", "member-005:occurrence"]
            pause = next(c for c in plan["claims"] if c["kind"] == "pause")
            assert pause["facts"]["gap_seconds"] == pytest.approx(15 * MIN - 2.1)
            lens = next(c for c in plan["claims"] if c["kind"] == "lens_change")
            assert (len(lens["facts"]["added"]), len(lens["facts"]["removed"])) == (1, 1)
            renditions = [c["facts"] for c in plan["claims"] if c["kind"] == "renditions"]
            assert renditions == [{"acts": 3, "files": 6}, {"acts": 1, "files": 2}]
            exposure = [c["facts"] for c in plan["claims"] if c["kind"] == "exposure_range"]
            assert [e["iso"] for e in exposure] == [[400, 400], [1600, 1600]]
            assert plan["phases"][1]["label_hint"] == "Phase 2 · after a pause · new lens"

            render = rendering.TemplateStoryRenderer("memory").render(
                document, plan, snap.sha256, planning.identity(plan)[1]
            )
            assert rendering.violations(render, plan, document, snap.sha256, planning.identity(plan)[1]) == []
            assert render["title"] == "4 photographs with the Canon EOS 5D Mark III from February 10, 2013"
            assert render["summary"] == "These 4 photographs were taken on February 10, 2013 and fall into 2 phases."
            texts = [b["text"] for s in render["sections"] for b in s["blocks"]]
            assert texts[0] == "3 photographs (6 files)."
            assert "A burst of 3 frames in 2.1 s." in texts
            assert "Shot with Canon EOS 5D Mark III through EF24-105mm." in texts
            assert "3 photographs here are kept as 6 files." in texts
            assert "The camera was down for 14 min 58 s before this phase." in texts
            assert "The lens changes here to EF70-200mm, from EF24-105mm." in texts
            assert not any("Exposure spans" in t for t in texts), "the memory profile does not surface exposure"
            technical = rendering.TemplateStoryRenderer("technical").render(
                document, plan, snap.sha256, planning.identity(plan)[1]
            )
            assert "Exposure spans ISO 400; f/4; 1/60 s; 50 mm." in [
                b["text"] for s in technical["sections"] for b in s["blocks"]
            ]
            # the generation planner refuses this subject; the capture planner refuses that one
            with pytest.raises(ValueError, match="capture sessions only"):
                planning.CaptureHistoryPlanner().plan(
                    {**document, "subject": {**document["subject"], "event_kind": "generation_session"}}, snap.sha256
                )
        finally:
            connect.close(conn)


@pytest.mark.slow
def test_a_capture_plan_is_durable_work_without_loading_any_weights(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    start = FEB_10 + 8 * HOUR
    _pair(root, "IMG_0001", start)
    _pair(root, "IMG_0002", start + 30)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        _library(client, root)
        conn = connect.connect(client.app.state.db_path)
        try:
            event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'capture_session'").fetchone()[0]
            snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
            conn.commit()
        finally:
            connect.close(conn)
        asked = client.post(
            "/stories/plans",
            json={"snapshot_id": snap.id, "planner": "capture_history", "settings": {"pause_minutes": 5}},
        )
        assert asked.status_code in (200, 201, 202), asked.text
        _drain(client)
        conn = connect.connect(client.app.state.db_path)
        try:
            job = conn.execute("SELECT state, error FROM job WHERE kind = 'story_plan'").fetchone()
            assert job == ("done", None), job
            items = conn.execute(
                "SELECT state, error FROM job_item WHERE job_id IN (SELECT id FROM job WHERE kind = 'story_plan')"
            ).fetchall()
            assert items == [("done", None)], items
            row = conn.execute("SELECT id, planner, similarity FROM story_plan").fetchone()
            assert row[1:] == ("capture_history", "none"), (
                "no engine identity leaks into a plan that compares no prompts"
            )
            plan = planning.load_plan(conn, row[0])
            assert plan["planner"]["settings"] == {"burst_seconds": 2.0, "pause_minutes": 5}
            assert [c["kind"] for c in plan["claims"]] == ["exposure_range", "equipment", "renditions"]
            ref = rendering.render_plan(conn, row[0], rendering.TemplateStoryRenderer("memory"), NOW + 40 * HOUR)
            conn.commit()
            render = rendering.load_render(conn, ref.id)
            assert render["title"].startswith("2 photographs with the Canon EOS 5D Mark III")
        finally:
            connect.close(conn)


def test_the_v3_grammar_stays_frozen_and_v4_rejects_bent_capture_facts():
    plan = {
        "v": 4,
        "snapshot_sha256": "a" * 64,
        "planner": {
            "kind": "capture_history",
            "version": 1,
            "settings": {"pause_minutes": 10, "burst_seconds": 2.0},
            "similarity": {"name": "none", "version": "1"},
        },
        "subject": {"kind": "capture_session", "sequenced": True, "label_hint": "x"},
        "phases": [
            {
                "id": "phase-001",
                "member_refs": ["member-000"],
                "representative_refs": ["member-000"],
                "label_hint": "Phase 1",
                "claim_refs": ["claim-001"],
            }
        ],
        "claims": [
            {
                "id": "claim-001",
                "kind": "burst",
                "confidence": 1.0,
                "evidence_refs": ["member-000:occurrence"],
                "facts": {"frames": 3, "span_seconds": 1.0, "frames_per_second": 2.0},
            }
        ],
        "unsupported": [],
    }
    assert planning.validate_story_plan(plan) == []
    assert any("not a v3" in why for why in planning.validate_story_plan({**plan, "v": 3})), "capture is v4's"
    bent = {
        **plan,
        "claims": [{**plan["claims"][0], "facts": {"frames": 2, "span_seconds": 1.0, "frames_per_second": 1.0}}],
    }
    assert any("do not fit burst" in why for why in planning.validate_story_plan(bent)), "two frames are not a burst"
    wrong_settings = {**plan, "planner": {**plan["planner"], "settings": {"phase_threshold": 0.5}}}
    assert any("planner.settings" in why for why in planning.validate_story_plan(wrong_settings))
    generation = {
        **plan,
        "planner": {**plan["planner"], "kind": "generation_history", "settings": {"phase_threshold": 0.5}},
        "subject": {**plan["subject"], "kind": "generation_session"},
        "claims": [],
        "phases": [{**plan["phases"][0], "claim_refs": []}],
    }
    assert planning.validate_story_plan(generation) == [], "a generation plan's own settings, exactly, under v4"


# --- the real library ---------------------------------------------------------------------------


@pytest.mark.skipif(not RAW.exists(), reason="the RAW sample library is not on this machine")
@pytest.mark.slow
def test_a_real_canon_day_becomes_acts_sessions_and_a_story(tmp_path):
    """2013-02-10: 356 files from one 5D Mark III -- CR2+JPG pairs and
    one MOV. Every pair is one act; the MOV is an act with the still's
    facts; the maker zone makes every claim an instant; sessions form
    over acts; the plan and the render say what the camera did."""
    root = RAW / "2013-02-10"
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        _library(client, root)
        conn = connect.connect(client.app.state.db_path)
        try:
            files = conn.execute("SELECT count(*) FROM file").fetchone()[0]
            occurrences = conn.execute(
                "SELECT count(*), count(DISTINCT act_key) FROM derived_media_occurrence WHERE kind = 'capture'"
            ).fetchone()
            assert occurrences[0] == files, "every file, the MOV included, has a capture occurrence"
            pairs = conn.execute(
                "SELECT count(*) FROM (SELECT act_key FROM derived_media_occurrence WHERE kind = 'capture'"
                " GROUP BY act_key HAVING count(*) = 2)"
            ).fetchone()[0]
            assert occurrences[1] == files - pairs
            assert pairs > 100
            zones = conn.execute(
                "SELECT DISTINCT tz_offset_min, time_precision FROM derived_media_occurrence WHERE kind = 'capture'"
            ).fetchall()
            assert zones == [(-300, "subsecond")], "the maker note's zone and the subsecond clock, on every file"
            supports = conn.execute(
                "SELECT supports, count(*) FROM derived_media_occurrence WHERE kind = 'capture' GROUP BY supports"
            ).fetchall()
            assert all('"maker_timezone"' in s and '"mtime_write_consistent"' in s for s, _ in supports), supports
            events = conn.execute(
                "SELECT e.id, (SELECT count(*) FROM derived_event_file ef WHERE ef.event_id = e.id)"
                " FROM derived_event e WHERE e.kind = 'capture_session' ORDER BY 2 DESC"
            ).fetchall()
            assert events
            assert sum(n for _, n in events) == files
            first_pair = conn.execute(
                "SELECT f.name FROM derived_event_file ef JOIN file f ON f.id = ef.file_id"
                " WHERE ef.event_id = ? ORDER BY ef.ordinal LIMIT 2",
                (events[0][0],),
            ).fetchall()
            assert first_pair[0][0].endswith(".CR2"), "inside one act the RAW leads"
            assert first_pair[0][0][:-4] == first_pair[1][0][:-4]
            snap = stories.snapshot_event(conn, events[0][0], NOW + 30 * HOUR)
            conn.commit()
            document = stories.load_snapshot(conn, snap.id)
            plan = planning.CaptureHistoryPlanner().plan(document, snap.sha256)
            assert planning.validate_current_plan(plan, document, snap.sha256) == []
            kinds = {c["kind"] for c in plan["claims"]}
            assert {"burst", "exposure_range", "equipment", "renditions", "video_clip"} <= kinds, kinds
            assert plan["subject"]["label_hint"].startswith("Canon EOS 5D Mark III session")
            render = rendering.TemplateStoryRenderer("memory").render(
                document, plan, snap.sha256, planning.identity(plan)[1]
            )
            assert rendering.violations(render, plan, document, snap.sha256, planning.identity(plan)[1]) == []
            assert "with the Canon EOS 5D Mark III from February 10, 2013" in render["title"]
        finally:
            connect.close(conn)
