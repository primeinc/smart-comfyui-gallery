"""Every file has a when -- the file's own claims, judged like the rest.

A screenshot, a download, a scan: no camera wrote EXIF, no generator
embedded a date. The file still says things. Its NAME may carry a
stamp in one of the grammars phones, screenshots and generators use;
its FOLDER may be a date; and the filesystem knows the earliest its
bytes existed. db/when.py judge_file takes the finest claim the file
itself makes and lets the filesystem support or dispute it -- the same
stack the generator's and the camera's claims go through, never a
different, lesser rule. With no claim at all the occurrence is the
earliest known existence, the smaller of mtime and btime, named for
what it is. Those occurrences form file sessions, so a folder of
screenshots is a session like any other.
"""

from __future__ import annotations

import datetime
import os
import pathlib

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, context, ingest, runner, stories, when
from sg_web.app import build_app

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0
DAY = 86400.0
JUNE_10 = 1_686_355_200.0  # 2023-06-10 00:00 as a wall clock
SWARM_MIXED = pathlib.Path("C:/ComfyUI/output/sample-datasets/swarm-mixed")


def _instant(wall: float) -> float:
    naive = datetime.datetime.fromtimestamp(wall, datetime.UTC).replace(tzinfo=None)
    return naive.astimezone().timestamp()


# --- the grammars ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "wall", "precision"),
    [
        ("IMG_20230610_142301.jpg", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
        ("PXL_20230610_142301123.jpg", JUNE_10 + 14 * HOUR + 23 * MIN + 1.123, "subsecond"),
        ("2023-06-10 14.23.01.jpg", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
        ("Screenshot 2023-06-10 at 14.23.01.png", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
        ("Screenshot_20230610-142301.png", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
        ("signal-2023-06-10-142301.jpg", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
        ("20230610_14h23m01s500ms_model.png", JUNE_10 + 14 * HOUR + 23 * MIN + 1.5, "subsecond"),
        ("20230610T142301001-hash-model.png", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
        ("IMG-20230610-WA0001.jpg", JUNE_10, "day"),
        ("2023-06-10.png", JUNE_10, "day"),
        ("1686406981123.jpg", JUNE_10 + 14 * HOUR + 23 * MIN + 1.123, "subsecond"),
        ("1686406981.jpg", JUNE_10 + 14 * HOUR + 23 * MIN + 1, "second"),
    ],
)
def test_a_stamped_name_is_read_at_the_precision_it_carries(name, wall, precision):
    told = when.name_stamp(name)
    assert told is not None, name
    assert told[0] == pytest.approx(wall)
    assert told[1] == precision


@pytest.mark.parametrize(
    "name",
    ["000324.jpg", "pic2_original.png", "0947001-c5afa607-m.png", "666A0200.CR2", "20230610_999999.jpg", "a.png"],
)
def test_a_name_that_carries_no_stamp_says_nothing(name):
    told = when.name_stamp(name)
    assert told is None or (name == "20230610_999999.jpg" and told[1] == "day"), (
        "a camera counter, a Swarm request prefix, a frame number: none is a date; 99:99:99 is not a time"
    )


def test_the_swarm_counter_after_the_t_is_order_not_milliseconds():
    assert when.name_stamp("20260718T094712001-x.png")[1] == "second"
    assert when.name_stamp("PXL_20260718_094712001.jpg")[1] == "subsecond", "the same digits after `_` are PXL's ms"


def test_a_dated_folder_is_a_day_claim_nearest_first():
    assert when.folder_day(["RAW", "2013-02-10"]) == 1_360_454_400.0
    assert when.folder_day(["RAW", "20130210"]) == 1_360_454_400.0
    assert when.folder_day(["x", "2013", "02", "10"]) == 1_360_454_400.0
    assert when.folder_day(["2013-02-10", "selects"]) == 1_360_454_400.0, "an ancestor names the day"
    assert when.folder_day(["2013-02-10", "2013-02-11"]) == 1_360_540_800.0, "the nearest folder wins"
    assert when.folder_day(["celeba", "img_align_celeba"]) is None
    assert when.folder_day(["2013-13-40"]) is None


# --- the judge ---------------------------------------------------------------------------------


def test_with_no_claim_the_occurrence_is_the_earliest_known_existence():
    copied = when.judge_file(name="000324.jpg", folders=["x"], mtime=NOW, btime=NOW + 30 * DAY)
    assert (copied.basis, copied.instant_at, copied.local_at, copied.precision) == ("mtime", NOW, None, "subsecond")
    assert copied.supports == (), "a copy made a month later is not consistent with anything"
    born_first = when.judge_file(name="000324.jpg", folders=["x"], mtime=NOW + HOUR, btime=NOW)
    assert (born_first.basis, born_first.instant_at) == ("btime", NOW), "an edit saved later: the birth is earlier"
    assert born_first.supports == ("btime_consistent",)
    assert when.judge_file(name="a.png", folders=[], mtime=None, btime=NOW).basis == "btime"
    assert when.judge_file(name="a.png", folders=[], mtime=None, btime=None) is None


def test_a_stamped_name_is_the_claim_and_the_filesystem_is_evidence_beside_it():
    at = JUNE_10 + 14 * HOUR + 23 * MIN + 1
    told = when.judge_file(
        name="IMG_20230610_142301.jpg", folders=["Camera"], mtime=_instant(at + 2), btime=_instant(at + 40 * DAY)
    )
    assert (told.basis, told.local_at, told.precision, told.instant_at) == ("filename", at, "second", None)
    assert told.supports == ("mtime_consistent", "btime_after_claim", "host_zone_assumed")
    assert told.conflicts == ()
    assert (told.quality, told.usable) == ("corroborated", True)
    assert told.finished_at == _instant(at + 2)
    resaved = when.judge_file(name="IMG_20230610_142301.jpg", folders=[], mtime=_instant(at + 3 * DAY), btime=None)
    assert resaved.conflicts[0].startswith("filesystem: mtime")
    assert "days after the claimed window" in resaved.conflicts[0]
    assert (resaved.quality, resaved.usable) == ("contested", True), "a re-save disputes; it does not demote"
    impossible = when.judge_file(name="IMG_20230610_142301.jpg", folders=[], mtime=_instant(at - DAY), btime=None)
    assert "before the claimed" in impossible.conflicts[0]


def test_a_dated_folder_is_a_day_claim_and_agrees_or_disputes_with_the_name():
    day = when.judge_file(
        name="scan-012.png", folders=["archive", "2023-06-10"], mtime=_instant(JUNE_10 + 9 * HOUR), btime=None
    )
    assert (day.basis, day.local_at, day.precision) == ("folder", JUNE_10, "day")
    assert day.supports == ("mtime_consistent", "host_zone_assumed")
    agreed = when.judge_file(name="IMG_20230610_142301.jpg", folders=["2023-06-10"], mtime=None, btime=None)
    assert agreed.supports == ("folder_day",)
    assert agreed.basis == "filename"
    split = when.judge_file(name="IMG_20230610_142301.jpg", folders=["2023-06-11"], mtime=None, btime=None)
    assert split.basis == "filename", "the name is finer and stands"
    assert split.conflicts[0].startswith("file: the folder says 2023-06-11 but the name says")
    assert split.usable is True


# --- the live library --------------------------------------------------------------------------


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
        rows = conn.execute(
            "SELECT f.name, f.id, f.folder_id FROM file f JOIN folder d ON d.id = f.folder_id ORDER BY f.id"
        ).fetchall()
        paths = dict(context._folder_names(conn).items())
        for name, file_id, folder_id in rows:
            sub = pathlib.Path(*paths[folder_id][1:]) if len(paths[folder_id]) > 1 else pathlib.Path()
            ingest.one(conn, file_id, root / sub / name, NOW)
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)


def _plain(path, at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), (30, 60, 90)).save(path)
    os.utime(path, (at, at))


def test_claimless_files_get_a_file_occurrence_and_form_file_sessions(tmp_path):
    """Three screenshots named to the second inside one afternoon, a
    scan in a dated folder, two downloads with nothing but mtime, one
    of them copied a month later (btime after mtime): every one has a
    `file` occurrence at its own basis; the named ones form a session
    on the wall clock; the claimless ones cluster on instants; and a
    captured photograph in the same library is NOT a file-session
    member -- its story is the camera's."""
    root = tmp_path / "lib"
    root.mkdir()
    at = JUNE_10 + 14 * HOUR
    for i in range(3):
        _plain(root / f"Screenshot 2023-06-10 at 14.0{i}.00.png", _instant(at + i * MIN + 5))
    _plain(root / "2023-06-10" / "scan-001.png", _instant(JUNE_10 + 9 * HOUR))
    _plain(root / "download-a.png", NOW)
    _plain(root / "download-b.png", NOW + 10 * MIN)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        _library(client, root)
        conn = connect.connect(client.app.state.db_path)
        try:
            rows = {
                name: (basis, local, instant, precision, supports)
                for name, basis, local, instant, precision, supports in conn.execute(
                    "SELECT f.name, o.basis, o.local_at, o.instant_at, o.time_precision, o.supports"
                    " FROM derived_media_occurrence o JOIN file f ON f.id = o.file_id WHERE o.kind = 'file'"
                )
            }
            assert len(rows) == 6, "every claimless file has a file occurrence"
            assert rows["Screenshot 2023-06-10 at 14.00.00.png"][:4] == ("filename", at, None, "second")
            assert '"mtime_consistent"' in rows["Screenshot 2023-06-10 at 14.00.00.png"][4]
            assert rows["scan-001.png"][:4] == ("folder", JUNE_10, None, "day")
            assert rows["download-a.png"][:4] == ("mtime", None, NOW, "subsecond")
            primary = dict(
                conn.execute(
                    "SELECT f.name, mc.time_basis FROM derived_media_context mc JOIN file f ON f.id = mc.file_id"
                ).fetchall()
            )
            assert primary["scan-001.png"] == "folder", "the human timeline reads the same claim"
            sessions = conn.execute(
                "SELECT e.kind, e.local_start IS NOT NULL, e.instant_start IS NOT NULL,"
                " (SELECT group_concat(f.name, '|') FROM derived_event_file ef JOIN file f ON f.id = ef.file_id"
                "   WHERE ef.event_id = e.id ORDER BY ef.ordinal)"
                " FROM derived_event e ORDER BY e.id"
            ).fetchall()
            assert sessions == [
                ("file_session", 0, 1, "download-a.png|download-b.png"),
                (
                    "file_session",
                    1,
                    0,
                    (
                        "Screenshot 2023-06-10 at 14.00.00.png|Screenshot 2023-06-10 at 14.01.00.png"
                        "|Screenshot 2023-06-10 at 14.02.00.png"
                    ),
                ),
            ], "instants cluster with instants, wall clocks with wall clocks; the day-fine scan is too coarse"
            snap = stories.snapshot_event(conn, 2, NOW + 30 * HOUR)
            conn.commit()
            document = stories.load_snapshot(conn, snap.id)
            assert document["subject"]["event_kind"] == "file_session"
            assert document["subject"]["claim"] == "file"
            assert document["members"][0]["occurrence"]["basis"] == "filename"
        finally:
            connect.close(conn)


@pytest.mark.skipif(not SWARM_MIXED.exists(), reason="the swarm-mixed sample library is not on this machine")
@pytest.mark.slow
def test_the_swarm_sidecar_family_gets_its_when_from_the_name(tmp_path):
    """The mp4, the swarmpreview.jpg/webp beside each Swarm clip carry
    no metadata of their own; their names carry the clip's stamp to the
    millisecond, so they are second-fine file occurrences, not silent."""
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        _library(client, SWARM_MIXED)
        conn = connect.connect(client.app.state.db_path)
        try:
            held = conn.execute(
                "SELECT f.name, o.basis, o.time_precision, o.local_at FROM derived_media_occurrence o"
                " JOIN file f ON f.id = o.file_id WHERE o.kind = 'file' ORDER BY f.name"
            ).fetchall()
            assert len(held) == 15, [h[0] for h in held]
            assert all(basis == "filename" and precision == "subsecond" for _, basis, precision, _ in held)
            clip = next(h for h in held if h[0].endswith(".mp4"))
            assert datetime.datetime.fromtimestamp(clip[3], datetime.UTC).strftime("%H:%M:%S") == "17:30:02"
        finally:
            connect.close(conn)
