"""The timeline draws any library, whatever its span: five minutes of
pictures or five centuries, the opening window and the whole extent
both draw, the scrubber fills its height exactly, every picture of
the extent is on the page, and the silence between days grows with the
days but never past a screen.

The spans are written into the interpretation (derived_media_context),
not the filesystem: a name claims a year only from 1990 (db/when.py
_YEARS), NTFS holds no time before 1601, and the timeline's contract is
with the moments the interpretation holds, whatever produced them."""

from __future__ import annotations

import itertools
import os
import pathlib
import re

import pytest
from PIL import Image

from db import connect, context, ingest
from tests.staging import staged
from tests.test_the_timeline_is_a_surface import NOW, _drain

PICTURES = 9
YEAR = 365.25 * 86_400
SPANS = {
    "5 minutes": 5 * 60,
    "5 days": 5 * 86_400,
    "5 weeks": 5 * 7 * 86_400,
    "5 months": 5 * 30 * 86_400,
    "5 years": 5 * YEAR,
    "55 years": 55 * YEAR,
    "555 years": 555 * YEAR,
}


def _library(root: pathlib.Path) -> None:
    for i in range(PICTURES):
        path = root / f"picture-{i:02d}.png"
        Image.new("RGB", (16 + i, 12), (20 * i, 60, 90)).save(path)
        os.utime(path, (NOW, NOW))


def _moments_of(span: float):
    """Interpret the library, then spread its moments evenly back over
    `span` from NOW, the newest at NOW. No grouping: a file session
    groups by when the files arrived, which is one moment here, and
    this is about where the pictures sit, not how they group."""

    def interpret(stage) -> None:
        client, root = stage.client, stage.root
        conn = stage.conn()
        try:
            chains = context._folder_names(conn)
            for name, file_id, folder_id in conn.execute("SELECT f.name, f.id, f.folder_id FROM file f").fetchall():
                sub = root.joinpath(*chains[folder_id][1:]) if len(chains[folder_id]) > 1 else root
                ingest.one(conn, file_id, sub / name, NOW)
            conn.commit()
        finally:
            connect.close(conn)
        client.post("/jobs/context")
        _drain(client)
        conn = stage.conn()
        try:
            ids = [row[0] for row in conn.execute("SELECT file_id FROM derived_media_context ORDER BY file_id")]
            assert len(ids) == PICTURES
            for i, file_id in enumerate(ids):
                at = NOW - span + (span * i) / (PICTURES - 1)
                conn.execute(
                    "UPDATE derived_media_context SET local_at = NULL, instant_at = ? WHERE file_id = ?", (at, file_id)
                )
            conn.commit()
        finally:
            connect.close(conn)

    return interpret


@pytest.fixture(params=list(SPANS), ids=list(SPANS))
def spanned(request, tmp_path_factory):
    name = f"span-{request.param.replace(' ', '-')}"
    with staged(tmp_path_factory, name, _library, _moments_of(SPANS[request.param])) as stage:
        yield stage.client, SPANS[request.param]


def _json(client, **params):
    told = client.get("/timeline", params=params, headers={"accept": "application/json"})
    assert told.status_code == 200, told.text
    return told.json()


def test_every_span_draws_its_opening_window_and_its_whole_extent(spanned):
    client, span = spanned
    opening = _json(client)
    assert opening["pictures_drawn"] >= 1
    assert opening["composition"] == "river", "an opening window is days of pictures"
    assert opening["river"], "the opening window holds at least one day of pictures"
    keys = [d["key"] for d in opening["river"]]
    assert keys == sorted(keys, reverse=True), "newest first"
    for day in opening["river"]:
        assert 0 <= day["silence"] <= 320
        assert (day["silence"] == 0) == (day["silent_days"] == 0)
        assert day["pictures"] == sum(len(g["pictures"]) for g in day["groups"])
    extent = opening["extent"]
    whole = _json(client, start=extent["start"], end=extent["end"] + 1)
    assert whole["pictures_total"] == PICTURES
    assert whole["pictures_drawn"] == PICTURES
    assert sum(b["pictures"] for b in whole["bins"]) == PICTURES, f"every picture is in a {whole['unit']} bar"
    assert len(whole["bins"]) <= 4_000
    assert sum(len(g["pictures"]) for g in whole["groups"]) == PICTURES
    if whole["composition"] == "river":
        silences = [d["silence"] for d in whole["river"]]
        assert silences[0] == 0, "the newest day has nothing above it"
        if span >= 5 * 7 * 86_400:
            assert all(s > 0 for s in silences[1:]), "days apart are drawn apart"
    elif whole["composition"] == "calendar":
        assert sum(m["pictures"] for m in whole["calendar"]) == PICTURES, "every picture is on a sheet"
    else:
        assert whole["composition"] == "years"
        assert sum(y["pictures"] for y in whole["years"]) == PICTURES, "every picture is in a year row"
        assert all(len(y["months"]) == 12 for y in whole["years"] if y["pictures"])
    for where in (opening, whole):
        segments = where["scrubber"]["segments"]
        assert segments
        assert sum(s["h"] for s in segments) == pytest.approx(1000, abs=1), "the scrubber fills its height exactly"
        assert all(s["h"] > 0 for s in segments)
        assert sum(s["pictures"] for s in segments) == PICTURES
        ats = [s["at"] for s in segments]
        assert ats == sorted(ats, reverse=True), "newest at the top"
        assert any(s["face"] for s in segments), "a month with pictures shows one"
        brush = where["scrubber"]["brush"]
        assert 0 <= brush["y"] <= 1000
        assert brush["h"] >= 4
        assert brush["y"] + brush["h"] <= 1001
    segments = whole["scrubber"]["segments"]
    held = [s for s in segments if s["pictures"]]
    gaps = [s for s in segments if not s["pictures"]]
    assert all(s["h"] >= 24 for s in held), "a month with pictures is never thinner than a thumb"
    assert all(s["h"] <= 10.01 for s in gaps), "a run of empty months is one short band, however long"
    assert all("without pictures" in s["label"] for s in gaps)
    assert not any(a["pictures"] == 0 and b["pictures"] == 0 for a, b in itertools.pairwise(segments)), (
        "empty bins run together"
    )
    assert all(s["strip"] and s["face"] == s["strip"][0] for s in held), "a segment carries a strip of its pictures"
    assert len(held) <= 40, "the unit is the finest that keeps the segments few"
    # the unit is the data's: the finest bin whose filled count keeps to 40
    # (a bin with nine pictures a minute apart scrubs by the minute), and
    # nothing coarser than the data needs
    names = ["minute", "quarter", "hour", "day", "week", "month", "year"]
    unit = whole["scrubber"]["unit"]
    finer = names[: names.index(unit)]
    if finer:
        asked = {"bin": finer[-1], "start": extent["start"], "end": extent["end"] + 1, "lean": "true"}
        answer = client.get("/timeline/density", params=asked, headers={"accept": "application/json"})
        assert answer.status_code == 400 or sum(1 for b in answer.json()["bins"] if b["pictures"]) > 40, (
            f"{finer[-1]} would have done and {unit} was chosen"
        )
    if span == 300:
        assert unit == "minute"
    if span >= 555 * YEAR:
        assert unit == "year", "centuries scrub by year"
        assert len(segments) < 600
    page = client.get(
        "/timeline", params={"start": extent["start"], "end": extent["end"] + 1}, headers={"accept": "text/html"}
    ).text
    if whole["composition"] == "river":
        assert len(set(re.findall(r'href="/i/([^"?]+)', page))) == PICTURES, "every picture is on the page"
    assert page.count("data-segment-at=") == len(whole["scrubber"]["segments"]), "every segment is on the page"
    assert page.count('class="bin"') == len(whole["bins"]), "every bar is on the page"
