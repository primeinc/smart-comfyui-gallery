"""What the benchmarks measured, on the page, said as a recording.

`just bench` writes JSON that only a terminal ever sees, and the
operations console is where a run's own facts already live. So the
throughput belongs there -- with two things the entry did not name and
the files made necessary.

**Only three of them are throughput.** There are twenty-three documents
under `benchmarks/results/` and NO key is shared by all of them:
calibration sweeps, recall tables, backend equivalence evidence. Three
agree on a shape because one script writes them, and those three are
the ones that answer "how fast did it go". A page that rendered the
rest would be a JSON viewer.

**Four of the twenty-three carry real filesystem paths.** Not the three
-- checked, and checked again here, because that is the kind of thing a
later benchmark adds without anyone noticing it reached a page.

**And they carry no timestamp of their own.** A rate on screen with no
date is a claim about whatever the tree looked like when somebody last
ran the recipe, and this tree moves: thumbnails went from 4.64 to 23.55
files a second in one afternoon. The file's mtime is shown and the
panel says the numbers were recorded rather than observed.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from litestar.testing import TestClient

from sg_web import operations
from sg_web.app import build_app

pytestmark = pytest.mark.slow

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def _page(tmp_path) -> str:
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        return client.get("/operations", headers={"accept": "text/html"}).text


def test_the_console_says_how_fast_each_measured_job_went(tmp_path):
    """The entry: throughput is something you look at rather than
    something you run."""
    page = _page(tmp_path)
    assert "data-operations-measured" in page
    for kind in ("scan", "embed", "annotate"):
        assert f'data-measured="{kind}"' in page, f"{kind} was measured and is not shown"


def test_a_rate_is_shown_with_the_day_it_was_recorded(tmp_path):
    """These files carry no timestamp of their own, so the mtime is the
    only honest answer to "when". A rate without one is a claim about
    code that has moved since."""
    page = _page(tmp_path)
    assert "data-measured-at" in page
    assert "recorded" in page
    assert "not observed now" in page, "the panel does not say these are recordings"


def test_it_says_where_the_time_went(tmp_path):
    """The rate alone says a job is slow; the phases say which part."""
    page = _page(tmp_path)
    assert "data-measured-phase" in page
    told = operations._recorded()
    biggest = max(told, key=lambda one: len(one.phases))
    assert biggest.phases[0].share >= biggest.phases[-1].share, "phases are not ordered by cost"
    assert 0 < biggest.phases[0].share <= 1


def test_no_filesystem_path_reaches_the_page(tmp_path):
    """Four of the twenty-three result documents hold real paths from
    the machine that ran them. None of the three shown do, and this is
    the assertion that notices if a later benchmark starts."""
    page = _page(tmp_path)
    # ONE backslash: the page renders decoded values, so a Windows path
    # arrives as `C:\ComfyUI`. The doubled form is what JSON SOURCE holds
    # and matches nothing here -- checked, because a probe that cannot
    # match is a test that cannot fail.
    for leak in ("C:" + chr(92), "C:/", "/home/", "/Users/", "ComfyUI", "sample-datasets"):
        assert leak not in page, f"the console rendered {leak!r} out of a benchmark result"


def test_the_three_it_reads_really_are_free_of_paths(tmp_path):
    """The same check against the FILES rather than the page, so a
    benchmark that starts recording paths fails here even if the
    template happens not to render that field."""
    for name in operations.MEASURED:
        one = RESULTS / name
        if not one.is_file():
            continue
        # TWO backslashes here: this is JSON SOURCE, where a Windows path
        # is escaped. Both forms are checked, since a result written by
        # another tool may not escape at all.
        body = one.read_text(encoding="utf-8")
        for leak in ("C:" + chr(92) * 2, "C:" + chr(92), "C:/", "/home/", "/Users/", "ComfyUI"):
            assert leak not in body, f"{name} now records a path; it must not reach the console"


def test_a_library_with_no_benchmarks_shows_no_panel(tmp_path):
    """An installed copy that ships no benchmarks, or a fresh checkout
    that has run none: the panel is absent rather than empty."""
    assert operations._recorded(root=tmp_path / "nothing-here") == []


def test_an_unreadable_result_is_skipped_and_not_a_500(tmp_path):
    """A half-written file -- a benchmark killed mid-write -- must cost
    its own row and not the console."""
    empty = tmp_path / "results"
    empty.mkdir()
    (empty / "job_scan.json").write_text("{ this is not json", encoding="utf-8")
    (empty / "embed_job.json").write_text(
        json.dumps({"items_per_second": 5.0, "files": 10, "wall_ms": 2000.0, "by_phase": {}}), encoding="utf-8"
    )
    told = operations._recorded(root=empty)
    assert [one.kind for one in told] == ["embed"], "the unreadable one took the readable one with it"


def test_a_result_with_no_rate_is_not_shown(tmp_path):
    """`items_per_second` is what makes a document a throughput
    measurement. Without it there is nothing to say."""
    empty = tmp_path / "results"
    empty.mkdir()
    (empty / "job_scan.json").write_text(json.dumps({"files": 10, "by_phase": {}}), encoding="utf-8")
    assert operations._recorded(root=empty) == []
