"""What the surface tells you about itself, and what it used to get wrong.

Every test here is a defect somebody found by LOOKING at the page. None
of them was caught by a test, because each one is about what a sentence
says rather than about whether a route answers -- and a route answering
200 with the word `None` in it is a route that answers.

The one that matters most is the first. A scope-narrowed timeline said
`showing only None` for seven of its eight scopes, on a live page, for
as long as the scope line has existed.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import re
import time

import pytest
from PIL import ExifTags, Image

from db import connect, runner
from tests.staging import staged

pytestmark = pytest.mark.slow

AS_BROWSER = {"accept": "text/html"}
AS_JSON = {"accept": "application/json"}


def _drain(client) -> None:
    """A scan finds files; the CONTEXT job is what gives them moments.
    Without it the surface has nothing to date and says so instead of
    saying anything this file is about."""
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", time.time() + 86_400) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _library(root) -> None:
    # Wider than the opening window, and clustered at the end. A library
    # that fits INSIDE the last thirty days makes `all` current and hides
    # the case one of these tests is about; a few old pictures and a
    # recent cluster is both the ordinary shape and the one where the
    # tightened window matches no preset at all.
    when = datetime.datetime(2026, 5, 1, 10, tzinfo=datetime.UTC).timestamp()
    for i, day in enumerate((0, 30, 60, 95, 97, 99)):
        path = root / f"p{i}.png"
        Image.new("RGB", (32, 24), (10 * i, 90, 140)).save(path)
        os.utime(path, (when + day * 86_400, when + day * 86_400))

    # One picture whose NAME claims a date its EXIF contradicts, so the
    # surface has something contested to count and the line that counts
    # it is not skipped past. TWO CLAIMS are what it takes: an mtime that
    # disagrees is filesystem dissent, which cannot demote a claim and
    # never reads as a conflict (tests/corpus.py `_name_for`).
    #
    # The EXIF stamp is a LOCAL wall clock with no zone, which is what a
    # camera writes: spelling it in UTC on a machine that is not UTC
    # makes EVERY photograph disagree by the offset, and the one that
    # disagrees on purpose stops being visible among them
    # (tests/corpus.py `_exif_for`).
    named = datetime.datetime.fromtimestamp(when, datetime.UTC)
    muddled = root / f"IMG_{named:%Y%m%d}_{named:%H%M%S}.png"
    said = datetime.datetime.fromtimestamp(when - 400 * 86_400, datetime.UTC).astimezone()
    exif = Image.Exif()
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = said.strftime("%Y:%m:%d %H:%M:%S")
    Image.new("RGB", (32, 24), (200, 40, 40)).save(muddled, exif=exif)
    os.utime(muddled, (when, when))


def _dated(stage) -> None:
    """The jobs that give the pictures their moments, once."""
    _drain(stage.client)
    for job in ("/jobs/ingest", "/jobs/context"):
        stage.client.post(job)
        _drain(stage.client)


@pytest.fixture(scope="module")
def _shown_stage(tmp_path_factory):
    with staged(tmp_path_factory, "the_timeline_says_what_it_is_showing", _library, _dated) as stage:
        yield stage


@pytest.fixture
def shown(_shown_stage):
    """The dated library, restored between tests.

    Every test here reads a rendered timeline and changes nothing, but
    each was paying for six pictures, an application, a scan and two
    drained job queues -- a third of a second of setup for a read that
    costs a tenth.
    """
    _shown_stage.restore()
    return _shown_stage.client


def _scope_line(client, qs: str) -> str:
    page = client.get(f"/timeline?{qs}", headers=AS_BROWSER).text
    found = re.search(r"showing only (.*?) &middot;", page)
    assert found is not None, f"no scope line for {qs}"
    return found.group(1)


def test_a_narrowed_timeline_says_what_it_is_narrowed_to(shown):
    """The defect, and it was on a live page.

    `TimelineScopePart.spelled` defaults to None, so Jinja's
    `is defined` -- which was the test -- is ALWAYS true on a wire
    model, and the template printed the literal. The template had been
    written against the plain dict `_scope_told` returns, where a
    missing key really is Undefined; giving the model a default
    disarmed it silently.

    Seven of eight scopes: folder, album, person, artifact, kind,
    favorite, rating_min. Only a facet escaped, because a facet really
    does carry `spelled`.
    """
    for qs in ("kind=image", "favorite=1", "favorite=0", "rating_min=4"):
        said = _scope_line(shown, qs)
        assert "None" not in said, f"{qs} still says None: {said}"


def test_it_says_the_clause_in_words_and_never_as_a_key(shown):
    """The fallback was `key=value`, so a bool would have read
    `favorite=False`. What a clause is CALLED belongs to
    db/vocabulary.py -- a chip printing a key is the exact drift that
    module opens by describing."""
    assert _scope_line(shown, "kind=image") == "<code>kind image</code>"
    assert _scope_line(shown, "favorite=1") == "<code>favorite yes</code>"
    assert _scope_line(shown, "favorite=0") == "<code>favorite no</code>"
    assert _scope_line(shown, "rating_min=4") == "<code>rating from 4</code>"


def test_a_facet_reads_in_words_too(shown):
    """It was already right by accident -- a facet carries its own
    spelling -- but it spelled itself `tag:eq:beach`, which is the URL
    rather than the sentence."""
    assert _scope_line(shown, "f=tag%3Aeq%3Abeach") == "<code>keyword beach</code>"


def test_two_clauses_are_two_chips(shown):
    said = _scope_line(shown, "kind=image&favorite=1")
    assert said == "<code>kind image</code>, <code>favorite yes</code>"


def test_the_whole_library_shows_no_scope_line_at_all(shown):
    """Null scope is not an empty scope: an unnarrowed timeline should
    not say `showing only` followed by nothing."""
    page = shown.get("/timeline", headers=AS_BROWSER).text
    assert "showing only" not in page


def test_the_bars_say_what_their_height_means(shown):
    """ "each bar is a day" names the x unit, and nothing named the
    other one -- so a bar could be five pictures or five hundred and the
    only way to find out was to hover it."""
    told = shown.get("/timeline", headers=AS_JSON).json()
    assert "tallest" in told["note"], told["note"]
    tallest = max(one["pictures"] for one in told["bins"])
    assert f"tallest {tallest:,}" in told["note"]


def test_a_window_matching_no_preset_says_which_window_it_is(shown):
    """Every first visit lands here: the opening window is the last
    month TIGHTENED to where the pictures sit, so it is never exactly a
    month and no preset is ever current. The control looked like
    nothing was selected rather than like the window was its own."""
    told = shown.get("/timeline", headers=AS_JSON).json()
    assert not [one for one in told["presets"] if one["current"]], "a preset matched; pick a library where none does"
    page = shown.get("/timeline", headers=AS_BROWSER).text
    assert "data-custom" in page
    assert told["window_spelled"] in page


def test_a_preset_that_is_current_is_readable_without_colour():
    """`--imprint` is already the link colour, the major tick, the `now`
    marker and the bar's hover. Current-by-colour-alone read as
    clickable, so the box carries it and the colour confirms it."""
    css = (pathlib.Path(__file__).resolve().parent.parent / "sg_web" / "static" / "gallery.css").read_text(
        encoding="utf-8"
    )
    block = css[css.index(".surface-zoom a[data-current]") :][:220]
    assert "border" in block, "current is marked by colour alone"


def test_a_count_of_conflicts_carries_what_it_is_out_of(shown):
    """`752 with conflicting dates` reads as a broken library. `752 of
    1,721` reads as a fact, and it is the same fact."""
    page = shown.get("/timeline", headers=AS_BROWSER).text
    if "conflicting dates" not in page:
        pytest.skip("this library has no conflicts to count")
    assert re.search(r"[\d,]+ of [\d,]+ with conflicting dates", page), page


def test_a_thumbnail_is_centred_on_the_bar_it_belongs_to(shown):
    """A 40px thumbnail anchored at the left edge of a 2px bar sits
    entirely to one side of the thing it belongs to, and a reader pairs
    it with whichever bar it happens to overlap."""
    told = shown.get("/timeline", headers=AS_JSON).json()
    page = shown.get("/timeline", headers=AS_BROWSER).text
    for one in told["bins"]:
        if not one["samples"]:
            continue
        middle = (one["x"] + one["w"] / 2) / 10
        assert f"left:{middle}%" in page, f"bin at {one['x']} is not centred: expected {middle}%"
        break
    else:
        pytest.skip("no bin carried a thumbnail")
