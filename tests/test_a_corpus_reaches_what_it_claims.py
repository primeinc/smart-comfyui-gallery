"""The corpus is held to a measurement, not to an opinion.

"Make the unreachable reader branches reachable" is unfalsifiable until
something measures it, and an unfalsifiable goal cannot be finished --
the phrase names a set you re-derive every time you think about it, and
you think better each time, so the set grows and the work never closes.

So it was measured once, against the synthetic corpus, and frozen:
`tests/reach_baseline.json`, 1356 unreached lines across ten reader
modules. That file is the target. A frozen list can be exhausted. An
intuition cannot.

TWO RULES THESE TESTS OBEY, and they are why this file can be finished.

**Enumeration comes from the application; expectations never do.** What
must be covered is read out of `db/scan.KIND_BY_SUFFIX`, so the corpus
cannot silently drift out of step with what the app claims. But an
expectation derived from the code under test is a tautology: `assert
scan(f).kind == KIND_BY_SUFFIX[s]` passes by construction and proves
nothing. Expectations come from the FILE.

**No test here asserts a metadata value.** That is the closed form of
"the corpus no longer lies": one that states nothing about what is
inside a file cannot later be found to have stated it wrongly, which is
exactly what the previous corpus did with EXIF it had invented. The only
standing claims are a sha256, which rots loudly, and a sentence about
why a file was chosen, which no amount of format knowledge falsifies.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests import reach, sourced

pytestmark = pytest.mark.slow

HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE / "reach_baseline.json"
needs_mirror = pytest.mark.skipif(not sourced.available(), reason="the pinned exiftool mirror is not present")


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


# --- the pin ------------------------------------------------------------------


@needs_mirror
def test_the_sourced_files_are_the_bytes_that_were_locked():
    """The whole "fetch, don't vendor" contract, in one assertion.

    The bytes live in somebody else's GPL-3 repository and carry real
    coordinates, so they never enter ours. What we hold instead is a
    commit and a checksum -- and a checksum nobody verifies is a comment.
    """
    locked = json.loads(sourced.LOCKFILE.read_text(encoding="utf-8"))
    assert locked["source"]["rev"] == sourced.PINNED
    on_disk = {one.name: path for one, path in sourced.specimens()}
    assert on_disk, "the mirror is present but no specimen resolved"
    for held in locked["files"]:
        path = on_disk.get(held["name"])
        assert path is not None, f"{held['name']} is locked and missing"
        assert sourced.digest(path) == held["sha256"], f"{held['name']} is not the file that was locked"


@needs_mirror
def test_every_locked_file_says_why_it_is_here():
    """The intent label is the only claim this corpus makes about a file,
    and an unlabelled specimen is one nobody can decide to remove."""
    locked = json.loads(sourced.LOCKFILE.read_text(encoding="utf-8"))
    for held in locked["files"]:
        assert held["why"].strip(), f"{held['name']} has no stated reason to exist"


@needs_mirror
def test_the_lockfile_records_which_files_have_pixels_and_is_right():
    """16 of these 35 are truncated to their metadata on purpose. A test
    asserting a thumbnail on one would be asserting against somebody
    else's deliberate truncation.

    `renders` is MEASURED when the lock is written, never declared: when
    it was a field somebody typed it was wrong for ten of thirty-five --
    the same defect as inventing EXIF, one layer up. This checks the
    recorded measurement still holds against the files.
    """
    locked = json.loads(sourced.LOCKFILE.read_text(encoding="utf-8"))
    on_disk = {one.name: path for one, path in sourced.specimens()}
    wrong = [
        f"{held['name']}: locked renders={held['renders']}, now {sourced.decodes(on_disk[held['name']])}"
        for held in locked["files"]
        if held["name"] in on_disk and sourced.decodes(on_disk[held["name"]]) != held["renders"]
    ]
    assert wrong == [], wrong
    assert any(not held["renders"] for held in locked["files"]), (
        "every specimen decodes; the metadata-only files that reach the reader's failure arms are missing"
    )


# --- enumeration from the application's own declarations ----------------------


def test_what_must_be_covered_is_read_out_of_the_application():
    """A literal list of kinds would drift the day somebody adds one. The
    app declares its own surface; this reads it rather than restating
    it."""
    from db import scan

    kinds = set(scan.KIND_BY_SUFFIX.values())
    assert kinds == {"image", "animated_image", "video", "audio", "document"}, kinds
    assert len(scan.KIND_BY_SUFFIX) > 90, "the suffix table shrank; the corpus's target moved"


@needs_mirror
def test_the_corpus_covers_a_kind_the_synthetic_one_never_wrote():
    """Audio is the plain case: the synthetic corpus writes none at all,
    because writing a real audio container is not something it does. The
    kinds are counted through the app's own suffix table."""
    from db import scan

    reached = {scan.KIND_BY_SUFFIX.get(path.suffix.lower()) for _one, path in sourced.specimens()}
    reached.discard(None)
    assert "audio" in reached, "no audio specimen; the audio arm of the readers stays dark"
    assert len(reached) >= 4, reached


# --- the measurement ----------------------------------------------------------


@needs_mirror
def test_the_sourced_files_reach_lines_the_synthetic_corpus_cannot(frozen):
    """The obligation, stated as a number.

    The synthetic corpus closes 0 of its own frozen list, by
    construction. Anything the real files close is a line one writer
    could not produce and several writers can -- which is the whole
    argument for sourcing rather than generating more.
    """
    target = {name: set(lines) for name, lines in frozen["unreached"].items()}
    held = reach.over([path for _one, path in sourced.specimens()])
    closed = sum(len(target.get(name, set()) & got) for name, got in held.lines.items())
    assert closed > 100, f"the sourced files closed only {closed} of the frozen target"


@needs_mirror
def test_the_baseline_still_describes_these_readers(frozen):
    """The second coverage run the completion condition asks for.

    If the readers changed size the frozen list is about a different
    program, and every number derived from it is stale -- so that is
    checked before anything is concluded from it.
    """
    held = reach.over([path for _one, path in sourced.specimens()])
    tally = held.tally()
    assert tally, "nothing was measured"
    assert sum(e for _, e in tally.values()) == frozen["totals"]["executable"], (
        "the readers changed size; re-freeze tests/reach_baseline.json before trusting it"
    )


def test_the_corpus_modules_assert_no_metadata_values():
    """The closed form of "it no longer lies".

    Checked against the OTHER modules, never this one: the strings being
    searched for necessarily appear in the searcher, and a test that
    fails itself proves nothing about anybody else.
    """
    # Split so the literals are not themselves present as the phrases
    # they forbid -- the same reason this file is not its own subject.
    banned = ("Canon " + "EOS", "NIKON " + "D750", "iPhone " + "13", "FUJI" + "FILM X-T4", "dream" + "shaper")
    guilty = [
        f"{one.name} states a metadata value: {held!r}"
        for one in (HERE / "sourced.py", HERE / "reach.py")
        for held in banned
        if held in one.read_text(encoding="utf-8")
    ]
    assert guilty == [], guilty
