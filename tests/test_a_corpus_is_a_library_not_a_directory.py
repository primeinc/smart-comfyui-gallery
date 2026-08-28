"""The sample library says what it claims to say.

A corpus is a directory until something reads it, and a corpus that
lies is worse than none: every fixture built on it inherits the lie and
every number measured against it is wrong in the same direction.

This is the corpus held to its own description. Each test is one
sentence from `tests/corpus.py`'s docstring, made checkable.

The tests are cheap on purpose -- `small` scale, one scan, no models.
The expensive proof is `just corpus prove`, which writes, serves and
reports so a person can LOOK at the numbers rather than assert them.

One of these exists because it caught something. `DateTimeOriginal` is
LOCAL wall clock with no zone, and writing it in UTC made every
photograph disagree with its own mtime by the machine's offset: 146 of
146 files arrived contested, which would have buried the eight that are
contested on purpose and made the whole corpus look like a library in a
date crisis.
"""

from __future__ import annotations

import itertools

import pytest

from tests import corpus

pytestmark = pytest.mark.slow


def _as_wall(instant: float) -> float:
    """An instant as the LOCAL wall clock it shows, numbered as if UTC.

    The numbering `db/capture.py` uses for `captured_at`, and the only
    one in which an EXIF time and a filesystem time are comparable at
    all -- see the docstring of the test that needed it.
    """
    import datetime

    local = datetime.datetime.fromtimestamp(instant, datetime.UTC).astimezone()
    return local.replace(tzinfo=datetime.UTC).timestamp()


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus") / "library"
    told = corpus.spread(root, scale="small")
    return root, told


def test_it_spans_decades_rather_than_an_afternoon(written):
    """The whole reason it exists. Every `write_library` in this suite
    covers a fortnight at most, so nothing ever exercised a surface
    against the shape of a life."""
    root, _told = written
    stamps = [one.stat().st_mtime for one in root.rglob("*") if one.is_file()]
    assert stamps
    import datetime

    years = {datetime.datetime.fromtimestamp(one, datetime.UTC).year for one in stamps}
    assert max(years) - min(years) >= 20, sorted(years)


def test_it_leaves_holes_big_enough_to_collapse(written):
    """The gaps are the point: a surface that draws elapsed time spends
    itself on them, and a corpus with none proves nothing about it."""
    root, _told = written
    stamps = sorted(one.stat().st_mtime for one in root.rglob("*") if one.is_file())
    gaps = [b - a for a, b in itertools.pairwise(stamps)]
    year = 365 * 86_400
    assert max(gaps) > 3 * year, f"the widest hole is {max(gaps) / year:.1f} years"
    assert sum(1 for one in gaps if one > 30 * 86_400) >= 3, "too few holes to collapse at more than one zoom"


def test_it_speaks_more_than_one_generator_dialect(written):
    """A corpus of one dialect proves one adapter. A1111 infotext, a
    ComfyUI graph and a SwarmUI manifest are three different readers."""
    root, _told = written
    from metaparse import containers

    dialects = set()
    for one in sorted(root.rglob("generated/**/*.png")):
        raw = containers.load_raw(str(one))
        keys = set(getattr(raw, "text", {}) or {}) if raw else set()
        if "parameters" in keys:
            dialects.add("swarm" if one.name.startswith("swarm") else "a1111")
        if "prompt" in keys:
            dialects.add("comfy")
    assert dialects == {"a1111", "comfy", "swarm"}, dialects


def test_it_carries_capture_evidence_as_well_as_recipes(written):
    """Generated and captured together, which is the thesis of this
    application and the one mix a single-purpose fixture never has."""
    root, _told = written
    from db import capture

    cameras = set()
    for one in sorted(root.rglob("photos/**/*.png")):
        found = capture.read(one)
        if found and found.camera:
            cameras.add(found.camera)
    assert len(cameras) >= 3, cameras


def test_only_the_muddled_files_disagree_with_themselves(written):
    """The test that exists because it caught something.

    EXIF `DateTimeOriginal` is local wall clock with NO zone. Written in
    UTC on a machine that is not UTC, every photograph disagrees with
    its own mtime by the offset -- and arrives contested. Measured at
    146 of 146 before the fix, which would have buried the eight that
    are contested on purpose.

    Compared in ONE numbering, which is the trap this test fell into
    first: `captured_at` is the wall clock NUMBERED AS IF UTC (db/capture
    docstring), and the mtime is a true instant. Subtracting one from
    the other measures the machine's offset and calls it a conflict.
    """
    root, _told = written
    from db import capture

    # `photos/` only: the muddled era lives under `downloads/` and is
    # SUPPOSED to disagree, which is the next test.
    disagreed = []
    for one in sorted(root.rglob("photos/**/*.png")):
        found = capture.read(one)
        if found is None or found.captured_at is None:
            continue
        if abs(found.captured_at - _as_wall(one.stat().st_mtime)) > 120:
            disagreed.append(one.name)
    assert disagreed == [], f"{len(disagreed)} photographs disagree with their own mtime: {disagreed[:5]}"


def test_the_muddled_ones_really_do_disagree(written):
    """The other half: a corpus that conflicts with nothing cannot
    exercise the surface that counts conflicts."""
    root, _told = written
    from db import capture

    muddled = sorted(root.rglob("downloads/**/IMG_*"))
    if not muddled:  # `small` may not reach the muddled era
        pytest.skip("this scale wrote no muddled files")
    found = capture.read(muddled[0])
    assert found is not None
    assert found.captured_at is not None
    assert abs(found.captured_at - _as_wall(muddled[0].stat().st_mtime)) > 300 * 86_400


def _digest(root):
    import hashlib

    held = hashlib.sha256()
    for one in sorted(root.rglob("*")):
        if one.is_file():
            held.update(one.relative_to(root).as_posix().encode())
            held.update(one.read_bytes())
    return held.hexdigest()


def test_the_same_seed_writes_the_same_bytes(written, tmp_path):
    """Deterministic, or a fixture built from it cannot be compared with
    itself and every measurement drifts for reasons nobody can name.
    Compared against THE FIXTURE -- the claim's own subject -- so the
    proof costs one write, not two."""
    root, _told = written
    again = tmp_path / "again"
    corpus.spread(again, scale="small")
    assert _digest(root) == _digest(again)


def test_a_different_seed_writes_a_different_library(written, tmp_path):
    """And it is a seed rather than a decoration."""
    root, _told = written
    other = tmp_path / "other"
    corpus.spread(other, scale="small", seed=8)
    assert _digest(root) != _digest(other)


def test_it_holds_both_kinds_of_duplicate(written):
    """An exact copy can become one stored payload and a re-encode
    cannot, which is the distinction `/dupes` is built on. A corpus with
    only one of them proves only half of it."""
    import hashlib

    root, told = written
    assert told["duplicates"] > 0
    backups = sorted(root.rglob("backup/**/*"))
    exact = [one for one in backups if one.suffix == ".png"]
    alike = [one for one in backups if one.suffix == ".jpg"]
    assert exact, "no byte-identical copy: the consolidating case is untested"
    assert alike, "no re-encode: the case that CANNOT be consolidated is untested"

    twin = exact[0]
    source = next(one for one in root.rglob(f"photos/**/{twin.name}"))
    assert hashlib.sha256(twin.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    assert hashlib.sha256(alike[0].read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest()


def test_every_kind_the_scanner_knows_is_represented(written):
    """A library is not only stills, and the surfaces that break on a
    video break on a corpus that has none."""
    _root, told = written
    assert set(told["kinds"]) >= {"image", "video"}, told["kinds"]


def test_it_refuses_a_scale_it_does_not_have(tmp_path):
    with pytest.raises(ValueError, match="scale is one of"):
        corpus.spread(tmp_path / "x", scale="enormous")
