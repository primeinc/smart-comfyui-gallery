"""The corpus is held to the shape, and the shape is read out of the app.

`docs/CORPUS_SHAPE.md` names seven axes. `tests/needs.py` measures them and
writes `tests/needs.lock.json`. Neither of those can FAIL, and a measurement
that cannot fail is a report -- it can be satisfied by writing a larger
number into a file nobody reads.

This is the part that fails.

TWO RULES, and they are why this file cannot become theatre.

**The denominator comes from the application.** Every value tested here is
read from `db/scan.py KIND_BY_SUFFIX`, `metaparse/adapters.py`, or
`db/context.py`, at run time. Nothing is typed into this file, so a suffix
or a generator added to the application becomes a failure here without
anybody remembering to update a list -- and no value can be quietly dropped
from the denominator to make the number look better.

**There is no escape hatch.** Every declared value must be reached by a real
file, measured through the real readers. There was a table of allowed
exceptions here and it filled up with two different things wearing one word:
files nobody had written yet, and values the application declares and cannot
produce. Neither is an exception -- the first is work and the second is a
false claim -- so the table is gone and the only ways to make this test pass
are to write the file or to stop declaring the value.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from tests import needs

pytestmark = pytest.mark.slow

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", pathlib.Path(__file__).resolve().parent.parent.parent / "sg-corpus"))

#: THERE IS NO ESCAPE HATCH, deliberately.
#:
#: This was a table of values allowed to go unreached, each with a reason.
#: It filled up. Half its rows were things nobody had got to yet wearing the
#: word "blocked", and the other half were values the APPLICATION DECLARES
#: AND CANNOT PRODUCE -- `time_precision:hour`, `location_basis:sidecar`,
#: `location_basis:inferred`, `time_basis:first_seen` -- which is not a
#: corpus gap at all. No file closes those, so no amount of acquisition
#: makes the test pass, and leaving them excused made the vocabulary look
#: findable instead of false.
#:
#: Both halves have exactly one honest fix and it is never a table here:
#:
#:   the corpus lacks a file   -> write or fetch the file
#:   the app declares a lie    -> delete the declaration
#:
#: `docs/BACKLOG.md` reached the same conclusion about the suffix count
#: before this file existed: "an overclaim no corpus fixes -- only editing
#: the claim fixes the claim."


@pytest.fixture(scope="module")
def measured() -> dict:
    """The ledger, re-measured. Never the file on disk.

    Reading `needs.lock.json` would test whatever the last run happened to
    write, which is the failure this file exists to prevent: a stale number
    that agrees with itself.
    """
    if not CORPUS.is_dir():
        pytest.skip(f"no corpus at {CORPUS}; `just corpus all` writes one")
    return needs.measure(CORPUS)


#: Every state that is not "the application read a real file and produced
#: the thing". PARTIAL is in here deliberately.
#:
#: PARTIAL means the file is present and the reader got less than all of
#: it, and for a while it passed this gate -- which made it the same
#: escape hatch as the allowed-exceptions table, wearing a word that
#: sounded like progress. It hid a real finding: `.ari` sat PARTIAL
#: because a genuine CC0 Arri frame was in the corpus and LibRaw refused
#: it, and the answer was that LibRaw has no ARRI decoder and the
#: application should never have claimed the suffix.
#:
#: A reader that gets less than all of a file it claims is a defect. It
#: fails here.
NOT_REACHED = ("UNSATISFIED", "UNKNOWN_NOT_MEASURED", "PARTIAL")


def _unreached(held: dict, prefix: str) -> list[str]:
    return sorted(
        one["need"] for one in held["needs"] if one["need"].startswith(prefix) and one["state"] in NOT_REACHED
    )


def test_every_generator_the_app_reads_has_a_file(measured):
    """Eight adapters are registered; eight must be reachable.

    `Draw Things` was registered and had no fixture in the whole tree --
    found by asking the adapter registry instead of a list somebody
    maintained.
    """
    missing = _unreached(measured, "dialect:")
    assert not missing, f"registered generators no corpus file parses as: {missing}"


def test_every_kind_the_app_serves_has_a_file(measured):
    missing = _unreached(measured, "kind:")
    assert not missing, f"media kinds the corpus never produced: {missing}"


def test_every_suffix_the_app_claims_has_a_file(measured):
    """The biggest axis, and the one this file did not guard.

    `db/scan.py KIND_BY_SUFFIX` claims a hundred-odd suffixes and every
    other axis here was tested while this one was not, so a claimed
    suffix with no corpus file failed nothing. That is how `.cin`, `.ari`,
    `.cap` and `.k25` sat unreached without the gate saying a word.

    Two of the four were the application over-claiming -- LibRaw decodes
    neither Phantom Cine nor ARRI, and both are gone from
    `vision/decode.py RAW_SUFFIXES`. The other two are real formats
    LibRaw reads, with no sample in the corpus, and this is meant to fail
    until somebody finds one.
    """
    missing = _unreached(measured, "suffix:")
    assert not missing, (
        f"suffixes the application claims and no corpus file reads: {missing}."
        " Either the corpus needs the file, or the application should not claim it"
    )


def test_every_dating_rung_is_reached_or_named(measured):
    """The rungs a real library actually lands on.

    `mtime` and `btime` are the fallback for everything without EXIF, which
    is most of what people own, and a corpus of camera output never reaches
    them. This is the axis the corpus was missing entirely until it was
    measured.
    """
    missing = _unreached(measured, "time_basis:")
    assert not missing, f"dating rungs nothing in the library landed on: {missing}"


def test_every_precision_and_origin_is_reached_or_named(measured):
    missing = [
        one for prefix in ("time_precision:", "location_basis:", "origin:") for one in _unreached(measured, prefix)
    ]
    assert not missing, f"interpretation values nothing reached: {missing}"


def test_the_shape_is_read_from_the_application_not_from_a_list():
    """The denominator cannot be edited to make the numerator look better.

    If these ever stop matching the application's own declarations, every
    number this file produces is about a different program.
    """
    from db import context, scan
    from metaparse import adapters

    axes = needs.declared_axes()
    assert axes["kind"] == tuple(sorted(set(scan.KIND_BY_SUFFIX.values())))
    assert axes["time_basis"] == tuple(context.TIME_BASES)
    assert axes["location_basis"] == tuple(context.LOCATION_BASES)
    assert axes["origin"] == tuple(context.ORIGINS)
    assert axes["time_precision"], "time_precision was not read out of db/schema.sql"

    registered = {
        getattr(one, "tool", one.__name__) for one in (*adapters.MARKER_ADAPTERS, *adapters.HEURISTIC_ADAPTERS)
    }
    assert set(needs.declared_dialects()) == registered
    assert len(needs.declared_suffixes()) == len(scan.KIND_BY_SUFFIX)


def test_the_ledger_on_disk_is_not_stale(measured):
    """`needs.lock.json` is what a person reads. A lockfile describing a
    corpus that has since grown is a number that agrees with itself and
    with nothing else."""
    if not needs.LEDGER.is_file():
        pytest.skip("no ledger written yet; `just corpus needs` writes one")
    frozen = json.loads(needs.LEDGER.read_text(encoding="utf-8"))
    assert frozen["totals"] == measured["totals"], (
        "tests/needs.lock.json disagrees with a fresh measurement; re-run `just corpus needs`"
    )
