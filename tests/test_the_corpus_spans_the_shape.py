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

**A blocked need is a state, not a failure.** A test that is red in the
repository's intended state gates nothing: it cannot sit in a hook, and it
trains everybody to pass the bypass flag. So a value no obtainable file can
reach is BLOCKED_EXTERNALLY, declared in `tests/needs.py BLOCKED` with the
searches that failed and their positive controls -- and the register is held
from both sides by `test_an_excuse_cannot_outlive_its_gap`: a row without
evidence fails, and a row whose gap a corpus file has since closed fails.
An earlier `ALLOWED` table here had neither property and filled up with
deferrals wearing the word "blocked"; the reverse assertion and the evidence
requirement are what make this one a measurement instead of an excuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest

from tests import needs
from tests.staging import corpus_measurement

pytestmark = pytest.mark.slow

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", pathlib.Path(__file__).resolve().parent.parent.parent / "sg-corpus"))

#: The three honest fixes for an unreached value, in the order to try them:
#:
#:   the corpus lacks a file        -> write or fetch the file
#:   the app declares a lie         -> delete the declaration
#:     (`time_precision:hour` and three friends went this way in v44)
#:   the world offers no specimen   -> a `tests/needs.py BLOCKED` row,
#:     with evidence, held by test_an_excuse_cannot_outlive_its_gap
#:
#: An `ALLOWED` table here once tried to be the third fix without the
#: evidence or the reverse assertion, and it filled up with the first two
#: wearing the word "blocked". The register survives only because both
#: properties are enforced below.


@pytest.fixture(scope="module")
def measured() -> dict:
    """The ledger, re-measured. Never the file on disk.

    Reading `needs.lock.json` would test whatever the last run happened to
    write, which is the failure this file exists to prevent: a stale number
    that agrees with itself.

    Re-measuring is ~140 s -- every reader over every corpus file -- and
    the answer is a constant of (corpus, readers, measurer). So it is
    cached on exactly those three: `corpus_measurement` keys on the
    corpus's own listing and on the bytes of every module under db/,
    metaparse/ and vision/, and the name carries this measurer's digest.
    Move any of them and the number is measured again. That is the
    property `needs.lock.json` lacks, not the caching.
    """
    if not CORPUS.is_dir():
        pytest.skip(f"no corpus at {CORPUS}; `just corpus all` writes one")
    measurer = hashlib.sha256(pathlib.Path(needs.__file__).read_bytes()).hexdigest()[:12]
    return json.loads(corpus_measurement(CORPUS, f"corpus-needs-{measurer}", lambda: json.dumps(needs.measure(CORPUS))))


#: Every state that is not "the application read a real file and produced
#: the thing". PARTIAL means the file is present and the reader got less than
#: all of it, which is a defect in a reader claiming the suffix.
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
    LibRaw reads with no obtainable sample: BLOCKED_EXTERNALLY in
    `needs.BLOCKED`, evidence attached, retried by `tests/rawsamples.py`
    on every fetch.
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


def test_an_excuse_cannot_outlive_its_gap(measured):
    """`needs.BLOCKED` rows die the day their gap closes.

    A register of excuses is exactly the escape hatch this file once
    deleted -- unless it is held from both sides. Held here: a row must
    name a need the application still declares, must carry evidence with
    a source, a positive control and a date, and must still be blocked.
    A blocked suffix that a corpus file now reads is a stale excuse and
    fails the same as a gap; `tests/rawsamples.py` retries blocked
    suffixes on every fetch, so the challenge is standing, not annual.
    """
    by_need = {one["need"]: one for one in measured["needs"]}
    for need, held in needs.BLOCKED.items():
        assert need in by_need, f"{need} is excused and no longer declared; delete the BLOCKED row"
        assert held["evidence"], f"{need} is excused without evidence"
        for row in held["evidence"]:
            for field in ("source", "control", "checked"):
                assert row.get(field), f"{need}: evidence needs a {field}; got {row}"
        state = by_need[need]["state"]
        assert state == "BLOCKED_EXTERNALLY", (
            f"{need} is {state} but sits in needs.BLOCKED: the excuse outlived its gap; delete the row"
        )


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
