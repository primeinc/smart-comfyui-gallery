"""The compat suite's published artifacts, checked against each other.

`compat.just` lanes are not run by any hook: `justfile` `gates` and
`prove-push` invoke neither, so every gate the suite builds is unreachable
from a commit. CONTRIBUTING.md: "A check no hook runs does not gate anything."

These read only committed JSON and import nothing from `compat`, which runs in
its own environment (.venv-compat). They cost milliseconds and run under
`just test`.

They cannot re-derive the evidence; they check that the artifacts generated
from it do not contradict one another. `answer.json` and
`compatibility-matrix.json` classify the same primitives from the same
`cases.json` through two code paths, and have disagreed: 19 primitives were
once simultaneously MUST RETAIN and UNPROVEN, detectable only by diffing the
two files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

GENERATED = Path(__file__).resolve().parent.parent / "compat" / "generated"


def _load(name: str) -> dict[str, Any]:
    where = GENERATED / name
    if not where.is_file():
        pytest.skip(f"{where} has not been generated")
    return json.loads(where.read_text(encoding="utf-8"))


def test_no_primitive_is_both_necessary_and_unproven():
    """One primitive, one verdict, across both published views."""
    answer = _load("answer.json")
    matrix = _load("compatibility-matrix.json")

    keep = {one["name"] for one in answer["must_retain"]}
    unproven = {one["primitive"] for one in matrix["primitives"] if one["verdict"] == "UNPROVEN"}

    both = sorted(keep & unproven)
    assert not both, (
        f"{len(both)} primitive(s) are MUST RETAIN in answer.json and UNPROVEN in "
        f"compatibility-matrix.json, from the same cases.json: {both}"
    )


def test_every_retained_primitive_carries_a_measured_size():
    """A durable set with no size attached does not answer the question."""
    answer = _load("answer.json")
    free = sorted(one["name"] for one in answer["must_retain"] if not one["bytes_per_face"])
    assert not free, f"{len(free)} entry(s) in must_retain report 0 bytes: {free}"


def test_the_matrix_counts_match_its_own_rows():
    """The published totals are derived from the rows beside them."""
    matrix = _load("compatibility-matrix.json")
    rows = matrix["consumers"]
    totals = matrix["totals"]
    for status, key in (
        ("REPRODUCED", "reproduced"),
        ("NOT EXERCISED", "not_exercised"),
        ("DIVERGED", "diverged"),
        ("CONTRADICTED", "contradicted"),
    ):
        counted = sum(1 for one in rows if one["status"] == status)
        assert counted == totals[key], f"totals.{key} is {totals[key]}; {counted} row(s) read {status}"


def test_every_case_verdict_is_one_the_contract_defines():
    """A verdict outside the vocabulary is a writer and a reader disagreeing."""
    cases = _load("cases.json")
    # The serialised values, not the enum names: `Verdict.REPRODUCED` is
    # "PASS" and `Verdict.DIVERGED` is "FAIL" (compat/contracts/case.py).
    known = {
        "PASS",
        "FAIL",
        "CONTRADICTED",
        "UNSUPPORTED",
        "INCONCLUSIVE",
        "VENDOR_BASELINE_UNAVAILABLE",
    }
    seen = {one["verdict"] for one in cases["results"]}
    assert seen <= known, f"unknown verdict(s) in cases.json: {sorted(seen - known)}"


def test_no_consumer_answered_nothing():
    """A consumer whose every case raised is a boundary that vanished.

    The storage lane once returned 36 UNSUPPORTED, 0 PASS and exit 0.
    """
    cases = _load("cases.json")
    answered: dict[str, int] = {}
    total: dict[str, int] = {}
    for one in cases["results"]:
        who = one["consumer_id"]
        total[who] = total.get(who, 0) + 1
        answered[who] = answered.get(who, 0) + (one["verdict"] == "PASS")
    silent = sorted(who for who, count in total.items() if count and not answered.get(who))
    assert not silent, f"consumer(s) with cases and not one PASS: {silent}"


def test_both_views_name_the_same_unproven_primitives():
    """One set of open questions, counted the same way in both artifacts.

    `answer.json` once listed `(primitive, swap)` pairs beside bare primitives
    under `unproven`, so it reported seven where the matrix reported five.
    """
    answer = _load("answer.json")
    matrix = _load("compatibility-matrix.json")

    theirs = {one["primitive"] for one in matrix["primitives"] if one["verdict"] == "UNPROVEN"}
    ours = {one["name"] for one in answer["unproven"]}
    assert ours == theirs, f"answer.json unproven {sorted(ours)}; matrix UNPROVEN {sorted(theirs)}"

    pairs = [one["name"] for one in answer["unproven"] if " <- " in one["name"]]
    assert not pairs, f"substitution pairs listed as unproven primitives: {pairs}"
