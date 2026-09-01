from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness import lanes, ledger, provenance
from compat.harness.ledger import FAILED, STAGE_EVIDENCE, VERIFIED

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"


def _case(who: str) -> dict[str, Any]:
    # Carries every field the six stages read, including the `durable` record no
    # runner writes yet: the point is that the ledger verifies them SEPARATELY.
    return {
        "consumer_id": who,
        "case": f"{who}_boundary",
        "verdict": "PASS",
        "baseline": {"sha256": "a" * 64, "dtype": "uint8", "shape": [1]},
        "replay": {"sha256": "a" * 64, "dtype": "uint8", "shape": [1]},
        "retained_bytes": {"whole_reference_image": 1024},
        "durable": {"written_bytes": 1024, "read_back_sha256": "b" * 64, "thawed_keys": ["whole_reference_image"]},
        "ablations": [
            {"primitive": "whole_reference_image", "expect_breaks": True, "observed_break": True, "verdict": "PASS"},
            # One tolerated inconclusive carrying a DECLARED cause: without it
            # the cause condition judges an empty population and returns
            # not-applicable, which is not green.
            {
                "primitive": "kps_source_px",
                "expect_breaks": True,
                "observed_break": None,
                "verdict": "INCONCLUSIVE",
                "cause": "retained_state_lacks_primitive",
            },
        ],
    }


def green_fixture(*, digest: str) -> dict[str, Any]:
    # Stamped from a digest the CALLER captured, not from a fresh read. Required
    # and keyword-only for the same reason build()'s is: a default would let the
    # fixture and the build drift apart again without anyone writing a line.
    manifest = provenance.load_manifest()
    # The build's OWN row set, not a second spelling of it. A fixture declaring
    # the vendored 22 while the build made 28 rows reported "the control is not
    # green" about six rows it had never fed.
    declared, first_party = ledger.declared_consumers(manifest)
    now = digest
    return {
        "provenance.json": {
            "identity": now,
            # PINS FOR THE VENDORED ONLY. A first-party consumer carrying a pin
            # is the contradictory state, so pinning all 28 here would make the
            # green control assert the thing the grading calls an offence.
            "repos": [
                {"key": f"consumer:{who}", "paths": [{"path": "node.py", "present": True}], "failures": []}
                for who in declared
                if who not in first_party
            ],
            "weights": [{"pack": "antelopev2", "file": "glintr100.onnx", "state": "VERIFIED"}],
        },
        "cases.json": {"identity": {"digest": now}, "results": [_case(who) for who in declared]},
        # Every DECLARED lane, from compat.just: the condition's population is the
        # declaration, so a fixture naming only a few would be short by the rest.
        "lanes.json": {"identity": now, "lanes": dict.fromkeys(lanes.declared(), 0)},
    }


def _suppress(held: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    for row in held["cases.json"]["results"]:
        target: Any = row
        for part in parts[:-1]:
            target = target.get(part) if isinstance(target, dict) else None
        if isinstance(target, dict):
            # DELETED, not nulled. Presence-based detection is the point: a writer
            # that did not record the field omits it, and an empty dict left here
            # read as a real emission of nothing once `emits` stopped using truth.
            target.pop(parts[-1], None)


def _write(where: Path, held: dict[str, Any]) -> None:
    for name, body in held.items():
        with (where / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
            handle.write("\n")


def _states(where: Path, held: dict[str, Any], digest: str) -> dict[str, dict[str, str]]:
    _write(where, held)
    out = ledger.build(where, digest=digest)
    row = out["rows"][0]
    return dict(row["cells"])


@dataclass
class Result:
    name: str
    only_its_own_stage_red: bool
    detail: str


@dataclass
class SourceState:
    name: str
    consumer: str
    state: str
    reason: str
    as_expected: bool


def _source_of(where: Path, held: dict[str, Any], digest: str, who: str) -> dict[str, str]:
    _write(where, held)
    for row in ledger.build(where, digest=digest)["rows"]:
        if row["consumer"] == who:
            return dict(row["cells"]["source_provenance"])
    raise KeyError(f"the build made no row for {who}, so its source cell cannot be observed")


def overlap_is_refused() -> tuple[bool, str]:
    """A consumer declared BOTH vendored and first-party is refused by name.

    The four-state grading catches the contradiction only through a PRESENT
    pin row, and the source stage reads provenance.json with no freshness
    check -- so against stale pins a double-declared id grades VERIFIED and
    the manifest's "THIS IS A DECLARATION THAT CAN BE WRONG, AND IT IS
    CHECKED" stops being true exactly when it is load-bearing. Refusing at
    the reader makes it hold whatever provenance.json happens to contain.
    """
    manifest = provenance.load_manifest()
    _declared, first_party = ledger.declared_consumers(manifest)
    if not first_party:
        return False, "the manifest declares no first-party consumer, so the overlap is unreachable"
    doubled = min(first_party)
    spoiled = {**manifest, "consumers": [*manifest.get("consumers", []), {"id": doubled}]}
    try:
        ledger.declared_consumers(spoiled)
    except KeyError as why:
        return True, f"refused by name: {str(why)[:96]}"
    return False, f"{doubled} declared in BOTH tables was accepted; the manifest's own comment is not true"


def source_states(digest: str) -> tuple[list[SourceState], str, bool]:
    """G10's four states over the pair (declared first-party, pinned).

    Driven THROUGH build() on mutated fixtures, never by calling the
    predicate directly: the wiring from the manifest to the cell is
    exactly what a hand-called predicate leaves untested, and a
    classification that never reaches a row grades nothing.
    """
    manifest = provenance.load_manifest()
    declared, first_party = ledger.declared_consumers(manifest)
    vendored = [one for one in declared if one not in first_party]
    if not first_party or not vendored:
        return [], "the manifest declares no first-party consumer, or no vendored one, so a state is unreachable", False

    mine, theirs = min(first_party), vendored[0]
    out: list[SourceState] = []
    with tempfile.TemporaryDirectory(prefix="ledger_source_") as raw:
        where = Path(raw)
        base = green_fixture(digest=digest)

        def observed(name: str, held: dict[str, Any], who: str, ruled: str) -> None:
            cell = _source_of(where, held, digest, who)
            out.append(SourceState(name, who, cell["state"], cell["reason"], cell["state"] == ruled))

        # The two states the green fixture already stands in.
        observed("declared first-party, unpinned", base, mine, VERIFIED)
        observed("pinned, not declared first-party", base, theirs, VERIFIED)

        # Contradictory: the manifest calls it ours and a pin says it is fetched.
        held = copy.deepcopy(base)
        held["provenance.json"]["repos"].append(
            {"key": f"consumer:{mine}", "paths": [{"path": "node.py", "present": True}], "failures": []}
        )
        observed("declared first-party AND pinned", held, mine, FAILED)

        # Unclassified: the state a NEW consumer lands in, so it fails until
        # somebody classifies it rather than defaulting into the excused bucket.
        held = copy.deepcopy(base)
        held["provenance.json"]["repos"] = [
            row for row in held["provenance.json"]["repos"] if row["key"] != f"consumer:{theirs}"
        ]
        observed("neither declared nor pinned", held, theirs, FAILED)

    # Four distinct reasons, or two states are one cell wearing two names.
    return out, "", len({one.reason for one in out}) == len(out)


def run_all(digest: str) -> tuple[list[Result], str, bool]:
    out: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="ledger_attack_") as raw:
        where = Path(raw)
        base = green_fixture(digest=digest)
        control = _states(where, base, digest)

        green = [name for name, cell in control.items() if cell["state"] != VERIFIED]
        if green:
            return [], f"the control is not green: {green}", False

        # The tripwire, not the proof: six cells built from one boolean would share
        # a reason. Six distinct reasons means six distinct derivations were run.
        reasons = {stage: control[stage]["reason"] for stage in STAGE_EVIDENCE}
        distinct = len(set(reasons.values())) == len(reasons)

        for stage, evidence in STAGE_EVIDENCE.items():
            held = copy.deepcopy(base)
            _suppress(held, evidence.field)
            after = _states(where, held, digest)
            red = sorted(name for name, cell in after.items() if cell["state"] != VERIFIED)
            out.append(
                Result(
                    name=f"suppress {evidence.field}",
                    only_its_own_stage_red=red == [stage],
                    detail=f"-> {stage} {after[stage]['state']}" if red == [stage] else f"red: {red or 'NOTHING'}",
                )
            )
    return out, "", distinct


def main() -> int:
    # ONE capture, threaded into the fixture and every build below.
    digest = str(evidence_identity.identity()["digest"])
    results, why, distinct = run_all(digest)
    if why:
        print(why)
        return 1

    print("ledger per-stage evidence controls\n")
    print(f"{'suppressed field':<36} {'only its own stage red':<24} observed")
    for one in results:
        print(f"{one.name:<36} {one.only_its_own_stage_red!s:<24} {one.detail}")

    missed = [one.name for one in results if not one.only_its_own_stage_red]
    print(f"\n{len(results)} stage(s), {len(missed)} not independently derived: {missed or 'none'}")
    print(f"tripwire -- six distinct cell reasons: {distinct}")

    refused, overlap_told = overlap_is_refused()
    print(f"\nvendor n first-party overlap refused: {refused}  -- {overlap_told}")

    states, unreachable, states_distinct = source_states(digest)
    if unreachable:
        print(unreachable)
        return 1
    print("\nG10 source_provenance -- the four states over (declared first-party, pinned)\n")
    print(f"{'state':<34} {'consumer':<18} {'graded':<10} as ruled")
    for one in states:
        print(f"{one.name:<34} {one.consumer:<18} {one.state:<10} {one.as_expected}")
    misgraded = [one.name for one in states if not one.as_expected]
    print(f"\n{len(states)} state(s), {len(misgraded)} not graded as ruled: {misgraded or 'none'}")
    print(f"tripwire -- four distinct cell reasons: {states_distinct}")

    real = ledger.exercised_against_real_evidence(GENERATED)
    live = sorted(stage for stage, seen in real.items() if seen["carrying"])
    invented = sorted(stage for stage, seen in real.items() if not seen["carrying"])
    print(f"\nexercised against real evidence: {len(live)}/{len(real)}")
    for stage in sorted(real):
        seen = real[stage]
        mark = "FIXTURE ONLY" if not seen["carrying"] else "real        "
        print(f"  {mark}  {stage:<24} {seen['carrying']}/{seen['of']} case(s) carry {STAGE_EVIDENCE[stage].field}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {
        "identity": digest,
        "stages": sorted(STAGE_EVIDENCE),
        "results": [asdict(one) for one in results],
        "cells_distinct": distinct,
        "exercised_against_real_evidence": real,
        "fixture_only": invented,
        "failing": missed,
        "source_states": [asdict(one) for one in states],
        "source_states_distinct": states_distinct,
        "source_states_misgraded": misgraded,
    }
    with (GENERATED / "ledger_controls.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {GENERATED / 'ledger_controls.json'}")
    return 0 if not missed and distinct and not misgraded and states_distinct else 1


if __name__ == "__main__":
    raise SystemExit(main())
