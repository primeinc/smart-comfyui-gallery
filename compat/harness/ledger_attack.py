from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness import lanes, ledger, provenance
from compat.harness.ledger import STAGE_EVIDENCE, VERIFIED

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
            {"primitive": "whole_reference_image", "expect_breaks": True, "observed_break": True, "verdict": "PASS"}
        ],
    }


def green_fixture() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    declared = sorted(one["id"] for one in manifest.get("consumers", []))
    now = str(evidence_identity.identity()["digest"])
    return {
        "provenance.json": {
            "identity": now,
            "repos": [
                {"key": f"consumer:{who}", "paths": [{"path": "node.py", "present": True}], "failures": []}
                for who in declared
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
            target[parts[-1]] = {} if parts[-1] == "retained_bytes" else None


def _write(where: Path, held: dict[str, Any]) -> None:
    for name, body in held.items():
        with (where / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
            handle.write("\n")


def _states(where: Path, held: dict[str, Any]) -> dict[str, dict[str, str]]:
    _write(where, held)
    out = ledger.build(where)
    row = out["rows"][0]
    return dict(row["cells"])


@dataclass
class Result:
    name: str
    only_its_own_stage_red: bool
    detail: str


def run_all() -> tuple[list[Result], str, bool]:
    out: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="ledger_attack_") as raw:
        where = Path(raw)
        base = green_fixture()
        control = _states(where, base)

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
            after = _states(where, held)
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
    results, why, distinct = run_all()
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

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {
        "identity": str(evidence_identity.identity()["digest"]),
        "stages": sorted(STAGE_EVIDENCE),
        "results": [asdict(one) for one in results],
        "cells_distinct": distinct,
        "failing": missed,
    }
    with (GENERATED / "ledger_controls.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {GENERATED / 'ledger_controls.json'}")
    return 0 if not missed and distinct else 1


if __name__ == "__main__":
    raise SystemExit(main())
