from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"


@dataclass
class Condition:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _read(name: str, where: Path = GENERATED) -> dict[str, Any] | None:
    path = where / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp(held: dict[str, Any]) -> str:
    found = held.get("identity")
    if isinstance(found, dict):
        return str(found.get("digest", ""))
    return str(found or "")


def _every_lane_green(where: Path) -> Condition:
    # A lane's exit code was recorded and read by nothing: `attack` and `selftest`
    # could both exit 1 while this printed GREEN. The lane record is now evidence.
    held = _read("lanes.json", where)
    if held is None:
        return Condition("every lane exited 0", False, "no lanes.json: the run recorded no lane exit")

    found = held.get("lanes")
    if not isinstance(found, dict) or not found:
        return Condition("every lane exited 0", False, "lanes.json records no lane; an empty set is not a pass")

    red = sorted(name for name, code in found.items() if int(code) != 0)
    return Condition(
        "every lane exited 0",
        not red,
        f"{len(found)} lane(s), all 0" if not red else f"{len(red)} of {len(found)} red: {', '.join(red)}",
    )


def _one_tree(
    ledger: dict[str, Any],
    cases: dict[str, Any] | None,
    pins: dict[str, Any] | None,
    lanes: dict[str, Any] | None,
    where: Path,
) -> Condition:
    now = str(evidence_identity.identity()["digest"])
    stamped = {"ledger.json": _stamp(ledger)}
    for name, held in (("cases.json", cases), ("provenance.json", pins), ("lanes.json", lanes)):
        if held is not None:
            stamped[name] = _stamp(held)
    wrong = {name: held for name, held in stamped.items() if held != now}

    if where != GENERATED:
        return Condition("evidence from one tree", True, f"fixture directory {where.name}, identity not compared")
    return Condition(
        "evidence from one tree",
        not wrong,
        f"all artifacts stamped {now[:12]}"
        if not wrong
        else "; ".join(
            f"{name} was built under {held[:12] or 'no digest'}, tree is {now[:12]}" for name, held in wrong.items()
        ),
    )


def conditions(where: Path = GENERATED) -> list[Condition]:
    out: list[Condition] = []
    ledger = _read("ledger.json", where)
    pins = _read("provenance.json", where)
    cases = _read("cases.json", where)
    lanes = _read("lanes.json", where)

    if ledger is None:
        return [Condition("ledger present", False, "no ledger.json: the ledger lane did not run")]

    out.append(_one_tree(ledger, cases, pins, lanes, where))
    out.append(_every_lane_green(where))

    rows: list[dict[str, Any]] = ledger["rows"]
    out.append(
        Condition(
            "every declared member accounted for",
            bool(rows) and len(rows) == ledger["totals"]["declared"],
            f"{len(rows)} row(s) for {ledger['totals']['declared']} declared",
        )
    )

    weights: list[dict[str, Any]] = (pins or {}).get("weights", [])
    for state in ("MISSING", "MISMATCH", "UNATTESTED"):
        bad = [one for one in weights if one.get("state") == state]
        out.append(
            Condition(
                f"no weight {state}",
                not bad and pins is not None,
                "pins wrote nothing"
                if pins is None
                else (f"{len(bad)}: " + ", ".join(f"{o['pack']}/{o['file']}" for o in bad[:4]) if bad else "none"),
            )
        )

    blocked = [
        (row["consumer"], stage)
        for row in rows
        for stage in ledger["stages"]
        if row["cells"][stage]["state"] == "BLOCKED"
    ]
    out.append(
        Condition(
            "no ledger cell BLOCKED",
            not blocked,
            "none" if not blocked else f"{len(blocked)} cell(s), e.g. {blocked[0][0]}/{blocked[0][1]}",
        )
    )

    failed = [
        (row["consumer"], stage)
        for row in rows
        for stage in ledger["stages"]
        if row["cells"][stage]["state"] == "FAILED"
    ]
    out.append(
        Condition(
            "no ledger cell FAILED",
            not failed,
            "none" if not failed else f"{len(failed)} cell(s), e.g. {failed[0][0]}/{failed[0][1]}",
        )
    )

    skipped = (cases or {}).get("skipped", [])
    out.append(
        Condition(
            "no skipped input",
            cases is not None and not skipped,
            "cases wrote nothing" if cases is None else (f"{len(skipped)} input(s) skipped" if skipped else "none"),
        )
    )

    shards = (cases or {}).get("shards_failed", [])
    out.append(
        Condition(
            "no shard failed",
            cases is not None and not shards,
            "cases wrote nothing" if cases is None else (f"{len(shards)} shard(s)" if shards else "none"),
        )
    )
    return out


def main() -> int:
    held = conditions()
    print("closure conditions\n")
    for one in held:
        print(f"{one.mark} {one.name:<44} {one.detail}")

    closed = all(one.held for one in held)
    print(f"\nCLOSURE: {'GREEN' if closed else 'RED'}")
    if not closed:
        print("the missing-work ledger is compat/generated/ledger.md")

    with (GENERATED / "closure.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                {"closed": closed, "conditions": [{"name": o.name, "held": o.held, "detail": o.detail} for o in held]},
                indent=2,
                sort_keys=True,
            )
        )
        handle.write("\n")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
