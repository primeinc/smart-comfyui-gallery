from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import closure, closure_attack, ledger, ledger_attack
from compat.harness import identity as evidence_identity

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"

#: The lane block the adversary's demonstration used: everything green but the two
#: that audit the system itself. Before G1 this closed GREEN.
RED_LANES: Final[dict[str, int]] = {"attack": 1, "selftest": 1}


@dataclass
class Probe:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _verdict(where: Path, held: dict[str, Any]) -> tuple[bool, list[str]]:
    closure_attack._write(where, held)
    conditions = closure.conditions(where)
    return all(one.green for one in conditions), [f"{one.name} [{one.state}]" for one in conditions if not one.green]


def _lane_red_over_a_green_run() -> Probe:
    # The demonstration itself: everything else green, the attack and selftest
    # lanes red. This is the exact shape that printed GREEN before G1 landed.
    with tempfile.TemporaryDirectory(prefix="green_attack_") as raw:
        where = Path(raw)
        held = ledger_attack.green_fixture()
        held["lanes.json"]["lanes"] = {**held["lanes.json"]["lanes"], **RED_LANES}
        closed, red = _verdict(where, held)
    named = "every lane exited 0" in " ".join(red)
    return Probe(
        "a red attack lane turns the verdict RED",
        not closed and named,
        "closure RED on " + ", ".join(red[:3]) if not closed else "THE GATE STAYED GREEN WITH attack=1",
    )


def _shipped_evidence_restamped() -> Probe:
    # The adversary's original method against the REAL artifacts: re-stamp them with
    # the current identity, which is exactly what "the lane exited 0 on this tree"
    # would mean, then let ledger.build() derive every cell itself.
    cases, pins = GENERATED / "cases.json", GENERATED / "provenance.json"
    if not (cases.is_file() and pins.is_file()):
        return Probe("shipped evidence re-stamped cannot close", True, "no shipped cases.json/provenance.json to test")

    now = evidence_identity.identity()
    held = {
        "cases.json": {**json.loads(cases.read_text(encoding="utf-8")), "identity": now},
        "provenance.json": {**json.loads(pins.read_text(encoding="utf-8")), "identity": now},
        "lanes.json": {"identity": str(now["digest"]), "lanes": {"cases": 0, "pins": 0, **RED_LANES}},
    }
    with tempfile.TemporaryDirectory(prefix="green_attack_") as raw:
        closed, red = _verdict(Path(raw), held)
    return Probe(
        "shipped evidence re-stamped cannot close",
        not closed,
        "closure RED on " + ", ".join(red[:3]) if not closed else "THE GATE CLOSED ON RE-STAMPED EVIDENCE",
    )


def _no_waiver_to_defeat() -> Probe:
    # The original had to rebind closure.conditions.__defaults__ because
    # `where != GENERATED` returned the one-tree condition true for any scratch
    # directory. G4 deleted that waiver; there is nothing left to rebind past.
    source = (ROOT / "harness" / "closure.py").read_text(encoding="utf-8")
    return Probe(
        "no non-default-directory waiver remains",
        "where != GENERATED" not in source,
        "closure.py carries no directory waiver"
        if "where != GENERATED" not in source
        else "THE WAIVER IS BACK: one-tree is excused outside compat/generated",
    )


def run_all() -> list[Probe]:
    return [_lane_red_over_a_green_run(), _shipped_evidence_restamped(), _no_waiver_to_defeat()]


def main() -> int:
    held = run_all()
    print("phase-G standing negative control\n")
    print("Scope: this proves the green-with-red-lanes path. It re-stamps digests and")
    print("does NOT empty any population -- condition_audit covers what it cannot.\n")
    for one in held:
        print(f"{one.mark} {one.name:<44} {one.detail}")

    failing = [one.name for one in held if not one.held]
    print(f"\n{len(held)} probe(s), {len(failing)} failing: {failing or 'none'}")
    if not failing:
        print("\nThe demonstration that closed GREEN before Phase G now closes RED.")

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {
        "identity": str(evidence_identity.identity()["digest"]),
        "probes": [asdict(one) for one in held],
        "failing": failing,
        "ledger_stages": list(ledger.STAGES),
    }
    with (GENERATED / "green_controls.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {GENERATED / 'green_controls.json'}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
