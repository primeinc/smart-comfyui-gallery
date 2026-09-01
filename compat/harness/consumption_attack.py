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


@dataclass
class Probe:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _put(where: Path, name: str, body: dict[str, Any]) -> None:
    with (where / name).open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
        handle.write("\n")


def _red(where: Path) -> list[str]:
    return [f"{one.name} [{one.state}]" for one in closure.conditions(where) if not one.green]


def _control_is_green() -> tuple[Probe, dict[str, Any]]:
    base = ledger_attack.green_fixture()
    with tempfile.TemporaryDirectory(prefix="consumption_attack_") as raw:
        where = Path(raw)
        closure_attack._write(where, base)
        red = _red(where)
    return (
        Probe("the control closes", not red, "GREEN" if not red else f"NOT GREEN: {', '.join(red[:3])}"),
        base,
    )


def _regenerated_input_is_caught(base: dict[str, Any]) -> Probe:
    # E11 exactly: build the ledger against clean pins, then regenerate ONE input
    # without rebuilding. `just compat pins` alone after a full run does this, and
    # both artifacts remain individually stamped for the current tree.
    with tempfile.TemporaryDirectory(prefix="consumption_attack_") as raw:
        where = Path(raw)
        closure_attack._write(where, base)

        spoiled = json.loads(json.dumps(base["provenance.json"]))
        spoiled["weights"][0]["state"] = "SKIPPED"
        _put(where, "provenance.json", spoiled)

        red = _red(where)
    named = any(one.startswith("every consumed artifact unchanged") for one in red)
    return Probe(
        "an input regenerated after the ledger read it is caught",
        bool(red) and named,
        "closure RED on " + ", ".join(red[:3]) if named else f"NOT CAUGHT BY THE GRAPH: {red or 'GREEN'}",
    )


def _stamps_alone_would_not_catch_it(base: dict[str, Any]) -> Probe:
    # Regenerating the content leaves the stamp untouched, which is why one-tree
    # could not see it. Compared against the fixture's OWN stamp, never a freshly
    # computed one: this worktree is shared and the tree digest moves underneath.
    before = str(base["provenance.json"].get("identity") or "")
    spoiled = json.loads(json.dumps(base["provenance.json"]))
    spoiled["weights"][0]["state"] = "SKIPPED"
    after = str(spoiled.get("identity") or "")
    return Probe(
        "the regenerated input keeps its stamp",
        bool(before) and after == before,
        f"stamp {before[:12]} unchanged by the regeneration, so only the graph sees it"
        if after == before
        else f"the stamp moved: {before[:12]} -> {after[:12]}",
    )


def _graph_is_derived_from_the_opens() -> Probe:
    # Not a declared table: the digests are recorded where build() opens each file,
    # so an input added to the builder joins the graph without anyone updating a list.
    with tempfile.TemporaryDirectory(prefix="consumption_attack_") as raw:
        where = Path(raw)
        closure_attack._write(where, ledger_attack.green_fixture())
        built = ledger.build(where)
        consumed = set(built.get("consumed") or {})
    wanted = {"provenance.json", "cases.json", "lanes.json"}
    return Probe(
        "the graph names every artifact build() opened",
        consumed == wanted,
        f"recorded {sorted(consumed)}"
        if consumed == wanted
        else f"recorded {sorted(consumed)}, expected {sorted(wanted)}",
    )


def run_all() -> list[Probe]:
    control, base = _control_is_green()
    return [
        control,
        _graph_is_derived_from_the_opens(),
        _stamps_alone_would_not_catch_it(base),
        _regenerated_input_is_caught(base),
    ]


def main() -> int:
    held = run_all()
    print("consumption-graph controls\n")
    for one in held:
        print(f"{one.mark} {one.name:<48} {one.detail}")

    failing = [one.name for one in held if not one.held]
    print(f"\n{len(held)} probe(s), {len(failing)} failing: {failing or 'none'}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {
        "identity": str(evidence_identity.identity()["digest"]),
        "probes": [asdict(one) for one in held],
        "failing": failing,
    }
    with (GENERATED / "consumption_controls.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {GENERATED / 'consumption_controls.json'}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
