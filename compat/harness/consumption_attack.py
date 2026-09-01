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


def _captured() -> str:
    # ONE read of the tree identity, threaded into every fixture and build.
    from compat.harness import identity as evidence_identity

    return str(evidence_identity.identity()["digest"])


def _red(where: Path) -> list[str]:
    return [f"{one.name} [{one.state}]" for one in closure.conditions(where) if not one.green]


def _control_is_green() -> tuple[Probe, dict[str, Any]]:
    digest = _captured()
    base = ledger_attack.green_fixture(digest=digest)
    with tempfile.TemporaryDirectory(prefix="consumption_attack_") as raw:
        where = Path(raw)
        closure_attack._write(where, base, digest=_captured())
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
        closure_attack._write(where, base, digest=_captured())

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


def _opens_in_build() -> set[str]:
    """The artifact names build() actually opens, read out of its source.

    The literal set this replaced could not see the miss it existed to catch: a
    fourth _read leaves the recorded graph at three and `consumed == wanted`
    still passes, in the control whose own comment claims derivation.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ledger.build))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_read"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def _graph_is_derived_from_the_opens() -> Probe:
    # Derived twice over: `consumed` is a required parameter, so an open that
    # records nothing is a TypeError, and the expected set is read from build()'s
    # own source rather than named here.
    opens = _opens_in_build()
    with tempfile.TemporaryDirectory(prefix="consumption_attack_") as raw:
        where = Path(raw)
        digest = _captured()
        closure_attack._write(where, ledger_attack.green_fixture(digest=digest), digest=digest)
        built = ledger.build(where, digest=digest)
        consumed = set(built.get("consumed") or {})
    tripwire = {"provenance.json", "cases.json", "lanes.json"}
    return Probe(
        "the graph names every artifact build() opened",
        bool(opens) and consumed == opens and consumed >= tripwire,
        f"{len(consumed)} recorded, matching the {len(opens)} open(s) in build()'s source"
        if consumed == opens
        else f"recorded {sorted(consumed)}, build() opens {sorted(opens)}",
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
