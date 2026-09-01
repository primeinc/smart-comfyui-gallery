from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from compat.harness import closure, closure_attack
from compat.harness import identity as evidence_identity

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"

ARTIFACTS: Final[tuple[str, ...]] = ("ledger.json", "cases.json", "provenance.json", "lanes.json")


@dataclass(frozen=True)
class Source:
    #: Emptying this on disk is what proves a condition over it cannot hold on
    #: nothing. It acts on the written artifacts, because the ledger is DERIVED
    #: from the inputs rather than supplied, so there is no input dict to edit.
    empty: Callable[[Path], None]

    #: The SECOND degenerate shape: a member that is present and unclassifiable.
    #: Empty was the audit's only word, so a denylist condition holding over
    #: "WOBBLE" was certified sound by a check that never showed it one.
    malform: Callable[[Path], None]

    #: The BOUNDARY RULE: a field traced only to its writer stops one screen above
    #: the guard, which is where the guarantee usually lives. Both are cited so a
    #: reviewer can check the trace rather than take this table's word for it.
    writer: str
    validator: str


def _rewrite(where: Path, name: str, change: Callable[[dict[str, Any]], None]) -> None:
    path = where / name
    if not path.is_file():
        return
    held: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    change(held)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(held, indent=2, sort_keys=True, default=str))
        handle.write("\n")


def _strip_stamps(where: Path) -> None:
    for name in ARTIFACTS:
        _rewrite(where, name, lambda held: held.pop("identity", None))


def _empty(artifact: str, field_name: str) -> Callable[[Path], None]:
    def emptied(where: Path) -> None:
        blank: Any = {} if field_name in ("lanes", "consumed", "exercised_against_real_evidence") else []
        _rewrite(where, artifact, lambda held: held.__setitem__(field_name, blank))

    return emptied


def _empty_ablations(where: Path) -> None:
    def strip(held: dict[str, Any]) -> None:
        for row in held.get("results") or []:
            row["ablations"] = []

    _rewrite(where, "cases.json", strip)


def _malform(artifact: str, field_name: str, make: Callable[[], Any]) -> Callable[[Path], None]:
    def spoiled(where: Path) -> None:
        def change(held: dict[str, Any]) -> None:
            found = held.get(field_name)
            if isinstance(found, dict):
                found["a_member_nobody_classified"] = make()
            else:
                # An absent field is not an excuse to inject nothing: a malformer
                # that no-ops reports the condition sound without ever testing it.
                held[field_name] = [*(found or []), make()]

        _rewrite(where, artifact, change)

    return spoiled


def _malform_ablation(where: Path) -> None:
    # TWO members: two conditions read this population and judge different
    # fields. A WOBBLE verdict is invisible to the cause condition, which
    # looks only at INCONCLUSIVE rows, so that one needs its own member.
    def change(held: dict[str, Any]) -> None:
        rows = held.get("results") or []
        if rows:
            rows[0].setdefault("ablations", []).append({"primitive": "p", "verdict": "WOBBLE"})
            rows[0]["ablations"].append(
                {"primitive": "p", "verdict": "INCONCLUSIVE", "cause": "a_cause_nobody_declared"}
            )

    _rewrite(where, "cases.json", change)


def _malform_stamp(where: Path) -> None:
    _rewrite(where, "cases.json", lambda held: held.__setitem__("identity", {"digest": "not-a-digest"}))


def _malform_case_offences(where: Path) -> None:
    # The row-coverage condition reads consumer_id: a case attributed to a
    # consumer nothing declares is this population's unclassifiable member.
    # `skipped` and `shards_failed` now have their own populations below.
    def change(held: dict[str, Any]) -> None:
        held["results"] = [*(held.get("results") or []), {"consumer_id": "a_consumer_nobody_declared"}]

    _rewrite(where, "cases.json", change)


def _empty_considered(field_name: str) -> Callable[[Path], None]:
    # EVERYTHING CONSIDERED: the successes and the field the condition reads.
    # Emptying only `results` left the offending members in place, which is
    # how a total failure read as not-applicable rather than as a failure.
    def emptied(where: Path) -> None:
        def change(held: dict[str, Any]) -> None:
            held["results"] = []
            held[field_name] = []

        _rewrite(where, "cases.json", change)

    return emptied


def _malform_considered(field_name: str, make: Callable[[], Any]) -> Callable[[Path], None]:
    def spoiled(where: Path) -> None:
        def change(held: dict[str, Any]) -> None:
            held[field_name] = [*(held.get(field_name) or []), make()]

        _rewrite(where, "cases.json", change)

    return spoiled


def _malform_ledger(where: Path) -> None:
    # An extra ROW carrying an unrecognised cell state, so it reaches every
    # condition over this population: the count one AND the two cell ones.
    def change(held: dict[str, Any]) -> None:
        stages = held.get("stages") or []
        rows = held.get("rows")
        if isinstance(rows, list) and stages:
            rows.append(
                {
                    "consumer": "a_consumer_nobody_declared",
                    "cells": {one: {"state": "WOBBLE", "reason": "a state nothing recognises"} for one in stages},
                }
            )

    _rewrite(where, "ledger.json", change)


def _drop_ledger(where: Path) -> None:
    (where / "ledger.json").unlink(missing_ok=True)


SOURCES: Final[dict[str, Source]] = {
    "generated:identity stamps": Source(
        empty=_strip_stamps,
        malform=_malform_stamp,
        writer="each artifact's own main(): ledger.py:build, sharded.py:112, lanes.py:main, provenance.py:main",
        validator="identity.py:identity() recomputes the tree digest; closure._one_tree compares every stamp to it",
    ),
    "lanes.json:lanes": Source(
        empty=_empty("lanes.json", "lanes"),
        # A real unclassifiable member now that the condition classifies instead
        # of coercing. It used to inject `1` -- an ordinary failing exit code the
        # condition already rejects -- because the honest value made it raise.
        malform=_malform("lanes.json", "lanes", lambda: "WOBBLE"),
        writer="compat.just `run` appends `<lane> <code>` per lane; lanes.py:_recorded parses it",
        validator="NONE beyond the int parse -- the exit code is the shell's, and nothing re-derives it",
    ),
    "ledger.json:consumed": Source(
        empty=_empty("ledger.json", "consumed"),
        malform=_malform("ledger.json", "consumed", lambda: "not-the-digest-that-was-read"),
        writer="ledger.py:_read records a sha256 at every open build() performs",
        validator="closure._consumption_agrees re-digests each named artifact and compares",
    ),
    "ledger.json:rows": Source(
        empty=_empty("ledger.json", "rows"),
        malform=_malform_ledger,
        writer="ledger.py:build() -- one row per manifest consumer",
        validator="ledger.py:stages_are_covered() + STAGE_EVIDENCE: each cell derives from its own case field",
    ),
    "provenance.json:weights": Source(
        empty=_empty("provenance.json", "weights"),
        malform=_malform("provenance.json", "weights", lambda: {"pack": "p", "file": "f", "state": "WOBBLE"}),
        writer="provenance.py:weight_identity() -> main(), state from weight_state() at :505-517",
        validator="provenance.weight_is_verified() -- the one predicate closure and ledger now share",
    ),
    "cases.json:ablations": Source(
        empty=_empty_ablations,
        malform=_malform_ablation,
        writer="run.py:run_ablation() -> run_case(); tallied by run.py:ablation_tally",
        validator="run.py:116 compares observed_break against the declared expect_breaks",
    ),
    "cases.json:results": Source(
        empty=_empty("cases.json", "results"),
        malform=_malform_case_offences,
        writer="sharded.py:main() for the shipped lane; run.py:main() when run whole",
        validator="assertions/arrays.py:compare() decides each verdict; run.py:run_case records it",
    ),
    "generated:ledger": Source(
        empty=_drop_ledger,
        malform=_malform_ledger,
        writer="ledger.py:main() writes ledger.json",
        validator="closure.conditions() returns only `ledger present` when the file is absent",
    ),
    "cases.json:results+skipped": Source(
        empty=_empty_considered("skipped"),
        malform=_malform_considered("skipped", lambda: {"why": "an input nobody classified"}),
        writer="run.py records a skip through case.note_skip; sharded.py:main collects them",
        validator="NONE re-derives a skip -- which is why every member of it is an offence",
    ),
    "cases.json:results+shards_failed": Source(
        empty=_empty_considered("shards_failed"),
        malform=_malform_considered("shards_failed", lambda: "a shard nobody classified"),
        writer="sharded.py:main names a shard that did not complete",
        validator="NONE -- a shard that did not finish leaves nothing to re-derive it from",
    ),
    "ledger_controls.json:exercised_against_real_evidence": Source(
        empty=_empty("ledger_controls.json", "exercised_against_real_evidence"),
        # A stage nothing derives. The comparison walks BOTH key sets, so this
        # member is examined rather than riding along beside the six real ones.
        malform=_malform(
            "ledger_controls.json",
            "exercised_against_real_evidence",
            lambda: {"carrying": 1, "of": 1},
        ),
        writer="ledger_attack.py:main() records ledger.exercised_against_real_evidence(GENERATED)",
        validator="closure._controls_describe_this_evidence re-derives it from cases.json and compares",
    ),
}


@dataclass
class Finding:
    kind: str
    condition: str
    population: str
    detail: str


@dataclass
class Audit:
    declared: dict[str, str] = field(default_factory=dict)
    emptied: dict[str, list[str]] = field(default_factory=dict)
    #: What the SECOND degenerate shape COST, per population -- not which
    #: conditions it asked, which is `emptied` again under another name. A
    #: population in `emptied` and empty here was asked and cost nothing.
    malformed: dict[str, list[str]] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings


Provider = Callable[[Path], list[closure.Condition]]


def run(asked: Provider = closure.conditions) -> Audit:
    import tempfile

    out = Audit()
    with tempfile.TemporaryDirectory(prefix="condition_audit_") as raw:
        where = Path(raw)
        digest = closure_attack._captured()
        base = closure_attack.green_fixture(digest=digest)
        closure_attack._write(where, base, digest=digest)
        answered = asked(where)
        control = {one.name: one for one in answered}
        if len(control) != len(answered):
            seen: set[str] = set()
            twice = sorted({one.name for one in answered if one.name in seen or seen.add(one.name)})
            out.findings.append(
                Finding(
                    "duplicate condition name", ", ".join(twice), "", "keyed by name, so one silently replaced another"
                )
            )

        for name, one in control.items():
            out.declared[name] = one.population
            if not one.population:
                out.findings.append(Finding("undeclared population", name, "", "the condition declares no population"))
            elif one.population not in SOURCES:
                out.findings.append(
                    Finding("no emptier", name, one.population, "this audit cannot empty it, so it is unproven")
                )
            if not one.green:
                out.findings.append(Finding("control not green", name, one.population, one.detail))

        for population, source in SOURCES.items():
            over = sorted(n for n, one in control.items() if one.population == population)
            out.emptied[population] = over

            closure_attack._write(where, base, digest=digest)
            source.empty(where)
            after = {one.name: one for one in asked(where)}

            for name in over:
                found = after.get(name)
                if found is not None and found.green:
                    out.findings.append(
                        Finding(
                            "held over an empty population", name, population, f"still {found.state}: {found.detail}"
                        )
                    )

            # A source no condition claims in the green state is not thereby exempt:
            # `ledger present` exists only on the early-return path. Emptying any
            # source must cost the verdict, or the source is not load-bearing at all.
            if all(one.green for one in after.values()):
                out.findings.append(
                    Finding("empties to no effect", "", population, "the verdict stayed green with it emptied")
                )

            # The second degenerate shape. Empty was the only word this audit knew,
            # so a denylist condition holding over an unclassifiable member was
            # certified sound by a check that never showed it one.
            closure_attack._write(where, base, digest=digest)
            source.malform(where)
            spoiled = {one.name: one for one in asked(where)}
            # PER CONDITION, not per verdict. The weaker "the verdict must cost"
            # rule was written here and withdrawn: it lets a vacuous condition hide
            # behind a real failure elsewhere, and the self-control caught that.
            for name in over:
                found = spoiled.get(name)
                if found is not None and found.green:
                    out.findings.append(
                        Finding(
                            "held over an unclassifiable member",
                            name,
                            population,
                            f"still {found.state} with a member nothing classifies",
                        )
                    )
            out.malformed[population] = sorted(
                name for name in over if (spoiled.get(name) is not None and not spoiled[name].green)
            )
    return out


def allowance_is_not_inflated() -> Finding | None:
    # The mutation that guards the allowance builds ALLOWANCE+1 rows FROM the
    # allowance, so it passes at any value. Shipped evidence is the number the
    # constant cannot move: a ratchet may not be set above what was observed.
    shipped = GENERATED / "cases.json"
    if not shipped.is_file():
        return Finding("allowance unverifiable", "every ablation concluded", "cases.json:ablations", "no evidence yet")
    held = json.loads(shipped.read_text(encoding="utf-8"))
    observed = sum(
        1
        for row in (held.get("results") or [])
        for one in (row.get("ablations") or [])
        if one.get("verdict") == "INCONCLUSIVE"
    )
    if observed < closure.ABLATION_INCONCLUSIVE_ALLOWANCE:
        return Finding(
            "allowance above what was observed",
            "every ablation concluded",
            "cases.json:ablations",
            f"allowance {closure.ABLATION_INCONCLUSIVE_ALLOWANCE} exceeds the {observed} in shipped evidence",
        )
    return None


def _with_a_vacuous_condition(where: Path) -> list[closure.Condition]:
    # A condition that holds whatever its population contains -- the exact shape this
    # audit exists to catch. If the audit cannot go red on this, it cannot go red.
    return [
        *closure.conditions(where),
        closure.Condition("a condition that cannot fail", closure.HELD, "holds regardless", "lanes.json:lanes"),
    ]


def self_control() -> tuple[bool, str]:
    caught = run(_with_a_vacuous_condition)
    wanted = {"held over an empty population", "held over an unclassifiable member"}
    found = {one.kind for one in caught.findings if one.condition == "a condition that cannot fail"}
    missing = sorted(wanted - found)
    return not missing, f"caught by {sorted(found)}" if not missing else f"NOT caught by {missing}"


def main() -> int:
    caught, named = self_control()
    if not caught:
        print("SELF-CONTROL FAILED: a deliberately vacuous condition was not caught")
        print(f"  the audit reported: {named}")
        return 1
    print("self-control: a deliberately vacuous condition IS caught\n")

    out = run()
    inflated = allowance_is_not_inflated()
    if inflated is not None:
        out.findings.append(inflated)
    print(f"conditions audited: {len(out.declared)}   sources: {len(SOURCES)}\n")
    for name, population in sorted(out.declared.items()):
        print(f"  {name:<38} over {population}")
    print()
    for population, over in sorted(out.emptied.items()):
        cost = out.malformed.get(population, [])
        print(
            f"  emptying {population:<28} -> {len(over)} condition(s) must stop holding"
            f"   |  a member nothing classifies cost {len(cost)}"
        )

    if out.findings:
        print(f"\n{len(out.findings)} FINDING(S):")
        for one in out.findings:
            print(f"  {one.kind:<30} {one.condition or '-':<38} {one.detail}")
    else:
        print("\nVALIDATED EMPTY: every condition declares a population, every population can be")
        print("emptied, and emptying it stops every condition over it from holding.")

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {
        # Stamped for the same reason closure.json now is: an audit left by an
        # earlier tree reads as this one's, and F28 named both files together.
        "identity": str(evidence_identity.identity()["digest"]),
        "clean": out.clean,
        "declared": out.declared,
        "emptied": out.emptied,
        "malformed": out.malformed,
        "findings": [asdict(one) for one in out.findings],
        "sources": {name: {"writer": one.writer, "validator": one.validator} for name, one in sorted(SOURCES.items())},
    }
    with (GENERATED / "condition_audit.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"\nwrote {GENERATED / 'condition_audit.json'}")
    return 0 if out.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
