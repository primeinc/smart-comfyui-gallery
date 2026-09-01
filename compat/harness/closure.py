from __future__ import annotations

import json
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"

HELD: Final[str] = "held"
FAILED: Final[str] = "failed"
NOT_APPLICABLE: Final[str] = "not-applicable"


#: Ablations the runner could not drive to a conclusion. A RATCHET, not a budget:
#: it may only ever be lowered. 497 of 1037 shipped green while ablation verdicts
#: reached no gate at all -- this number exists so that stops being invisible.
ABLATION_INCONCLUSIVE_ALLOWANCE: Final[int] = 497

ABLATION_INCONCLUSIVE: Final[str] = "INCONCLUSIVE"

#: Known-good ablation verdicts. Anything else -- including null, absent and a
#: value nobody anticipated -- is unconcluded, because a denylist admits the rest.
ABLATION_CONCLUDED: Final[frozenset[str]] = frozenset({"PASS", "INCONCLUSIVE"})


@dataclass
class Condition:
    name: str
    state: str
    detail: str

    #: `<artifact>:<field>` that must be non-empty for this condition to mean
    #: anything. condition_audit empties exactly this and asserts the condition
    #: stops holding; a population it cannot empty fails that audit.
    population: str = ""

    @property
    def green(self) -> bool:
        return self.state == HELD

    @property
    def mark(self) -> str:
        return {HELD: "ok ", FAILED: "RED", NOT_APPLICABLE: "N/A"}[self.state]


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


def _over(name: str, population: str, over: Sized, offending: list[str], clean: str) -> Condition:
    # `over` is the COLLECTION, not a count: with size passed separately the
    # declared population could name one collection while the verdict came from
    # another, and the audit then proved a set the condition never reads.
    size = len(over)
    # The empty-population rule. A condition that examined nothing has not held:
    # green must mean "checked, and clean", never "found no reason to complain".
    if size == 0:
        return Condition(name, NOT_APPLICABLE, f"{population} is empty; nothing was checked", population)
    if offending:
        return Condition(name, FAILED, f"{len(offending)} of {size}: {', '.join(offending[:4])}", population)
    return Condition(name, HELD, clean, population)


def _one_tree(stamped: dict[str, str]) -> Condition:
    now = str(evidence_identity.identity()["digest"])
    wrong = [f"{name} built under {held[:12] or 'no digest'}" for name, held in sorted(stamped.items()) if held != now]
    return _over(
        "evidence from one tree",
        "generated:identity stamps",
        stamped,
        wrong,
        f"{len(stamped)} artifact(s) stamped {now[:12]}",
    )


def _every_lane_green(lanes: dict[str, Any] | None) -> Condition:
    # A lane's exit code was recorded and read by nothing: `attack` and `selftest`
    # could both exit 1 while this printed GREEN. The lane record is now evidence,
    # and its population is compat.just's declared list rather than its own keys.
    from compat.harness import lanes as lane_record

    found = (lanes or {}).get("lanes")
    held = found if isinstance(found, dict) else {}
    want = lane_record.declared()
    if not want:
        # `want or held` would fall back to judging the record by its own keys --
        # the exact defect this condition was repaired to remove. A declaration
        # that cannot be read is a failure, never a quiet reversion.
        return Condition(
            "every lane exited 0",
            FAILED,
            "compat.just's lane loop could not be parsed, so there is no declared set to judge against",
            "lanes.json:lanes",
        )

    offending = sorted(f"{name} exited {code}" for name, code in held.items() if int(code) != 0)
    offending += [f"{name} declared but never recorded" for name in want if name not in held]
    return _over(
        "every lane exited 0",
        "lanes.json:lanes",
        want or held,
        offending,
        f"{len(held)} lane(s) recorded for {len(want)} declared, all 0",
    )


def _ablations_concluded(cases: dict[str, Any] | None) -> Condition:
    # The only mechanism proving a retained field is load-bearing, and its verdicts
    # were aggregated nowhere. CONTRADICTED gets no allowance: it means an ablation
    # behaved opposite to its declaration, which is a finding, not a shortfall.
    held = [one for row in ((cases or {}).get("results") or []) for one in (row.get("ablations") or [])]
    inconclusive = [one for one in held if one.get("verdict") == ABLATION_INCONCLUSIVE]
    # Allowlist, matching provenance.weight_is_verified. An enumerated-bad-values
    # filter admits every shape it did not think of: null, absent, "", "WOBBLE".
    unconcluded = [one for one in held if one.get("verdict") not in ABLATION_CONCLUDED]

    offending = [f"{one.get('primitive', '?')} {one.get('verdict') or 'no verdict'}" for one in unconcluded[:4]]
    if len(inconclusive) > ABLATION_INCONCLUSIVE_ALLOWANCE:
        offending.append(f"{len(inconclusive)} INCONCLUSIVE over an allowance of {ABLATION_INCONCLUSIVE_ALLOWANCE}")
    return _over(
        "every ablation concluded",
        "cases.json:ablations",
        held,
        offending,
        f"{len(held)} ablation(s), all concluded, {len(inconclusive)} inconclusive "
        f"within the allowance of {ABLATION_INCONCLUSIVE_ALLOWANCE}",
    )


def _consumption_agrees(ledger: dict[str, Any], where: Path) -> Condition:
    # One-tree compares each artifact to the TREE, never to each other, so two
    # artifacts individually current for the tree but not built from one another
    # both passed. `just compat pins` alone after a full run does exactly that.
    consumed: dict[str, str] = ledger.get("consumed") or {}
    drifted: list[str] = []
    for name, digest in sorted(consumed.items()):
        path = where / name
        now = evidence_identity.sha256_of(path.read_bytes()) if path.is_file() else ""
        if now != digest:
            drifted.append(f"{name} was {digest[:8]}, is {now[:8] or 'absent'}")
    return _over(
        "every consumed artifact unchanged",
        "ledger.json:consumed",
        consumed,
        drifted,
        f"{len(consumed)} artifact(s) still the bytes the ledger read",
    )


def conditions(where: Path = GENERATED) -> list[Condition]:
    out: list[Condition] = []
    ledger = _read("ledger.json", where)
    pins = _read("provenance.json", where)
    cases = _read("cases.json", where)
    lanes = _read("lanes.json", where)

    if ledger is None:
        return [Condition("ledger present", FAILED, "no ledger.json: the ledger lane did not run", "generated:ledger")]

    stamped: dict[str, str] = {"ledger.json": _stamp(ledger)}
    for name, held in (("cases.json", cases), ("provenance.json", pins), ("lanes.json", lanes)):
        if held is not None:
            stamped[name] = _stamp(held)
    out.append(_one_tree(stamped))
    out.append(_every_lane_green(lanes))
    out.append(_consumption_agrees(ledger, where))

    rows: list[dict[str, Any]] = ledger.get("rows") or []
    declared = int(ledger.get("totals", {}).get("declared", 0))
    out.append(
        _over(
            "every declared member accounted for",
            "ledger.json:rows",
            rows,
            [] if len(rows) == declared else [f"{len(rows)} row(s) for {declared} declared"],
            f"{len(rows)} row(s) for {declared} declared",
        )
    )

    weights: list[dict[str, Any]] = (pins or {}).get("weights") or []
    unverified = [
        f"{one.get('pack', '?')}/{one.get('file', '?')} {one.get('state') or 'no state'}"
        for one in weights
        if not provenance.weight_is_verified(one)
    ]
    out.append(
        _over(
            "every weight VERIFIED",
            "provenance.json:weights",
            weights,
            unverified,
            f"{len(weights)} weight(s) VERIFIED",
        )
    )

    stages: list[str] = ledger.get("stages") or []
    for state in ("BLOCKED", "FAILED"):
        cells = [
            f"{row['consumer']}/{stage}" for row in rows for stage in stages if row["cells"][stage]["state"] == state
        ]
        out.append(
            _over(
                f"no ledger cell {state}",
                "ledger.json:rows",
                [(row["consumer"], stage) for row in rows for stage in stages],
                cells,
                f"{len(rows) * len(stages)} cell(s), none {state}",
            )
        )

    results: list[dict[str, Any]] = (cases or {}).get("results") or []
    skipped = [str(one) for one in ((cases or {}).get("skipped") or [])]
    out.append(
        _over("no skipped input", "cases.json:results", results, skipped, f"{len(results)} case(s), none skipped")
    )

    out.append(_ablations_concluded(cases))

    shards = [str(one) for one in ((cases or {}).get("shards_failed") or [])]
    out.append(
        _over("no shard failed", "cases.json:results", results, shards, f"{len(results)} case(s), no shard failed")
    )
    return out


def main() -> int:
    held = conditions()
    print("closure conditions\n")
    for one in held:
        print(f"{one.mark} {one.name:<38} {one.detail}")

    closed = all(one.green for one in held)
    print(f"\nCLOSURE: {'GREEN' if closed else 'RED'}")
    if not closed:
        print("the missing-work ledger is compat/generated/ledger.md")

    with (GENERATED / "closure.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                {
                    "closed": closed,
                    "conditions": [
                        {
                            "name": o.name,
                            "state": o.state,
                            "green": o.green,
                            "detail": o.detail,
                            "population": o.population,
                        }
                        for o in held
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        handle.write("\n")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
