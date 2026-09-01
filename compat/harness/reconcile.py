from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from compat.harness import observe

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

AGREED: Final[str] = "AGREED"
UNEXERCISED: Final[str] = "UNEXERCISED"
POPULATION_DEFECT: Final[str] = "POPULATION_DEFECT"


@dataclass
class Reconciled:
    identity: str
    verdict: str
    consumers: list[str] = field(default_factory=list)
    loaders: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    detail: str = ""


def _stems(identity: str) -> set[str]:
    held = identity.strip().strip("/").replace("\\", "/")
    parts = [one for one in held.split("/") if one]
    out = {held, parts[-1] if parts else held}
    if len(parts) >= 2:
        out.add("/".join(parts[-2:]))
    tail = parts[-1] if parts else held
    if "." in tail:
        out.add(tail.rsplit(".", 1)[0])
    return {one for one in out if one}


def _resolve(identity: str, claimed: dict[str, set[str]]) -> tuple[str, set[str]]:
    for stem in _stems(identity):
        holders = claimed.get(stem)
        if not holders:
            continue
        if len(holders) > 1:
            return "", holders
        return next(iter(holders)), set()
    return "", set()


def reconcile(population: dict[str, Any], observed: list[dict[str, Any]]) -> list[Reconciled]:
    static: dict[str, Reconciled] = {}
    claimed: dict[str, set[str]] = {}
    for edge in population.get("edges", []):
        identity = str(edge.get("artifact_logical_identity", ""))

        if not identity or identity.startswith(("UNRESOLVED_ARTIFACT:", "UNRESOLVED_CALL:")):
            continue
        held = static.setdefault(identity, Reconciled(identity, UNEXERCISED))
        consumer, status = str(edge.get("consumer_id", "")), str(edge.get("discovery_status", ""))
        if consumer and consumer not in held.consumers:
            held.consumers.append(consumer)
        if status and status not in held.statuses:
            held.statuses.append(status)
        for stem in _stems(identity):
            claimed.setdefault(stem, set()).add(identity)

    out: list[Reconciled] = []
    for row in observed:
        identity = str(row.get("identity", ""))
        loader = str(row.get("loader", ""))

        if loader == observe.NATIVE_UNSEEN:
            continue
        matched, ambiguous = _resolve(identity, claimed)
        if ambiguous:
            out.append(
                Reconciled(
                    identity=identity,
                    verdict=POPULATION_DEFECT,
                    loaders=[loader],
                    detail=f"{identity!r} matches {sorted(ambiguous)}; the population names it more than once",
                )
            )
            continue
        if matched:
            held = static[matched]
            held.verdict = AGREED
            if loader and loader not in held.loaders:
                held.loaders.append(loader)
            held.detail = f"observed as {identity!r} through {loader}"
            continue
        out.append(
            Reconciled(
                identity=identity,
                verdict=POPULATION_DEFECT,
                loaders=[loader],
                detail=f"{loader} opened it and static discovery names no such artifact",
            )
        )

    for held in static.values():
        if held.verdict == UNEXERCISED:
            held.detail = "discovered statically; no observation of it in this run"
        out.append(held)
    return sorted(out, key=lambda one: (one.verdict, one.identity))


def build() -> dict[str, Any]:
    generated = ROOT / "generated"
    population = json.loads((generated / "artifact_population.json").read_text(encoding="utf-8"))
    cases = json.loads((generated / "cases.json").read_text(encoding="utf-8"))
    observed = list(cases.get("observed", []))
    trees = _one_tree(population, cases)
    rows = reconcile(population, observed)
    return {
        "observed_artifacts": len(observed),
        "rows": [asdict(one) for one in rows],
        "totals": {
            one: sum(1 for row in rows if row.verdict == one) for one in (AGREED, UNEXERCISED, POPULATION_DEFECT)
        },
        "observer_is_complete": _observer_complete(generated, cases),
        "evidence_from_one_tree": trees,
    }


def _one_tree(population: dict[str, Any], cases: dict[str, Any]) -> dict[str, Any]:
    from compat.harness import identity as evidence_identity

    now = str(evidence_identity.identity()["digest"])
    held = {
        "artifact_population.json": str(population.get("identity") or ""),
        "cases.json": str((cases.get("identity") or {}).get("digest", "")),
    }
    wrong = {name: one for name, one in held.items() if one != now}
    return {
        "tree": now,
        "agree": not wrong,
        "detail": "both artifacts stamped for this tree"
        if not wrong
        else "; ".join(f"{name} was built under {one[:12] or 'no digest'}" for name, one in wrong.items()),
    }


def _observer_complete(generated: Path, cases: dict[str, Any]) -> dict[str, Any]:
    ran = int(cases.get("cases", 0)) > 0 and not cases.get("shards_failed")
    where = generated / "observer_controls.json"
    if not where.is_file():
        return {"known": False, "cases_ran": ran, "detail": "observe-attack has not run; coverage is unmeasured"}
    held = json.loads(where.read_text(encoding="utf-8"))
    failing = list(held.get("failing", []))
    if not ran:
        return {
            "known": True,
            "complete": not failing,
            "cases_ran": False,
            "failing_probes": failing,
            "detail": "the cases did not run, so nothing here is evidence about what loaded",
        }
    return {
        "known": True,
        "complete": not failing,
        "cases_ran": True,
        "failing_probes": failing,
        "detail": (
            "every probe holds and the cases ran, so an unobserved artifact is evidence it did not load"
            if not failing
            else f"{len(failing)} probe(s) red, so UNEXERCISED means 'no evidence', never 'did not load'"
        ),
    }


def main() -> int:
    out = build()
    totals = out["totals"]
    print(f"observed artifacts              : {out['observed_artifacts']}")
    print(f"AGREED                          : {totals[AGREED]}")
    print(f"UNEXERCISED                     : {totals[UNEXERCISED]}")
    print(f"POPULATION_DEFECT               : {totals[POPULATION_DEFECT]}\n")

    for row in out["rows"]:
        if row["verdict"] == AGREED:
            continue
        print(f"{'!! ' if row['verdict'] == POPULATION_DEFECT else '-- '}{row['verdict']:<18}{row['identity'][:60]}")
        print(f"    {row['detail'][:150]}")

    coverage = out["observer_is_complete"]
    print(f"\nobserver coverage: {coverage['detail']}")

    target = ROOT / "generated" / "reconciliation.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")

    return 0 if not totals[POPULATION_DEFECT] else 1


if __name__ == "__main__":
    raise SystemExit(main())
