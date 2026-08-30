"""Static discovery against what the run actually opened.

`population.py` reads the pinned source and finds what a loader COULD resolve.
`observe.py` records what a process DID resolve. Neither is the population on
its own, and the difference between them is the finding:

    AGREED             discovered statically and observed loading
    UNEXERCISED        discovered, never observed -- a variant nothing ran
    POPULATION_DEFECT  observed, never discovered -- discovery missed it

`Edge.dynamic_observed` existed as a field that nothing ever set, and
`observe.write` had no caller. So every edge read `dynamic_observed = False`,
which looked like a finding and was actually the absence of a measurement.

WHAT UNEXERCISED IS ALLOWED TO MEAN
-----------------------------------
It depends on the observer's own controls, and this reads them rather than
assuming either way. `_observer_complete` reports what `observe-attack`
measured on this run: while any probe is red, UNEXERCISED is "no evidence it
ran" and never "evidence it did not", because an observer with a blind spot
cannot be quoted on absence.

POPULATION_DEFECT does not depend on that, and the asymmetry is the reason: a
blind spot can only make the observer MISS an artifact, so anything it did see
and discovery did not is a hole in discovery either way. That is why it is the
only verdict here that reds the lane.
"""

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
    """One artifact identity, and how the two views of it compare."""

    identity: str
    verdict: str
    consumers: list[str] = field(default_factory=list)
    loaders: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    detail: str = ""


def _stems(identity: str) -> set[str]:
    """Every spelling one artifact answers to.

    Static discovery records what the SOURCE says -- `antelopev2`,
    `org/model`, `weights/det.onnx` -- and the observer records what the
    PROCESS resolved, which is a filename or a repository id. Matching them
    exactly would report a defect for every artifact whose two views spell it
    differently, so both sides are reduced to the same set of stems.
    """
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
    """The one population artifact this observation names, or the collision.

    A stem is a bare filename among other things, and the manifest carries two
    `models/image_encoder/model.safetensors` under different consumers. Taking
    the first would credit an observation of one to the other and leave the
    other reading UNEXERCISED, silently.
    """
    for stem in _stems(identity):
        holders = claimed.get(stem)
        if not holders:
            continue
        if len(holders) > 1:
            return "", holders
        return next(iter(holders)), set()
    return "", set()


def reconcile(population: dict[str, Any], observed: list[dict[str, Any]]) -> list[Reconciled]:
    """Every artifact from both sides, classified."""
    static: dict[str, Reconciled] = {}
    claimed: dict[str, set[str]] = {}
    for edge in population.get("edges", []):
        identity = str(edge.get("artifact_logical_identity", ""))
        # An UNRESOLVED row names no artifact -- it names the fact that one
        # could not be resolved -- so pairing it against an observation would
        # be matching a finding to a file.
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
        # A native-opener marker names a SYMBOL, not an artifact. Comparing it
        # against the population would report `_wfopen` as a model discovery
        # missed, which is the opposite of what it says.
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
        # The observer's own controls decide whether UNEXERCISED can be read as
        # absence. While any probe is red it cannot, and the number is carried
        # here so a reader of this file does not have to go and find out.
        "observer_is_complete": _observer_complete(generated, cases),
        "evidence_from_one_tree": trees,
    }


def _one_tree(population: dict[str, Any], cases: dict[str, Any]) -> dict[str, Any]:
    """Both sides describe the same tree, or the comparison means nothing.

    This lane's whole output is a difference between two artifacts. Nothing
    required them to be built under one tree, so a stale population against
    current observations would have read as discovery defects and unexercised
    variants that are neither.
    """
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
    """Whether an unobserved artifact may be read as one that did not load.

    Two things have to hold, and only one of them is about the observer. The
    other is that the cases ACTUALLY RAN: a run where every shard died records
    no observations at all, and reading that as "nothing loaded" turns a dead
    run into a finding about the artifacts. Observed here: six shards failed
    on one KeyError, and this reported twelve artifacts as not loaded.
    """
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

    # A POPULATION_DEFECT is a hole in discovery and reds this lane. An
    # UNEXERCISED row does not, while the observer is known incomplete: it
    # would be reporting the observer's blindness as the population's fault.
    return 0 if not totals[POPULATION_DEFECT] else 1


if __name__ == "__main__":
    raise SystemExit(main())
