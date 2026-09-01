from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from compat.assertions.arrays import Comparison, compare
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    CaseResult,
    CaseVerdict,
    Measurement,
    MissingPrimitive,
    Registry,
    RetainedState,
    Tier,
    Verdict,
)
from compat.contracts.case import (
    considered as case_considered,
)
from compat.contracts.case import (
    skipped as case_skips,
)
from compat.harness import failfast, observe, provenance
from compat.harness import identity as evidence_identity

ROOT: Path = Path(__file__).resolve().parent.parent


class Runner(Protocol):
    consumer_id: str

    def cases(self) -> tuple[Case, ...]: ...
    def retained_for(self, case: Case) -> RetainedState: ...
    def baseline(self, case: Case) -> Artifact: ...
    def replay(self, case: Case, retained: RetainedState) -> Artifact: ...
    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement: ...


@runtime_checkable
class Ablating(Runner, Protocol):
    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState: ...


def _values_of(artifact: Artifact) -> Any:
    if artifact.values is None:
        raise ValueError(f"artifact {artifact.name!r} carries no values to compare")
    return artifact.values


def _inconclusive(declared: Ablation, why: str) -> Ablation:
    return Ablation(
        primitive=declared.primitive,
        expect_breaks=declared.expect_breaks,
        kind=declared.kind,
        swap=declared.swap,
        observed_break=None,
        verdict=Verdict.INCONCLUSIVE,
        detail=f"INCONCLUSIVE: {why}",
    )


def run_ablation(
    runner: Ablating, case: Case, retained: RetainedState, declared: Ablation, against: Artifact
) -> Ablation:
    broke: bool | None
    method = ""
    try:
        degraded = runner.ablate(case, retained, declared)
    except MissingPrimitive as problem:
        return _inconclusive(
            declared,
            f"{problem}. The replay indexes this key; it was not shown to need the value.",
        )
    except (KeyError, TypeError, ValueError, IndexError) as problem:
        return _inconclusive(
            declared,
            f"the runner could not build the {declared.swap or declared.primitive!r} state: "
            f"{type(problem).__name__}: {problem}",
        )

    try:
        if declared.kind == "substitution" and degraded.same_as(retained):
            return _inconclusive(
                declared,
                f"the {declared.swap!r} substitute is identical to the retained {declared.primitive!r}; "
                f"no degradation was applied and nothing could be observed",
            )
        produced = runner.replay(case, degraded)
        result: Comparison = compare(
            _values_of(against),
            _values_of(produced),
            exact_bytes=case.exact_bytes,
            rtol=case.rtol,
            atol=case.atol,
        )
        broke = not result.equal

        detail = f"{result.method}: {result.detail}"
        method = result.method
    except MissingPrimitive as problem:
        broke = None
        detail = f"INCONCLUSIVE: {problem}. The replay indexes this key; it was not shown to need the value."
    except (KeyError, TypeError, ValueError, IndexError) as problem:
        broke = True
        detail = f"{type(problem).__name__}: {problem}"

    if broke is None:
        verdict = Verdict.INCONCLUSIVE
    else:
        verdict = Verdict.REPRODUCED if broke == declared.expect_breaks else Verdict.CONTRADICTED
    return Ablation(
        primitive=declared.primitive,
        expect_breaks=declared.expect_breaks,
        kind=declared.kind,
        compare_method=method,
        swap=declared.swap,
        observed_break=broke,
        verdict=verdict,
        detail=detail,
    )


def run_measurement(runner: Runner, case: Case, retained: RetainedState, name: str) -> Measurement:
    try:
        return runner.measure(case, retained, name)
    except (KeyError, TypeError, ValueError, IndexError) as problem:
        return Measurement(name=name, unit="", value=None, detail=f"{type(problem).__name__}: {problem}")


def run_case(runner: Runner, case: Case) -> CaseResult:
    began = time.perf_counter()

    try:
        baseline = runner.baseline(case)
        retained = runner.retained_for(case)
        replayed = runner.replay(case, retained)
    except (KeyError, TypeError, ValueError, IndexError, OSError, NotImplementedError) as problem:
        return CaseResult(
            case=case.name,
            consumer_id=case.consumer_id,
            tier=case.tier,
            verdict=CaseVerdict.DIVERGED,
            fixture_sha256=case.fixture.sha256,
            comparison=f"{type(problem).__name__}: {problem}",
            seconds=time.perf_counter() - began,
        )

    result = compare(
        _values_of(baseline),
        _values_of(replayed),
        exact_bytes=case.exact_bytes,
        rtol=case.rtol,
        atol=case.atol,
    )

    if case.ablations and not isinstance(runner, Ablating):
        raise TypeError(f"{case.name} declares {len(case.ablations)} ablations; {type(runner).__name__} cannot ablate")
    ablations = (
        tuple(run_ablation(runner, case, retained, one, baseline) for one in case.ablations)
        if isinstance(runner, Ablating)
        else ()
    )
    measurements = tuple(run_measurement(runner, case, retained, one) for one in case.measurements)

    verdict = CaseVerdict.REPRODUCED if result.equal else CaseVerdict.DIVERGED

    return CaseResult(
        case=case.name,
        consumer_id=case.consumer_id,
        tier=case.tier,
        verdict=verdict,
        fixture_sha256=case.fixture.sha256,
        baseline=Artifact(baseline.name, baseline.dtype, baseline.shape, baseline.sha256),
        replay=Artifact(replayed.name, replayed.dtype, replayed.shape, replayed.sha256),
        comparison=f"{result.method}: {result.detail}",
        max_abs_diff=result.max_abs_diff,
        retained_bytes=retained.sizes(),
        ablations=ablations,
        measurements=measurements,
        seconds=time.perf_counter() - began,
    )


def build_runners() -> tuple[Runner, ...]:
    from compat.consumers.aligned_crop import AlignedCropRunner
    from compat.consumers.consisid_facexlib import ConsisIDRunner
    from compat.consumers.control_stream import all_runners as control_runners
    from compat.consumers.embedding_spaces import all_runners as space_runners
    from compat.consumers.face_family import all_runners
    from compat.consumers.face_selection import all_runners as selection_runners
    from compat.consumers.gallery_storage import all_runners as storage_runners
    from compat.consumers.masked_reference import all_runners as masked_runners
    from compat.consumers.other_media import all_runners as media_runners
    from compat.consumers.producer_derivations import ProducerDerivationRunner
    from compat.consumers.reactor_face_model import ReactorFaceModelRunner
    from compat.consumers.reference_sets import all_runners as refset_runners
    from compat.consumers.whole_reference import all_runners as whole_runners
    from compat.vendor.conformance import all_runners as conformance_runners

    return (
        *storage_runners(),
        AlignedCropRunner(),
        ProducerDerivationRunner(),
        ReactorFaceModelRunner(),
        ConsisIDRunner(),
        *all_runners(),
        *whole_runners(),
        *media_runners(),
        *control_runners(),
        *selection_runners(),
        *refset_runners(),
        *masked_runners(),
        *conformance_runners(),
        *space_runners(),
    )


def _without_timing(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "seconds"}


def _without_seconds(out: dict[str, Any]) -> dict[str, Any]:

    return {key: value for key, value in out.items() if key != "seconds_by_case"}


def runners(only: str = "") -> tuple[Runner, ...]:
    if not only:
        return build_runners()
    wanted = {one.strip() for one in only.split(",") if one.strip()}
    built = build_runners()

    unknown = sorted(wanted - {one.consumer_id for one in built})
    if unknown:
        raise KeyError(
            f"no runner answers to {unknown}; the registry holds {sorted({one.consumer_id for one in built})}"
        )
    return tuple(one for one in built if one.consumer_id in wanted)


EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "runtime",
        "identity",
        "cases",
        "results",
        "skipped",
        "considered",
        "observed",
        "duplicated_cases",
        "shards_failed",
        "shards_exited_over_findings",
        "population",
        "verdicts",
        "ablation_verdicts",
    }
)


def evidence_shape(out: dict[str, Any], who: str) -> dict[str, Any]:
    held = set(out) - {"seconds_by_case"}
    if held != EVIDENCE_KEYS:
        raise KeyError(
            f"{who} emits {sorted(held ^ EVIDENCE_KEYS)} that the other evidence writer does not. "
            f"Both must serialise the same keys or `attack.evidence_not_reproducible` compares two shapes."
        )
    return out


def blocking_failures(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"diverged": [], "no case answered": []}
    cases: dict[str, int] = {}
    passes: dict[str, int] = {}
    for row in results:
        who = row["consumer_id"]
        cases[who] = cases.get(who, 0) + 1
        passes[who] = passes.get(who, 0) + (row["verdict"] == CaseVerdict.REPRODUCED.value)
        if row["verdict"] == CaseVerdict.DIVERGED.value:
            out["diverged"].append(f"{row['case']}: {row.get('comparison', '')[:100]}")

    for who, held in sorted(cases.items()):
        if held and not passes.get(who):
            out["no case answered"].append(f"{who}: {held} case(s), not one reproduced")
    return {why: names for why, names in out.items() if names}


#: Anything the Verdict members do not classify. Without it the tally silently
#: dropped null and unrecognised verdicts and summed to less than it counted.
UNCLASSIFIED: Final[str] = "UNCLASSIFIED"


def ablation_tally(results: list[dict[str, Any]]) -> dict[str, int]:
    held = [one for row in results for one in (row.get("ablations") or [])]
    known = {one.value for one in Verdict}
    out = {one.value: sum(1 for a in held if a.get("verdict") == one.value) for one in Verdict}
    out[UNCLASSIFIED] = sum(1 for a in held if a.get("verdict") not in known)
    if sum(out.values()) != len(held):
        raise ValueError(f"ablation tally sums to {sum(out.values())} over {len(held)} ablations")
    return out


def canonical(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda one: (one["consumer_id"], one["case"]))


def run_all(only: str = "") -> dict[str, Any]:
    registry = Registry()
    results: list[CaseResult] = []

    with observe.recording() as watching:
        for runner in runners(only):
            cases = runner.cases()
            registry.extend(cases)
            results.extend(run_case(runner, one) for one in cases)
    observed = watching.rows()

    manifest = provenance.load_manifest()
    declared = {one["id"] for one in manifest.get("consumers", [])}

    at_tier = {
        tier: {one.consumer_id for one in results if one.tier is tier} for tier in (Tier.PRIMITIVE, Tier.CONSUMER)
    }
    covered = at_tier[Tier.CONSUMER]

    out: dict[str, Any] = {
        "runtime": provenance.runtime_identity(),
        "identity": evidence_identity.identity(),
        "cases": len(registry),
        "skipped": [asdict(one) for one in case_skips()],
        "considered": [asdict(one) for one in case_considered()],
        "observed": observed,
        "shards_failed": [],
        "shards_exited_over_findings": [],
        "duplicated_cases": [],
        "results": canonical([_without_timing(asdict(one)) for one in results]),
        "seconds_by_case": {one.case: round(one.seconds, 3) for one in results},
        "population": {
            "declared": sorted(declared),
            "consumer_tier_covered": sorted(covered & declared),
            "primitive_tier_only": sorted(at_tier[Tier.PRIMITIVE] - covered),
            "unexercised": sorted(declared - covered),
        },
        "verdicts": {one.value: sum(1 for result in results if result.verdict is one) for one in CaseVerdict},
        "ablation_verdicts": ablation_tally([_without_timing(asdict(one)) for one in results]),
    }
    return evidence_shape(out, "run_all")


def report(out: dict[str, Any]) -> None:
    for row in out["results"]:
        print(f"{row['verdict']:<13} {row['case']:<38} {row['comparison']}")
        for one in row["ablations"]:
            mark = "ok " if one["verdict"] == CaseVerdict.REPRODUCED.value else "!! "
            expected = "breaks" if one["expect_breaks"] else "survives"
            observed = "broke" if one["observed_break"] else "survived"
            print(
                f"    {mark}ablate {one['primitive']:<22} expect {expected:<8} "
                f"observed {observed:<9} {one['detail'][:70]}"
            )
        for one in row["measurements"]:
            print(f"    == measure {one['name']:<21} {one['detail']}")

    print(f"\nverdicts: {out['verdicts']}")
    pop = out["population"]
    print(f"declared consumers      : {len(pop['declared'])}")
    print(f"covered at CONSUMER tier: {len(pop['consumer_tier_covered'])}  {pop['consumer_tier_covered']}")
    print(f"primitive tier only     : {len(pop['primitive_tier_only'])}  {pop['primitive_tier_only']}")
    print(f"NOT exercised           : {len(pop['unexercised'])}")

    from compat.corpus import loaded as corpus_loaded

    print(f"corpus memo             : {corpus_loaded.statistics()}")


def main(argv: list[str] | None = None) -> int:
    failfast.arm()
    args = list(argv if argv is not None else sys.argv[1:])

    partial_to: Path | None = None
    if "--json" in args:
        at = args.index("--json")
        partial_to = Path(args[at + 1])
        del args[at : at + 2]
    only = " ".join(args).strip()

    out = run_all(only)
    report(out)
    if only:
        if partial_to is None:
            print(f"\nPARTIAL RUN ({only}): no evidence written. Run the whole population for that.")
        else:
            partial_to.parent.mkdir(parents=True, exist_ok=True)
            with partial_to.open("w", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(_without_seconds(out), indent=2, sort_keys=True, default=str))
                handle.write("\n")
            print(f"wrote partial {partial_to}")

            beside = partial_to.with_suffix(".timings.json")
            with beside.open("w", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(out.get("seconds_by_case") or {}, indent=2, sort_keys=True))
                handle.write("\n")
        return 0 if not blocking_failures(out["results"]) else 1

    timings = {"runtime": out["runtime"], "seconds_by_case": out.pop("seconds_by_case")}
    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for name, body in (("cases.json", out), ("timings.json", timings)):
        target = generated / name
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
            handle.write("\n")
        print(f"wrote {target}")

    blocking = blocking_failures(out["results"])
    if blocking:
        print("\nBLOCKING:")
        for why, names in blocking.items():
            print(f"    {why}:")
            for one in names:
                print(f"        {one}")

    # Was `out["verdicts"][CONTRADICTED]`, which run_case cannot assign: constant
    # true. The ablation tally is where CONTRADICTED and INCONCLUSIVE actually land.
    unsettled = out["ablation_verdicts"]
    clean = not blocking and not unsettled.get(Verdict.CONTRADICTED.value, 0) and not unsettled.get(UNCLASSIFIED, 0)
    complete = not out["population"]["unexercised"]
    print(f"\ncases: {'clean' if clean else 'NOT clean'}   population: {'complete' if complete else 'INCOMPLETE'}")
    return 0 if (clean and complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
