"""Execute every case: baseline, replay, ablations, evidence.

One case runs three ways.

    BASELINE   original media -> pinned upstream path -> boundary artifact
    REPLAY     retained state -> reconstructed path   -> the same artifact
    ABLATION   replay again, one primitive removed

A replay matching the baseline proves the retained state is SUFFICIENT. It
says nothing about whether all of it was needed, which is what the ablations
are for: a primitive is durable truth only when removing it actually breaks
the replay. If the replay still matches without it, the primitive was
derivable from what remained and the verdict is CONTRADICTED -- a passing
ablation is a failing necessity claim.

A runner that raises during an ablation has BROKEN, which is the expected
outcome; the exception is recorded as the evidence. A runner that raises
during the un-ablated replay has failed outright.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from compat.assertions.arrays import Comparison, compare
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    CaseResult,
    Measurement,
    Registry,
    RetainedState,
    Tier,
    Verdict,
)
from compat.harness import provenance

ROOT: Path = Path(__file__).resolve().parent.parent


class Runner(Protocol):
    """What `run.py` needs from a consumer file.

    Wider than `ConsumerRunner` by two methods: the executor has to be able to
    build the retained state and to ablate it, and only the consumer knows what
    removing one of its primitives means.
    """

    consumer_id: str

    def cases(self) -> tuple[Case, ...]: ...
    def retained_for(self, case: Case) -> RetainedState: ...
    def baseline(self, case: Case) -> Artifact: ...
    def replay(self, case: Case, retained: RetainedState) -> Artifact: ...
    def ablate(self, case: Case, retained: RetainedState, primitive: str) -> RetainedState: ...
    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement: ...


def _values_of(artifact: Artifact) -> Any:
    if artifact.values is None:
        raise ValueError(f"artifact {artifact.name!r} carries no values to compare")
    return artifact.values


def run_ablation(runner: Runner, case: Case, retained: RetainedState, declared: Ablation) -> Ablation:
    """One primitive removed, and whether that actually broke anything."""
    try:
        degraded = runner.ablate(case, retained, declared.primitive)
        produced = runner.replay(case, degraded)
        against = runner.baseline(case)
        result: Comparison = compare(
            _values_of(against),
            _values_of(produced),
            exact_bytes=case.exact_bytes,
            rtol=case.rtol,
            atol=case.atol,
        )
        broke = not result.equal
        detail = result.detail
    except (KeyError, TypeError, ValueError, IndexError) as problem:
        # The replay could not proceed without the primitive. That IS the
        # break, and the exception is the evidence for it.
        broke = True
        detail = f"{type(problem).__name__}: {problem}"

    # CONTRADICTED is reserved for the case that matters: we claimed a
    # primitive was required, removed it, and nothing broke.
    verdict = Verdict.REPRODUCED if broke == declared.expect_breaks else Verdict.CONTRADICTED
    return Ablation(
        primitive=declared.primitive,
        expect_breaks=declared.expect_breaks,
        kind=declared.kind,
        observed_break=broke,
        verdict=verdict,
        detail=detail,
    )


def run_measurement(runner: Runner, case: Case, retained: RetainedState, name: str) -> Measurement:
    """One searched quantity, or the reason it could not be searched.

    A measurement that raises is recorded as a measurement with no value, not
    as a case failure: the sufficiency claim and the search for a minimum are
    separate questions, and a broken search must not be able to turn a
    reproduced replay into a divergence.
    """
    try:
        return runner.measure(case, retained, name)
    except (KeyError, TypeError, ValueError, IndexError) as problem:
        return Measurement(name=name, unit="", value=None, detail=f"{type(problem).__name__}: {problem}")


def run_case(runner: Runner, case: Case) -> CaseResult:
    """Baseline, replay, then every declared ablation."""
    began = time.perf_counter()
    try:
        baseline = runner.baseline(case)
        retained = runner.retained_for(case)
        replayed = runner.replay(case, retained)
    except (KeyError, TypeError, ValueError, IndexError, OSError) as problem:
        return CaseResult(
            case=case.name,
            consumer_id=case.consumer_id,
            tier=case.tier,
            verdict=Verdict.UNSUPPORTED,
            fixture_sha256=case.fixture.sha256,
            unsupported_reason=f"{type(problem).__name__}: {problem}",
            seconds=time.perf_counter() - began,
        )

    result = compare(
        _values_of(baseline),
        _values_of(replayed),
        exact_bytes=case.exact_bytes,
        rtol=case.rtol,
        atol=case.atol,
    )

    ablations = tuple(run_ablation(runner, case, retained, one) for one in case.ablations)
    measurements = tuple(run_measurement(runner, case, retained, one) for one in case.measurements)

    # A case whose replay matches but whose necessity claim is contradicted is
    # NOT a pass: the retained state is sufficient and too large, and reporting
    # that as success is how unnecessary state becomes permanent.
    verdict = Verdict.REPRODUCED if result.equal else Verdict.DIVERGED
    if verdict is Verdict.REPRODUCED and any(one.verdict is Verdict.CONTRADICTED for one in ablations):
        verdict = Verdict.CONTRADICTED

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
        ablations=ablations,
        measurements=measurements,
        seconds=time.perf_counter() - began,
    )


def runners() -> tuple[Runner, ...]:
    """Every consumer file that can run here.

    Imported rather than discovered by filename: a module that fails to import
    is a real condition, and swallowing it would silently shrink the
    population, which is the one thing this suite must never do.
    """
    from compat.consumers.aligned_crop import AlignedCropRunner
    from compat.consumers.consisid_facexlib import ConsisIDRunner
    from compat.consumers.face_family import all_runners
    from compat.consumers.other_media import all_runners as media_runners
    from compat.consumers.producer_derivations import ProducerDerivationRunner
    from compat.consumers.reactor_face_model import ReactorFaceModelRunner
    from compat.consumers.whole_reference import all_runners as whole_runners

    # ReActor appears twice on purpose and the two are not redundant. The
    # family runner covers the embedding it extracts like every other
    # consumer; this one covers its ACTUAL boundary -- a safetensors file
    # written and read back by upstream's own code -- which is the only
    # first-party loader in the population.
    return (
        AlignedCropRunner(),
        ProducerDerivationRunner(),
        ReactorFaceModelRunner(),
        ConsisIDRunner(),
        *all_runners(),
        *whole_runners(),
        *media_runners(),
    )


def run_all() -> dict[str, Any]:
    registry = Registry()
    results: list[CaseResult] = []
    for runner in runners():
        cases = runner.cases()
        registry.extend(cases)
        results.extend(run_case(runner, one) for one in cases)

    manifest = provenance.load_manifest()
    declared = {one["id"] for one in manifest.get("consumers", [])}

    # Tier is load-bearing in the count, not just a label. A PRIMITIVE case
    # proves one transform -- `norm_crop` at a size -- and a CONSUMER case
    # proves the path that consumer actually takes. Those are not the same
    # claim: IPAdapter's real path re-detects over a descending det_size
    # sweep (IPAdapterPlus.py:355-367) and crops from THAT detection's
    # keypoints, so proving the warp says nothing about whether the consumer
    # reproduces. Counting a primitive as consumer coverage is how a suite
    # reports 1 of 22 as though the one were finished.
    at_tier = {
        tier: {one.consumer_id for one in results if one.tier is tier} for tier in (Tier.PRIMITIVE, Tier.CONSUMER)
    }
    covered = at_tier[Tier.CONSUMER]

    return {
        "runtime": provenance.runtime_identity(),
        "cases": len(registry),
        "results": [asdict(one) for one in results],
        "population": {
            "declared": sorted(declared),
            "consumer_tier_covered": sorted(covered & declared),
            "primitive_tier_only": sorted(at_tier[Tier.PRIMITIVE] - covered),
            # Anything declared and not covered AT CONSUMER TIER stays
            # visible. A consumer that disappears when nothing runs it is how
            # a suite reports success it did not earn.
            "unexercised": sorted(declared - covered),
        },
        "verdicts": {one.value: sum(1 for result in results if result.verdict is one) for one in Verdict},
    }


def report(out: dict[str, Any]) -> None:
    for row in out["results"]:
        print(f"{row['verdict']:<13} {row['case']:<38} {row['comparison']}")
        if row["unsupported_reason"]:
            print(f"    -- {row['unsupported_reason']}")
        for one in row["ablations"]:
            mark = "ok " if one["verdict"] == Verdict.REPRODUCED.value else "!! "
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


def main() -> int:
    out = run_all()
    report(out)

    where = ROOT / "generated" / "cases.json"
    where.parent.mkdir(parents=True, exist_ok=True)
    with where.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"\nwrote {where}")

    # The suite is green only when every case reproduced AND every consumer in
    # the frozen population was exercised. One unexercised consumer is a red
    # gate, not a footnote.
    clean = out["verdicts"][Verdict.DIVERGED.value] == 0 and out["verdicts"][Verdict.CONTRADICTED.value] == 0
    complete = not out["population"]["unexercised"]
    print(f"\ncases: {'clean' if clean else 'NOT clean'}   population: {'complete' if complete else 'INCOMPLETE'}")
    return 0 if (clean and complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
