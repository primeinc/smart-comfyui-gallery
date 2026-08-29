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
import sys
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
from compat.harness import identity as evidence_identity
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


def run_ablation(
    runner: Runner, case: Case, retained: RetainedState, declared: Ablation, against: Artifact
) -> Ablation:
    """One primitive removed, and whether that actually broke anything.

    `against` is passed in rather than recomputed. It was being rebuilt on
    every ablation, which is wasteful -- a `kps_render` case ran full
    detection on a 4896x6528 photograph five times -- and quietly wrong in
    principle: an ablation compared against a freshly-derived baseline is not
    compared against the one the case's own verdict used, and if a baseline
    ever stopped being invariant nothing here would notice.
    """
    try:
        degraded = runner.ablate(case, retained, declared.primitive)
        produced = runner.replay(case, degraded)
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

    ablations = tuple(run_ablation(runner, case, retained, one, baseline) for one in case.ablations)
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


def build_runners() -> tuple[Runner, ...]:
    """Every consumer file that can run here.

    Imported rather than discovered by filename: a module that fails to import
    is a real condition, and swallowing it would silently shrink the
    population, which is the one thing this suite must never do.
    """
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

    # ReActor appears twice on purpose and the two are not redundant. The
    # family runner covers the embedding it extracts like every other
    # consumer; this one covers its ACTUAL boundary -- a safetensors file
    # written and read back by upstream's own code -- which is the only
    # first-party loader in the population.
    # `gallery_storage` is FIRST because everything after it is quoted against
    # it. Every other runner builds its retained state from the producer held
    # in memory, which proves a conditional whose antecedent -- that this
    # application durably keeps those values -- nothing else here tests.
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
        # The same family, over the VENDOR's own reference images rather
        # than our corpus. Case names carry a `vendor_` shot label, so
        # these do not collide with the corpus cases.
        *conformance_runners(),
        # Three recognition models over ONE aligned crop: whether the
        # stored vector serves consumers that use a different one.
        *space_runners(),
    )


def _without_timing(row: dict[str, Any]) -> dict[str, Any]:
    """One evidence row with the wall clock taken out."""
    return {key: value for key, value in row.items() if key != "seconds"}


def _without_seconds(out: dict[str, Any]) -> dict[str, Any]:
    """One run's evidence with the wall clock removed.

    A shard's partial carries the same shape as a whole run so the merge does
    not special-case it, minus timings for the same reason the whole run drops
    them: two identical runs must serialise to identical bytes.
    """
    return {key: value for key, value in out.items() if key != "seconds_by_case"}


def runners(only: str = "") -> tuple[Runner, ...]:
    """Every runner, or just the ones whose `consumer_id` matches `only`.

    Constructed LAZILY per selection rather than all at once. Sixteen lanes
    now load model packs in their constructors -- two insightface packs, the
    face family per vendor setup, three recognition models, facexlib, torch --
    and building every one of them in a single process exhausted memory and
    the run died with no traceback partway through model loading.

    Selecting is therefore an operational need, not a convenience: a lane that
    cannot be run on its own cannot be debugged either.
    """
    if not only:
        return build_runners()
    wanted = {one.strip() for one in only.split(",") if one.strip()}
    return tuple(one for one in build_runners() if one.consumer_id in wanted)


def canonical(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Case results in an order that belongs to the RESULTS, not the producer.

    Two entrypoints write this evidence and they disagreed about order:
    `run_all` emits in `runners()` order, while `sharded.merge` concatenates
    shard by shard in `SHARDS` order. Both are legitimate; neither is
    canonical; so the bytes of `cases.json` depended on which one wrote it.

    That is not a cosmetic difference. `attack.evidence_not_reproducible`
    re-runs the executor in-process and compares against the file on disk --
    which `just compat::cases` writes via the SHARDED path. It was therefore
    comparing two different orderings of the same results and reporting the
    harness as non-deterministic. The attack was right that the bytes did not
    match and wrong about why, and a green there would have been just as
    uninformative as the red.

    `(consumer_id, case)` is unique per row: `case` already carries the
    boundary label that distinguishes rows within a consumer.
    """
    return sorted(results, key=lambda one: (one["consumer_id"], one["case"]))


def run_all(only: str = "") -> dict[str, Any]:
    registry = Registry()
    results: list[CaseResult] = []
    for runner in runners(only):
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
        # Every answer-changing input, so a runner edited under unchanged pins
        # makes the evidence provably stale rather than silently wrong.
        "identity": evidence_identity.identity(),
        "cases": len(registry),
        # Timing is stripped from the EVIDENCE and reported beside it. It is a
        # property of the machine, not of the observation, and leaving it in
        # made the evidence non-reproducible by construction: two identical
        # runs produced different files, so "the evidence is unchanged" was
        # never a statement anyone could make. Now it is, and `attack.py`
        # asserts it.
        "results": canonical([_without_timing(asdict(one)) for one in results]),
        # Reported beside the evidence, never inside it. `main` moves this into
        # timings.json; `attack.py` re-runs the executor and asserts what is
        # left serialises to the same bytes, which is only a question that can
        # be asked once the wall clock is out of the way.
        "seconds_by_case": {one.case: round(one.seconds, 3) for one in results},
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


def main(argv: list[str] | None = None) -> int:
    """`python -m compat.harness.run [consumer_id[,consumer_id...]]`.

    With an argument the run is PARTIAL: it writes no evidence, because a
    cases.json covering one lane would read as a full pass over a shrunken
    population -- the exact failure `unexercised` exists to prevent.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    # `--json <path>` writes this shard's partial where the caller asks, so
    # `compat.harness.sharded` can merge several processes into one evidence
    # file. Without a path a selected run stays partial and writes nothing.
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
        return 0 if out["verdicts"][Verdict.DIVERGED.value] == 0 else 1

    # Timings leave the evidence and land beside it, so two identical runs
    # produce byte-identical cases.json and a diff of that file means
    # something. Before this, every run rewrote it with new wall clocks and
    # "the evidence is unchanged" was not a statement anyone could make.
    timings = {"runtime": out["runtime"], "seconds_by_case": out.pop("seconds_by_case")}
    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for name, body in (("cases.json", out), ("timings.json", timings)):
        target = generated / name
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
            handle.write("\n")
        print(f"wrote {target}")

    # The suite is green only when every case reproduced AND every consumer in
    # the frozen population was exercised. One unexercised consumer is a red
    # gate, not a footnote.
    clean = out["verdicts"][Verdict.DIVERGED.value] == 0 and out["verdicts"][Verdict.CONTRADICTED.value] == 0
    complete = not out["population"]["unexercised"]
    print(f"\ncases: {'clean' if clean else 'NOT clean'}   population: {'complete' if complete else 'INCOMPLETE'}")
    return 0 if (clean and complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
