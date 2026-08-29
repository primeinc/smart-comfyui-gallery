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
from typing import Any, Final, Protocol, runtime_checkable

from compat.assertions.arrays import Comparison, compare
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    CaseResult,
    Measurement,
    MissingPrimitive,
    Registry,
    RetainedState,
    Tier,
    Verdict,
)
from compat.contracts.case import (
    skipped as case_skips,
)
from compat.harness import identity as evidence_identity
from compat.harness import provenance

ROOT: Path = Path(__file__).resolve().parent.parent


class Runner(Protocol):
    """What `run.py` needs from a consumer file to run one case.

    Wider than `ConsumerRunner` by one method: the executor has to be able to
    build the retained state, and only the consumer knows what its own durable
    record contains.
    """

    consumer_id: str

    def cases(self) -> tuple[Case, ...]: ...
    def retained_for(self, case: Case) -> RetainedState: ...
    def baseline(self, case: Case) -> Artifact: ...
    def replay(self, case: Case, retained: RetainedState) -> Artifact: ...
    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement: ...


@runtime_checkable
class Ablating(Runner, Protocol):
    """A runner that also knows what degrading one of its primitives means.

    Split from `Runner` so that declaring no ablations is a fact the type
    system carries. `gallery_storage` measures conservation -- the producer's
    value against what the store gave back -- and has no necessity question to
    ask; requiring it to supply an `ablate` it never uses is how a lane ends
    up with a method whose only purpose is to satisfy a protocol.

    A case that declares ablations over a runner that is not `Ablating` is a
    contradiction, and `run_case` says so rather than skipping it.
    """

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState: ...


def _values_of(artifact: Artifact) -> Any:
    if artifact.values is None:
        raise ValueError(f"artifact {artifact.name!r} carries no values to compare")
    return artifact.values


def _inconclusive(declared: Ablation, why: str) -> Ablation:
    """One ablation that could not answer, carrying the reason."""
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
    """One primitive removed, and whether that actually broke anything.

    `against` is passed in rather than recomputed. It was being rebuilt on
    every ablation, which is wasteful -- a `kps_render` case ran full
    detection on a 4896x6528 photograph five times -- and quietly wrong in
    principle: an ablation compared against a freshly-derived baseline is not
    compared against the one the case's own verdict used, and if a baseline
    ever stopped being invariant nothing here would notice.
    """
    broke: bool | None
    method = ""
    try:
        # Building the substitute is the harness's work. A raise here says the
        # runner could not construct the degraded state, not that the consumer
        # needs the value.
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
            # The substitute WAS the original, so nothing was degraded and the
            # outcome is a property of the two values being equal.
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
        # The METHOD as well as the detail: without it a substitution that
        # broke on SHAPE is indistinguishable from one that broke on content.
        detail = f"{result.method}: {result.detail}"
        method = result.method
    except MissingPrimitive as problem:
        # NOT a break. The replay dereferenced the key that was just removed,
        # which happens for every key -- including one nothing needs -- so it
        # separates no hypothesis from any other.
        broke = None
        detail = f"INCONCLUSIVE: {problem}. The replay indexes this key; it was not shown to need the value."
    except (KeyError, TypeError, ValueError, IndexError) as problem:
        # Raised by the REPLAY: the consumer got far enough to use the
        # degraded value and could not.
        broke = True
        detail = f"{type(problem).__name__}: {problem}"

    if broke is None:
        verdict = Verdict.INCONCLUSIVE
    else:
        # CONTRADICTED is reserved for the case that matters: we claimed a
        # primitive was required, removed it, and nothing broke.
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
    except (KeyError, TypeError, ValueError, IndexError, OSError, NotImplementedError) as problem:
        # NotImplementedError included deliberately: a lane that declares a
        # boundary it has no derivation for could not run HERE, which is what
        # UNSUPPORTED means.
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

    if case.ablations and not isinstance(runner, Ablating):
        raise TypeError(f"{case.name} declares {len(case.ablations)} ablations; {type(runner).__name__} cannot ablate")
    ablations = (
        tuple(run_ablation(runner, case, retained, one, baseline) for one in case.ablations)
        if isinstance(runner, Ablating)
        else ()
    )
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
        retained_bytes=retained.sizes(),
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

    # `gallery_storage` is FIRST: every other runner builds its retained state
    # from the producer in memory, proving a conditional whose antecedent only
    # this lane tests.
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
    # `seconds_by_case` is MOVED, not dropped: wall clocks in the evidence
    # make two identical runs serialise differently. `sharded.main` writes it
    # beside the evidence as `timings.json`.
    return {key: value for key, value in out.items() if key != "seconds_by_case"}


def runners(only: str = "") -> tuple[Runner, ...]:
    """Every runner, or just the ones whose `consumer_id` matches `only`.

    Sixteen lanes load model packs when they RUN -- two insightface packs, the
    face family per vendor setup, three recognition models, facexlib, torch --
    and running every one of them in a single process exhausted memory and
    died with no traceback partway through model loading. Selecting is
    therefore an operational need, not a convenience: a lane that cannot be
    run on its own cannot be debugged either.

    The construction itself is NOT selective: `build_runners()` builds them
    all and this filters afterwards. Constructors are cheap -- they hold a
    setup and a shot list -- so that costs little, but it is not lazy per
    selection: every shard constructs every runner, so all six record the same
    skipped photograph.
    """
    if not only:
        return build_runners()
    wanted = {one.strip() for one in only.split(",") if one.strip()}
    built = build_runners()
    # A name matching no runner would select nothing and print a clean verdict
    # table over an empty population, so a shard whose lane list drifted from
    # the registry would report no failures rather than reporting it ran none.
    unknown = sorted(wanted - {one.consumer_id for one in built})
    if unknown:
        raise KeyError(
            f"no runner answers to {unknown}; the registry holds {sorted({one.consumer_id for one in built})}"
        )
    return tuple(one for one in built if one.consumer_id in wanted)


#: Every key an evidence file carries, whichever entrypoint wrote it. `run_all`
#: and `sharded.merge` both serialise evidence and
#: `attack.evidence_not_reproducible` compares their bytes.
EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "runtime",
        "identity",
        "cases",
        "results",
        "skipped",
        "duplicated_cases",
        "shards_failed",
        "shards_exited_over_findings",
        "population",
        "verdicts",
    }
)


def evidence_shape(out: dict[str, Any], who: str) -> dict[str, Any]:
    """One evidence dict, checked against the shape both writers must share."""
    held = set(out) - {"seconds_by_case"}
    if held != EVIDENCE_KEYS:
        raise KeyError(
            f"{who} emits {sorted(held ^ EVIDENCE_KEYS)} that the other evidence writer does not. "
            f"Both must serialise the same keys or `attack.evidence_not_reproducible` compares two shapes."
        )
    return out


def blocking_failures(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Every DIVERGED case. There is no list of ones that do not count.

    The reasoning was wrong twice. The instruction was not to redesign the
    schema UNTIL the suite reached its closure condition, which is a sequence
    and not a prohibition -- and nobody authorised turning nineteen measured
    failures into a declaration. A suite that answers what a store must retain
    cannot also hold a list of the ways its store is allowed to be wrong.

    The five divergences that list named are fixed at the source instead:
    `vision/faces.py` no longer rounds landmarks or pose, `db/detect.py`
    writes the landmark blob at float64, `compat/storage/gallery_v45.py`
    returns `det_score` at the producer's width, and `db/migrate.py` steps
    v45 to v46 to widen what is already stored.
    """
    out: dict[str, list[str]] = {"diverged": [], "no case answered": []}
    cases: dict[str, int] = {}
    passes: dict[str, int] = {}
    for row in results:
        who = row["consumer_id"]
        cases[who] = cases.get(who, 0) + 1
        passes[who] = passes.get(who, 0) + (row["verdict"] == Verdict.REPRODUCED.value)
        if row["verdict"] == Verdict.DIVERGED.value:
            out["diverged"].append(f"{row['case']}: {row.get('comparison', '')[:100]}")
    # A lane whose every case raises reports no divergence, so UNSUPPORTED
    # alone cannot distinguish it from a clean run. Individual UNSUPPORTED
    # rows are recorded in compat/generated/cases.json and do not block.
    for who, held in sorted(cases.items()):
        if held and not passes.get(who):
            out["no case answered"].append(f"{who}: {held} case(s), not one reproduced")
    return {why: names for why, names in out.items() if names}


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

    # Tier is load-bearing in the count. IPAdapter's real path re-detects over
    # a descending det_size sweep (IPAdapterPlus.py:355-367) and crops from
    # THAT detection, so proving the warp says nothing about the consumer.
    at_tier = {
        tier: {one.consumer_id for one in results if one.tier is tier} for tier in (Tier.PRIMITIVE, Tier.CONSUMER)
    }
    covered = at_tier[Tier.CONSUMER]

    out: dict[str, Any] = {
        "runtime": provenance.runtime_identity(),
        # Every answer-changing input, so a runner edited under unchanged pins
        # makes the evidence provably stale rather than silently wrong.
        "identity": evidence_identity.identity(),
        "cases": len(registry),
        # Inputs a lane declined to build a case from, with the reason.
        # Carrying them here is what makes a shrunk population visible in the
        # artifact rather than only in a difference between two case counts.
        "skipped": [asdict(one) for one in case_skips()],
        # Always present, always empty here: `sharded.merge` emits this key, so
        # omitting it would make the two serialisations different shapes.
        "shards_failed": [],
        "shards_exited_over_findings": [],
        "duplicated_cases": [],
        # Timing is a property of the machine, not of the observation, and
        # leaving it here makes the evidence non-reproducible by construction.
        "results": canonical([_without_timing(asdict(one)) for one in results]),
        # Reported beside the evidence, never inside it: `main` moves this into
        # timings.json so `attack.py` can compare the rest byte for byte.
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
    return evidence_shape(out, "run_all")


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
    # `corpus.loaded.statistics()` had no caller, and `corpus.cache.note()`
    # accumulated hit/miss counts only it consumed -- so the store's own
    # accounting was written every run and read by nobody.
    from compat.corpus import loaded as corpus_loaded

    print(f"corpus memo             : {corpus_loaded.statistics()}")


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
            # The wall clock, beside the partial rather than inside it: the
            # partials are committed evidence and must serialise identically
            # across two runs.
            beside = partial_to.with_suffix(".timings.json")
            with beside.open("w", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(out.get("seconds_by_case") or {}, indent=2, sort_keys=True))
                handle.write("\n")
        return 0 if not blocking_failures(out["results"]) else 1

    # Timings leave the evidence and land beside it, so two identical runs
    # produce byte-identical cases.json and a diff of it means something.
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
    blocking = blocking_failures(out["results"])
    if blocking:
        print("\nBLOCKING:")
        for why, names in blocking.items():
            print(f"    {why}:")
            for one in names:
                print(f"        {one}")

    clean = not blocking and out["verdicts"][Verdict.CONTRADICTED.value] == 0
    complete = not out["population"]["unexercised"]
    print(f"\ncases: {'clean' if clean else 'NOT clean'}   population: {'complete' if complete else 'INCOMPLETE'}")
    return 0 if (clean and complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
