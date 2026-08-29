"""Corrupt the harness deliberately and assert it notices.

A gate that has never gone red is not known to work. `attack.py` covers pins,
blobs, weights and evidence bytes; these cover the mechanisms added since:

    positive_control            change NOTHING           -> no drift
    missing_consumer            drop a declared consumer -> population INCOMPLETE
    changed_pin                 mutate a commit          -> identity drift
    changed_weight              mutate a weight digest   -> identity drift
    changed_manifest            mutate the manifest      -> identity drift
    changed_runner_source       edit a runner's bytes    -> identity drift
    changed_application_source  edit vision/ or db/      -> identity drift
    changed_corpus              re-hash a photograph     -> identity drift
    dropped_persisted_primitive remove a stored value    -> replay breaks
    changed_selection_rule      swap first/largest       -> different face
    changed_reference_order     reverse a stacked set    -> different artifact
    stale_evidence              old identity vs current  -> drift reported
    changed_vendor_fixture      same path, other bytes   -> different sha256
    vendor_acceptance_not_faked accepted set must equal the set that RAN
    vendor_round_trip_lossless  plant one lost key       -> aggregate goes red
    vendor_boundary_repeats     plant a second digest    -> stability goes red
    vendor_pack_declaration_checked  contradict the declared pack -> disagreement
    cache_never_crosses_identity     entry under another identity -> not served
    cache_preserves_every_value_type round-trip a Face            -> types survive

The positive control is not decoration. Every other row asserts a mutation IS
seen; without a row asserting an unmutated tree is seen as clean, a detector
that returned "changed" unconditionally would pass every one of them.

WHAT EACH ROW ACTUALLY EXERCISES
--------------------------------
Stated because it was not, and the difference decides what a green here is
worth.

    through the production function
        missing_consumer                 matrices.build
        dropped_persisted_primitive      RetainedState._require + run_ablation
        vendor_pack_declaration_checked  acceptance.declared_against_observed
        changed_selection_rule           the selection rules themselves
        changed_reference_order          reference_sets.combine
        cache_never_crosses_identity     corpus.cache read and write paths
        cache_preserves_every_value_type the same, over a real Face
        changed_vendor_fixture           re-reads the bytes and re-hashes them

    through `evidence.compare_to`, over a mutated copy of `identity()`
        positive_control, changed_pin, changed_weight, changed_manifest,
        changed_runner_source, changed_application_source, changed_corpus,
        stale_evidence
        -- the drift checker is real; no file is touched and no lane is run.

    over the RECORDED ARTIFACT only, re-implementing the aggregation
        vendor_acceptance_not_faked, vendor_round_trip_lossless,
        vendor_boundary_repeats
        -- these assert the artifact is internally consistent. They do NOT
        call `acceptance.py`'s aggregation, so a hardcoded aggregate would
        pass them. `determinism` now re-runs each vendor in a fresh
        interpreter, which a selftest cannot afford to call; the honest
        position is that these three are artifact checks and are labelled as
        such rather than counted as gate coverage.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.harness import identity as evidence

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


@dataclass
class Attack:
    """One deliberate corruption and whether the harness saw it."""

    name: str
    mechanism: str
    detected: bool
    detail: str
    applicable: bool = True
    """False when the attack cannot be mounted HERE -- the cache is off, a
    corpus is absent -- as opposed to mounted and missed. `ok` passes an
    inapplicable row, because a suite whose selftest reds when its own cache
    is disabled cannot be run cache-free, and cache-free is the control
    condition `attack.evidence_not_reproducible` now depends on."""

    @property
    def ok(self) -> bool:
        return self.detected or not self.applicable


def _tampered(mutate: Callable[[dict[str, Any]], None]) -> tuple[bool, str]:
    """Apply `mutate` to a copy of the current identity and diff it back.

    The digest is recomputed through `evidence.digest_of`, the same function
    `identity()` uses. Leaving the honest digest in place would make every row
    "detect" a mismatch it did not cause.
    """
    tampered = copy.deepcopy(evidence.identity())
    mutate(tampered)
    tampered["digest"] = evidence.digest_of(tampered)
    drift = evidence.compare_to(tampered)
    return bool(drift), "; ".join(drift[:3]) or "no drift reported"


def positive_control() -> Attack:
    """Change nothing. Drift MUST be empty.

    `detected` is inverted here on purpose: it is True when the checker
    reports CLEAN, because a checker that cries wolf on an untouched tree is
    as broken as one that sleeps through a real change.
    """
    drift = evidence.compare_to(evidence.identity())
    return Attack(
        name="positive_control",
        mechanism="nothing changed; the checker must report clean",
        detected=not drift,
        detail="clean" if not drift else f"FALSE POSITIVE: {'; '.join(drift[:3])}",
    )


def changed_pin() -> Attack:
    def mutate(held: dict[str, Any]) -> None:
        held["repos"][next(iter(held["repos"]))] = "0" * 40

    seen, detail = _tampered(mutate)
    return Attack("changed_pin", "one pinned commit replaced with zeros", seen, detail)


def changed_weight() -> Attack:
    if not evidence.identity()["weights"]:
        return Attack("changed_weight", "no weights declared", False, "nothing to corrupt")

    def mutate(held: dict[str, Any]) -> None:
        held["weights"][next(iter(held["weights"]))] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("changed_weight", "one weight digest replaced with zeros", seen, detail)


def changed_manifest() -> Attack:
    def mutate(held: dict[str, Any]) -> None:
        held["manifest"] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("changed_manifest", "manifest digest altered", seen, detail)


def changed_runner_source() -> Attack:
    def mutate(held: dict[str, Any]) -> None:
        key = next(one for one in held["sources"] if one.startswith("consumers/") and not one.endswith("__init__.py"))
        held["sources"][key] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("changed_runner_source", "one consumer's source digest altered", seen, detail)


def stale_evidence() -> Attack:
    """Evidence recorded before a harness edit must read as stale."""

    def mutate(held: dict[str, Any]) -> None:
        key = next(one for one in held["sources"] if one.startswith("harness/") and not one.endswith("__init__.py"))
        held["sources"][key] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("stale_evidence", "evidence recorded against an older harness source", seen, detail)


def missing_consumer() -> Attack:
    """Drop a declared consumer and re-derive the population.

    Reads the generated evidence rather than re-running: the claim under test
    is that `unexercised` is computed from the FROZEN manifest, not from
    whatever happened to run.
    """
    cases = ROOT / "generated" / "cases.json"
    if not cases.is_file():
        return Attack("missing_consumer", "no cases.json", False, "run the case lane first")
    held: dict[str, Any] = json.loads(cases.read_text(encoding="utf-8"))
    declared = set(held["population"]["declared"])
    covered = set(held["population"]["consumer_tier_covered"])
    if not covered:
        return Attack("missing_consumer", "no consumer-tier coverage", False, "nothing to drop")
    dropped = min(covered)
    # Through `matrices.build`, not set arithmetic: asserting
    # `declared - (covered - {dropped})` non-empty holds for any non-empty
    # `covered` and calls none of the code under test.
    from compat.harness import matrices

    thinned = copy.deepcopy(held)
    thinned["results"] = [one for one in thinned["results"] if one["consumer_id"] != dropped]
    thinned["population"]["consumer_tier_covered"] = sorted(covered - {dropped})
    thinned["population"]["unexercised"] = sorted(declared - (covered - {dropped}))
    generated = ROOT / "generated"
    built = matrices.build(
        matrices.Evidence(
            cases=thinned,
            provenance=json.loads((generated / "provenance.json").read_text(encoding="utf-8")),
            producer=json.loads((generated / "producer_inventory.json").read_text(encoding="utf-8")),
            union={},
        )
    )
    row = next((one for one in built["consumers"] if one["consumer"] == dropped), None)
    seen = row is not None and row["status"] == "NOT EXERCISED" and built["totals"]["not_exercised"] > 0
    return Attack(
        "missing_consumer",
        f"{dropped} dropped, then rebuilt through matrices.build",
        seen,
        f"{dropped} reads {row['status'] if row else 'ABSENT FROM THE TABLE'}; "
        f"not_exercised={built['totals']['not_exercised']}",
    )


def dropped_persisted_primitive() -> Attack:
    """A replay that merely INDEXES a removed key must not read as necessity.

    The old row asserted that a dict raises on a missing key. That is a
    property of dicts, and asserting it is how 22 of 23 primitives came to
    report `survives: 0`: `run_ablation` caught the KeyError and recorded it
    as a break, so `answer.json` listed pose, age, gender, bbox, det_score,
    kps and landmark_2d_106 as durable state on the strength of a dict lookup.

    The assertion that matters now is the one that replaced it: `_require`
    raises the typed `MissingPrimitive`, and `run_ablation` turns THAT into
    INCONCLUSIVE rather than into evidence. Both halves are checked, because a
    typed exception nobody special-cases is the same bug with a longer name.
    """
    from compat.contracts.case import (
        Ablation,
        Artifact,
        Case,
        Fixture,
        Measurement,
        MissingPrimitive,
        RetainedState,
        Tier,
        Verdict,
    )
    from compat.harness.run import run_ablation

    kps = np.zeros((5, 2), dtype=np.float32)
    typed = False
    try:
        RetainedState(kps=kps).without("kps").points("kps")
    except MissingPrimitive:
        typed = True
    except KeyError:
        typed = False

    def art(values: np.ndarray) -> Artifact:
        return Artifact(name="out", dtype=str(values.dtype), shape=values.shape, sha256="x", values=values)

    class IndexesTheKey:
        """The shape found in twelve consumer modules, reduced to its essence.

        Every member of `Ablating` is present because the executor's own
        protocol is what this row is testing against; a stub that satisfied
        only the two methods it uses would be checked against a weaker
        contract than a real runner is.
        """

        consumer_id = "selftest"

        def cases(self) -> tuple[Case, ...]:
            return ()

        def retained_for(self, case: Case) -> RetainedState:
            del case
            return RetainedState(kps=kps)

        def baseline(self, case: Case) -> Artifact:
            del case
            return art(kps)

        def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
            del case, retained
            raise KeyError(name)

        def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
            del case
            return retained.without(ablation.primitive)

        def replay(self, case: Case, retained: RetainedState) -> Artifact:
            del case
            return art(retained.points("kps"))

    case = Case(
        name="selftest_indexing",
        consumer_id="selftest",
        tier=Tier.PRIMITIVE,
        boundary="out",
        fixture=Fixture(name="selftest", path="", sha256="0" * 64, kind="synthetic"),
        exact_bytes=True,
        rtol=0.0,
        atol=0.0,
    )
    recorded = run_ablation(
        IndexesTheKey(), case, RetainedState(kps=kps), Ablation(primitive="kps", expect_breaks=True), art(kps)
    )
    inconclusive = recorded.observed_break is None and recorded.verdict is Verdict.INCONCLUSIVE
    return Attack(
        "dropped_persisted_primitive",
        "a replay that only indexes the removed key, through run_ablation",
        typed and inconclusive,
        f"MissingPrimitive raised={typed}; recorded observed_break={recorded.observed_break} "
        f"verdict={recorded.verdict}",
    )


def changed_selection_rule() -> Attack:
    """Swap first for largest-bbox-area on a two-face detection.

    Synthetic HERE and only here: the claim is about the rules' arithmetic,
    not about a detector. `face_selection.py` runs the same two rules over
    real group photographs; this asserts they are distinguishable at all,
    which is the precondition for that lane meaning anything.
    """
    from compat.consumers.face_selection import select

    class Found:
        def __init__(self, bbox: tuple[float, float, float, float]) -> None:
            self.bbox = np.asarray(bbox, dtype=np.float32)

    # First is deliberately the SMALLER box, so the rules must disagree.
    found = [Found((0.0, 0.0, 10.0, 10.0)), Found((0.0, 0.0, 100.0, 100.0))]
    first = select(found, "first").bbox
    largest = select(found, "largest_bbox_area").bbox
    return Attack(
        "changed_selection_rule",
        "first against largest_bbox_area on a two-face detection",
        not np.array_equal(first, largest),
        f"first {first.tolist()} against largest {largest.tolist()}",
    )


def changed_reference_order() -> Attack:
    """Reverse a set through both combiners.

    Asserts BOTH directions: a stack must change under reversal and a mean
    must not. Checking only one would accept a combiner that is neither.
    """
    from compat.consumers.reference_sets import combine

    rng = np.random.default_rng(20260828)
    vectors = [rng.standard_normal(512).astype(np.float32) for _ in range(3)]
    stack_ordered = not np.array_equal(combine(vectors, "stack"), combine(vectors[::-1], "stack"))
    mean_unordered = np.allclose(combine(vectors, "mean"), combine(vectors[::-1], "mean"), rtol=0.0, atol=1e-6)
    return Attack(
        "changed_reference_order",
        "set reversed through stack and through mean",
        stack_ordered and mean_unordered,
        f"stack order-sensitive: {stack_ordered}; mean order-insensitive: {mean_unordered}",
    )


def changed_vendor_fixture() -> Attack:
    """Different bytes under the same fixture path must not pass.

    The vendor lane resolves every fixture by sha256 out of a pinned commit,
    so a file swapped at the same address is a different fixture. Asserted on
    the recorded evidence rather than by touching the corpus: the media is
    third-party and this suite never rewrites it.
    """
    held = ROOT / "generated" / "vendor_fixtures.json"
    if not held.is_file():
        return Attack("changed_vendor_fixture", "no vendor_fixtures.json", False, "run the vendor lane first")
    rows: list[dict[str, Any]] = json.loads(held.read_text(encoding="utf-8"))["fixtures"]
    present = [one for one in rows if one["present"]]
    if not present:
        return Attack("changed_vendor_fixture", "no resolved fixtures", False, "nothing to corrupt")
    one = present[0]
    # Re-hashed for real: this fetches the bytes the index points at through
    # the same reader the conformance lane uses, checks the recorded digest
    # against them, then flips one byte and requires the digest to move.
    from compat.vendor import conformance

    blob = conformance._read(one)
    if blob is None:
        return Attack(
            "changed_vendor_fixture",
            f"{one['path']} could not be re-read",
            False,
            "neither on disk nor at the pinned commit on this machine",
            applicable=False,
        )
    matches = hashlib.sha256(blob).hexdigest() == one["sha256"]
    moved = hashlib.sha256(bytes([blob[0] ^ 0xFF]) + blob[1:]).hexdigest() != one["sha256"]
    return Attack(
        "changed_vendor_fixture",
        f"{one['path']} re-hashed, then one byte flipped",
        matches and moved,
        f"recorded digest matches the bytes={matches}; one flipped byte moves it={moved}",
    )


def vendor_acceptance_not_faked() -> Attack:
    """A consumer may not read as vendor-accepted without an executed run.

    The layer this checks is the one most worth faking: `ran` is the only
    thing separating "the vendor's own entrypoint produced this" from "our
    adapter produced something and we called it a reference".

    Over the RECORDED rows. `vendor_accepted` means "reproduced the shape
    upstream declares" rather than "ran and produced a boundary", so an
    accepted vendor must carry an `against_upstream` row that AGREES -- and a
    vendor that ran without such a row is VENDOR_BASELINE_UNAVAILABLE and must
    not appear in the accepted set.
    """
    held = ROOT / "generated" / "vendor_acceptance.json"
    if not held.is_file():
        return Attack("vendor_acceptance_not_faked", "no vendor_acceptance.json", False, "run the acceptance lane")
    out: dict[str, Any] = json.loads(held.read_text(encoding="utf-8"))
    accepted = set(out["population"]["vendor_accepted"])
    agreed = {one["consumer_id"] for one in out["against_upstream"] if one["agrees"] is True}
    unstated = {one["consumer_id"] for one in out["against_upstream"] if not one["stated"]}
    consistent = accepted == agreed and not (accepted & unstated)
    return Attack(
        "vendor_acceptance_not_faked",
        "vendor_accepted must equal the set whose boundary matched upstream's declared one",
        consistent,
        f"accepted {sorted(accepted)}; agreed-with-upstream {sorted(agreed)}; no upstream statement {sorted(unstated)}",
    )


def vendor_round_trip_lossless() -> Attack:
    """A vendor's own storage losing a key must not read as a clean round trip.

    ReActor is the one vendor here that persists a face, so its acceptance is
    the only place a real storage answer is measured rather than reasoned
    about. `every_key_survives` is the aggregate that would be quoted; this
    plants a lost key into the recorded per-key flags and asserts the
    aggregate follows, so a hardcoded True cannot survive.

    Over the RECORDED rows, not through `acceptance.py`'s aggregation: a
    hardcoded aggregate would pass this. It asserts the artifact is
    internally consistent, which is weaker than exercising the gate.
    """
    held = ROOT / "generated" / "vendor_acceptance.json"
    if not held.is_file():
        return Attack("vendor_round_trip_lossless", "no vendor_acceptance.json", False, "run the acceptance lane")
    out: dict[str, Any] = json.loads(held.read_text(encoding="utf-8"))
    rows = [one for one in out["acceptance"] if one["ran"] and "round_trip" in one["boundary"]]
    if not rows:
        return Attack("vendor_round_trip_lossless", "no vendor persists a face", False, "nothing to corrupt")
    row = rows[0]
    keys = list(row["boundary"]["round_trip"])
    clean = all(row["boundary"]["round_trip"][one]["survives_round_trip"] for one in keys)
    planted = copy.deepcopy(row["boundary"]["round_trip"])
    planted[keys[0]]["survives_round_trip"] = False
    after = all(planted[one]["survives_round_trip"] for one in keys)
    return Attack(
        "vendor_round_trip_lossless",
        f"{row['consumer_id']}: {keys[0]} planted as lost",
        clean and row["boundary"]["every_key_survives"] and not after,
        f"{len(keys)} keys recorded; clean={clean}; with one lost={after}",
    )


def vendor_boundary_repeats() -> Attack:
    """A boundary that moves between identical runs must not read as stable.

    This is the load-bearing one under every "our adapter matches" claim in
    the suite: an unstable vendor boundary is a moving reference, and a
    comparison against a moving reference proves nothing. The mechanism is
    real -- two ConsisID runs once produced different id_cond bytes from the
    same fixture, weights and code under onnxruntime's EXHAUSTIVE convolution
    search -- so the aggregate is asserted to follow a planted disagreement
    rather than trusted.

    Over the RECORDED rows. `determinism` now re-runs every vendor in a
    fresh interpreter, so calling it here would cost eight model loads;
    this checks the artifact it wrote rather than the function.
    """
    held = ROOT / "generated" / "vendor_acceptance.json"
    if not held.is_file():
        return Attack("vendor_boundary_repeats", "no vendor_acceptance.json", False, "run the acceptance lane")
    out: dict[str, Any] = json.loads(held.read_text(encoding="utf-8"))
    measured = {name: one for name, one in out["determinism"]["vendors"].items() if one["digests"]}
    if not measured:
        return Attack("vendor_boundary_repeats", "no vendor was measured twice", False, "nothing to corrupt")
    name = min(measured)
    planted = copy.deepcopy(measured[name]["digests"])
    planted[-1] = "0" * 64
    after = len(set(planted)) == 1
    return Attack(
        "vendor_boundary_repeats",
        f"{name}: last of {len(planted)} digests replaced",
        out["determinism"]["stable"] and not out["determinism"]["unstable"] and not after,
        f"{len(measured)} vendors measured; recorded stable={out['determinism']['stable']}; with one changed={after}",
    )


def vendor_pack_declaration_checked() -> Attack:
    """A manifest pack that contradicts the run must not read as agreement.

    The gate this proves exists because four `pack` values in this manifest
    were wrong and NOTHING caught them. The citations lane passed -- each was
    cited to a line that really does construct a FaceAnalysis. The
    substitution ablation passed too, because it derives `expect_breaks` FROM
    `pack`: with the wrong pack it expects no break and observes none. Both
    checks are satisfied by a well-formed claim rather than a true one.

    Run through `acceptance.declared_against_observed` itself. The earlier
    version copied a recorded row, flipped its `declared` field and asserted
    `planted["declared"] == planted["observed"]` was False -- a tautology by
    construction that never called the function it claimed to exercise. The
    rows are rehydrated into `Acceptance` objects and the production
    aggregation decides, so a change to how `pack` is compared is caught here
    rather than agreed with.
    """
    from compat.vendor.acceptance import Acceptance, declared_against_observed

    held = ROOT / "generated" / "vendor_acceptance.json"
    if not held.is_file():
        return Attack(
            "vendor_pack_declaration_checked",
            "no vendor_acceptance.json",
            False,
            "run the acceptance lane first",
            applicable=False,
        )
    out: dict[str, Any] = json.loads(held.read_text(encoding="utf-8"))
    fields = {one.name for one in dataclasses.fields(Acceptance)}
    rows = [Acceptance(**{k: v for k, v in one.items() if k in fields}) for one in out.get("acceptance", [])]
    observed = [
        one for one in declared_against_observed(rows) if one.get("field") is None and one["agrees"] is not None
    ]
    if not observed:
        return Attack(
            "vendor_pack_declaration_checked",
            "no acceptance observed a pack",
            False,
            "nothing to contradict; the gate is unexercised",
            applicable=False,
        )
    clean = all(one["agrees"] for one in observed)

    # Contradict the RUN, then re-run the production comparison. The pack is
    # planted into the boundary the analyser reported, which is the side the
    # function reads as fact.
    victim = min((one for one in rows if one.boundary.get("observed_pack")), key=lambda one: one.consumer_id)
    planted = copy.deepcopy(rows)
    for one in planted:
        if one.consumer_id != victim.consumer_id:
            continue
        was = one.boundary["observed_pack"]["pack"]
        one.boundary["observed_pack"] = {
            **was_dict(was, one),
            "pack": "buffalo_l" if was != "buffalo_l" else "antelopev2",
        }
    after = declared_against_observed(planted)
    caught = any(one["consumer_id"] == victim.consumer_id and one["agrees"] is False for one in after)
    return Attack(
        "vendor_pack_declaration_checked",
        f"{victim.consumer_id}: the pack the run reported replaced, through declared_against_observed",
        clean and caught,
        f"{len(observed)} packs observed, all agree={clean}; with one contradicted the function reports "
        f"disagreement={caught}",
    )


def was_dict(pack: str, row: Any) -> dict[str, Any]:
    """The observed-pack record, with its other fields kept."""
    held = dict(row.boundary.get("observed_pack") or {})
    held.setdefault("pack", pack)
    held.setdefault("modules", {})
    return held


def cache_never_crosses_identity() -> Attack:
    """An entry written under another identity must never be served.

    This is the only property that makes a persistent memo safe here. The
    store is namespaced by `identity()["digest"]`, which covers the manifest,
    every pinned commit, every weight sha256, the runtime and the sha256 of
    every compat source file -- so an entry computed before any of that moved
    lives in a directory the current run does not look in.

    Asserted both ways. A store that returned None unconditionally would pass
    the negative half and be useless, so the positive half runs too.
    """
    from compat.corpus import cache

    if not cache.enabled():
        return Attack(
            "cache_never_crosses_identity",
            "COMPAT_CACHE=0",
            False,
            "the store is off, so there is nothing to attack",
            applicable=False,
        )

    sha = "c0" * 32
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    mine = cache.slot("frame", f"{sha}.npy")
    foreign = cache.CACHE_ROOT / ("f" * 64) / "frame" / f"{sha}.npy"
    try:
        foreign.parent.mkdir(parents=True, exist_ok=True)
        with foreign.open("wb") as handle:
            np.save(handle, frame, allow_pickle=False)
        crossed = cache.frame_get(sha) is not None

        cache.frame_put(sha, frame)
        held = cache.frame_get(sha)
        served = held is not None and np.array_equal(held, frame)
    finally:
        mine.unlink(missing_ok=True)
        foreign.unlink(missing_ok=True)

    return Attack(
        "cache_never_crosses_identity",
        "one entry under a foreign identity digest, one under this run's",
        not crossed and served,
        f"foreign entry served={crossed} (must be False); own entry served={served} (must be True)",
    )


def cache_preserves_every_value_type() -> Attack:
    """A held face must come back with every value's TYPE, not just its bytes.

    `producers/insightface_pass._describe` branches on `isinstance(value,
    np.ndarray)` before it looks at anything else, so an np.float32 returning
    as a 0-d array would be inventoried as an ndarray of 4 bytes where the
    producer reported a float of 8. The storage evidence is a byte-cost table
    over those fields: a cache that lost the distinction would not be slow,
    it would be wrong.

    An unreadable entry is checked in the same pass. It must read as a miss,
    because a store that can raise can fail a run it was only meant to speed
    up.
    """
    from insightface.app.common import Face

    from compat.corpus import cache

    if not cache.enabled():
        return Attack(
            "cache_preserves_every_value_type",
            "COMPAT_CACHE=0",
            False,
            "the store is off, so there is nothing to attack",
            applicable=False,
        )

    sha = "c1" * 32
    face = Face(
        bbox=np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        det_score=np.float32(0.87),
        embedding=np.arange(512, dtype=np.float32),
        gender=1,
        age=34,
        label="x",
    )
    body, kinds_at = cache.slot("ours", f"{sha}.npz"), cache.slot("ours", f"{sha}.json")
    try:
        cache.face_put(sha, face)
        back = cache.face_get(sha)
        kept = back is not None and all(
            type(face[key]) is type(back[key])
            and (np.array_equal(face[key], back[key]) if isinstance(face[key], np.ndarray) else face[key] == back[key])
            for key in face
        )
        # The same entry, truncated. A reader must return None rather than
        # raise: the caller can always recompute.
        body.write_bytes(b"PK not a zip")
        survived = cache.face_get(sha) is None
    finally:
        body.unlink(missing_ok=True)
        kinds_at.unlink(missing_ok=True)

    return Attack(
        "cache_preserves_every_value_type",
        "round-trip a Face of arrays, an np.float32, two ints and a str; then truncate it",
        kept and survived,
        f"types and bytes preserved={kept}; a corrupt entry read as a miss={survived}",
    )


def changed_application_source() -> Attack:
    """The storage lane runs `vision.faces` and `db.*`; editing them must drift.

    `storage/gallery_v45.py` is the only lane that reaches out of `compat/`,
    and it reaches into the code under test. Until the identity covered them,
    an edit to `vision/faces.py` changed the headline answer with every pin,
    weight and compat source unmoved, and the staleness lane reported
    "evidence is current".
    """
    if not evidence.identity()["application"]:
        return Attack(
            "changed_application_source",
            "no application sources digested",
            False,
            "vision/ and db/ were not found beside compat/",
            applicable=False,
        )

    def mutate(held: dict[str, Any]) -> None:
        held["application"][min(held["application"])] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("changed_application_source", "one application source digest altered", seen, detail)


def changed_corpus() -> Attack:
    """A corpus photograph swapped must invalidate the evidence.

    `corpus/loaded.shots()` selects four by min-sha256 within each
    (identity, role) bucket, so adding or editing one changes WHICH
    photographs every baseline was computed from.
    """
    if not evidence.identity()["corpus"]:
        return Attack(
            "changed_corpus",
            "no corpus digested",
            False,
            "the KYC corpus is not on this machine",
            applicable=False,
        )

    def mutate(held: dict[str, Any]) -> None:
        held["corpus"][min(held["corpus"])] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("changed_corpus", "one corpus photograph re-hashed", seen, detail)


def retained_bytes_cannot_be_asserted() -> Attack:
    """A durable size the array does not encode to must be refused.

    `sizes()` prefers `_durable` over `ndarray.nbytes`, and `answer.py`
    publishes the result as the storage cost of a primitive. Before `priced()`
    validated, a runner naming any integer decided that cost.

    Two-sided through `RetainedState.priced` itself: the encoded length is
    accepted and a planted one is refused. A one-sided check passes against a
    function that accepts everything.
    """
    import numpy as np

    from compat.contracts.case import RetainedState
    from compat.storage import derivatives

    frame = (np.arange(64 * 64 * 3, dtype=np.uint8) % 251).reshape(64, 64, 3)
    encoded = derivatives.lossless_bytes(frame)

    honest = False
    try:
        priced = RetainedState(whole_reference_image=frame).priced({"whole_reference_image": encoded})
        honest = priced.sizes()["whole_reference_image"] == encoded
    except (KeyError, TypeError, ValueError):
        honest = False

    planted: list[bool] = []
    for bad in (0, encoded + 1, frame.nbytes):
        try:
            RetainedState(whole_reference_image=frame).priced({"whole_reference_image": bad})
            planted.append(False)
        except ValueError:
            planted.append(True)

    return Attack(
        "retained_bytes_cannot_be_asserted",
        "a durable size that is not the array's encoded length, through RetainedState.priced",
        honest and all(planted),
        f"encoded {encoded} B accepted={honest}; refused 0, +1 and nbytes={planted}",
    )


def every_attack() -> tuple[Attack, ...]:
    return (
        positive_control(),
        missing_consumer(),
        changed_pin(),
        changed_weight(),
        changed_manifest(),
        changed_runner_source(),
        dropped_persisted_primitive(),
        changed_selection_rule(),
        changed_reference_order(),
        stale_evidence(),
        changed_vendor_fixture(),
        vendor_acceptance_not_faked(),
        vendor_round_trip_lossless(),
        vendor_boundary_repeats(),
        vendor_pack_declaration_checked(),
        changed_application_source(),
        changed_corpus(),
        cache_never_crosses_identity(),
        cache_preserves_every_value_type(),
        retained_bytes_cannot_be_asserted(),
    )


def main() -> int:
    attacks = every_attack()
    for one in attacks:
        print(f"{'ok ' if one.ok else '!! '}{one.name:<28} {one.mechanism}")
        print(f"    {one.detail[:150]}")

    missed = [one.name for one in attacks if not one.ok]
    print(f"\nattacks: {len(attacks)}   undetected: {len(missed)}  {missed}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "selftest.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                {"attacks": [asdict(one) for one in attacks], "undetected": missed},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        handle.write("\n")
    print(f"wrote {target}")

    # An attack the harness does NOT see is a red result: every green ever
    # reported for that mechanism was uninformative.
    return 0 if not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
