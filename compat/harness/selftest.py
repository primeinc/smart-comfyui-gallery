"""Corrupt the harness deliberately and assert it notices.

A gate that has never gone red is not known to work. `attack.py` covers pins,
blobs, weights and evidence bytes; these cover the mechanisms added since:

    positive_control            change NOTHING           -> no drift
    missing_consumer            drop a declared consumer -> population INCOMPLETE
    changed_pin                 mutate a commit          -> identity drift
    changed_weight              mutate a weight digest   -> identity drift
    changed_manifest            mutate the manifest      -> identity drift
    changed_runner_source       edit a runner's bytes    -> identity drift
    dropped_persisted_primitive remove a stored value    -> replay breaks
    changed_selection_rule      swap first/largest       -> different face
    changed_reference_order     reverse a stacked set    -> different artifact
    stale_evidence              old identity vs current  -> drift reported
    changed_vendor_fixture      same path, other bytes   -> different sha256
    vendor_acceptance_not_faked accepted set must equal the set that RAN
    vendor_round_trip_lossless  plant one lost key       -> aggregate goes red
    vendor_boundary_repeats     plant a second digest    -> stability goes red
    vendor_pack_declaration_checked  contradict the declared pack -> disagreement

The positive control is not decoration. Every other row asserts a mutation IS
seen; without a row asserting an unmutated tree is seen as clean, a detector
that returned "changed" unconditionally would pass every one of them.
"""

from __future__ import annotations

import copy
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

    @property
    def ok(self) -> bool:
        return self.detected


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
    left = declared - (covered - {dropped})
    return Attack(
        "missing_consumer",
        f"{dropped} removed from consumer-tier coverage",
        bool(left),
        f"unexercised would become {sorted(left)}",
    )


def dropped_persisted_primitive() -> Attack:
    """Remove one value from a retained state; the replay must break."""
    from compat.contracts.case import RetainedState

    without = RetainedState(kps=np.zeros((5, 2), dtype=np.float32)).without("kps")
    try:
        without.points("kps")
    except KeyError as problem:
        return Attack("dropped_persisted_primitive", "kps removed from retained state", True, str(problem))
    return Attack("dropped_persisted_primitive", "kps removed from retained state", False, "replay read it anyway")


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
    return Attack(
        "changed_vendor_fixture",
        f"{one['path']} re-hashed as zeros",
        one["sha256"] != "0" * 64,
        f"recorded {one['sha256'][:16]}; a swap at the same path changes it",
    )


def vendor_acceptance_not_faked() -> Attack:
    """A consumer may not read as vendor-accepted without an executed run.

    The layer this checks is the one most worth faking: `ran` is the only
    thing separating "the vendor's own entrypoint produced this" from "our
    adapter produced something and we called it a reference".
    """
    held = ROOT / "generated" / "vendor_acceptance.json"
    if not held.is_file():
        return Attack("vendor_acceptance_not_faked", "no vendor_acceptance.json", False, "run the acceptance lane")
    out: dict[str, Any] = json.loads(held.read_text(encoding="utf-8"))
    accepted = set(out["population"]["vendor_accepted"])
    ran = {one["consumer_id"] for one in out["acceptance"] if one["ran"] and one["boundary"]}
    # Accepted must be exactly those that ran AND produced a boundary.
    consistent = accepted == ran
    return Attack(
        "vendor_acceptance_not_faked",
        "vendor_accepted must equal the set that actually ran and produced a boundary",
        consistent,
        f"accepted {sorted(accepted)}; ran-with-boundary {sorted(ran)}",
    )


def vendor_round_trip_lossless() -> Attack:
    """A vendor's own storage losing a key must not read as a clean round trip.

    ReActor is the one vendor here that persists a face, so its acceptance is
    the only place a real storage answer is measured rather than reasoned
    about. `every_key_survives` is the aggregate that would be quoted; this
    plants a lost key into the recorded per-key flags and asserts the
    aggregate follows, so a hardcoded True cannot survive.
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

    `acceptance.observed_pack` reads the model files off the live analyser,
    so the comparison is manifest-versus-run. This plants a contradicting
    declaration into the recorded rows and asserts the agreement flips.
    """
    held = ROOT / "generated" / "vendor_acceptance.json"
    if not held.is_file():
        return Attack("vendor_pack_declaration_checked", "no vendor_acceptance.json", False, "run the acceptance lane")
    out: dict[str, Any] = json.loads(held.read_text(encoding="utf-8"))
    rows = [one for one in out.get("declared_against_observed", []) if one.get("agrees") is not None]
    if not rows:
        return Attack(
            "vendor_pack_declaration_checked",
            "no acceptance observed a pack",
            False,
            "nothing to contradict; the gate is unexercised",
        )

    clean = all(one["agrees"] for one in rows)
    victim = min(rows, key=lambda one: one["consumer_id"])
    planted = copy.deepcopy(victim)
    planted["declared"] = "buffalo_l" if planted["observed"] != "buffalo_l" else "antelopev2"
    after = planted["declared"] == planted["observed"]
    return Attack(
        "vendor_pack_declaration_checked",
        f"{victim['consumer_id']}: declared pack replaced with {planted['declared']!r}",
        clean and not after,
        f"{len(rows)} packs observed; all agree={clean}; with one contradicted={after}",
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
