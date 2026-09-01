from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.harness import identity as evidence
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


@dataclass
class Attack:
    name: str
    mechanism: str
    detected: bool
    detail: str
    applicable: bool = True

    @property
    def ok(self) -> bool:
        return self.detected


def _tampered(mutate: Callable[[dict[str, Any]], None]) -> tuple[bool, str]:
    tampered = copy.deepcopy(evidence.identity())
    mutate(tampered)
    tampered["digest"] = evidence.digest_of(tampered)
    drift = evidence.compare_to(tampered)
    return bool(drift), "; ".join(drift[:3]) or "no drift reported"


def positive_control() -> Attack:
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

    def mutate(held: dict[str, Any]) -> None:
        key = next(one for one in held["sources"] if one.startswith("harness/") and not one.endswith("__init__.py"))
        held["sources"][key] = "0" * 64

    seen, detail = _tampered(mutate)
    return Attack("stale_evidence", "evidence recorded against an older harness source", seen, detail)


def missing_consumer() -> Attack:
    cases = ROOT / "generated" / "cases.json"
    if not cases.is_file():
        return Attack("missing_consumer", "no cases.json", False, "run the case lane first")
    held: dict[str, Any] = json.loads(cases.read_text(encoding="utf-8"))
    declared = set(held["population"]["declared"])
    covered = set(held["population"]["consumer_tier_covered"])
    if not covered:
        return Attack("missing_consumer", "no consumer-tier coverage", False, "nothing to drop")
    dropped = min(covered)

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
    from compat.consumers.face_selection import select

    class Found:
        def __init__(self, bbox: tuple[float, float, float, float]) -> None:
            self.bbox = np.asarray(bbox, dtype=np.float32)

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
    held = ROOT / "generated" / "vendor_fixtures.json"
    if not held.is_file():
        return Attack("changed_vendor_fixture", "no vendor_fixtures.json", False, "run the vendor lane first")
    rows: list[dict[str, Any]] = json.loads(held.read_text(encoding="utf-8"))["fixtures"]
    present = [one for one in rows if one["present"]]
    if not present:
        return Attack("changed_vendor_fixture", "no resolved fixtures", False, "nothing to corrupt")
    one = present[0]

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
    held = dict(row.boundary.get("observed_pack") or {})
    held.setdefault("pack", pack)
    held.setdefault("modules", {})
    return held


def cache_never_crosses_identity() -> Attack:
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


def _facexlib_asset() -> dict[str, Any] | None:
    for row in provenance.load_manifest().get("weights", []):
        for one in row.get("attestations", []):
            if one.get("source_class") == "github_release_asset" and one.get("revision"):
                return dict(one)
    return None


def vendor_asset_swap_detected() -> Attack:
    located = _facexlib_asset()
    if located is None:
        return Attack("vendor_asset_swap", "no release-asset attestation declared", False, "nothing to attack")
    refs = (ROOT.parent / provenance.load_manifest()["refs_root"]).resolve()

    honest = provenance._resolve_github_release_asset(located, refs)
    ident, _, stamp = located["revision"].partition("@")
    caught: list[str] = []
    for swapped in (f"{ident}@2099-01-01T00:00:00Z", f"{int(ident) + 1}@{stamp}", ident, f"@{stamp}"):
        verdict = provenance._resolve_github_release_asset({**located, "revision": swapped}, refs)
        if verdict["evidence"] != provenance.EVIDENCE_UNRESOLVABLE or verdict["resolved_sha256"]:
            caught.append(f"{swapped} -> {verdict['evidence']}")
    return Attack(
        "vendor_asset_swap",
        "a release asset re-uploaded under the same name, through the pinned id and mtime",
        honest["evidence"] == provenance.EVIDENCE_PROVEN and not caught,
        f"pinned locator -> {honest['evidence']}; four moved pins refused" if not caught else f"ACCEPTED {caught}",
    )


def vendor_asset_size_checked() -> Attack:
    located = _facexlib_asset()
    if located is None:
        return Attack("vendor_asset_size", "no release-asset attestation declared", False, "nothing to attack")
    refs = (ROOT.parent / provenance.load_manifest()["refs_root"]).resolve()
    ident = located["revision"].partition("@")[0]
    real = provenance._release_asset_cache(refs) / located["repo_id"] / ident
    name = located["path"].rsplit("/", 1)[-1]
    if not (real / "asset.json").is_file():
        return Attack("vendor_asset_size", "no mirrored asset", False, "run the pins lane first")

    with tempfile.TemporaryDirectory() as where:
        fake = Path(where) / "_release_assets" / located["repo_id"] / ident
        fake.mkdir(parents=True)
        shutil.copyfile(real / "asset.json", fake / "asset.json")
        (fake / name).write_bytes(b"truncated")
        verdict = provenance._resolve_github_release_asset(located, Path(where))
    return Attack(
        "vendor_asset_size",
        "a truncated mirror of a release asset, through the release's own byte count",
        verdict["evidence"] == provenance.EVIDENCE_UNRESOLVABLE and not verdict["resolved_sha256"],
        f"9-byte mirror -> {verdict['evidence']}: {verdict['detail']}",
    )


def vendor_asset_host_checked() -> Attack:
    refused: list[str] = []
    for url in (
        "file:///C:/Windows/System32/drivers/etc/hosts",
        "http://github.com/x/y/releases/download/v1/w.pth",
        "https://evil.example.com/w.pth",
        "https://github.com.evil.example.com/w.pth",
    ):
        try:
            provenance._checked(url)
            refused.append(url)
        except ValueError:
            pass
    allowed = True
    try:
        provenance._checked("https://objects.githubusercontent.com/x")
    except ValueError:
        allowed = False
    return Attack(
        "vendor_asset_host",
        "a download url off https or off GitHub, through the fetcher's scheme and host check",
        not refused and allowed,
        "four refused, the real host allowed" if not refused and allowed else f"ACCEPTED {refused} allowed={allowed}",
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
        vendor_asset_swap_detected(),
        vendor_asset_size_checked(),
        vendor_asset_host_checked(),
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

    return 0 if not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
