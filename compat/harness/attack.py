"""Attack the suite. A gate that has never gone red is not known to work.

Every other module here tries to establish that something holds. This one
tries to break the machinery that establishes it, because a harness reporting
green has two possible causes and only one of them is good news.

Seven attacks, each on a different load-bearing claim:

    pin_mutated          change a pinned commit -> provenance must FAIL
    blob_mutated         change a recorded blob digest -> must FAIL
    weight_moved         point a weight row at an absent file -> must FAIL
    necessary_removed    a primitive an ablation says is required ->
                         that case must stop reproducing
    population_shrunk    delete a consumer's cases -> the matrix must report
                         it NOT EXERCISED rather than omitting it
    evidence_not_reproducible
                         run the whole executor again -> the evidence must
                         serialise to the same bytes
    positive_control     the unmodified suite must still pass, so a red
                         result above is the attack and not a broken machine

Nothing here writes to the repository. Manifests and evidence are copied to a
scratch directory, mutated there, and the real checks are run against the
copy -- an attack that damaged the tree to prove a point would be a worse
failure than the one it was testing for.

An attack that does NOT produce the failure it is aiming at is itself a red
result: it means the gate cannot see that class of problem, and every green
this suite has ever reported for that class was uninformative.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compat.harness import matrices, provenance

ROOT: Path = Path(__file__).resolve().parent.parent
GENERATED: Path = ROOT / "generated"


@dataclass
class Attack:
    """One deliberate corruption, and whether the gate noticed."""

    name: str
    targets: str
    expected: str
    detected: bool = False
    observed: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.detected


def _workspace() -> Path:
    """A throwaway copy of the manifest and the evidence."""
    where = Path(tempfile.mkdtemp(prefix="compat_attack_"))
    shutil.copy2(ROOT / "manifest.toml", where / "manifest.toml")
    (where / "generated").mkdir()
    for name in ("cases.json", "provenance.json", "producer_inventory.json"):
        source = GENERATED / name
        if source.is_file():
            shutil.copy2(source, where / "generated" / name)
    return where


def _manifest(where: Path) -> dict[str, Any]:
    return provenance.load_manifest(where / "manifest.toml")


def attack_pin_mutated(where: Path, repo_root: Path) -> Attack:
    """A commit nobody has: provenance must refuse it."""
    out = Attack(
        name="pin_mutated",
        targets="manifest [[consumers]].commit",
        expected="provenance FAILS: HEAD is not the pinned commit",
    )
    text = (where / "manifest.toml").read_text(encoding="utf-8")
    original = 'commit = "6ad6b35a4df250d14cb2abf0808c9ffedf59f747"'
    if original not in text:
        out.observed = "the ReActor pin is not where this attack expected it; attack could not be mounted"
        return out
    (where / "manifest.toml").write_text(
        text.replace(original, 'commit = "0000000000000000000000000000000000000000"', 1),
        encoding="utf-8",
        newline="",
    )
    result = provenance.verify_all(_manifest(where), repo_root)
    out.detected = not result["provenance_ok"]
    failures = [one for repo in result["repos"] for one in repo["failures"]]
    out.observed = failures[0] if failures else "provenance still reported PASS"
    return out


def attack_weight_moved(where: Path, repo_root: Path) -> Attack:
    """A weight file that is not there: provenance must refuse it."""
    out = Attack(
        name="weight_moved",
        targets="manifest [[weights]].file",
        expected="provenance FAILS: the model file is absent",
    )
    text = (where / "manifest.toml").read_text(encoding="utf-8")
    (where / "manifest.toml").write_text(
        text.replace('file = "glintr100.onnx"', 'file = "glintr100-does-not-exist.onnx"', 1),
        encoding="utf-8",
        newline="",
    )
    result = provenance.verify_all(_manifest(where), repo_root)
    missing = [one for one in result["weights"] if not one["present"]]
    out.detected = not result["provenance_ok"] and bool(missing)
    out.observed = f"{len(missing)} weight row(s) reported absent" if missing else "every weight still reported present"
    return out


def attack_blob_mutated(where: Path, repo_root: Path) -> Attack:
    """A recorded blob digest that no longer matches the commit.

    Recomputed rather than trusted: the check reads the blob from git every
    run, so a doctored digest in the evidence must not survive a re-run.
    """
    out = Attack(
        name="blob_mutated",
        targets="generated/provenance.json blob_sha256",
        expected="a re-run recomputes the digest and does not reproduce the doctored one",
    )
    evidence = json.loads((where / "generated" / "provenance.json").read_text(encoding="utf-8"))
    doctored = "0" * 64
    changed = False
    for repo in evidence["repos"]:
        for path in repo["paths"]:
            if path.get("blob_sha256"):
                path["blob_sha256"] = doctored
                changed = True
                break
        if changed:
            break
    if not changed:
        out.observed = "no blob digest to doctor"
        return out
    (where / "generated" / "provenance.json").write_text(json.dumps(evidence), encoding="utf-8", newline="")

    fresh = provenance.verify_all(_manifest(where), repo_root)
    digests = {one["blob_sha256"] for repo in fresh["repos"] for one in repo["paths"] if one.get("blob_sha256")}
    out.detected = doctored not in digests
    out.observed = (
        f"re-run produced {len(digests)} digests, none of them the doctored value"
        if out.detected
        else ("the doctored digest survived a re-run")
    )
    return out


def attack_necessary_removed(where: Path) -> Attack:
    """Something, somewhere, must be shown necessary. Otherwise nothing is.

    The property is unchanged and the evidence for it moved. This counted
    REMOVALS that broke, which was right while a removal was the necessity
    test -- and every removal in this suite now ends INCONCLUSIVE, because
    taking away the one key a replay indexes shows only that the replay
    indexes it. Necessity is established by SUBSTITUTION instead: the same
    value offered in a cheaper storable form, with the consumer's own boundary
    measured under it.

    So both are counted, and they are counted SEPARATELY -- a suite where only
    removals broke and a suite where only substitutions did are different
    suites, and one number would hide which one this is.
    """
    out = Attack(
        name="necessary_removed",
        targets="a primitive whose removal or degradation broke the replay",
        expected="the case that needs it stops reproducing",
    )
    evidence = json.loads((where / "generated" / "cases.json").read_text(encoding="utf-8"))
    removals: list[tuple[str, str, str]] = []
    swaps: list[tuple[str, str, str]] = []
    for result in evidence["results"]:
        for ablation in result["ablations"]:
            if not ablation["observed_break"]:
                continue
            if ablation.get("kind", "removal") == "removal":
                removals.append((result["case"], ablation["primitive"], ablation["detail"]))
            else:
                swaps.append((result["case"], f"{ablation['primitive']} <- {ablation['swap']}", ablation["detail"]))

    out.detected = bool(removals or swaps)
    if out.detected:
        case, primitive, detail = (removals or swaps)[0]
        out.observed = (
            f"{len(removals)} removal(s) and {len(swaps)} substitution(s) broke their replay, "
            f"e.g. {case} on {primitive}"
        )
        out.notes.append(f"{primitive}: {detail[:120]}")
    else:
        out.observed = "NO recorded ablation broke anything -- nothing in this suite is shown to be necessary"
    return out


def attack_population_shrunk(where: Path) -> Attack:
    """Delete a consumer's results; the matrix must still list it."""
    out = Attack(
        name="population_shrunk",
        targets="generated/cases.json results",
        expected="the matrix reports the consumer NOT EXERCISED instead of dropping it",
    )
    evidence = json.loads((where / "generated" / "cases.json").read_text(encoding="utf-8"))
    victim = "reactor"
    before = len(evidence["results"])
    evidence["results"] = [one for one in evidence["results"] if one["consumer_id"] != victim]
    if len(evidence["results"]) == before:
        out.observed = f"{victim} had no results to delete; attack could not be mounted"
        return out
    (where / "generated" / "cases.json").write_text(json.dumps(evidence), encoding="utf-8", newline="")

    built = matrices.build(matrices.Evidence.load(where / "generated"))
    row = next((one for one in built["consumers"] if one["consumer"] == victim), None)
    out.detected = row is not None and row["status"] == "NOT EXERCISED" and built["totals"]["not_exercised"] > 0
    if row is None:
        out.observed = f"{victim} VANISHED from the matrix entirely"
    else:
        out.observed = (
            f"{victim} listed as {row['status']}; matrix totals not_exercised={built['totals']['not_exercised']}"
        )
    return out


def attack_evidence_not_reproducible() -> Attack:
    """Two runs of the same inputs must produce the same evidence bytes.

    Not a corruption like the others. It requires wall clocks to stay out of
    `cases.json`: a file rewritten with fresh timings on every run makes "the
    evidence is unchanged" uncheckable and a diff between two runs noise.

    Re-runs the case executor in-process, with `COMPAT_CACHE=0` so the
    producer and the decoder actually run, and compares the serialisation it
    would write against the serialisation on disk. A difference here means
    something in the pipeline is not deterministic, and every byte-exact
    claim this suite makes rests on that not being true.
    """
    from compat.harness import run as case_runner

    out = Attack(
        name="evidence_not_reproducible",
        targets="generated/cases.json",
        expected="a second run serialises to the same bytes as the first",
    )
    on_disk = (GENERATED / "cases.json").read_text(encoding="utf-8")

    # The rebuild runs with the persistent store OFF: a rebuild that could
    # read `corpus/cache.py` would be handed the first run's answers and
    # re-execute neither cv2.imdecode nor the ONNX pass.
    was = os.environ.get("COMPAT_CACHE")
    os.environ["COMPAT_CACHE"] = "0"
    try:
        fresh = case_runner.run_all()
    finally:
        if was is None:
            os.environ.pop("COMPAT_CACHE", None)
        else:
            os.environ["COMPAT_CACHE"] = was

    fresh.pop("seconds_by_case", None)
    rebuilt = json.dumps(fresh, indent=2, sort_keys=True, default=str) + chr(10)

    out.detected = rebuilt == on_disk
    if out.detected:
        out.observed = f"a second run reproduced all {len(on_disk):,} bytes of the evidence"
    else:
        first = next(
            (i for i, (a, b) in enumerate(zip(on_disk, rebuilt, strict=False)) if a != b),
            min(len(on_disk), len(rebuilt)),
        )
        out.observed = (
            f"the two runs differ from byte {first:,}: "
            f"{on_disk[first : first + 50]!r} against {rebuilt[first : first + 50]!r}"
        )
    return out


def attack_positive_control(repo_root: Path) -> Attack:
    """The untouched suite must still pass, or the reds above prove nothing."""
    out = Attack(
        name="positive_control",
        targets="the real manifest and the real evidence",
        expected="provenance PASSES and the matrix reports every consumer exercised",
    )
    result = provenance.verify_all(provenance.load_manifest(), repo_root)
    built = matrices.build(matrices.Evidence.load())
    # Through `matrices.blocking`, the same function the matrix lane gates on.
    # A control re-implementing one of its conditions cannot fail while that
    # lane passes, and does not follow the lane when it gains more.
    bad = matrices.blocking(built)
    out.detected = bool(result["provenance_ok"]) and not bad
    out.observed = (
        f"provenance {'PASS' if result['provenance_ok'] else 'FAIL'}; "
        f"{built['totals']['reproduced']}/{built['totals']['declared']} reproduced; "
        f"blocking totals: {bad or 'none'}"
    )
    return out


def attack_producer_inventory_stale() -> Attack:
    """The recorded producer inventory must name every key the pass emits.

    `gallery_storage.unlisted_keys()` exists for exactly this and had no
    caller anywhere in the tree. Its docstring says a non-empty result means
    `generated/producer_inventory.json` is stale against the live pass -- and
    `storage/contract.emitted_keys()` reads that file to decide which keys the
    storage lane builds cases for, so a producer that gained a key would be
    measured as though it had not.
    """
    from compat.consumers.gallery_storage import unlisted_keys

    out = Attack(
        name="producer_inventory_stale",
        targets="generated/producer_inventory.json against the live producer",
        expected="every key the pass emits is named in the recorded inventory",
    )
    missing = sorted(unlisted_keys())
    out.detected = not missing
    out.observed = (
        "the recorded inventory names every key the pass emitted"
        if not missing
        else f"the live pass emits {len(missing)} key(s) the inventory does not name: {missing}"
    )
    return out


def run_all() -> list[Attack]:
    repo_root = ROOT.parent
    attacks: list[Attack] = []
    for build in (attack_pin_mutated, attack_weight_moved, attack_blob_mutated):
        where = _workspace()
        try:
            attacks.append(build(where, repo_root))
        finally:
            shutil.rmtree(where, ignore_errors=True)
    for build_local in (attack_necessary_removed, attack_population_shrunk):
        where = _workspace()
        try:
            attacks.append(build_local(where))
        finally:
            shutil.rmtree(where, ignore_errors=True)
    attacks.append(attack_evidence_not_reproducible())
    attacks.append(attack_producer_inventory_stale())
    attacks.append(attack_positive_control(repo_root))
    return attacks


def main() -> int:
    attacks = run_all()
    print(f"{'attack':<22} {'gate saw it':<12} observed")
    for one in attacks:
        print(f"{one.name:<22} {'YES' if one.ok else 'NO -- BLIND':<12} {one.observed}")
        for note in one.notes:
            print(f"{'':<22} {'':<12} {note}")

    out = {
        "runtime": provenance.runtime_identity(),
        "attacks": [
            {
                "name": one.name,
                "targets": one.targets,
                "expected": one.expected,
                "detected": one.detected,
                "observed": one.observed,
                "notes": one.notes,
            }
            for one in attacks
        ],
        "all_detected": all(one.ok for one in attacks),
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    where = GENERATED / "attack.json"
    with where.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")

    blind = [one.name for one in attacks if not one.ok]
    print(f"\nwrote {where}")
    print(
        "every attack was detected"
        if not blind
        else f"BLIND to {len(blind)}: {blind} -- greens for those classes are uninformative"
    )
    return 0 if not blind else 1


if __name__ == "__main__":
    raise SystemExit(main())
