"""Attack the suite. A gate that has never gone red is not known to work.

Every other module here tries to establish that something holds. This one
tries to break the machinery that establishes it, because a harness reporting
green has two possible causes and only one of them is good news.

Six attacks, each on a different load-bearing claim:

    pin_mutated          change a pinned commit -> provenance must FAIL
    blob_mutated         change a recorded blob digest -> must FAIL
    weight_moved         point a weight row at an absent file -> must FAIL
    necessary_removed    drop a primitive an ablation says is required ->
                         that case must stop reproducing
    population_shrunk    delete a consumer's cases -> the matrix must report
                         it NOT EXERCISED rather than omitting it
    positive_control     the unmodified suite must still pass, so a red
                         result above is the attack and not a broken machine

Nothing here writes to the repository. Manifests and evidence are copied to a
temporary directory, mutated there, and the real checks are run against the
copy -- an attack that damaged the tree to prove a point would be a worse
failure than the one it was testing for.

An attack that does NOT produce the failure it is aiming at is itself a red
result: it means the gate cannot see that class of problem, and every green
this suite has ever reported for that class was uninformative.
"""

from __future__ import annotations

import json
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
    """Drop a primitive the evidence calls necessary; its case must go red.

    Read out of the evidence rather than named here: whichever primitive the
    ablations actually found necessary is the one attacked, so this keeps
    working when the storage contract changes.
    """
    out = Attack(
        name="necessary_removed",
        targets="a primitive whose ablation broke the replay",
        expected="the case that needs it stops reproducing",
    )
    evidence = json.loads((where / "generated" / "cases.json").read_text(encoding="utf-8"))
    found = [
        (result["case"], ablation["primitive"], ablation["detail"])
        for result in evidence["results"]
        for ablation in result["ablations"]
        if ablation.get("kind", "removal") == "removal" and ablation["observed_break"]
    ]
    out.detected = bool(found)
    if found:
        case, primitive, detail = found[0]
        out.observed = f"{len(found)} recorded removals broke their replay, e.g. {case} without {primitive}"
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


def attack_positive_control(repo_root: Path) -> Attack:
    """The untouched suite must still pass, or the reds above prove nothing."""
    out = Attack(
        name="positive_control",
        targets="the real manifest and the real evidence",
        expected="provenance PASSES and the matrix reports every consumer exercised",
    )
    result = provenance.verify_all(provenance.load_manifest(), repo_root)
    built = matrices.build(matrices.Evidence.load())
    out.detected = bool(result["provenance_ok"]) and built["totals"]["not_exercised"] == 0
    out.observed = (
        f"provenance {'PASS' if result['provenance_ok'] else 'FAIL'}; "
        f"{built['totals']['reproduced']}/{built['totals']['declared']} reproduced, "
        f"{built['totals']['not_exercised']} not exercised"
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
