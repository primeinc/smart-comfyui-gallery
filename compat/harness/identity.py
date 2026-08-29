"""What the evidence is an answer TO.

A generated file that does not name its inputs cannot be stale, because
nothing can disagree with it. This computes one digest over every input that
can change a case's outcome, and `run.py` writes it into the evidence. A
mismatch between the recorded digest and the current one means the evidence
answers a question nobody asked any more.

Six inputs, and dropping any one of them makes the digest a decoration:

    fixtures      every fixture sha256 the suite is holding
    repos         every pinned repo and its FULL commit
    weights       every model file's sha256
    runtime       interpreter, platform, and the package versions that run
    manifest      the manifest bytes, so a threshold edit is visible
    runners       the source of every runner and preprocessor

`runners` is the one that closes the hole the others leave open. A runner
edited in place changes what a case DOES while every pin, fixture and weight
stays identical, so without hashing our own source the suite would keep
serving a green it can no longer earn.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Every directory whose source decides what a case computes. `harness` is
#: included: the executor's own comparison and verdict logic is as
#: answer-changing as any runner's.
SOURCE_DIRS: Final[tuple[str, ...]] = (
    "assertions",
    "consumers",
    "contracts",
    "corpus",
    "harness",
    "primitives",
    "producers",
    "storage",
    "vendor",
)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


#: The inputs the digest is taken over. Named once so `identity` and any
#: checker that rebuilds a digest cannot disagree about the input set.
PARTS: Final[tuple[str, ...]] = ("manifest", "repos", "weights", "runtime", "sources")


def digest_of(parts: dict[str, Any]) -> str:
    """The canonical digest over `PARTS`, computed one way only."""
    held = {key: parts[key] for key in PARTS}
    canonical = json.dumps(held, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_of(canonical)


def source_digests() -> dict[str, str]:
    """sha256 per source file, by repo-relative path.

    Per file rather than one rolled-up digest: a mismatch should say WHICH
    runner changed, or the signal is a bare "something moved".
    """
    out: dict[str, str] = {}
    for name in SOURCE_DIRS:
        folder = ROOT / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            out[path.relative_to(ROOT).as_posix()] = sha256_of(path.read_bytes())
    return out


def manifest_digest() -> str:
    return sha256_of(provenance.MANIFEST.read_bytes())


def pinned_repos() -> dict[str, str]:
    """Every pinned repo and its FULL commit, from the manifest.

    Full, never abbreviated: a short sha is a prefix, and a prefix is not an
    identity.
    """
    manifest = provenance.load_manifest()
    out: dict[str, str] = {}
    for key, entry in (manifest.get("upstreams") or {}).items():
        out[f"upstream:{key}"] = str(entry.get("commit", ""))
    for row in manifest.get("consumers", []):
        if row.get("commit"):
            out[f"consumer:{row['id']}"] = str(row["commit"])
    return out


def weight_digests() -> dict[str, str]:
    """Every declared model file's sha256, as `provenance` computes it."""
    out: dict[str, str] = {}
    for row in provenance.weight_identity(provenance.load_manifest()):
        key = f"{row.get('pack', '?')}/{row.get('file', '?')}"
        out[key] = str(row.get("sha256") or row.get("state") or "absent")
    return out


def fixture_digests() -> dict[str, str]:
    """Every fixture the generated evidence is holding, by case name."""
    cases = ROOT / "generated" / "cases.json"
    if not cases.is_file():
        return {}
    held: dict[str, Any] = json.loads(cases.read_text(encoding="utf-8"))
    return {row["case"]: row["fixture_sha256"] for row in held.get("results", [])}


def identity() -> dict[str, Any]:
    """Every answer-changing input, and one digest over all of them."""
    parts: dict[str, Any] = {
        "manifest": manifest_digest(),
        "repos": pinned_repos(),
        "weights": weight_digests(),
        "runtime": provenance.runtime_identity(),
        "sources": source_digests(),
    }
    return {**parts, "digest": digest_of(parts)}


def compare_to(recorded: dict[str, Any]) -> list[str]:
    """What changed between recorded evidence and the tree as it stands.

    Returns one line per differing input, most specific first. An empty list
    means the evidence is current.
    """
    now = identity()
    if recorded.get("digest") == now["digest"]:
        return []

    drift: list[str] = [key + " changed" for key in ("manifest", "runtime") if recorded.get(key) != now[key]]
    for key in ("repos", "weights", "sources"):
        was: dict[str, str] = recorded.get(key) or {}
        has: dict[str, str] = now[key]
        drift.extend(
            f"{key}: {name} {was.get(name, 'absent')[:12]} -> {has.get(name, 'absent')[:12]}"
            for name in sorted(set(was) | set(has))
            if was.get(name) != has.get(name)
        )
    return drift or ["digest changed but no field differs: the digest input set is incomplete"]


def main() -> int:
    now = identity()
    print(f"evidence identity: {now['digest']}")
    print(f"  manifest : {now['manifest'][:16]}")
    print(f"  repos    : {len(now['repos'])} pinned")
    print(f"  weights  : {len(now['weights'])} files")
    print(f"  sources  : {len(now['sources'])} files")

    cases = ROOT / "generated" / "cases.json"
    if not cases.is_file():
        print("\nno generated/cases.json: nothing to check against")
        return 0
    held: dict[str, Any] = json.loads(cases.read_text(encoding="utf-8"))
    recorded = held.get("identity")
    if not recorded:
        print("\ngenerated/cases.json carries NO identity: it cannot be checked for staleness")
        return 1
    drift = compare_to(recorded)
    if not drift:
        print("\nevidence is current")
        return 0
    print(f"\nevidence is STALE, {len(drift)} input(s) changed:")
    for line in drift:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
