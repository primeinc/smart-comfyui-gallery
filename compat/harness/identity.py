"""What the evidence is an answer TO.

A generated file that does not name its inputs cannot be stale, because
nothing can disagree with it. This computes one digest over every input that
can change a case's outcome, and `run.py` writes it into the evidence. A
mismatch between the recorded digest and the current one means the evidence
answers a question nobody asked any more.

Seven inputs, and dropping any one of them makes the digest a decoration:

    manifest      the manifest bytes, so a threshold edit is visible
    repos         every pinned repo and its FULL commit
    weights       every model file's sha256
    runtime       interpreter, platform, and the package versions that run
    sources       the source of every runner and preprocessor
    application   the source of the application code the storage lane RUNS
    corpus        every corpus photograph, by content

`sources` closes the hole the pins leave open: a runner edited in place
changes what a case DOES while every pin and weight stays identical.

`application` and `corpus` close the two it leaves open in turn. The storage
lane executes `vision.faces` and `db.*` through `storage/gallery_v45.py`, so
an edit there changes the headline answer with nothing else moving; and
`corpus/loaded.shots()` selects four photographs by min-sha256 per bucket, so
adding one to the tree changes which four every baseline was computed from.
Both were outside the digest, and the staleness lane reported "evidence is
current" across either.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Every directory whose source decides what a case computes, `harness`
#: included: the executor's comparison and verdict logic is as answer-changing
#: as any runner's. Application packages are hashed whole, not by entry point.
APP_DIRS: Final[tuple[str, ...]] = ("vision", "db")

#: Repository-root modules the suite executes. `proc.py` carries every lane's
#: git reads, every shard launch and every timeout, and sits in no source
#: directory, so nothing else here notices when it changes.
ROOT_SOURCES: Final[tuple[str, ...]] = ("proc.py",)

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
PARTS: Final[tuple[str, ...]] = (
    "manifest",
    "repos",
    "weights",
    "runtime",
    "sources",
    "application",
    "corpus",
)


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
        # *.json as well as *.py: a checked-in fixture or table under a source
        # directory decides what a case computes exactly as much as the code
        # that reads it, and a glob that only saw code would not notice it move.
        for path in sorted(one for pattern in ("*.py", "*.json") for one in folder.rglob(pattern)):
            if "__pycache__" in path.parts:
                continue
            out[path.relative_to(ROOT).as_posix()] = sha256_of(path.read_bytes())
    return out


def application_digests() -> dict[str, str]:
    """sha256 per application source file the compat suite executes.

    `compat/storage/gallery_v45.py` is the only lane that reaches out of
    `compat/`, and it reaches into the code under test. Without this the one
    question the suite exists to answer could change with no input moving.
    """
    repo = ROOT.parent
    out: dict[str, str] = {}
    for name in APP_DIRS:
        folder = repo / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            out[path.relative_to(repo).as_posix()] = sha256_of(path.read_bytes())
    for name in ROOT_SOURCES:
        one = repo / name
        if one.is_file():
            out[name] = sha256_of(one.read_bytes())
    return out


#: Memoised per process: the scan streams a sha256 over every KYC image, and
#: the tree it reads is one this suite never writes.
_corpus: dict[str, dict[str, str]] = {}


def corpus_digests() -> dict[str, str]:
    """Every corpus photograph, by absolute path and content.

    The WHOLE scan, not the four shots a run uses: `loaded.shots()` chooses by
    min-sha256 within each (identity, role) bucket, so a photograph added to a
    bucket can displace the one every baseline was computed from without being
    used itself.

    Absent corpus records an empty mapping rather than raising -- the case
    lanes already report UNSUPPORTED for that, and an identity that could not
    be computed is not the place to find out.
    """
    if "held" not in _corpus:
        from compat.corpus import index as corpus

        _corpus["held"] = {one.path: one.sha256 for one in corpus.scan_kyc()} if corpus.KYC.is_dir() else {}
    return _corpus["held"]


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
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    for row in provenance.weight_identity(manifest, refs_root):
        key = f"{row.get('pack', '?')}/{row.get('file', '?')}"
        out[key] = str(row.get("sha256") or row.get("state") or "absent")
    return out


def identity() -> dict[str, Any]:
    """Every answer-changing input, and one digest over all of them."""
    parts: dict[str, Any] = {
        "manifest": manifest_digest(),
        "repos": pinned_repos(),
        "weights": weight_digests(),
        "runtime": provenance.runtime_identity(),
        "sources": source_digests(),
        "application": application_digests(),
        "corpus": corpus_digests(),
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
    for key in ("repos", "weights", "sources", "application", "corpus"):
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
    print(f"  app code : {len(now['application'])} files")
    print(f"  corpus   : {len(now['corpus'])} photographs")

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
