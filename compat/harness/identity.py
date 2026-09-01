from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


APP_DIRS: Final[tuple[str, ...]] = ("vision", "db")


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


#: What decides WHETHER a gate runs and what it enforces. None of it was digested,
#: so a lane could be deleted from a .just module and every staleness check would
#: still report the evidence current. Globbed, because a named list misses modules.
GATE_GLOBS: Final[tuple[str, ...]] = ("*.just", "justfile", "pyproject.toml", "uv.lock", "conftest.py")

GATE_DIRS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("tests", ("*.py", "*.sql")),
    ("metaparse", ("*.py",)),
)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PARTS: Final[tuple[str, ...]] = (
    "manifest",
    "repos",
    "weights",
    "runtime",
    "sources",
    "application",
    "corpus",
    "gates",
)


def digest_of(parts: dict[str, Any]) -> str:
    held = {key: parts[key] for key in PARTS}
    canonical = json.dumps(held, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_of(canonical)


def source_digests() -> dict[str, str]:
    out: dict[str, str] = {}

    # compat's own root: __init__.py, and the ty/pyrefly configs that decide what
    # the `check` lane enforces. Only the named subdirectories were walked before.
    for path in sorted(one for pattern in ("*.py", "*.json", "*.toml") for one in ROOT.glob(pattern)):
        out[path.relative_to(ROOT).as_posix()] = sha256_of(path.read_bytes())

    for name in SOURCE_DIRS:
        folder = ROOT / name
        if not folder.is_dir():
            continue

        for path in sorted(one for pattern in ("*.py", "*.json", "*.toml") for one in folder.rglob(pattern)):
            if "__pycache__" in path.parts:
                continue
            out[path.relative_to(ROOT).as_posix()] = sha256_of(path.read_bytes())
    return out


def gate_digests() -> dict[str, str]:
    repo = ROOT.parent
    out: dict[str, str] = {}
    for pattern in GATE_GLOBS:
        for path in sorted(repo.glob(pattern)):
            if path.is_file():
                out[path.relative_to(repo).as_posix()] = sha256_of(path.read_bytes())
    for name, patterns in GATE_DIRS:
        folder = repo / name
        if not folder.is_dir():
            continue
        for path in sorted(one for pattern in patterns for one in folder.rglob(pattern)):
            if "__pycache__" in path.parts:
                continue
            out[path.relative_to(repo).as_posix()] = sha256_of(path.read_bytes())
    return out


def application_digests() -> dict[str, str]:
    repo = ROOT.parent
    out: dict[str, str] = {}
    for name in APP_DIRS:
        folder = repo / name
        if not folder.is_dir():
            continue
        for path in sorted(one for pattern in ("*.py", "*.sql") for one in folder.rglob(pattern)):
            if "__pycache__" in path.parts:
                continue
            out[path.relative_to(repo).as_posix()] = sha256_of(path.read_bytes())
    for name in ROOT_SOURCES:
        one = repo / name
        if one.is_file():
            out[name] = sha256_of(one.read_bytes())
    return out


_corpus: dict[str, dict[str, str]] = {}


def corpus_digests() -> dict[str, str]:
    if "held" not in _corpus:
        from compat.corpus import index as corpus

        _corpus["held"] = {one.path: one.sha256 for one in corpus.scan_kyc()} if corpus.KYC.is_dir() else {}
    return _corpus["held"]


def manifest_digest() -> str:
    return sha256_of(provenance.MANIFEST.read_bytes())


def pinned_repos() -> dict[str, str]:
    manifest = provenance.load_manifest()
    out: dict[str, str] = {}
    for key, entry in (manifest.get("upstreams") or {}).items():
        out[f"upstream:{key}"] = str(entry.get("commit", ""))
    for row in manifest.get("consumers", []):
        if row.get("commit"):
            out[f"consumer:{row['id']}"] = str(row["commit"])
    return out


def weight_digests() -> dict[str, str]:
    out: dict[str, str] = {}
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    for row in provenance.weight_identity(manifest, refs_root):
        key = f"{row.get('pack', '?')}/{row.get('file', '?')}"
        out[key] = str(row.get("sha256") or row.get("state") or "absent")
    return out


def identity() -> dict[str, Any]:
    parts: dict[str, Any] = {
        "manifest": manifest_digest(),
        "repos": pinned_repos(),
        "weights": weight_digests(),
        "runtime": provenance.runtime_identity(),
        "sources": source_digests(),
        "application": application_digests(),
        "corpus": corpus_digests(),
        "gates": gate_digests(),
    }
    return {**parts, "digest": digest_of(parts)}


def compare_to(recorded: dict[str, Any]) -> list[str]:
    now = identity()
    if recorded.get("digest") == now["digest"]:
        return []

    drift: list[str] = [key + " changed" for key in ("manifest", "runtime") if recorded.get(key) != now[key]]
    for key in ("repos", "weights", "sources", "application", "corpus", "gates"):
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
    print(f"  gates    : {len(now['gates'])} gate/config/test files")

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
