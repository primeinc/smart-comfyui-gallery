from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


#: Top-level directories the identity does NOT cover, each with its reason.
#: EVERYTHING ELSE git tracks at the top level is covered.
#:
#: Exclusion form, never an include list: a remembered list can omit a tree
#: silently -- even a gate's own policy, while carrying the tests that check
#: it -- and then editing rules moves nothing while editing tests moves it.
#:
#: compat/ is absent because `sources` already carries it, keyed relative to
#: compat/ rather than to the repo root.
DIRS_IGNORED: Final[dict[str, str]] = {
    "compat": "carried by `sources`, keyed relative to compat/ rather than the repo root",
    "docs": (
        "prose, and the only tree nothing reads: the vale lane names db, vision, sg_web, metaparse, "
        "sglint, story_renderers, tests and benchmarks, and not docs. No gate's behaviour and no "
        "producer's output depends on a word in it, so a ruling recorded here would otherwise stale "
        "every artifact in the tree."
    ),
}


ROOT_SOURCES: Final[tuple[str, ...]] = ("proc.py",)

#: Directories under compat/ the identity does not cover. Exclusion form,
#: for the same reason as DIRS_IGNORED: a load-bearing decision is stated
#: here with its reason, never carried silently by an enumeration elsewhere.
COMPAT_IGNORED: Final[dict[str, str]] = {
    "generated": (
        "the evidence itself. Digesting it would make the identity self-referential -- regenerating "
        "any artifact would change the identity that stamps it, and no run could ever be current."
    ),
}


#: Root gate/config files are digested BY EXCLUSION -- everything at the
#: repo root minus this list. lefthook.yml decides whether the gates run at
#: all; pytest.ini, .vale.ini and biome.json decide what they enforce.
GATE_IGNORED: Final[frozenset[str]] = frozenset(
    {
        "LICENSE",  # legal text; cannot change what any gate does
        "faceefind.png",  # a screenshot asset
    }
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

    # compat's own root, ON DISK like gate_digests and with no extension
    # list: an untracked file here still changes what the `check` lane
    # enforces, and a pattern set silently skips whatever it did not name.
    for path in sorted(one for one in ROOT.iterdir() if one.is_file()):
        out[path.name] = sha256_of(path.read_bytes())

    for relative in sorted(_tracked(ROOT.parent)):
        inside = relative.removeprefix("compat/")
        if inside == relative or "/" not in inside or inside.split("/", 1)[0] in COMPAT_IGNORED:
            continue
        path = ROOT.parent / relative
        if not path.is_file():
            raise RuntimeError(
                f"git tracks {relative} but it is not a readable file; the identity cannot skip it silently"
            )
        out[inside] = sha256_of(path.read_bytes())
    return out


def tracked_root_files(repo: Path) -> list[str]:
    """Everything git tracks at the repo root. The tree is the authority on what
    exists; a list in this file is only ever the authority on what someone
    remembered. An unreadable index is a failure, never an empty gate set."""
    import proc

    code, out, err = proc.text(["git", "-C", str(repo), "ls-files", "--", ":(glob)*"], timeout=proc.LOCAL_SECONDS)
    if code != 0:
        raise RuntimeError(f"git ls-files failed ({code}) enumerating the gate surface: {err.strip()[:200]}")
    return sorted(one for one in out.splitlines() if one and "/" not in one)


def _tracked(repo: Path) -> list[str]:
    import proc

    # core.quotepath=false: a non-ASCII tracked name arrives as literal
    # UTF-8 rather than a C-quoted string no path on disk resolves to.
    code, out, err = proc.text(
        ["git", "-C", str(repo), "-c", "core.quotepath=false", "ls-files"], timeout=proc.LOCAL_SECONDS
    )
    if code != 0:
        raise RuntimeError(f"git ls-files failed ({code}) enumerating the tree: {err.strip()[:200]}")
    return [one for one in out.splitlines() if one]


def tracked_trees(repo: Path) -> list[str]:
    """Every top-level directory git tracks, minus the declared exclusions.

    The tree is the authority on what exists; a list here is only ever the
    authority on what somebody remembered -- and a remembered list can
    forget even a gate's own policy while carrying the tests that check it.
    """
    return sorted({one.split("/", 1)[0] for one in _tracked(repo) if "/" in one} - set(DIRS_IGNORED))


def gate_digests() -> dict[str, str]:
    repo = ROOT.parent
    out: dict[str, str] = {}
    # ON DISK, not git-tracked: `just` reads *.just from the filesystem whether or
    # not git knows about it, so an untracked lane module changes what runs while
    # being invisible to the index. iterdir also sees dotfiles like .vale.ini.
    for path in sorted(repo.iterdir()):
        if path.is_file() and path.name not in GATE_IGNORED:
            out[path.name] = sha256_of(path.read_bytes())
    return out


def application_digests() -> dict[str, str]:
    """Every tracked file under every tracked top-level tree, by EXCLUSION.

    Not an extension list either: a pattern set reads as coverage while a
    shell script, a .ts, or a style rule inside a covered tree changes what
    the gates do without moving the identity. What git tracks is what ships.
    """
    repo = ROOT.parent
    out: dict[str, str] = {}
    covered = set(tracked_trees(repo))
    for relative in sorted(_tracked(repo)):
        if "/" not in relative or relative.split("/", 1)[0] not in covered:
            continue
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(
                f"git tracks {relative} but it is not a readable file; the identity cannot skip it silently"
            )
        out[relative] = sha256_of(path.read_bytes())
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


_memo: dict[str, dict[str, Any]] = {}


def forget() -> None:
    # G3 widened this to the whole gate/test surface, so per-call recomputation is
    # no longer cheap. A control that mutates a file on disk must invalidate it;
    # everything else gets one snapshot per run, which a shared worktree wants too.
    _memo.clear()
    _corpus.clear()


def identity() -> dict[str, Any]:
    if "held" in _memo:
        return _memo["held"]
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
    _memo["held"] = {**parts, "digest": digest_of(parts)}
    return _memo["held"]


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
