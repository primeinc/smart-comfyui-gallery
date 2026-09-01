from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
import tomllib
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import requests

import proc

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent
MANIFEST: Path = ROOT / "manifest.toml"


@dataclass
class PathProof:
    path: str
    symbol: str
    present: bool
    blob_sha256: str | None = None
    symbol_present: bool | None = None
    locator_only: bool = False
    note: str | None = None


@dataclass
class RepoProof:
    key: str
    repo: str
    pinned_commit: str
    clone: str
    exists: bool
    head_commit: str | None = None
    at_pin: bool = False
    clean: bool | None = None
    dirty_paths: int = 0
    paths: list[PathProof] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _git(where: Path | None, *args: str) -> tuple[int, str]:
    argv: list[str] = ["git"]
    if where is not None:
        argv += ["-C", str(where)]
    argv += list(args)
    code, out, err = proc.text(argv, timeout=proc.LOCAL_SECONDS)
    if code == proc.TIMED_OUT:
        return code, f"timed out after {proc.LOCAL_SECONDS}s: {' '.join(argv)}"
    return code, (out + err).strip()


def _git_bytes(where: Path, *args: str) -> tuple[int, bytes]:
    argv: list[str] = ["git", "-C", str(where), *args]
    code, out, err = proc.run(argv, timeout=proc.LOCAL_SECONDS)
    if code == proc.TIMED_OUT:
        return code, f"timed out after {proc.LOCAL_SECONDS}s: {' '.join(argv)}".encode()
    return code, (out if code == 0 else err)


def defines_symbol(source: str, symbol: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def defined_in(body: list[ast.stmt], names: list[str]) -> bool:
        head, rest = names[0], names[1:]
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == head:
                return not rest or defined_in(node.body, rest)

            if not rest and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(one, ast.Name) and one.id == head for one in targets):
                    return True
        return False

    return defined_in(tree.body, symbol.split("."))


def clone_dir(refs_root: Path, repo_url: str) -> Path:
    tail: str = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    org, _, name = tail.partition("/")
    return refs_root / org / name


def declared_paths(entry: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    spellings: list[Any] = [entry.get("entrypoint"), *entry.get("also_read", []), *entry.get("read", [])]
    for spelling in spellings:
        if not spelling:
            continue
        path, _, symbol = str(spelling).partition("::")
        if path:
            out.append((path, symbol))
    return out


def verify_repo(
    key: str,
    entry: dict[str, Any],
    refs_root: Path,
    *,
    paths_from: dict[str, Any] | None = None,
) -> RepoProof:
    repo_url: str = entry["repo"]
    pinned: str = entry["commit"]
    where: Path = clone_dir(refs_root, repo_url)
    proof = RepoProof(key=key, repo=repo_url, pinned_commit=pinned, clone=str(where), exists=where.is_dir())

    if not proof.exists:
        proof.failures.append(f"clone absent at {where}")
        return proof

    code, head = _git(where, "rev-parse", "HEAD")
    if code != 0:
        proof.failures.append(f"rev-parse failed: {head}")
        return proof
    proof.head_commit = head
    proof.at_pin = head == pinned
    if not proof.at_pin:
        proof.failures.append(f"HEAD {head} is not the pinned {pinned}")

    code, dirty = _git(where, "status", "--porcelain")
    if code != 0:
        proof.failures.append(f"status failed: {dirty}")
    else:
        lines: list[str] = [one for one in dirty.splitlines() if one.strip()]
        proof.dirty_paths = len(lines)
        proof.clean = not lines
        if lines:
            proof.failures.append(f"{len(lines)} uncommitted path(s): the commit does not describe the tree")

    source: dict[str, Any] = paths_from if paths_from is not None else entry
    for path, symbol in declared_paths(source):
        code, blob = _git(where, "show", f"{pinned}:{path}")
        if code != 0:
            proof.paths.append(PathProof(path=path, symbol=symbol, present=False, note=blob[:200]))
            proof.failures.append(f"{path} absent at {pinned[:12]}")
            continue
        digest: str = hashlib.sha256(blob.encode("utf-8", errors="surrogateescape")).hexdigest()

        locator = not path.endswith(".py") or symbol in {"", "__main__", "quickstart", "preprocess"}
        found: bool | None = None
        if not locator:
            found = defines_symbol(blob, symbol)
            if not found:
                proof.failures.append(f"{path} exists at {pinned[:12]} but does not define {symbol}")

        proof.paths.append(
            PathProof(
                path=path,
                symbol=symbol,
                present=True,
                blob_sha256=digest,
                symbol_present=found,
                locator_only=locator,
            )
        )

    return proof


LIBRARIES: tuple[str, ...] = (
    "onnx",
    "numpy",
    "opencv-python",
    "opencv-contrib-python",
    "scikit-image",
    "torch",
    "insightface",
    "mediapipe",
    "face-alignment",
)


IMPORTED: tuple[str, ...] = ("cv2", "numpy", "skimage", "torch", "PIL")


def backend_identity() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None

    imported: dict[str, Any] = {}
    for name in IMPORTED:
        try:
            module = importlib.import_module(name)
        except ImportError as why:
            imported[name] = f"ABSENT: {why}"
            continue
        imported[name] = {
            "version": str(getattr(module, "__version__", "present")),
            "file": str(getattr(module, "__file__", "?")),
        }
    out["imported"] = imported

    engine: dict[str, Any] = {}
    try:
        import onnxruntime

        engine["version"] = onnxruntime.get_version_string()
        engine["build_info"] = onnxruntime.get_build_info()
        engine["available_providers"] = list(onnxruntime.get_available_providers())
        engine["device"] = onnxruntime.get_device()
    except (ImportError, OSError, AttributeError) as why:
        engine["error"] = f"{type(why).__name__}: {why}"
    out["onnxruntime"] = engine
    return out


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _weight_root(row: dict[str, Any]) -> Path:
    package = row.get("root_package")
    if package:
        found = importlib.util.find_spec(package)
        if found is None or not found.origin:
            return Path(row.get("root", f"<{package} not importable>"))
        return Path(found.origin).parent / row.get("root_subdir", "")
    return Path(row["root"])


VENDOR_ROOT: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures"


def vendor_weight_identity(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in manifest.get("vendor_weights", []):
        where = VENDOR_ROOT / row["file"]
        found = where.is_file()
        measured = digest_file(where) if found else None
        size = where.stat().st_size if found else None
        recorded = str(row.get("sha256") or "")
        out.append(
            {
                "consumer": row.get("consumer", "?"),
                "file": row["file"],
                "source": row.get("source", ""),
                "present": found,
                "sha256": measured,
                "recorded_sha256": recorded,
                "bytes": size,
                "recorded_bytes": row.get("bytes"),
                "matches": None if not found else (measured == recorded and size == row.get("bytes")),
            }
        )
    return out


WEIGHT_VERIFIED: Final[str] = "VERIFIED"
WEIGHT_UNATTESTED: Final[str] = "UNATTESTED"
WEIGHT_MISMATCH: Final[str] = "MISMATCH"
WEIGHT_MISSING: Final[str] = "MISSING"


WEIGHT_CONTRADICTED: Final[str] = "CONTRADICTED"


WEIGHT_CORROBORATED: Final[str] = "CORROBORATED"


CANONICAL_AUTHORITY: Final[frozenset[str]] = frozenset({"canonical_vendor", "first_party_consumer_snapshot"})


EVIDENCE_PROVEN: Final[str] = "PROVEN"
EVIDENCE_OBSERVED: Final[str] = "OBSERVED"
EVIDENCE_UNRESOLVABLE: Final[str] = "UNRESOLVABLE"


_LFS_OID = re.compile(r"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
_LFS_SIZE = re.compile(r"^size (\d+)$", re.MULTILINE)
_SHA256 = re.compile(r"\b([0-9a-f]{64})\b")


def _blob_at(where: Path, revision: str, path: str) -> str | None:
    if not where.is_dir():
        return None
    code, out = _git(where, "show", f"{revision}:{path}")
    return out if code == 0 else None


def _resolve_huggingface_snapshot(one: dict[str, Any], refs_root: Path) -> dict[str, Any]:
    where = refs_root / one["repo_id"]
    body = _blob_at(where, one["revision"], one["path"])
    if body is None:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"no {one['repo_id']} clone at {one['revision'][:12]} holding {one['path']}",
        }
    found = _LFS_OID.search(body)
    if not found:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": "the blob at that path is not an lfs pointer",
        }
    size = _LFS_SIZE.search(body)
    return {
        "evidence": EVIDENCE_PROVEN,
        "resolved_sha256": found.group(1),
        "detail": f"lfs oid at {one['revision'][:12]}" + (f", size {size.group(1)}" if size else ""),
    }


def _resolve_git_published_checksum(one: dict[str, Any], refs_root: Path) -> dict[str, Any]:
    where = refs_root / one["repo_id"]
    file, _, line_no = one["path"].partition(":")
    revision = one["revision"] or "HEAD"
    body = _blob_at(where, revision, file)
    if body is None:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"no {one['repo_id']} blob {file} at {revision[:12]}",
        }
    lines = body.splitlines()
    if not line_no.isdigit() or not (1 <= int(line_no) <= len(lines)):
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"{one['path']} is not a line in that blob",
        }
    found = _SHA256.search(lines[int(line_no) - 1])
    if not found:
        return {"evidence": EVIDENCE_UNRESOLVABLE, "resolved_sha256": "", "detail": f"{one['path']} carries no sha256"}

    evidence = EVIDENCE_PROVEN if one["revision"] else EVIDENCE_OBSERVED
    return {"evidence": evidence, "resolved_sha256": found.group(1), "detail": f"read from {one['path']}"}


_GITHUB_API: Final[str] = "https://api.github.com"
ASSET_SECONDS: float = 900.0


def _release_asset_cache(refs_root: Path) -> Path:
    return refs_root / "_release_assets"


ASSET_HOSTS: Final[frozenset[str]] = frozenset({"api.github.com", "github.com", "objects.githubusercontent.com"})


def _checked(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ASSET_HOSTS:
        raise ValueError(f"refusing to fetch {url!r}: not https on {sorted(ASSET_HOSTS)}")
    return url


def _read_json(url: str) -> Any:
    answer = requests.get(
        _checked(url),
        headers={"Accept": "application/vnd.github+json", "User-Agent": "compat"},
        timeout=ASSET_SECONDS,
    )
    answer.raise_for_status()
    return answer.json()


def _fetch_asset(url: str, into: Path) -> None:
    into.parent.mkdir(parents=True, exist_ok=True)
    partial = into.with_suffix(into.suffix + ".partial")
    with requests.get(
        _checked(url),
        headers={"Accept": "application/octet-stream", "User-Agent": "compat"},
        timeout=ASSET_SECONDS,
        stream=True,
    ) as body:
        body.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in body.iter_content(1 << 20):
                handle.write(chunk)
    partial.replace(into)


def _resolve_github_release_asset(one: dict[str, Any], refs_root: Path) -> dict[str, Any]:
    tag, _, name = one["path"].rpartition("/")
    tag = tag.rsplit("/", 1)[-1]
    pinned_id, _, pinned_stamp = one["revision"].partition("@")
    if not pinned_id.isdigit() or not pinned_stamp:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"revision {one['revision']!r} is not <asset_id>@<updated_at>",
        }

    cache = _release_asset_cache(refs_root) / one["repo_id"] / pinned_id
    meta_file = cache / "asset.json"
    try:
        if meta_file.is_file():
            asset = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            release = _read_json(f"{_GITHUB_API}/repos/{one['repo_id']}/releases/tags/{tag}")
            matched = [row for row in release.get("assets", []) if row.get("name") == name]
            if not matched:
                return {
                    "evidence": EVIDENCE_UNRESOLVABLE,
                    "resolved_sha256": "",
                    "detail": f"{one['repo_id']} {tag} publishes no asset named {name}",
                }
            asset = matched[0]
            cache.mkdir(parents=True, exist_ok=True)
            meta_file.write_text(json.dumps(asset, indent=2, sort_keys=True), encoding="utf-8", newline="")
    except (requests.RequestException, TimeoutError, OSError, ValueError) as problem:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"could not read {one['repo_id']} {tag}: {type(problem).__name__}: {problem}",
        }

    if str(asset.get("id")) != pinned_id or str(asset.get("updated_at")) != pinned_stamp:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": (
                f"{name} is now asset {asset.get('id')} at {asset.get('updated_at')}, "
                f"not {pinned_id} at {pinned_stamp}: the vendor replaced it"
            ),
        }

    where = cache / name
    try:
        if not where.is_file():
            _fetch_asset(str(asset["browser_download_url"]), where)
    except (requests.RequestException, TimeoutError, OSError, ValueError, KeyError) as problem:
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"could not fetch {name}: {type(problem).__name__}: {problem}",
        }

    size = where.stat().st_size
    if size != int(asset.get("size", -1)):
        return {
            "evidence": EVIDENCE_UNRESOLVABLE,
            "resolved_sha256": "",
            "detail": f"mirrored {name} is {size} bytes; the release says {asset.get('size')}",
        }
    return {
        "evidence": EVIDENCE_PROVEN,
        "resolved_sha256": digest_file(where),
        "detail": f"vendor asset {pinned_id} at {pinned_stamp}, {size} bytes, hashed from the mirror",
    }


RESOLVERS: Final[dict[str, Any]] = {
    "huggingface_snapshot": _resolve_huggingface_snapshot,
    "git_published_checksum": _resolve_git_published_checksum,
    "github_release_asset": _resolve_github_release_asset,
}


def _attestations(row: dict[str, Any], refs_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for one in row.get("attestations", []):
        located = {
            "source_class": one.get("source_class", ""),
            "repo_id": one.get("repo_id", ""),
            "revision": one.get("revision", ""),
            "path": one.get("path", ""),
            "authority": one.get("authority", ""),
            "note": one.get("note", ""),
        }
        resolve = RESOLVERS.get(located["source_class"])
        derived = (
            resolve(located, refs_root)
            if resolve
            else {
                "evidence": EVIDENCE_UNRESOLVABLE,
                "resolved_sha256": "",
                "detail": f"no resolver for source_class {located['source_class']!r}",
            }
        )
        out.append({**located, **derived})
    return out


def weight_state(measured: str | None, found: bool, attestations: list[dict[str, Any]]) -> str:
    if not found or measured is None:
        return WEIGHT_MISSING
    resolved = [one for one in attestations if one.get("resolved_sha256")]
    digests = {one["resolved_sha256"] for one in resolved}
    if len(digests) > 1:
        return WEIGHT_CONTRADICTED
    if digests and digests != {measured}:
        return WEIGHT_MISMATCH
    proven = [one for one in resolved if one["evidence"] == EVIDENCE_PROVEN]
    if not proven:
        return WEIGHT_UNATTESTED

    if any(one["authority"] in CANONICAL_AUTHORITY for one in proven):
        return WEIGHT_VERIFIED
    return WEIGHT_CORROBORATED


def weight_identity(manifest: dict[str, Any], refs_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in manifest.get("weights", []):
        where = _weight_root(row) / row["file"]
        found = where.is_file()
        measured = digest_file(where) if found else None
        attestations = _attestations(row, refs_root)
        out.append(
            {
                "pack": row["pack"],
                "file": row["file"],
                "path": str(where),
                "role": row.get("role", ""),
                "present": found,
                "bytes": where.stat().st_size if found else 0,
                "sha256": measured,
                "attestations": attestations,
                "state": weight_state(measured, found, attestations),
            }
        )
    return out


def runtime_identity() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backends": backend_identity(),
    }


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


@dataclass
class RuntimeProof:
    package: str
    path: str
    pinned_commit: str
    installed_file: str | None = None
    installed_sha256: str | None = None
    pinned_sha256: str | None = None
    matches: bool = False
    note: str | None = None


def installed_module_root(package: str) -> Path | None:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def verify_runtime(package: str, repo: Path, commit: str, path: str, installed_path: str) -> RuntimeProof:
    proof = RuntimeProof(package=package, path=path, pinned_commit=commit)

    root = installed_module_root(package)
    if root is None:
        proof.note = f"{package} is not importable in this environment"
        return proof

    where = root.parent / installed_path
    proof.installed_file = str(where)
    if not where.is_file():
        proof.note = f"pinned path is not present in the installed package at {where}"
        return proof

    code, blob = _git_bytes(repo, "show", f"{commit}:{path}")
    if code != 0:
        proof.note = f"git show failed: {blob.decode('utf-8', 'replace')[:200]}"
        return proof

    proof.pinned_sha256 = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
    proof.installed_sha256 = hashlib.sha256(where.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    proof.matches = proof.installed_sha256 == proof.pinned_sha256
    if not proof.matches:
        proof.note = "the installed package is NOT the pinned commit; evidence from it would cite the wrong source"
    return proof


def verify_runtimes(manifest: dict[str, Any], refs_root: Path) -> list[RuntimeProof]:
    out: list[RuntimeProof] = []
    for row in manifest.get("runtimes", []):
        upstream = manifest["upstreams"][row["upstream"]]
        out.append(
            verify_runtime(
                package=row["package"],
                repo=clone_dir(refs_root, upstream["repo"]),
                commit=upstream["commit"],
                path=row["path"],
                installed_path=row["installed_path"],
            )
        )
    return out


def verify_all(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    refs_root: Path = (repo_root / manifest["refs_root"]).resolve()
    proofs: list[RepoProof] = []

    upstreams: dict[str, Any] = manifest.get("upstreams", {})
    for key, entry in upstreams.items():
        proofs.append(verify_repo(f"upstream:{key}", entry, refs_root))

    consumers: list[dict[str, Any]] = manifest.get("consumers", [])
    for consumer in consumers:
        host: str | None = consumer.get("entrypoint_in")
        if host:
            entry = dict(upstreams[host])
            proofs.append(verify_repo(f"consumer:{consumer['id']}@{host}", entry, refs_root, paths_from=consumer))
        else:
            proofs.append(verify_repo(f"consumer:{consumer['id']}", consumer, refs_root))

    runtimes: list[RuntimeProof] = verify_runtimes(manifest, refs_root)
    weights: list[dict[str, Any]] = weight_identity(manifest, refs_root)
    vendor_weights: list[dict[str, Any]] = vendor_weight_identity(manifest)

    from compat.harness import identity as evidence_identity

    return {
        "manifest_version": manifest["version"],
        "recorded_at": manifest["recorded_at"],
        "identity": evidence_identity.identity()["digest"],
        "runtime": runtime_identity(),
        "refs_root": str(refs_root),
        "repos": [asdict(one) for one in proofs],
        "runtimes": [asdict(one) for one in runtimes],
        "weights": weights,
        "population": {
            "total": len(consumers),
        },
        "vendor_weights": vendor_weights,
        "provenance_ok": (
            all(one.ok for one in proofs)
            and all(one.matches for one in runtimes)
            and all(one["present"] for one in weights)
            and all(one["matches"] is not False for one in vendor_weights)
            and all(one["state"] == WEIGHT_VERIFIED for one in weights)
        ),
    }


def report(out: dict[str, Any]) -> None:
    repos: list[dict[str, Any]] = out["repos"]
    for repo in repos:
        mark: str = "ok  " if not repo["failures"] else "FAIL"
        print(f"{mark} {repo['key']:<46} {repo['pinned_commit'][:12]}  paths={len(repo['paths'])}")
        for failure in repo["failures"]:
            print(f"       ! {failure}")
        for one in repo["paths"]:
            if not one["present"]:
                print(f"       - {one['path']}::{one['symbol']}  ABSENT AT PIN")
                continue
            if one["locator_only"]:
                grade = "locator"
            elif one["symbol_present"]:
                grade = "symbol "
            else:
                grade = "NO SYM "
            print(f"       {grade} {one['path']}::{one['symbol']}  blob {one['blob_sha256'][:12]}")

    for row in out["runtimes"]:
        mark = "ok  " if row["matches"] else "FAIL"
        state = "matches pin" if row["matches"] else (row["note"] or "does not match pin")
        print(f"{mark} runtime:{row['package']:<38} {row['pinned_commit'][:12]}  {state}")

    for row in out["weights"]:
        mark = "ok  " if row["state"] == WEIGHT_VERIFIED else "FAIL"
        digest = row["sha256"][:12] if row["sha256"] else "ABSENT"
        print(f"{mark} weight:{row['pack']}/{row['file']:<32} {digest}  {row['bytes']:>12,} bytes  {row['state']}")
        for one in row["attestations"]:
            where = f"{one['repo_id']}@{one['revision'][:12]}" if one["revision"] else one["repo_id"]
            print(f"         {one['evidence']:<9} {one['source_class']:<24} {where} {one['path']}")

    tally: dict[str, int] = {}
    for one in out["weights"]:
        tally[one["state"]] = tally.get(one["state"], 0) + 1
    print("\nweights by attestation state: " + ", ".join(f"{count} {state}" for state, count in sorted(tally.items())))

    pop: dict[str, Any] = out["population"]
    print(f"\npopulation: {pop['total']} consumers")
    print(f"\nprovenance: {'PASS' if out['provenance_ok'] else 'FAIL'}")


def main() -> int:
    repo_root: Path = ROOT.parent
    manifest: dict[str, Any] = load_manifest()
    out: dict[str, Any] = verify_all(manifest, repo_root)
    report(out)

    where: Path = ROOT / "generated" / "provenance.json"
    where.parent.mkdir(parents=True, exist_ok=True)

    with where.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"\nwrote {where}")

    return 0 if out["provenance_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
