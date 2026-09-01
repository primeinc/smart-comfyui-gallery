from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import requests

import proc
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


OID: Final[re.Pattern[bytes]] = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
SIZE: Final[re.Pattern[bytes]] = re.compile(rb"^size (\d+)$", re.MULTILINE)


MOVING: Final[frozenset[str]] = frozenset({"HEAD", "main", "master", "latest", "dev", "trunk"})


@dataclass(frozen=True)
class Pointer:
    repo: str
    revision: str
    path: str
    sha256: str
    size: int

    @property
    def cite(self) -> str:
        return f"{self.repo}@{self.revision}:{self.path}"


@dataclass
class Resolution:
    subject: str
    kind: str
    sha256: str
    verdict: str = ""
    detail: str = ""
    cites: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "RESOLVED"


def _git_bytes(where: Path, *args: str) -> bytes | None:
    code, out, _ = proc.run(["git", "-C", str(where), *args], timeout=proc.LOCAL_SECONDS)
    return out if code == 0 else None


def _blobs(where: Path, names: list[str]) -> dict[str, bytes]:
    if not names:
        return {}
    payload = ("\n".join(names) + "\n").encode("utf-8", errors="surrogateescape")
    code, out, _ = proc.run(["git", "-C", str(where), "cat-file", "--batch"], timeout=proc.LOCAL_SECONDS, stdin=payload)
    if code != 0:
        return {}
    found: dict[str, bytes] = {}
    at = 0
    while at < len(out):
        end = out.find(b"\n", at)
        if end < 0:
            break
        header = out[at:end].split(b" ")
        at = end + 1
        if len(header) < 3:
            continue
        try:
            length = int(header[2])
        except ValueError:
            continue
        found[header[0].decode("ascii", errors="surrogateescape")] = out[at : at + length]

        at += length + 1
    return found


def _text(where: Path, *args: str) -> str:
    out = _git_bytes(where, *args)
    return out.decode("utf-8", errors="surrogateescape").strip() if out else ""


HF_MIRROR: Final[str] = "huggingface.co"


def clone_for(repo_id: str, refs_root: Path) -> tuple[Path | None, str]:
    explicit = refs_root / HF_MIRROR / repo_id
    if (explicit / ".git").exists():
        return explicit, ""
    plain = refs_root / repo_id
    if (plain / ".git").exists():
        return plain, ""
    return None, f"{repo_id}: not cloned at {explicit} or {plain}"


def source_repos(manifest: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for row in manifest.get("vendor_weights", []):
        org = ""
        for part in str(row.get("source", "")).split("+"):
            name = part.strip().removeprefix("datasets/")
            if not name or "?" in name:
                continue

            if "." in name.split("/", 1)[0]:
                continue
            if name.count("/") == 1:
                org = name.split("/", 1)[0]
                found.add(name.split(" ", 1)[0])
            elif org and " " not in name:
                found.add(f"{org}/{name}")
    for row in manifest.get("weights", []):
        for one in row.get("attestations", []):
            repo = str(one.get("repo_id", ""))
            if repo.count("/") == 1:
                found.add(repo)
    return sorted(found)


def index_of(repos: list[str], refs_root: Path) -> tuple[dict[str, list[Pointer]], list[str]]:
    index: dict[str, list[Pointer]] = defaultdict(list)
    unreadable: list[str] = []
    for repo in repos:
        where, why = clone_for(repo, refs_root)
        if where is None:
            unreadable.append(why)
            continue
        revision = _text(where, "rev-parse", "HEAD")
        if not revision or revision in MOVING:
            unreadable.append(f"{repo}: no resolvable revision")
            continue

        listing = _git_bytes(where, "rev-list", "--objects", "--all")
        if listing is None:
            unreadable.append(f"{repo}: cannot enumerate objects")
            continue
        seen: dict[str, str] = {}
        for raw in listing.decode("utf-8", errors="surrogateescape").splitlines():
            name, _, path = raw.partition(" ")
            path = path.strip()
            if not path or name in seen:
                continue
            seen[name] = path
        bodies = _blobs(where, list(seen))
        for name, path in seen.items():
            blob = bodies.get(name)
            if not blob:
                continue
            found = OID.search(blob)
            if found is not None:
                size = SIZE.search(blob)
                digest, length = found.group(1).decode("ascii"), int(size.group(1)) if size else 0
            else:
                digest, length = hashlib.sha256(blob).hexdigest(), len(blob)
            index[digest].append(Pointer(repo=repo, revision=revision, path=path, sha256=digest, size=length))
    return dict(index), unreadable


PUBLISHED_SECONDS: Final[float] = 300.0


def _fetch(url: str, into: Path) -> None:
    into.parent.mkdir(parents=True, exist_ok=True)
    partial = into.with_suffix(into.suffix + ".partial")
    with requests.get(url, timeout=PUBLISHED_SECONDS, stream=True) as body:
        body.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in body.iter_content(1 << 20):
                handle.write(chunk)
    partial.replace(into)


def published_urls(manifest: dict[str, Any], refs_root: Path) -> tuple[dict[str, list[Pointer]], list[str]]:
    out: dict[str, list[Pointer]] = defaultdict(list)
    problems: list[str] = []
    cache = refs_root / "_published_urls"
    for row in manifest.get("vendor_weights", []):
        url, source = str(row.get("url", "")), str(row.get("source", ""))
        if not url:
            continue
        if not source or not url.startswith(f"https://{source}/"):
            problems.append(f"{row.get('file')}: {url} is not under https://{source}/")
            continue
        where = cache / url.removeprefix("https://")
        try:
            if not where.is_file():
                _fetch(url, where)
        except (requests.RequestException, TimeoutError, OSError, ValueError) as problem:
            problems.append(f"{row.get('file')}: {url} -- {type(problem).__name__}: {problem}")
            continue
        digest = provenance.digest_file(where)
        out[digest].append(
            Pointer(repo=source, revision="published", path=url, sha256=digest, size=where.stat().st_size)
        )
    return dict(out), problems


def recorded(manifest: dict[str, Any]) -> list[Resolution]:
    out: list[Resolution] = []
    for row in manifest.get("weights", []):
        where = provenance._weight_root(row) / row["file"]
        digest = provenance.digest_file(where) if where.is_file() else ""
        out.append(
            Resolution(
                subject=f"{row['pack']}/{row['file']}",
                kind="weight",
                sha256=digest,
                verdict="" if digest else "LOCAL_MISSING",
                detail="" if digest else f"{where} is not on this machine",
            )
        )
    out.extend(
        Resolution(
            subject=f"{row['consumer']}: {row['file']}",
            kind="vendor_weight",
            sha256=str(row.get("sha256", "")),
        )
        for row in manifest.get("vendor_weights", [])
    )
    return out


def survey() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    repos = source_repos(manifest)
    index, unreadable = index_of(repos, refs_root)
    served, unfetched = published_urls(manifest, refs_root)
    unreadable.extend(unfetched)
    for digest, pointers in served.items():
        index.setdefault(digest, []).extend(pointers)

    rows = recorded(manifest)
    for one in rows:
        if one.verdict:
            continue
        if not one.sha256:
            one.verdict, one.detail = "NO_DIGEST", "the manifest records no sha256 for it"
            continue
        found = index.get(one.sha256, [])
        if found:
            one.verdict = "RESOLVED"
            one.cites = [p.cite for p in found]
            one.detail = f"published at {len(found)} path(s); {found[0].size:,} B"
        else:
            one.verdict = "UNRESOLVED"
            one.detail = "no indexed lfs pointer publishes these bytes"

    counts: dict[str, int] = {}
    for one in rows:
        counts[one.verdict] = counts.get(one.verdict, 0) + 1
    return {
        "refs_root": str(refs_root),
        "repos_indexed": repos,
        "unreadable_repos": unreadable,
        "digests_indexed": len(index),
        "pointers_indexed": sum(len(v) for v in index.values()),
        "resolutions": [asdict(one) for one in rows],
        "counts": counts,
        "clean": bool(rows) and not unreadable and all(one.ok for one in rows),
    }


def main() -> int:
    out = survey()
    print(f"refs root: {out['refs_root']}")
    print(f"repositories indexed: {len(out['repos_indexed'])}  {out['repos_indexed']}")
    print(f"lfs pointers indexed: {out['pointers_indexed']} over {out['digests_indexed']} distinct digests\n")

    for row in out["resolutions"]:
        mark = "ok  " if row["verdict"] == "RESOLVED" else "FAIL"
        print(f"{mark} {row['subject'][:66]:<66} {row['verdict']}")
        if row["cites"]:
            print(f"       {row['cites'][0]}")
        elif row["detail"]:
            print(f"       {row['detail']}")

    for one in out["unreadable_repos"]:
        print(f"FAIL source repository unreadable: {one}")

    print(f"\nhf resolutions: {out['counts']}")
    if not out["resolutions"]:
        print("NO digests were found to resolve; that is a red result, not a green one")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "hf_citations.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if out["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
