"""Resolve every weight digest in the manifest against the revision it came from.

`cite_check.py` does this for source: `org/repo@sha:path:line`, read with
`git show` out of the pinned commit, so a moved HEAD cannot silently pass. A
weight is the same claim about a different kind of file, and needs one thing
source does not -- the CONTENT DIGEST.

A Hugging Face repository is a git repository whose large files are git-lfs
pointers, and the pointer body IS the content address:

    version https://git-lfs.github.com/spec/v1
    oid sha256:4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf
    size 260665334

So a clone with `GIT_LFS_SKIP_SMUDGE=1` attests a 260 MB weight while holding
none of its bytes: the revision is immutable and the pointer names the content.

WHY THIS INDEXES RATHER THAN LOOKS UP
-------------------------------------
`[[vendor_weights]]` records 30 digests against a `source` naming a repository
and no revision, and `vendor_weight_identity` compared each to the digest the
manifest itself recorded -- our hash against our own note of our hash, which
cannot come out the other way. Every pointer in every cloned source repo is
read once into a digest index, and each recorded digest is resolved through it.
That turns "we wrote this number down" into "this revision of this repository
publishes these bytes at this path", and it finds the path without anyone
typing it.

A digest nothing publishes is UNRESOLVED. That is a red result and not a
missing feature: it means the only thing asserting those bytes is this
repository.

Exit 1 if any recorded digest is unresolved, or if the index is empty -- a
checker over nothing exits 0 and reports that everything resolved, which is the
false pass this file exists to refuse.
"""

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


#: The pointer body. Anchored per line so a file merely MENTIONING an oid
#: cannot be read as one.
OID: Final[re.Pattern[bytes]] = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.MULTILINE)
SIZE: Final[re.Pattern[bytes]] = re.compile(rb"^size (\d+)$", re.MULTILINE)

#: A ref that moves is not a pin: it names whatever the clone is at today, so
#: the claim can never be re-checked and never goes stale visibly.
MOVING: Final[frozenset[str]] = frozenset({"HEAD", "main", "master", "latest", "dev", "trunk"})


@dataclass(frozen=True)
class Pointer:
    """One file, at one revision, and the content it names."""

    repo: str
    revision: str
    path: str
    sha256: str
    size: int

    @property
    def cite(self) -> str:
        """The citation form `cite_check` uses: org/repo@sha:path."""
        return f"{self.repo}@{self.revision}:{self.path}"


@dataclass
class Resolution:
    """One recorded digest, and where -- if anywhere -- it is published."""

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
    """One git call whose output is bytes, never decoded.

    Bytes, not text: a weights repository holds real binaries, and decoding
    every blob as utf-8 raises inside a reader thread on the first one.
    Decoding is not needed anyway -- the pointer is ASCII and the regex runs
    on bytes.

    Through `shell.run`, which writes the streams to files rather than pipes.
    This lane is where the pipe deadlock was found: the call had already timed
    out and the lane still never returned.
    """
    code, out, _ = proc.run(["git", "-C", str(where), *args], timeout=proc.LOCAL_SECONDS)
    return out if code == 0 else None


def _blobs(where: Path, names: list[str]) -> dict[str, bytes]:
    """Every named object's contents, from ONE git process.

    This used to spawn `git cat-file blob <name>` once per object, over every
    object in every cloned weight repository. That is tens of thousands of
    process creations for a survey whose whole job is to read them, and on
    Windows each one costs more than the read does -- the lane took longer to
    spawn processes than to do anything with what they returned.

    `cat-file --batch` is git's own answer: names on stdin, and for each a
    header line `<oid> <type> <size>` followed by exactly `size` bytes and a
    newline. The size is read from the header rather than by scanning, so a
    blob containing a newline cannot desynchronise the parse. A name git does
    not have answers `<name> missing`, which is skipped rather than guessed at.
    """
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
            # `<oid> missing`, or a line this parse does not understand. Either
            # way there is no length to skip, so the next line is the next
            # record and nothing is consumed here.
            continue
        try:
            length = int(header[2])
        except ValueError:
            continue
        found[header[0].decode("ascii", errors="surrogateescape")] = out[at : at + length]
        # The body, then the newline git writes after it.
        at += length + 1
    return found


def _text(where: Path, *args: str) -> str:
    out = _git_bytes(where, *args)
    return out.decode("utf-8", errors="surrogateescape").strip() if out else ""


#: Where a Hugging Face MODEL repo is mirrored, when its `<org>/<name>` is
#: also a GitHub repository. `refs/<org>/<name>` carries no host, so the two
#: cannot both live there.
HF_MIRROR: Final[str] = "huggingface.co"


def clone_for(repo_id: str, refs_root: Path) -> tuple[Path | None, str]:
    """The clone that answers for a weight source, and why there is none.

    `ByteDance/InfiniteYou` and `TencentARC/PhotoMaker` each name a GitHub
    code repository AND a Hugging Face model repository, and only one of them
    fits at `refs/<org>/<name>`. Reading whichever was there made the lane say
    "no indexed lfs pointer publishes these bytes" about weights the vendor
    does publish -- a false claim about the vendor rather than about the
    mirror.

    So the host-explicit path is preferred where it exists. A source with only
    the plain clone is still indexed: several are GitHub repositories that
    publish digests and inline configs, and refusing them would trade one
    silent wrong answer for a louder one.
    """
    explicit = refs_root / HF_MIRROR / repo_id
    if (explicit / ".git").exists():
        return explicit, ""
    plain = refs_root / repo_id
    if (plain / ".git").exists():
        return plain, ""
    return None, f"{repo_id}: not cloned at {explicit} or {plain}"


def source_repos(manifest: dict[str, Any]) -> list[str]:
    """Every repository the manifest names as a weight source.

    Taken from the manifest, never a second list. A repo this misses is never
    indexed, so its digests read as unpublished -- the same silent pass this
    file exists to prevent, moved up one level.

    A `source` may name more than one repository (`h94/IP-Adapter-FaceID +
    h94/IP-Adapter`) and may carry a `datasets/` prefix, which is part of the
    URL and not of the clone path.
    """
    found: set[str] = set()
    for row in manifest.get("vendor_weights", []):
        org = ""
        for part in str(row.get("source", "")).split("+"):
            name = part.strip().removeprefix("datasets/")
            if not name or "?" in name:
                continue
            # A hostname is not a repository. `storage.googleapis.com/...`
            # is a download URL, and no clone answers for it.
            if "." in name.split("/", 1)[0]:
                continue
            if name.count("/") == 1:
                org = name.split("/", 1)[0]
                found.add(name.split(" ", 1)[0])
            elif org and " " not in name:
                # `TencentARC/PhotoMaker + PhotoMaker-V2` -- the second name
                # inherits the org, and dropping it left PhotoMaker's two
                # weights reading as published by nobody.
                found.add(f"{org}/{name}")
    for row in manifest.get("weights", []):
        for one in row.get("attestations", []):
            repo = str(one.get("repo_id", ""))
            if repo.count("/") == 1:
                found.add(repo)
    return sorted(found)


def index_of(repos: list[str], refs_root: Path) -> tuple[dict[str, list[Pointer]], list[str]]:
    """Every lfs pointer in every cloned source repo, by content digest.

    Returns the index and the repositories that could not be read, which are
    reported rather than skipped: an absent clone makes its digests look
    unpublished, and that is a different fact from nobody publishing them.
    """
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
        # EVERY object, not the tip tree: a weight that moved or was
        # superseded is still published by the revision that held it.
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
                # NOT a pointer, so the blob IS the content: a file under
                # the repository's lfs threshold is stored inline.
                digest, length = hashlib.sha256(blob).hexdigest(), len(blob)
            index[digest].append(Pointer(repo=repo, revision=revision, path=path, sha256=digest, size=length))
    return dict(index), unreadable


#: Seconds a published-file fetch may take. Bigger than a local call and
#: smaller than a model download: these are the small files vendors publish
#: loose rather than in a repository.
PUBLISHED_SECONDS: Final[float] = 300.0


def _fetch(url: str, into: Path) -> None:
    """Stream one published file to disk, `.partial` until whole."""
    into.parent.mkdir(parents=True, exist_ok=True)
    partial = into.with_suffix(into.suffix + ".partial")
    with requests.get(url, timeout=PUBLISHED_SECONDS, stream=True) as body:
        body.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in body.iter_content(1 << 20):
                handle.write(chunk)
    partial.replace(into)


def published_urls(manifest: dict[str, Any], refs_root: Path) -> tuple[dict[str, list[Pointer]], list[str]]:
    """Weights the vendor serves as a FILE at a URL, hashed from those bytes.

    Not every vendor uses a repository. Google publishes `face_landmarker.task`
    from `storage.googleapis.com`, where there is no lfs pointer to read, and
    an index built only from clones reported it as published by nobody.

    The URL must sit under the row's own `source`, which is what makes this an
    attestation rather than a fetch: a locator free to name any host would
    hash whatever it was pointed at and report it as the vendor's.

    A fetch that fails is RETURNED, never swallowed. Dropping it silently
    leaves the row reading UNRESOLVED for a reason that is not the true one.
    """
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
    """Every digest the manifest asserts, from both weight tables."""
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
        # An empty survey is not a clean one, and neither is one whose sources
        # could not be read.
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
