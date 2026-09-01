from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import requests

from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


FETCH_SECONDS: Final[float] = 3600.0


@dataclass
class Provisioned:
    consumer: str
    kind: str
    source: str
    path: str
    verdict: str
    detail: str = ""
    bytes: int = 0

    @property
    def ok(self) -> bool:
        return self.verdict in ("PRESENT", "FETCHED")


def root_of(manifest: dict[str, Any]) -> Path:
    block: dict[str, Any] = manifest.get("provisioned", {})
    override = os.environ.get(str(block.get("override_env", "")))
    return Path(override or str(block.get("root", ""))).resolve()


def _under(root: Path, path: str) -> Path | None:
    where = (root / path).resolve()
    if where == root.resolve() or root.resolve() not in where.parents:
        return None
    return where


def _already_here(kind: str, where: Path) -> bool:
    return kind != "hf_snapshot" and where.exists() and bool(_size(where))


def _size(where: Path) -> int:
    if where.is_file():
        return where.stat().st_size
    return sum(one.stat().st_size for one in where.rglob("*") if one.is_file())


def _fetch_url(url: str, into: Path) -> None:
    into.parent.mkdir(parents=True, exist_ok=True)
    partial = into.with_suffix(into.suffix + ".partial")
    with requests.get(url, timeout=FETCH_SECONDS, stream=True) as body:
        body.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in body.iter_content(1 << 20):
                handle.write(chunk)
    partial.replace(into)


def _fetch_hf_file(repo: str, name: str, into: Path) -> None:
    from huggingface_hub import hf_hub_download

    into.parent.mkdir(parents=True, exist_ok=True)
    got = Path(hf_hub_download(repo, name))
    partial = into.with_suffix(into.suffix + ".partial")
    partial.write_bytes(got.read_bytes())
    partial.replace(into)


def _fetch_hf_snapshot(repo: str, into: Path) -> None:
    from huggingface_hub import snapshot_download

    into.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(into))


def fetch(row: dict[str, Any], root: Path) -> Provisioned:
    kind, source, path = str(row.get("kind", "")), str(row.get("source", "")), str(row.get("path", ""))
    held = Provisioned(str(row.get("consumer", "")), kind, source, path, "UNDECLARED")
    if not path:
        held.detail = "no destination path declared"
        return held

    bounded = _under(root, path)
    if bounded is None:
        held.detail = f"{path!r} does not resolve inside {root}"
        return held
    where = bounded
    if _already_here(kind, where):
        held.verdict, held.bytes = "PRESENT", _size(where)
        return held

    try:
        if kind == "url":
            url = str(row.get("url", ""))
            if not url.startswith(f"https://{source}/"):
                held.detail = f"{url} is not under https://{source}/"
                return held
            _fetch_url(url, where)
        elif kind == "hf_file":
            _fetch_hf_file(source, str(row.get("file", "")), where)
        elif kind == "hf_snapshot":
            _fetch_hf_snapshot(source, where)
        else:
            held.detail = f"no downloader for kind {kind!r}"
            return held
    except (requests.RequestException, TimeoutError, OSError, ValueError, KeyError) as problem:
        held.verdict = "FAILED"
        held.detail = f"{type(problem).__name__}: {problem}"
        if row.get("gated"):
            held.detail += " -- this source is gated; run `hf auth login` once, then re-run"
        return held

    held.bytes = _size(where)
    if not held.bytes:
        held.verdict, held.detail = "FAILED", "the download left nothing on disk"
        return held
    held.verdict = "FETCHED"
    return held


def _verified(where: Path, expected: str) -> str:
    if not expected:
        where.unlink(missing_ok=True)
        return "nothing declares a digest for this artifact; the bytes were not kept"
    got = provenance.digest_file(where)
    if got == expected:
        return ""
    where.unlink(missing_ok=True)
    return f"downloaded bytes hash {got[:16]}, not the declared {expected[:16]}; removed"


def _from_hub(repo: str, revision: str, path: str, into: Path, expected: str) -> Provisioned:
    held = Provisioned("", "hf_pinned", f"{repo}@{revision[:12]}", str(into), "FETCHED")
    if not expected:
        held.verdict = "UNRESOLVED"
        held.detail = "no declared digest to verify against, so the download is refused"
        return held
    try:
        from huggingface_hub import hf_hub_download

        got = Path(hf_hub_download(repo, path, revision=revision or None))
        into.parent.mkdir(parents=True, exist_ok=True)
        partial = into.with_suffix(into.suffix + ".partial")
        partial.write_bytes(got.read_bytes())
        partial.replace(into)
    except (requests.RequestException, TimeoutError, OSError, ValueError, KeyError) as problem:
        held.verdict, held.detail = "FAILED", f"{type(problem).__name__}: {problem}"
        return held
    wrong = _verified(into, expected)
    if wrong:
        held.verdict, held.detail = "FAILED", wrong
        return held
    held.bytes = _size(into)
    return held


def pack_weights(manifest: dict[str, Any]) -> list[Provisioned]:
    out: list[Provisioned] = []
    for row in manifest.get("weights", []):
        where = provenance._weight_root(row) / row["file"]
        held = Provisioned(row.get("pack", ""), "weight", "", str(where), "PRESENT")
        if where.is_file():
            held.bytes = _size(where)
            out.append(held)
            continue
        pinned = [
            one
            for one in row.get("attestations", [])
            if one.get("source_class") == "huggingface_snapshot" and one.get("revision") and one.get("path")
        ]
        if not pinned:
            held.verdict = "UNRESOLVED"
            held.detail = "no attestation names a repository, revision and path to fetch from"
            out.append(held)
            continue
        first = pinned[0]

        got = _from_hub(
            str(first["repo_id"]),
            str(first["revision"]),
            str(first["path"]),
            where,
            str(first.get("resolved_sha256") or first.get("sha256") or ""),
        )
        got.consumer, got.path = row.get("pack", ""), str(where)
        out.append(got)
    return out


def vendor_weights(manifest: dict[str, Any], index: dict[str, list[Any]]) -> list[Provisioned]:
    out: list[Provisioned] = []
    for row in manifest.get("vendor_weights", []):
        where = provenance.VENDOR_ROOT / row["file"]
        digest = str(row.get("sha256", ""))
        held = Provisioned(row.get("consumer", ""), "vendor_weight", str(row.get("source", "")), row["file"], "PRESENT")
        if where.is_file():
            held.bytes = _size(where)
            out.append(held)
            continue
        found = index.get(digest, [])
        if not found:
            held.verdict = "UNRESOLVED"
            held.detail = f"no indexed source publishes {digest[:16]}, so there is nowhere to fetch it from"
            out.append(held)
            continue
        pointer = found[0]

        if pointer.revision == "published":
            got = fetch(
                {
                    "consumer": row.get("consumer", ""),
                    "kind": "url",
                    "source": pointer.repo,
                    "url": pointer.path,
                    "path": row["file"],
                },
                provenance.VENDOR_ROOT,
            )
            wrong = _verified(where, digest) if got.ok else ""
            if wrong:
                got.verdict, got.detail = "FAILED", wrong
        else:
            got = _from_hub(pointer.repo, pointer.revision, pointer.path, where, digest)
        got.consumer, got.path = row.get("consumer", ""), row["file"]
        out.append(got)
    return out


def survey() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    root = root_of(manifest)
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()

    from compat.harness import hf_check

    index, _ = hf_check.index_of(hf_check.source_repos(manifest), refs_root)
    served, _ = hf_check.published_urls(manifest, refs_root)
    for one, pointers in served.items():
        index.setdefault(one, []).extend(pointers)

    rows = [
        *pack_weights(manifest),
        *vendor_weights(manifest, index),
        *(fetch(one, root) for one in manifest.get("provisioned", {}).get("artifacts", [])),
    ]
    return {
        "root": str(root),
        "artifacts": [asdict(one) for one in rows],
        "missing": [one.path for one in rows if not one.ok],
    }


def main() -> int:
    out = survey()
    print(f"provisioned root: {out['root']}\n")
    for one in out["artifacts"]:
        mark = "ok " if one["verdict"] in ("PRESENT", "FETCHED") else "!! "
        print(f"{mark}{one['verdict']:<11}{one['path']:<44}{one['bytes']:>15,} B  {one['source']}")
        if one["detail"]:
            print(f"    {one['detail'][:160]}")

    print(f"\nartifacts: {len(out['artifacts'])}   missing: {len(out['missing'])}  {out['missing']}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "provisioned.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if not out["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
