from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

import numpy as np

import proc
from compat.contracts.case import Fixture, UInt8Array
from compat.corpus.loaded import Shot
from compat.vendor import fixtures

FACE_ROLES: Final[frozenset[str]] = frozenset({"single_reference", "reference_set"})


IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


SHOTS: Final[int] = 2


def _decode(blob: bytes) -> UInt8Array:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2 could not decode the vendor fixture bytes")
    return np.asarray(bgr, dtype=np.uint8)


def vendor_shots(consumer_id: str, limit: int = SHOTS) -> list[Shot]:
    resolved = fixtures.resolve()
    seen: set[str] = set()
    out: list[Shot] = []
    for row in resolved["fixtures"]:
        if len(out) >= limit:
            break
        if row["consumer_id"] != consumer_id:
            continue
        if not row["present"] or row["role"] not in FACE_ROLES:
            continue
        path = Path(row["path"])
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        if row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        blob = _read(row)
        if blob is None:
            continue
        out.append(
            Shot(
                label=f"vendor_{consumer_id}_{path.stem}",
                fixture=Fixture(
                    name=f"vendor_{consumer_id}_{path.stem}",
                    path=row["path"],
                    sha256=row["sha256"],
                    kind="vendor_reference",
                    note=f"{consumer_id}: {row['cited']}"[:200],
                ),
                frame=_decode(blob),
            )
        )
    return out


def without_vendor_fixture() -> list[str]:
    from compat.consumers.face_family import vendor_setups
    from compat.consumers.whole_reference import whole_setups

    return sorted(
        consumer_id for consumer_id in (set(vendor_setups()) | set(whole_setups())) if not vendor_shots(consumer_id)
    )


def _read(row: dict[str, Any]) -> bytes | None:
    path = Path(row["path"])
    if path.is_file():
        held = path.read_bytes()

        if hashlib.sha256(held).hexdigest() == row["sha256"]:
            return held

    manifest = fixtures.provenance.load_manifest()
    refs_root = (fixtures.ROOT.parent / manifest["refs_root"]).resolve()
    upstreams = manifest.get("upstreams", {})
    pinned = {one["id"]: one for one in manifest.get("consumers", []) if one.get("repo")}
    for sample in fixtures.SAMPLES:
        if sample.consumer_id != row["consumer_id"] or row["path"] not in sample.inputs:
            continue
        holder = upstreams.get(sample.from_upstream) or pinned.get(sample.from_upstream) or pinned[sample.consumer_id]
        clone = fixtures.provenance.clone_dir(refs_root, holder["repo"])
        code, out, _ = proc.run(
            ["git", "-C", str(clone), "cat-file", "blob", f"{holder['commit']}:{row['path']}"],
            timeout=proc.LOCAL_SECONDS,
        )
        return out if code == 0 else None
    return None


def all_runners() -> list[Any]:
    from compat.consumers.face_family import FaceFamilyRunner, vendor_setups
    from compat.consumers.whole_reference import WholeReferenceRunner, whole_setups

    out: list[Any] = []
    for consumer_id, face_setup in vendor_setups().items():
        found = vendor_shots(consumer_id)

        if found:
            out.append(FaceFamilyRunner(face_setup, found))
    for consumer_id, whole_setup in whole_setups().items():
        found = vendor_shots(consumer_id)
        if found:
            out.append(WholeReferenceRunner(whole_setup, found))
    return out
