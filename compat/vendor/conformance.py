"""Adapter conformance: our adapter's boundary, on the VENDOR's own fixture.

Three proof layers stand between "our code runs" and "this consumer is
proven":

    VENDOR ACCEPTANCE    does the pinned upstream reproduce its own example?
    ADAPTER CONFORMANCE  does our adapter reproduce that path on the
                         upstream's own data?
    STORAGE COMPATIBILITY can our persisted state serve that proven adapter?

This module is the SECOND. Everything in `compat/consumers/` runs on our KYC
corpus, which is our data: a consumer passing there shows our adapter is
self-consistent, not that it agrees with the vendor. Running the same adapter
over the vendor's own committed reference image is what makes the two
comparable.

MECHANISM
---------
`face_family.all_runners` takes an injectable `Shot` list
(compat/consumers/face_family.py:654-658), so the entire family runs against
vendor images with no change to the family itself. Case names carry the
shot label, and vendor labels are prefixed `vendor_`, so these never collide
with the corpus cases.

WHAT THIS IS NOT
----------------
It is NOT vendor acceptance. Nothing here executes the vendor's own script
end to end -- that needs FLUX, SDXL and per-consumer checkpoints which are
not on this machine. A consumer is PROVEN only when layer one has also run;
until then this lane establishes that our adapter and the vendor agree about
the vendor's own input, which is strictly more than the corpus lane shows and
strictly less than proof.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.contracts.case import Fixture, UInt8Array
from compat.corpus.loaded import Shot
from compat.vendor import fixtures

#: Vendor fixtures that are a single face photograph, so the face family's
#: detect/crop/embed boundaries apply. Video, audio, masks and reference SETS
#: are covered by their own lanes and are deliberately absent here.
FACE_ROLES: Final[frozenset[str]] = frozenset({"single_reference", "reference_set"})

#: Suffixes the face family can decode.
IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: How many vendor photographs to exercise. Every consumer runs against every
#: shot, so this multiplies the case count by the family size.
SHOTS: Final[int] = 4


def _decode(blob: bytes) -> UInt8Array:
    """Vendor bytes to the BGR frame the family expects.

    `cv2.imdecode` over a byte buffer, the same call
    `compat/producers/insightface_pass.py:decode` makes. `PIL.Image.open` is
    banned in this tree (pyproject.toml TID251): it skips plugin registration,
    so the same file answers differently depending on what ran first.
    """
    import cv2

    bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2 could not decode the vendor fixture bytes")
    return np.asarray(bgr, dtype=np.uint8)


def vendor_shots(limit: int = SHOTS) -> list[Shot]:
    """One `Shot` per vendor face fixture, by content.

    Reads the resolved fixture index rather than re-deriving it, so a fixture
    that moved or changed bytes is caught by `compat.vendor.fixtures` first
    and this lane cannot silently run on something else.
    """
    resolved = fixtures.resolve()
    seen: set[str] = set()
    out: list[Shot] = []
    for row in resolved["fixtures"]:
        if len(out) >= limit:
            break
        if not row["present"] or row["role"] not in FACE_ROLES:
            continue
        path = Path(row["path"])
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # One shot per distinct image: the same bytes appear under several
        # consumers (lecun.jpg is both PuLID's and InstantID's), and running
        # it twice would double the case count without adding evidence.
        if row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        blob = _read(row)
        if blob is None:
            continue
        out.append(
            Shot(
                label=f"vendor_{path.stem}",
                fixture=Fixture(
                    name=f"vendor_{row['consumer_id']}_{path.stem}",
                    path=row["path"],
                    sha256=row["sha256"],
                    kind="vendor_reference",
                    note=f"{row['consumer_id']}: {row['cited']}"[:200],
                ),
                frame=_decode(blob),
            )
        )
    return out


def _read(row: dict[str, Any]) -> bytes | None:
    """The fixture's bytes, from the pinned commit or the fetched cache."""
    import subprocess

    path = Path(row["path"])
    if path.is_file():
        return path.read_bytes()
    # A repo-relative path: re-extract from the commit the index resolved it
    # against, rather than trusting a working tree that may be dirty.
    manifest = fixtures.provenance.load_manifest()
    refs_root = (fixtures.ROOT.parent / manifest["refs_root"]).resolve()
    upstreams = manifest.get("upstreams", {})
    pinned = {one["id"]: one for one in manifest.get("consumers", []) if one.get("repo")}
    for sample in fixtures.SAMPLES:
        if sample.consumer_id != row["consumer_id"] or row["path"] not in sample.inputs:
            continue
        holder = upstreams.get(sample.from_upstream) or pinned.get(sample.from_upstream) or pinned[sample.consumer_id]
        clone = fixtures.provenance.clone_dir(refs_root, holder["repo"])
        argv: list[str] = ["git", "-C", str(clone), "cat-file", "blob", f"{holder['commit']}:{row['path']}"]
        done = subprocess.run(argv, capture_output=True, check=False)
        return done.stdout if done.returncode == 0 else None
    return None


def all_runners() -> list[Any]:
    """Both families, pointed at vendor photographs.

    The face family covers detect/crop/embed; the whole-reference family
    covers the consumers that take the framed picture. Running only the first
    would leave the whole-reference half of the population measured on OUR
    corpus alone -- the exact gap this module exists to close.

    Both accept an injectable `Shot` list
    (face_family.py:654-658, whole_reference.py:284-286), so neither family
    changes to be pointed at vendor data.
    """
    from compat.consumers.face_family import all_runners as family
    from compat.consumers.whole_reference import all_runners as whole

    found = vendor_shots()
    if not found:
        return []
    return [*family(found), *whole(found)]
