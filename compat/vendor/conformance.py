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
Both families take an injectable `Shot` list, so they run against vendor
images with no change to the family itself. A runner is built PER CONSUMER
with that consumer's OWN committed fixtures -- `vendor_shots(consumer_id)` --
because handing one global list to every runner is how all 21 consumers came
to be measured on InstantID's four example photographs.

Case names carry the shot label, and vendor labels are prefixed
`vendor_<consumer_id>_`, so these never collide with the corpus cases and a
row names the vendor whose data produced it.

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

import hashlib
from pathlib import Path
from typing import Any, Final

import numpy as np

import proc
from compat.contracts.case import Fixture, UInt8Array
from compat.corpus.loaded import Shot
from compat.vendor import fixtures

#: Vendor fixtures that are a single face photograph, so the face family's
#: detect/crop/embed boundaries apply. Video, audio, masks and reference SETS
#: are covered by their own lanes and are deliberately absent here.
FACE_ROLES: Final[frozenset[str]] = frozenset({"single_reference", "reference_set"})

#: Suffixes the face family can decode.
IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: How many of a consumer's OWN vendor photographs to exercise. Per consumer,
#: not globally: a single global list is how every consumer came to be
#: measured on InstantID's four examples.
SHOTS: Final[int] = 2


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


def vendor_shots(consumer_id: str, limit: int = SHOTS) -> list[Shot]:
    """One `Shot` per face fixture THIS consumer committed, by content.

    Filtered by `consumer_id`, which is the whole point of the lane. Taking
    the first N face fixtures in index order instead returned InstantID's four
    `examples/*_resize.jpg` for everybody, so a claim that our adapter agrees
    with the vendor about the vendor's own input was, for 20 of 21 consumers,
    a claim about a different vendor's input.

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
        if row["consumer_id"] != consumer_id:
            continue
        if not row["present"] or row["role"] not in FACE_ROLES:
            continue
        path = Path(row["path"])
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # One shot per distinct image: the same bytes appear more than once
        # under one consumer (uniportrait cites stylegan2-ffhq-0293 twice),
        # and running it twice would double the case count without adding
        # evidence.
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
    """Consumers in either family with no committed face photograph of their own.

    Reported rather than skipped. A lane that contributes no cases for a
    consumer contributes no failures for it either, and the difference between
    "agreed with the vendor" and "the vendor ships nothing to agree about" is
    the difference this suite's VENDOR_BASELINE_UNAVAILABLE verdict exists for.
    """
    from compat.consumers.face_family import vendor_setups
    from compat.consumers.whole_reference import whole_setups

    return sorted(
        consumer_id for consumer_id in (set(vendor_setups()) | set(whole_setups())) if not vendor_shots(consumer_id)
    )


def _read(row: dict[str, Any]) -> bytes | None:
    """The fixture's bytes, from the pinned commit or the fetched cache."""
    path = Path(row["path"])
    if path.is_file():
        held = path.read_bytes()
        # Checked against the digest the index recorded. A repo-relative
        # fixture path resolves against the CWD, so this branch could return
        # some other file's bytes while the `Shot` carried the pinned sha256 --
        # a frame and a provenance stamp describing two different images.
        if hashlib.sha256(held).hexdigest() == row["sha256"]:
            return held
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
        code, out, _ = proc.run(
            ["git", "-C", str(clone), "cat-file", "blob", f"{holder['commit']}:{row['path']}"],
            timeout=proc.LOCAL_SECONDS,
        )
        return out if code == 0 else None
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
    from compat.consumers.face_family import FaceFamilyRunner, vendor_setups
    from compat.consumers.whole_reference import WholeReferenceRunner, whole_setups

    # Two loops rather than one over a pair: a setup and its runner class are
    # matched types, and zipping them into one iterable erases that -- ty
    # rejects `build(setup, ...)` because `setup` becomes the union.
    out: list[Any] = []
    for consumer_id, face_setup in vendor_setups().items():
        found = vendor_shots(consumer_id)
        # Not silently dropped: `without_vendor_fixture()` names these, and
        # the fixtures lane already records what upstream does and does not
        # ship.
        if found:
            out.append(FaceFamilyRunner(face_setup, found))
    for consumer_id, whole_setup in whole_setups().items():
        found = vendor_shots(consumer_id)
        if found:
            out.append(WholeReferenceRunner(whole_setup, found))
    return out
