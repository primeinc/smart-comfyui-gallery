"""Decode once, detect once, keyed by content.

Four consumer modules were each defining their own `shots()` and each
decoding the same four photographs -- one of them 4896x6528 -- and then each
re-running detection over them. The suite spent most of its wall clock
recomputing things it had already computed, which is a strange way for a
harness built entirely on content addressing to behave.

Everything here is memoised on a key that names EVERY input that can change
the answer:

    frames        the file's sha256
    detections    (image sha256, pack, allowed_modules, det_size sequence,
                  selection rule)

That list is the whole contract. A cache keyed on less than what the result
depends on does not make a suite faster, it makes it wrong -- and wrong in
the worst available way, because it would return a confident answer computed
for a different question. The det_size sequence is in the key because
consumers sweep differently and the first size that finds a face decides the
keypoints; the selection rule is in the key because `face[0]` and
largest-by-area disagree the moment a photograph has two people in it.

Nothing is persisted between processes. The point is to stop repeating work
within one run, not to carry an answer across a change nobody re-derived --
evidence that survives its own inputs is the failure the rest of this suite
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.contracts.case import Fixture, UInt8Array
from compat.corpus import index as corpus
from compat.producers import insightface_pass as producer

#: Corpus photographs the consumer runners draw on: both capture paths for
#: the first identities. An ID document and a selfie went through different
#: optics, and a detector agreeing on one says nothing about the other.
CORPUS_IMAGES: Final[int] = 4


@dataclass(frozen=True)
class Shot:
    """One corpus photograph, decoded once and shared."""

    label: str
    fixture: Fixture
    frame: UInt8Array

    @property
    def frame_wh(self) -> tuple[int, int]:
        height, width = self.frame.shape[:2]
        return int(width), int(height)


_frames: dict[str, tuple[UInt8Array, str]] = {}
_shots: dict[int, list[Shot]] = {}
_detections: dict[tuple[Any, ...], Any] = {}


def frame_of(path: Path) -> tuple[UInt8Array, str]:
    """The decoded frame and the file's digest, decoded at most once."""
    key = str(path)
    if key not in _frames:
        _frames[key] = producer.decode(path)
    return _frames[key]


def shots(limit: int = CORPUS_IMAGES) -> list[Shot]:
    """The corpus slice every consumer runner shares.

    Chosen by digest rather than by filename so the selection does not move
    when a directory listing does, and so the same slice comes back on any
    machine holding the same corpus.
    """
    if limit in _shots:
        return _shots[limit]
    if not corpus.KYC.is_dir():
        _shots[limit] = []
        return []

    buckets: dict[tuple[str, str], list[corpus.Sample]] = {}
    for one in corpus.scan_kyc():
        buckets.setdefault((one.identity, one.role), []).append(one)
    chosen = [min(buckets[key], key=lambda one: one.sha256) for key in sorted(buckets)][:limit]

    out: list[Shot] = []
    for one in chosen:
        frame, sha = frame_of(Path(one.path))
        out.append(
            Shot(
                label=f"{one.identity}_{one.role}",
                fixture=Fixture(
                    name=f"corpus_{one.identity}_{one.role}",
                    path=one.path,
                    sha256=sha,
                    kind="corpus_photograph",
                    note=f"{corpus.LICENCE}, not vendored",
                ),
                frame=frame,
            )
        )
    _shots[limit] = out
    return out


def our_face(shot: Shot) -> Any:
    """The face OUR producer finds: what the database row would describe.

    Separate from a vendor's detection on purpose. This is the observation the
    application actually stores, and several runners need it to build their
    retained state -- so it is computed once per photograph rather than once
    per case.
    """
    key = ("ours", shot.fixture.sha256)
    if key not in _detections:
        faces = producer.analysis().get(shot.frame)
        if not faces:
            raise ValueError(f"our own producer found no face in {shot.label}")
        _detections[key] = max(faces, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
    return _detections[key]


def our_kps(shot: Shot) -> np.ndarray:
    return np.asarray(our_face(shot).kps, dtype=np.float32)


def vendor_face(
    shot: Shot,
    *,
    pack: str,
    allowed_modules: tuple[str, ...],
    sizes: tuple[int, ...],
    select: str,
    rebuild: Any,
) -> Any:
    """A vendor's own detection, computed once per distinct question.

    `rebuild` is the callable that actually detects when the cache misses. It
    is passed in rather than imported so this module stays free of the
    consumer layer -- `face_family` owns what a vendor's sweep means, and this
    owns only the memo.

    Every element of the key changes the answer. Dropping any one of them
    would let a cached detection from one consumer be served to another that
    asked a different question, which is the one failure a cache must never
    have in a suite whose whole output is claims about what was observed.
    """
    key = ("vendor", shot.fixture.sha256, pack, allowed_modules, sizes, select)
    if key not in _detections:
        _detections[key] = rebuild(shot.frame)
    return _detections[key]


def statistics() -> dict[str, int]:
    """What the memos actually saved, for the run report."""
    return {
        "frames_decoded": len(_frames),
        "detections_computed": len(_detections),
    }
