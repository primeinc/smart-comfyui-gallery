from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.contracts.case import Fixture, UInt8Array
from compat.corpus import cache
from compat.corpus import index as corpus
from compat.producers import insightface_pass as producer

CORPUS_IMAGES: Final[int] = 4


@dataclass(frozen=True)
class Shot:
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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_of(path: Path) -> tuple[UInt8Array, str]:
    key = str(path)
    if key in _frames:
        return _frames[key]

    sha = sha256_of(path)
    held = cache.frame_get(sha)
    cache.note("frame", held is not None)
    if held is None:
        frame, decoded = producer.decode(path)

        if decoded != sha:
            raise ValueError(f"{path} changed while it was being read: {sha} then {decoded}")
        cache.frame_put(sha, frame)
    else:
        frame = held

    _frames[key] = (frame, sha)
    return _frames[key]


def shots(limit: int = CORPUS_IMAGES) -> list[Shot]:
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


def best_face(faces: list[Any]) -> Any:
    return max(faces, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))


def our_face(shot: Shot) -> Any:
    key = ("ours", shot.fixture.sha256)
    if key in _detections:
        return _detections[key]

    held = cache.face_get(shot.fixture.sha256)
    cache.note("ours", held is not None)
    if held is None:
        faces = producer.detect(shot.frame)
        if not faces:
            raise ValueError(f"our own producer found no face in {shot.label}")
        held = best_face(faces)
        cache.face_put(shot.fixture.sha256, held)

    _detections[key] = held
    return _detections[key]


def our_recovery_face(shot: Shot) -> Any | None:
    key = ("recovery", shot.fixture.sha256)
    if key not in _detections:
        _detections[key] = producer.detect_padded(shot.frame)
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
    key = ("vendor", shot.fixture.sha256, pack, allowed_modules, sizes, select)
    if key not in _detections:
        _detections[key] = rebuild(shot.frame)
    return _detections[key]


def statistics() -> dict[str, Any]:
    return {
        "frames_decoded": len(_frames),
        "detections_computed": len(_detections),
        "store": cache.statistics(),
    }
