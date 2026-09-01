from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from compat.contracts.case import Ablation, Artifact, Case, Fixture, Measurement, RetainedState, Tier, note_skip
from compat.corpus import index as corpus
from compat.producers import insightface_pass as producer
from compat.storage import precision

CONSUMER_ID: Final[str] = "insightface_producer"


CORPUS_IMAGES: Final[int] = 6


XY_PLACES: Final[int] = 5
Z_PLACES: Final[int] = 2


@dataclass(frozen=True)
class Observation:
    label: str
    fixture: Fixture
    width: int
    height: int
    embedding: npt.NDArray[np.float32]
    normed_embedding: npt.NDArray[np.float32]
    landmark_3d_68: npt.NDArray[np.float32]
    pose: npt.NDArray[np.float32]


def pose_from_landmarks(landmark_3d_68: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    from insightface.data import get_object
    from insightface.utils import transform

    mean_lmk = get_object("meanshape_68.pkl")
    matrix = transform.estimate_affine_matrix_3d23d(mean_lmk, np.asarray(landmark_3d_68, dtype=np.float32))
    _scale, rotation, _translation = transform.P2sRt(matrix)
    rx, ry, rz = transform.matrix2angle(rotation)
    return np.array([rx, ry, rz], dtype=np.float32)


def through_todays_storage(landmark_3d_68: npt.NDArray[np.float32], width: int, height: int) -> npt.NDArray[np.float32]:
    out = np.asarray(landmark_3d_68, dtype=np.float64).copy()
    out[:, 0] = np.round(np.clip(out[:, 0] / width, 0.0, 1.0), XY_PLACES) * width
    out[:, 1] = np.round(np.clip(out[:, 1] / height, 0.0, 1.0), XY_PLACES) * height
    out[:, 2] = np.round(out[:, 2], Z_PLACES)
    return out.astype(np.float32)


def observations(limit: int = CORPUS_IMAGES) -> list[Observation]:
    if not corpus.KYC.is_dir():
        return []

    samples = corpus.scan_kyc()
    buckets: dict[tuple[str, str], list[corpus.Sample]] = {}
    for one in samples:
        buckets.setdefault((one.identity, one.role), []).append(one)
    chosen = [min(buckets[key], key=lambda one: one.sha256) for key in sorted(buckets)][:limit]

    app = producer.analysis()
    out: list[Observation] = []
    for one in chosen:
        frame, sha = producer.decode(Path(one.path))
        faces = app.get(frame)
        if not faces:
            note_skip(CONSUMER_ID, one.path, "our producer found no face in this photograph")
            continue
        best = max(faces, key=lambda face: float(face.det_score))
        landmarks = best.get("landmark_3d_68")
        pose = best.get("pose")
        if landmarks is None or pose is None:
            note_skip(CONSUMER_ID, one.path, "the 1k3d68 head produced no landmarks or no pose")
            continue
        height, width = frame.shape[:2]
        out.append(
            Observation(
                label=f"{one.identity}_{one.role}",
                fixture=Fixture(
                    name=f"corpus_{one.identity}_{one.role}",
                    path=one.path,
                    sha256=sha,
                    kind="corpus_photograph",
                    note=f"{corpus.LICENCE}, not vendored; antelopev2 1k3d68 + glintr100, CPU",
                ),
                width=int(width),
                height=int(height),
                embedding=np.asarray(best.embedding, dtype=np.float32).reshape(-1),
                normed_embedding=np.asarray(best.normed_embedding, dtype=np.float32).reshape(-1),
                landmark_3d_68=np.asarray(landmarks, dtype=np.float32),
                pose=np.asarray(pose, dtype=np.float32),
            )
        )
    return out


def _artifact(name: str, values: npt.NDArray[np.float32]) -> Artifact:
    return Artifact(
        name=name,
        dtype=str(values.dtype),
        shape=tuple(int(one) for one in values.shape),
        sha256=producer.digest_array(values),
        values=values,
    )


class ProducerDerivationRunner:
    consumer_id: str = CONSUMER_ID

    def __init__(self, found: list[Observation] | None = None) -> None:
        self._by_label: dict[str, Observation] = {
            one.label: one for one in (found if found is not None else observations())
        }

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for label in self._by_label:
            out.append(
                Case(
                    name=f"pose_from_landmark_3d_68_{label}",
                    consumer_id=self.consumer_id,
                    tier=Tier.PRIMITIVE,
                    fixture=self._by_label[label].fixture,
                    boundary=f"pose|{label}",
                    exact_bytes=True,
                    rtol=0.0,
                    atol=0.0,
                    retained=("landmark_3d_68",),
                    ablations=(
                        Ablation(primitive="landmark_3d_68", expect_breaks=True),
                        Ablation(
                            primitive="landmark_3d_68",
                            swap="half_precision",
                            expect_breaks=True,
                            kind="substitution",
                        ),
                    ),
                    measurements=("pose_error_through_todays_storage",),
                    note="upstream's own estimate_affine_matrix_3d23d/P2sRt/matrix2angle, no pixels",
                )
            )
            out.append(
                Case(
                    name=f"normed_embedding_from_embedding_{label}",
                    consumer_id=self.consumer_id,
                    tier=Tier.PRIMITIVE,
                    fixture=self._by_label[label].fixture,
                    boundary=f"normed_embedding|{label}",
                    exact_bytes=True,
                    rtol=0.0,
                    atol=0.0,
                    retained=("embedding",),
                    ablations=(
                        Ablation(primitive="embedding", expect_breaks=True),
                        Ablation(
                            primitive="embedding",
                            swap="half_precision",
                            expect_breaks=True,
                            kind="substitution",
                        ),
                    ),
                    measurements=("norm_is_not_recoverable",),
                    note="Face.normed_embedding is a property; the norm it divides away is not in the result",
                )
            )
        return tuple(out)

    def _parts(self, case: Case) -> tuple[str, Observation]:
        kind, _, label = case.boundary.partition("|")
        return kind, self._by_label[label]

    def retained_for(self, case: Case) -> RetainedState:
        kind, found = self._parts(case)
        if kind == "pose":
            return RetainedState(landmark_3d_68=found.landmark_3d_68.copy())
        return RetainedState(embedding=found.embedding.copy())

    def baseline(self, case: Case) -> Artifact:
        kind, found = self._parts(case)
        if kind == "pose":
            return _artifact(case.boundary, found.pose)
        return _artifact(case.boundary, found.normed_embedding)

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        kind, _ = self._parts(case)
        if kind == "pose":
            return _artifact(case.boundary, pose_from_landmarks(retained.points("landmark_3d_68")))
        raw = retained.points("embedding")
        return _artifact(case.boundary, (raw / np.linalg.norm(raw)).astype(np.float32))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "half_precision":
            return retained.replacing(ablation.primitive, precision.half(retained.array(ablation.primitive)))
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        _kind, found = self._parts(case)

        if name == "pose_error_through_todays_storage":
            stored = through_todays_storage(found.landmark_3d_68, found.width, found.height)
            degraded = pose_from_landmarks(stored)
            worst = float(np.max(np.abs(degraded.astype(np.float64) - found.pose.astype(np.float64))))
            return Measurement(
                name=name,
                unit="degrees",
                value=worst,
                basis=(
                    f"landmark_3d_68 normalised by {found.width}x{found.height}, xy rounded to {XY_PLACES} "
                    f"places, z to {Z_PLACES}, clamped to [0,1], expanded back, then re-derived"
                ),
                detail=(
                    f"worst axis error {worst:.6f} deg against the producer's own pose "
                    f"(pitch {found.pose[0]:.2f} yaw {found.pose[1]:.2f} roll {found.pose[2]:.2f})"
                ),
            )

        if name == "norm_is_not_recoverable":
            magnitude = float(np.linalg.norm(found.embedding))
            back = found.normed_embedding * magnitude
            worst = float(np.max(np.abs(back.astype(np.float64) - found.embedding.astype(np.float64))))
            return Measurement(
                name=name,
                unit="l2_norm",
                value=magnitude,
                basis="norm(embedding); reconstruction needs a number the unit vector does not carry",
                detail=(
                    f"norm {magnitude:.4f}; raw recovers to {worst:.3e} WITH the norm supplied "
                    f"externally, and not at all without it"
                ),
            )

        raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")
