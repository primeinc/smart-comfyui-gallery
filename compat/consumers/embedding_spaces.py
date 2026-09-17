from __future__ import annotations

from typing import Final

import numpy as np

from compat.assertions.arrays import digest
from compat.consumers.face_family import embed_with, norm_crop112
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Float32Array,
    Measurement,
    RetainedState,
    Tier,
)
from compat.corpus.loaded import Shot, our_face, shots

CONSUMER_ID: Final[str] = "embedding_spaces"


SPACES: Final[dict[str, str]] = {
    "glintr100": "antelopev2 recognition; what derived_face_instance.embedding holds",
    "w600k_r50": "photomaker_v2 embedding_model; buffalo_l recognition",
    "facexlib_arcface": "infiniteyou; init_recognition_model('arcface'), torch",
}


STORED: Final[str] = "glintr100"


def cosine(left: Float32Array, right: Float32Array) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / scale) if scale else 0.0


class EmbeddingSpaceRunner:
    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        self._shots = {one.label: one for one in shots()}
        self._crops: dict[str, np.ndarray] = {}
        self._vectors: dict[tuple[str, str], Float32Array] = {}

    def _parts(self, case: Case) -> tuple[str, Shot]:
        space, _, label = case.boundary.partition("|")
        return space, self._shots[label]

    def crop(self, shot: Shot) -> np.ndarray:
        if shot.label not in self._crops:
            kps = np.asarray(our_face(shot).kps, dtype=np.float32)
            self._crops[shot.label] = norm_crop112(shot.frame, kps)
        return self._crops[shot.label]

    def vector(self, space: str, shot: Shot) -> Float32Array:
        key = (space, shot.label)
        if key not in self._vectors:
            self._vectors[key] = embed_with(space, self.crop(shot))
        return self._vectors[key]

    def _production_embed(self, crop: np.ndarray) -> Float32Array:
        # The application's own forward pass, mirrored from vision/faces.py
        # (blob 1/127.5, mean 127.5, swapRB) over the SAME glintr100 file the
        # producer's onnxruntime session reads.
        import cv2

        from compat.producers.insightface_pass import MODELS_ROOT, PACK

        net = cv2.dnn.readNetFromONNX(str(MODELS_ROOT / "models" / PACK / "glintr100.onnx"))
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        net.setInput(blob)
        return np.asarray(net.forward(), dtype=np.float32).reshape(-1)

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for shot in self._shots.values():
            # The consumer boundary: the application's cv2.dnn embedding and
            # the producer's onnxruntime embedding are the same space over
            # the same weights, within cross-runtime numerics.
            out.append(
                Case(
                    name=f"space_production_glintr100_{shot.label}",
                    consumer_id=CONSUMER_ID,
                    tier=Tier.CONSUMER,
                    fixture=shot.fixture,
                    boundary=f"production_glintr100|{shot.label}",
                    exact_bytes=False,
                    rtol=0.0,
                    atol=1e-3,
                    retained=("aligned_crop_112",),
                    measurements=("agreement_with_stored",),
                    note="vision/faces.py forward against insightface get_feat on the same glintr100 bytes",
                )
            )
            for space in SPACES:
                ablations = [
                    Ablation(primitive="aligned_crop_112", expect_breaks=True),
                ]
                if space != STORED:
                    ablations.append(
                        Ablation(
                            primitive="aligned_crop_112",
                            swap="stored_glintr100",
                            expect_breaks=True,
                            kind="substitution",
                        )
                    )
                out.append(
                    Case(
                        name=f"space_{space}_{shot.label}",
                        consumer_id=CONSUMER_ID,
                        tier=Tier.PRIMITIVE,
                        fixture=shot.fixture,
                        boundary=f"{space}|{shot.label}",
                        exact_bytes=True,
                        rtol=0.0,
                        atol=0.0,
                        retained=("aligned_crop_112",),
                        ablations=tuple(ablations),
                        measurements=("agreement_with_stored",),
                        note=SPACES[space],
                    )
                )
        return tuple(out)

    def retained_for(self, case: Case) -> RetainedState:
        _, shot = self._parts(case)
        return RetainedState(aligned_crop_112=self.crop(shot).copy())

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)

    def baseline(self, case: Case) -> Artifact:
        space, shot = self._parts(case)
        if space == "production_glintr100":
            return self._artifact(case.boundary, self.vector(STORED, shot))
        return self._artifact(case.boundary, self.vector(space, shot))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        space, _ = self._parts(case)
        if retained.has("substituted_vector"):
            return self._artifact(case.boundary, retained.points("substituted_vector"))
        if space == "production_glintr100":
            return self._artifact(case.boundary, self._production_embed(retained.pixels("aligned_crop_112")))
        return self._artifact(case.boundary, embed_with(space, retained.pixels("aligned_crop_112")))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "stored_glintr100":
            _, shot = self._parts(case)
            return retained.replacing("substituted_vector", self.vector(STORED, shot))
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name != "agreement_with_stored":
            raise KeyError(f"{CONSUMER_ID} has no measurement called {name!r}")
        del retained
        space, shot = self._parts(case)
        mine = self._production_embed(self.crop(shot)) if space == "production_glintr100" else self.vector(space, shot)
        stored = self.vector(STORED, shot)
        agreement = cosine(mine, stored)
        return Measurement(
            name=name,
            unit="cosine",
            value=agreement,
            basis=f"{space} against {STORED} on the same norm_crop@112",
            detail=(
                f"{shot.label}: {space} |v|={float(np.linalg.norm(mine)):.3f} against "
                f"{STORED} |v|={float(np.linalg.norm(stored)):.3f}, cosine {agreement:+.4f}"
                + (" -- the same model" if space == STORED else "")
            ),
        )


def all_runners() -> tuple[EmbeddingSpaceRunner, ...]:
    return (EmbeddingSpaceRunner(),)
