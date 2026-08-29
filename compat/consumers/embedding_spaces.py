"""Whether one stored vector serves every consumer, measured.

The manifest declares three recognition models across the population:

    glintr100        the default; antelopev2's, and what the gallery stores
    w600k_r50        photomaker_v2, manifest embedding_model
    facexlib_arcface infiniteyou, init_recognition_model('arcface'), a torch
                     model rather than an insightface one

All three take the SAME input -- one norm_crop@112 built from the same five
keypoints -- and all three return 512 floats. That shared shape is exactly why
this needs measuring rather than assuming: a 512-vector from one is
substitutable for another only if the spaces agree, and nothing about the
shape says whether they do.

WHAT EACH CASE ASSERTS
----------------------
Baseline is the vector THIS consumer's own model produces. The replay rebuilds
it from the retained crop, which must reproduce exactly.

The ablation is the question: `glintr100_substituted` puts the gallery's stored
vector where the consumer's own belongs. It MUST break. If it does not, one
space serves everyone and a single stored embedding is sufficient; if it does,
the gallery either stores a vector per space or stores enough source pixels to
derive them.

The measurement records cosine similarity between the spaces, so the answer
is a number rather than a verdict: two spaces at 0.99 and two at 0.02 are
different findings and a pass/fail cannot tell them apart.
"""

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
from compat.storage import derivatives

CONSUMER_ID: Final[str] = "embedding_spaces"

#: The model each consumer's own path uses, from its manifest row. glintr100
#: is what the gallery stores, so it is both a row here and the substitute
#: every other row is tested against.
SPACES: Final[dict[str, str]] = {
    "glintr100": "antelopev2 recognition; what derived_face_instance.embedding holds",
    "w600k_r50": "photomaker_v2 embedding_model; buffalo_l recognition",
    "facexlib_arcface": "infiniteyou; init_recognition_model('arcface'), torch",
}

#: What the gallery actually keeps.
STORED: Final[str] = "glintr100"


def cosine(left: Float32Array, right: Float32Array) -> float:
    """Cosine similarity, on the raw vectors.

    Raw rather than normalised because the norm is part of what a space
    carries: two vectors can point the same way with different magnitudes,
    and a consumer taking `face.embedding` rather than `normed_embedding`
    receives both.
    """
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / scale) if scale else 0.0


class EmbeddingSpaceRunner:
    """Every declared recognition model, over one shared aligned crop."""

    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        self._shots = {one.label: one for one in shots()}
        self._crops: dict[str, np.ndarray] = {}
        self._vectors: dict[tuple[str, str], Float32Array] = {}

    def _parts(self, case: Case) -> tuple[str, Shot]:
        space, _, label = case.boundary.partition("|")
        return space, self._shots[label]

    def crop(self, shot: Shot) -> np.ndarray:
        """The one 112 crop every model here is handed.

        Shared on purpose: a difference between two vectors must be the
        model's, and giving each its own crop would let an alignment
        difference masquerade as a space difference.
        """
        if shot.label not in self._crops:
            kps = np.asarray(our_face(shot).kps, dtype=np.float32)
            self._crops[shot.label] = norm_crop112(shot.frame, kps)
        return self._crops[shot.label]

    def vector(self, space: str, shot: Shot) -> Float32Array:
        key = (space, shot.label)
        if key not in self._vectors:
            self._vectors[key] = embed_with(space, self.crop(shot))
        return self._vectors[key]

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for shot in self._shots.values():
            for space in SPACES:
                ablations = [
                    Ablation(primitive="aligned_crop_112", expect_breaks=True),
                    # The crop as this application's own encoder keeps it.
                    # `vision/thumbs` writes every raster variant as WebP at
                    # quality 82, the avatar crop included, so a store that
                    # kept the aligned crop would keep exactly these bytes.
                    Ablation(
                        primitive="aligned_crop_112",
                        swap="webp_encoded",
                        expect_breaks=True,
                        kind="substitution",
                    ),
                ]
                if space != STORED:
                    # The claim under test, per space: the gallery's stored
                    # glintr100 vector cannot stand in for this model's.
                    ablations.append(
                        Ablation(
                            # The primitive is the RETAINED key, which is the
                            # crop; the swap is the vector offered instead of
                            # keeping it. `substituted_vector` named neither --
                            # nothing retains it, so `answer.json` listed it
                            # beside `kps` as a column and could not price it.
                            # It survives below only as the state key the
                            # replay reads, which is a mechanism, not a claim.
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
        """The CROP, not the vector.

        Deliberate: if the crop is retained then every space is derivable and
        the gallery needs one image region rather than three vectors. Whether
        that holds is what the ablations decide.
        """
        _, shot = self._parts(case)
        return RetainedState(aligned_crop_112=self.crop(shot).copy())

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)

    def baseline(self, case: Case) -> Artifact:
        space, shot = self._parts(case)
        return self._artifact(case.boundary, self.vector(space, shot))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        space, _ = self._parts(case)
        if retained.has("substituted_vector"):
            return self._artifact(case.boundary, retained.points("substituted_vector"))
        return self._artifact(case.boundary, embed_with(space, retained.pixels("aligned_crop_112")))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "webp_encoded":
            return retained.replacing("aligned_crop_112", derivatives.encoded(retained.pixels("aligned_crop_112"))[0])
        if ablation.swap == "stored_glintr100":
            _, shot = self._parts(case)
            return retained.replacing("substituted_vector", self.vector(STORED, shot))
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """How close this space is to the one the gallery stores."""
        if name != "agreement_with_stored":
            raise KeyError(f"{CONSUMER_ID} has no measurement called {name!r}")
        del retained
        space, shot = self._parts(case)
        mine = self.vector(space, shot)
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
