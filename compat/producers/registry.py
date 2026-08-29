"""Every landmark producer, so the vocabulary is a union rather than one pack.

The storage lane grades a candidate against what a producer emits. With ONE
producer that is honest and still circular: `compat/producers/` held only
antelopey2's pack, so "producer-derived" meant "antelopev2-derived" -- nine
keys, three of them landmark sets (5, 106, 68x3). A schema with room for
exactly those keys would score perfectly and prove nothing about the next
backend.

So a producer is a plug. Each one declares what it needs, reports whether it
can run here, and returns a record keyed the way IT names things. The union of
those keys is the vocabulary a storage candidate has to answer for.

WHAT EACH ONE CONTRIBUTES THAT THE OTHERS DO NOT
------------------------------------------------
    insightface/antelopev2  5-point kps, 106 2D, 68 3D, pose, gender, age,
                            512-d recognition embedding
    insightface/buffalo_l   the SAME detector file as antelopev2
                            (det_10g.onnx and scrfd_10g_bnkps.onnx are
                            byte-identical) with a DIFFERENT recognition
                            model, w600k_r50 -- so it separates "the
                            keypoints moved" from "the space changed"
    face_alignment          68 iBUG points in 2D and in 3D. The 3D set is the
                            same 68 points with a regressed z
                            (1adrianb/face-alignment api.py:265-279), which is
                            a different axis count for the same scheme name --
                            the case a `points`/`axes` pair exists for
    mediapipe               478 points (468 mesh + 2x5 iris; the counts are
                            FACEMESH_NUM_LANDMARKS 468 and
                            FACEMESH_NUM_LANDMARKS_WITH_IRISES 478 in
                            google-ai-edge/mediapipe@7387fbc6f0fe
                            mediapipe/python/solutions/face_mesh.py:55-56),
                            52 blendshape scores and a 4x4 facial transform
                            matrix. None of the three has anywhere to land in
                            a face row.

A producer whose runtime or weights are absent reports UNSUPPORTED with the
reason. It is never dropped: a vocabulary that shrinks when a backend is
missing would make the storage question easier by forgetting what it was
about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from compat.contracts.case import UInt8Array


@dataclass(frozen=True)
class Emission:
    """One producer's record for one face, keyed the producer's own way."""

    producer: str
    values: dict[str, npt.NDArray[np.generic]]
    note: str = ""

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))


@dataclass
class Availability:
    """Whether a producer can run on this machine, and why not."""

    producer: str
    ready: bool
    reason: str = ""
    identity: dict[str, str] = field(default_factory=dict)


class Producer(Protocol):
    """One landmark/observation producer.

    `name` is how the evidence refers to it. `available()` answers before any
    weight is loaded, so a missing backend is a recorded fact rather than an
    exception in the middle of a case.
    """

    name: str

    def available(self) -> Availability: ...

    def observe(self, frame: UInt8Array) -> list[Emission]:
        """Every face this producer finds, in its own vocabulary."""
        ...


def _versions(*modules: str) -> dict[str, str]:
    """Installed version per module, or the reason it could not be imported."""
    import importlib

    out: dict[str, str] = {}
    for name in modules:
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "present"))
        except ImportError as problem:
            out[name] = f"ABSENT: {problem}"
    return out


class InsightFacePack:
    """One insightface pack. Two packs are two producers, not one with a flag.

    `buffalo_l` and `antelopev2` ship byte-identical detector files and
    different recognition models, so running both is what separates a
    keypoint difference from an embedding-space difference.
    """

    def __init__(self, pack: str) -> None:
        self.name = f"insightface/{pack}"
        self._pack = pack

    def available(self) -> Availability:
        from compat.producers import insightface_pass as producer

        root = producer.MODELS_ROOT / "models" / self._pack
        if not root.is_dir():
            return Availability(self.name, False, f"pack absent at {root}")
        return Availability(self.name, True, identity=_versions("insightface", "onnxruntime"))

    def observe(self, frame: UInt8Array) -> list[Emission]:
        from compat.producers import insightface_pass as producer

        app = producer.analysis(pack=self._pack)
        return [
            Emission(
                producer=self.name,
                # Iterated, never named: a pack that emits a key this suite
                # has not heard of must still reach the vocabulary.
                values={str(key): np.asarray(face[key]) for key in face if face[key] is not None},
                note=f"pack {self._pack}",
            )
            for face in app.get(frame)
        ]


class FaceAlignment68:
    """1adrianb/face-alignment: 68 iBUG points, 2D and 3D.

    Two emissions from one model family with the same scheme NAME and a
    different axis count -- `landmark_68` at 2 axes and at 3. A schema keyed
    by scheme name alone cannot hold both.
    """

    name = "face_alignment/68"

    def available(self) -> Availability:
        held = _versions("face_alignment", "torch")
        missing = [key for key, value in held.items() if value.startswith("ABSENT")]
        if missing:
            return Availability(self.name, False, f"{', '.join(missing)} not importable", held)
        return Availability(self.name, True, identity=held)

    def _detector(self, kind: str) -> Any:
        import face_alignment

        types = {
            "2d": face_alignment.LandmarksType.TWO_D,
            "3d": face_alignment.LandmarksType.THREE_D,
        }
        # `compile=False`: 1.5.0 wraps the net in `torch.compile` by default
        # (api.py:87, :118-126) and Inductor needs MSVC, which this machine has
        # no `cl` for -- the run died with InvalidCxxCompiler. Eager is the
        # same arithmetic, and a compiled kernel is not what the storage
        # question is about.
        return face_alignment.FaceAlignment(types[kind], device="cpu", flip_input=False, compile=False)

    def observe(self, frame: UInt8Array) -> list[Emission]:
        # RGB: the library's own examples read with `io.imread`, which is RGB,
        # while everything else in this suite carries BGR.
        rgb = np.asarray(frame[:, :, ::-1], dtype=np.uint8)
        out: list[Emission] = []
        for kind in ("2d", "3d"):
            found = self._detector(kind).get_landmarks(rgb)
            if not found:
                continue
            for points in found:
                held = np.asarray(points, dtype=np.float32)
                out.append(
                    Emission(
                        producer=self.name,
                        values={f"landmark_{held.shape[1]}d_{held.shape[0]}": held},
                        note=f"iBUG 68, {kind}",
                    )
                )
        return out


class MediaPipeFaceMesh:
    """MediaPipe Face Landmarker: 478 points, 52 blendshapes, a 4x4 transform.

    mediapipe 1.0 removed the legacy `mp.solutions` API -- the installed
    package exposes only `Image`, `ImageFormat` and `tasks` -- so this runs
    the Tasks `FaceLandmarker`, which needs a downloaded `.task` bundle.

    The blendshapes and the transform matrix are the interesting part for the
    storage question: neither is a landmark, neither is derivable from one,
    and no column in a face row names either.
    """

    name = "mediapipe/face_landmarker"

    #: Google's published bundle, cached outside the repository. Recorded by
    #: digest so a bundle swapped under the same filename is a different
    #: producer rather than a silent change of answer.
    BUNDLE: Final[Path] = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "sg-vendor-fixtures"
        / "mediapipe"
        / "face_landmarker.task"
    )
    BUNDLE_SHA256: Final[str] = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

    def available(self) -> Availability:
        held = _versions("mediapipe")
        if held["mediapipe"].startswith("ABSENT"):
            return Availability(self.name, False, held["mediapipe"], held)
        if not self.BUNDLE.is_file():
            return Availability(self.name, False, f"bundle absent at {self.BUNDLE}", held)
        import hashlib

        digest = hashlib.sha256(self.BUNDLE.read_bytes()).hexdigest()
        if digest != self.BUNDLE_SHA256:
            return Availability(self.name, False, f"bundle digest {digest[:16]} is not the pinned one", held)
        return Availability(self.name, True, identity={**held, "bundle_sha256": digest})

    def observe(self, frame: UInt8Array) -> list[Emission]:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.BUNDLE)),
            num_faces=8,
            # Both ON: they are the two outputs a face row has no column for,
            # which is the whole reason this producer is in the union.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        rgb = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            answer = landmarker.detect(image)

        height, width = frame.shape[:2]
        out: list[Emission] = []
        blendshapes = list(answer.face_blendshapes or [])
        matrices = list(answer.facial_transformation_matrixes or [])
        for index, face in enumerate(answer.face_landmarks or []):
            points = np.asarray([[one.x * width, one.y * height, one.z] for one in face], dtype=np.float32)
            values: dict[str, npt.NDArray[np.generic]] = {f"landmark_3d_{points.shape[0]}": points}
            if index < len(blendshapes):
                values["blendshape_scores"] = np.asarray([one.score for one in blendshapes[index]], dtype=np.float32)
            if index < len(matrices):
                values["facial_transform_matrix"] = np.asarray(matrices[index], dtype=np.float32)
            out.append(Emission(producer=self.name, values=values, note="468 mesh + 2x5 iris"))
        return out


def every_producer() -> tuple[Producer, ...]:
    """Every producer this suite knows, available or not."""
    return (
        InsightFacePack("antelopev2"),
        InsightFacePack("buffalo_l"),
        FaceAlignment68(),
        MediaPipeFaceMesh(),
    )
