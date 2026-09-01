from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from compat.contracts.case import UInt8Array


@dataclass(frozen=True)
class Emission:
    producer: str
    values: dict[str, npt.NDArray[np.generic]]
    note: str = ""

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))


@dataclass
class Availability:
    producer: str
    ready: bool
    reason: str = ""
    identity: dict[str, str] = field(default_factory=dict)


class Producer(Protocol):
    name: str

    def available(self) -> Availability: ...

    def observe(self, frame: UInt8Array) -> list[Emission]: ...


def _versions(*modules: str) -> dict[str, str]:
    import importlib

    out: dict[str, str] = {}
    for name in modules:
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "present"))
        except ImportError as problem:
            out[name] = f"ABSENT: {problem}"
    return out


class InsightFacePack:
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
                values={str(key): np.asarray(face[key]) for key in face if face[key] is not None},
                note=f"pack {self._pack}",
            )
            for face in app.get(frame)
        ]


class OpenCVYuNet:
    name = "opencv/yunet+arcface"

    def available(self) -> Availability:
        from compat.producers import insightface_pass as producer
        from vision import faces

        try:
            faces.OpenCVFaceBackend(str(producer.MODELS_ROOT.parent))
        except faces.BackendUnavailable as why:
            return Availability(self.name, False, str(why))
        return Availability(self.name, True, identity=_versions("cv2"))

    def observe(self, frame: UInt8Array) -> list[Emission]:
        import cv2
        from PIL import Image

        from compat.producers import insightface_pass as producer
        from vision import faces, facestore

        backend = faces.OpenCVFaceBackend(str(producer.MODELS_ROOT.parent))
        image = Image.fromarray(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))
        out: list[Emission] = []
        for found in backend.detect(image):
            if found.native is None:
                raise ValueError("the OpenCV backend handed over a detection with no native record")
            record = facestore.thaw(found.native).record
            out.append(
                Emission(
                    producer=self.name,
                    values={key: np.asarray(value) for key, value in record.items() if value is not None},
                    note="YuNet row + recognizer feature, through the application's own capture",
                )
            )
        return out


class FaceAlignment68:
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

        return face_alignment.FaceAlignment(types[kind], device="cpu", flip_input=False, compile=False)

    def observe(self, frame: UInt8Array) -> list[Emission]:

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
    name = "mediapipe/face_landmarker"

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
    return (
        InsightFacePack("antelopev2"),
        InsightFacePack("buffalo_l"),
        OpenCVYuNet(),
        FaceAlignment68(),
        MediaPipeFaceMesh(),
    )
