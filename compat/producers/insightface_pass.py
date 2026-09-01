from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from compat.assertions.arrays import digest
from vision import faces as faces_module

MODELS_ROOT: Path = Path(os.environ.get("COMPAT_INSIGHTFACE_ROOT", "C:/ComfyUI/output/.AImodels/insightface"))
PACK: str = "antelopev2"


PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)


DET_SIZE: tuple[tuple[int, int], ...] = tuple((one, one) for one in faces_module.DET_SIZES)


@dataclass
class FieldReport:
    key: str
    kind: str
    dtype: str
    shape: tuple[int, ...] | None
    bytes_raw: int
    note: str = ""


@dataclass
class FaceReport:
    index: int
    fields: list[FieldReport] = field(default_factory=list)
    det_score: float = 0.0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    embedding_sha256: str = ""
    embedding_norm: float = 0.0
    normed_is_derivable: bool | None = None
    normed_max_abs_diff: float | None = None
    pose: dict[str, float] = field(default_factory=dict)


@dataclass
class ImageReport:
    path: str
    sha256: str
    height: int
    width: int
    faces: list[FaceReport] = field(default_factory=list)
    seconds: float = 0.0


digest_array = digest


_prepared: dict[tuple[str, str], Any] = {}


def analysis(root: Path = MODELS_ROOT, pack: str = PACK) -> Any:
    from insightface.app import FaceAnalysis

    key = (str(root), pack)
    if key not in _prepared:
        app = FaceAnalysis(name=pack, root=str(root), providers=list(PROVIDERS))
        app.prepare(ctx_id=-1, det_size=DET_SIZE[0])
        _prepared[key] = app
    return _prepared[key]


def detect(bgr: npt.NDArray[np.uint8], root: Path = MODELS_ROOT, pack: str = PACK) -> list[Any]:
    return list(faces_module.first_hit_descending(analysis(root, pack), bgr))


def detect_padded(bgr: npt.NDArray[np.uint8], root: Path = MODELS_ROOT, pack: str = PACK) -> Any | None:
    return faces_module.padded_recovery(analysis(root, pack), bgr)


def loaded_models(app: Any) -> dict[str, str]:
    return {str(taskname): type(model).__name__ for taskname, model in app.models.items()}


def _describe(key: str, value: object) -> FieldReport | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return FieldReport(
            key=key,
            kind="ndarray",
            dtype=str(value.dtype),
            shape=tuple(int(one) for one in value.shape),
            bytes_raw=value.nbytes,
        )
    if isinstance(value, (bool, np.bool_)):
        return FieldReport(key=key, kind="bool", dtype="bool", shape=None, bytes_raw=1)
    if isinstance(value, (int, np.integer)):
        return FieldReport(key=key, kind="int", dtype="int64", shape=None, bytes_raw=8)
    if isinstance(value, (float, np.floating)):
        return FieldReport(key=key, kind="float", dtype="float64", shape=None, bytes_raw=8)
    if isinstance(value, str):
        return FieldReport(key=key, kind="str", dtype="utf-8", shape=None, bytes_raw=len(value.encode("utf-8")))
    return FieldReport(key=key, kind=type(value).__name__, dtype="?", shape=None, bytes_raw=0, note="not measured")


def four_floats(values: Any) -> tuple[float, float, float, float]:
    out = [float(one) for one in values]
    if len(out) != 4:
        raise ValueError(f"bbox has {len(out)} elements, not 4: {out!r}")
    return (out[0], out[1], out[2], out[3])


def inventory_face(face: Any, index: int) -> FaceReport:
    report = FaceReport(index=index)

    for key in sorted(face.keys()):
        described = _describe(str(key), face[key])
        if described is not None:
            report.fields.append(described)

    report.det_score = float(face.det_score)
    report.bbox = four_floats(face.bbox)

    raw = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
    report.embedding_sha256 = digest_array(raw)
    report.embedding_norm = float(np.linalg.norm(raw))

    upstream_normed = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
    derived = raw / np.linalg.norm(raw)
    report.normed_max_abs_diff = float(np.max(np.abs(upstream_normed - derived)))
    report.normed_is_derivable = np.array_equal(upstream_normed, derived)

    pose = face.get("pose")
    if pose is not None:
        report.pose = {"pitch": float(pose[0]), "yaw": float(pose[1]), "roll": float(pose[2])}

    return report


def decode(path: Path) -> tuple[npt.NDArray[np.uint8], str]:
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2 could not decode {path}")
    return bgr.astype(np.uint8, copy=False), hashlib.sha256(raw.tobytes()).hexdigest()


def run_image(app: Any, path: Path) -> ImageReport:
    bgr, sha = decode(path)
    began = time.perf_counter()

    faces = faces_module.first_hit_descending(app, bgr)
    seconds = time.perf_counter() - began

    height, width = bgr.shape[:2]
    return ImageReport(
        path=str(path),
        sha256=sha,
        height=int(height),
        width=int(width),
        faces=[inventory_face(one, index) for index, one in enumerate(faces)],
        seconds=seconds,
    )


def field_costs(reports: list[ImageReport]) -> dict[str, dict[str, Any]]:
    gathered: dict[str, dict[str, Any]] = {}
    for image in reports:
        for face in image.faces:
            for one in face.fields:
                row = gathered.setdefault(
                    one.key,
                    {"kind": one.kind, "dtype": one.dtype, "shapes": set(), "sizes": [], "faces": 0},
                )
                row["shapes"].add(one.shape)
                row["sizes"].append(one.bytes_raw)
                row["faces"] += 1

    out: dict[str, dict[str, Any]] = {}
    for key, row in gathered.items():
        sizes: list[int] = row["sizes"]
        per_face = max(sizes) if sizes else 0
        out[key] = {
            "kind": row["kind"],
            "dtype": row["dtype"],
            "shapes": sorted(str(one) for one in row["shapes"]),
            "faces": row["faces"],
            "bytes_per_face": per_face,
            "bytes_min": min(sizes) if sizes else 0,
            "at_1k": per_face * 1_000,
            "at_22k": per_face * 22_000,
            "at_1m": per_face * 1_000_000,
        }
    return out
