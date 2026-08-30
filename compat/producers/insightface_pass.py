"""What one expensive observation pass actually produces, measured.

The storage question cannot be answered from the schema or from the backend's
own docstring. It needs the pass to run on real photographs and the outputs to
be inventoried by shape, dtype and byte cost -- including the outputs nothing
currently reads, which are the ones at risk of being dropped precisely because
no column names them.

This runs upstream `FaceAnalysis` directly rather than through
`vision/faces.py`. Two reasons, and both are the point:

  * the application's backend NORMALISES every coordinate to a fraction of
    width and height, ROUNDS landmarks to 5 places and pose to 2, and CLAMPS
    to [0, 1]. Those are conservation decisions. Measuring them requires
    seeing what came in before they were applied.
  * `Face.normed_embedding` is a property computed on access. Asking the
    application what it stores cannot show whether the normalised form is
    derivable from the raw one -- only running both can.

The pack is antelopev2: SCRFD-10GF detection, glintr100 recognition, plus the
genderage, 2d106det and 1k3d68 heads. Which heads ran is recorded, because an
inventory that does not say which models produced it is a list of numbers.
"""

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

#: The provisioned pack root, the directory ABOVE `models/`, as
#: `vision/weights.py` resolves it. Machine-local, so it takes an env var and
#: the resolved path is recorded in the evidence.
MODELS_ROOT: Path = Path(os.environ.get("COMPAT_INSIGHTFACE_ROOT", "C:/ComfyUI/output/.AImodels/insightface"))
PACK: str = "antelopev2"

#: CPU only: a CUDA reduction order is not fixed across driver versions, and a
#: number that changes with the GPU makes every downstream digest a claim about
#: this machine's driver.
PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)

#: The detection sizes THIS APPLICATION uses, in its order. Imported rather
#: than retyped, so a ladder that moves in `vision/faces.py` moves the
#: evidence with it.
DET_SIZE: tuple[tuple[int, int], ...] = tuple((one, one) for one in faces_module.DET_SIZES)


@dataclass
class FieldReport:
    """One value the producer emitted, by cost and by shape."""

    key: str
    kind: str
    dtype: str
    shape: tuple[int, ...] | None
    bytes_raw: int
    note: str = ""


@dataclass
class FaceReport:
    """Every field of one detected face, plus the derived relations tested."""

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
    """One corpus image through one pass."""

    path: str
    sha256: str
    height: int
    width: int
    faces: list[FaceReport] = field(default_factory=list)
    seconds: float = 0.0


#: The suite's one array digest. Four byte-identical copies of it lived
#: in four modules; `compat/assertions/arrays.py` holds the definition
#: and this is the name this module has always exported.
digest_array = digest


#: One prepared pack per (root, pack): five ONNX sessions are slow to stand up
#: and several runners want the same producer. Keyed rather than global so a
#: test can hold a differently-configured pack.
_prepared: dict[tuple[str, str], Any] = {}


def analysis(root: Path = MODELS_ROOT, pack: str = PACK) -> Any:
    """Upstream `FaceAnalysis`, prepared, over the provisioned pack.

    Imported inside the function: this tree bans a module-level import of
    insightface, and the ban is right -- the package pulls onnxruntime, and
    `compat` must stay outside anything `db/` or `sg_web/` can reach.
    """
    from insightface.app import FaceAnalysis

    key = (str(root), pack)
    if key not in _prepared:
        app = FaceAnalysis(name=pack, root=str(root), providers=list(PROVIDERS))
        app.prepare(ctx_id=-1, det_size=DET_SIZE[0])
        _prepared[key] = app
    return _prepared[key]


def detect(bgr: npt.NDArray[np.uint8], root: Path = MODELS_ROOT, pack: str = PACK) -> list[Any]:
    """Every face the application would find in this frame.

    The one entry point for `our_face`: `analysis().get(frame)` detects at
    whatever single size the pack was prepared with, and the application
    descends `DET_SIZES` until one finds a face.
    """
    return list(faces_module.first_hit_descending(analysis(root, pack), bgr))


def detect_padded(bgr: npt.NDArray[np.uint8], root: Path = MODELS_ROOT, pack: str = PACK) -> Any | None:
    """The padded-recovery face the application stores beside the primary."""
    return faces_module.padded_recovery(analysis(root, pack), bgr)


def loaded_models(app: Any) -> dict[str, str]:
    """Which heads the pack actually supplied, by taskname.

    An inventory that does not say which models ran is a list of numbers: the
    same pack minus `1k3d68` produces no pose and no 3D landmarks at all, and
    the absence would otherwise read as "the producer does not emit them".
    """
    return {str(taskname): type(model).__name__ for taskname, model in app.models.items()}


def _describe(key: str, value: object) -> FieldReport | None:
    """One emitted value as cost and shape, or None if it is not storable."""
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
    """A bbox as exactly four floats, checked by length rather than assumed.

    A comprehension produces a tuple of unknown arity as far as any checker is
    concerned, and a bbox that arrived with three or five elements should say
    so here rather than three frames on.
    """
    out = [float(one) for one in values]
    if len(out) != 4:
        raise ValueError(f"bbox has {len(out)} elements, not 4: {out!r}")
    return (out[0], out[1], out[2], out[3])


def inventory_face(face: Any, index: int) -> FaceReport:
    """Every key the producer put on one Face, and the derivability checks.

    Iterating the record rather than asking for named fields is the whole
    point: an allowlist reports exactly the fields somebody already thought
    of, which is the defect this suite exists to measure.
    """
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

    # The claim under test: is the normalised vector derivable from the raw
    # one? If so, the raw form is the one to keep, since its norm is not
    # recoverable from the unit vector.
    upstream_normed = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
    derived = raw / np.linalg.norm(raw)
    report.normed_max_abs_diff = float(np.max(np.abs(upstream_normed - derived)))
    report.normed_is_derivable = np.array_equal(upstream_normed, derived)

    pose = face.get("pose")
    if pose is not None:
        # Upstream order is [pitch, yaw, roll] -- landmark.py, the 1k3d68
        # head. Named here so the inventory cannot be read positionally.
        report.pose = {"pitch": float(pose[0]), "yaw": float(pose[1]), "roll": float(pose[2])}

    return report


def decode(path: Path) -> tuple[npt.NDArray[np.uint8], str]:
    """A BGR frame and the file's digest.

    `np.fromfile` then `cv2.imdecode` rather than `cv2.imread`: the latter
    takes a str path and silently returns None for a non-ASCII one on Windows,
    and a corpus is exactly where that turns into a false absence.
    """
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2 could not decode {path}")
    return bgr.astype(np.uint8, copy=False), hashlib.sha256(raw.tobytes()).hexdigest()


def run_image(app: Any, path: Path) -> ImageReport:
    """One image, decoded the way the consumers decode it, through the pass."""
    bgr, sha = decode(path)
    began = time.perf_counter()
    # The application's own ladder, not one `app.get` picks: the record this
    # producer inventories is the record `vision/faces.py` stores.
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
    """Per-field byte cost across every face observed, for the storage view."""
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
