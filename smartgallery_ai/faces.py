"""Face detection, embedding, and generated-identity clustering (WI-31).

IMPORTANT: `ai_face_clusters` groups faces that recur across a *generated*
image collection because their embeddings are close in a similarity space
learned by a face-recognition network. This is clustering for browsing
convenience only -- it is NOT real-world identity recognition, verification,
or attribution, and a cluster's `label` is a free-text nickname a human
attaches to a bucket of similar-looking generated faces, never a claim about
who (if anyone real) a face resembles.

`StubFaceBackend` is a TEST/DEV stub: it returns pre-programmed detections
and does not look at pixels at all. Two real pipelines are deployed and
config-swappable (`AI_DAM_FACE_BACKEND`): `InsightFaceBackend` (upstream
FaceAnalysis over the provisioned antelopev2 pack -- SCRFD detection,
glintr100 embedding, genderage attributes; preferred by `auto`) and
`OpenCVFaceBackend` (YuNet detection plus ArcFace-glintr100-via-cv2.dnn
or SFace embedding). All models load only from local files under
`models_dir`, never downloaded here. Backends self-report
`BackendUnavailable` instead of raising when runtime or weights are
missing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.faiss_runtime import import_faiss

_logger = logging.getLogger(__name__)

# insightface 1.0.1 aligns faces through skimage's pre-2.2 estimate()
# API; skimage 0.26 deprecates it with a FutureWarning that fires on
# EVERY alignment. Silence exactly that warning at its source module —
# every other warning stays visible.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"insightface\.utils\.face_align", message=r".*`estimate` is deprecated.*"
)

__all__ = [
    "BackendUnavailable",
    "FaceBackend",
    "FaceDetection",
    "OpenCVFaceBackend",
    "StubFaceBackend",
    "cluster_faces",
    "get_face_backend",
    "replace_faces_for_file",
]

_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"  # detector ONNX, expected directly under models_dir
_SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"  # recognizer ONNX, expected directly under models_dir
# glintr100 lives inside the provisioned antelopev2 pack (FaceAnalysis
# layout: <models_dir>/insightface/models/antelopev2/); the cv2 arcface
# embedder reads it from there so the weights exist exactly once.
_INSIGHTFACE_ROOT = "insightface"  # models_dir-relative FaceAnalysis root
_ARCFACE_FILENAME = os.path.join(_INSIGHTFACE_ROOT, "models", "antelopev2", "glintr100.onnx")

# ArcFace canonical 112x112 5-landmark template
# (insightface python-package/insightface/utils/face_align.py: arcface_dst).
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991) mapping `src`
    points onto `dst`, as a 2x3 affine matrix. Matches
    skimage.transform.SimilarityTransform.estimate — the estimator
    insightface's norm_crop uses — to ~1e-6 (verified over 200 random
    landmark sets) without a scikit-image dependency."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, d = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    sign = np.ones(d)
    if np.linalg.det(cov) < 0:
        sign[-1] = -1
    rot = u @ np.diag(sign) @ vt
    var_src = (src_c**2).sum() / n
    scale = (s * sign).sum() / var_src
    t = dst_mean - scale * (rot @ src_mean)
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = t
    return m


def _arcface_norm_crop(bgr: np.ndarray, landmarks_px: np.ndarray) -> np.ndarray:
    """Warp the face to the canonical ArcFace 112x112 crop from its five
    detector landmarks (insightface face_align.norm_crop)."""
    m = _umeyama_similarity(landmarks_px, _ARCFACE_DST)
    return cv2.warpAffine(bgr, m, (112, 112), borderValue=0.0)


_SIM_CHUNK_SIZE = 256  # rows per similarity block; caps clustering memory at O(chunk * n)
_LABEL_MATCH_THRESHOLD = 0.9  # min centroid cosine for a recomputed cluster to inherit an old label


@dataclass
class FaceDetection:
    """One detected face. Coordinates are normalized [0, 1] per schema.py."""

    bbox: tuple  # (x, y, w, h), normalized
    landmarks: list  # list[(x, y)], normalized, may be empty
    det_score: float  # detector confidence; higher is more face-like
    embedding: np.ndarray | None  # float32 1-D, or None
    dim: int | None = None  # embedding length; derived from `embedding` when one is present
    attributes: dict | None = None  # per-face model attributes, e.g. {"age": 27, "sex": "M"}

    def __post_init__(self) -> None:
        """Coerce fields to plain floats / float32 and keep `dim` consistent
        with the embedding actually carried."""
        self.bbox = tuple(float(v) for v in self.bbox)
        self.landmarks = [tuple(float(v) for v in pt) for pt in self.landmarks]
        if self.embedding is not None:
            arr = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
            self.embedding = arr
            self.dim = int(arr.shape[0])


class FaceBackend(ABC):
    """Face detector + per-face embedder over a single image."""

    model_id: str  # provenance recorded on every ai_face_instances row
    model_version: str  # scopes stored instances and clusters; versions never mix
    default_cluster_threshold: float = 0.55  # per-embedder operating point
    # No face backend declares itself callable from two threads at once:
    # cv2.FaceDetectorYN and insightface's FaceAnalysis both carry per-call
    # state inside the native object. `smartgallery_ai.backends` therefore
    # leases them exclusively. See SemanticEmbedder.thread_safe.
    thread_safe: bool = False

    @abstractmethod
    def detect(self, img: Image.Image) -> list:
        """Detect faces in `img`. Returns a list of `FaceDetection`."""


def resolve_cluster_threshold(config: AIConfig, backend: FaceBackend) -> float:
    """The clustering threshold to use: the explicit config value when set
    (AI_DAM_FACE_CLUSTER_THRESHOLD), else the backend's per-embedder
    default — embedders occupy different cosine scales, so one global
    number cannot serve both."""
    if config.face_cluster_threshold is not None:
        return config.face_cluster_threshold
    return getattr(backend, "default_cluster_threshold", 0.55)


class StubFaceBackend(FaceBackend):
    """TEST/DEV STUB -- pre-programmed, deterministic detections.

    `source` is either:
      - a callable `source(img) -> list[FaceDetection]`, or
      - a mapping from `image_key(img)` to a pre-programmed
        `list[FaceDetection]` (images whose key is absent detect as empty).
    """

    model_id = "stub-face"
    model_version = "stub-v1"

    def __init__(self, source: Callable[[Image.Image], list] | Mapping):
        self._source = source

    def detect(self, img: Image.Image) -> list:
        """Replay the pre-programmed detections for `img`; unknown images
        detect as no faces."""
        if callable(self._source):
            return list(self._source(img))
        return list(self._source.get(image_key(img), []))


def image_key(img: Image.Image) -> str:
    """A deterministic key for an image's pixel content, for use with
    `StubFaceBackend`'s mapping form (content-based, not object identity)."""

    digest = hashlib.sha256(img.tobytes()).hexdigest()
    return f"{img.mode}:{img.size[0]}x{img.size[1]}:{digest}"


def _clamp01(value: float) -> float:
    """Clamp to [0, 1], the normalized-coordinate range stored in the DB."""
    return 0.0 if value < 0.0 else (min(value, 1.0))


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert to the BGR uint8 array layout OpenCV expects."""
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class OpenCVFaceBackend(FaceBackend):
    """YuNet detector + a per-face recognizer, all through OpenCV, loaded
    only from local ONNX files under `models_dir`.

    Recognizers ('embedder'):
      - 'arcface' — antelopev2 glintr100 (ResNet100@Glint360K, 512-d),
        aligned via the canonical 5-landmark Umeyama warp. Best of the
        three-way labeled A/B (benchmarks/results/face_embedder_ab.json);
        weights are non-commercial research license
        (deepinsight/insightface).
      - 'sface'   — OpenCV FaceRecognizerSF (128-d), its own alignCrop.
      - 'auto'    — arcface when its weights are present, else sface.

    Raises `BackendUnavailable` (never crashes) when the cv2 build lacks
    these APIs or the model files are not present.
    """

    def __init__(
        self,
        models_dir: str,
        min_det_score: float = 0.5,
        min_face_px: int = 24,
        detect_max_side: int = 1600,
        embedder: str = "auto",
    ):
        """Load the detector and the selected recognizer. `min_det_score`
        is the minimum detector confidence for a face to be reported;
        `min_face_px` is the minimum face box side in detect-input pixels —
        YuNet detects down to ~10px, and detections near that floor are
        featureless, embed into one generic region, and chain unrelated
        clusters together. `detect_max_side` caps the detection input:
        images larger than N px on their longest side are downscaled first,
        keeping large faces inside YuNet's ~10-300px training band
        (measured: >=300px-face recall 55%->97%, false positives 7x down,
        detection 3.7x faster — docs/FACE_CLUSTERING.md). 0 disables the
        cap. A forced `embedder` whose weights are missing raises instead
        of silently falling back."""
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise BackendUnavailable("this OpenCV build lacks FaceDetectorYN/FaceRecognizerSF")
        detector_path = os.path.join(models_dir, _YUNET_FILENAME)
        recognizer_path = os.path.join(models_dir, _SFACE_FILENAME)
        arcface_path = os.path.join(models_dir, _ARCFACE_FILENAME)
        if not os.path.isfile(detector_path):
            raise BackendUnavailable(f"YuNet model not found at {detector_path}")
        if embedder == "auto":
            embedder = "arcface" if os.path.isfile(arcface_path) else "sface"
        if embedder == "arcface":
            if not os.path.isfile(arcface_path):
                raise BackendUnavailable(f"ArcFace model not found at {arcface_path}")
        elif embedder == "sface":
            if not os.path.isfile(recognizer_path):
                raise BackendUnavailable(f"SFace model not found at {recognizer_path}")
        else:
            raise ValueError(f"unknown face embedder: {embedder!r}")
        self._embedder = embedder
        self.model_id = f"opencv/yunet+{embedder}"
        self.model_version = (
            "yunet-2023mar+arcface-glintr100-ms1600"
            if embedder == "arcface"
            else "yunet-2023mar+sface-2021dec-v2-ms1600"
        )
        # Operating points from the labeled three-way A/B sweep
        # (benchmarks/face_embedder_ab.py, 175 faces / 31 identities):
        # glintr100 pairwise-F1 is flat 0.926-0.933 across 0.30-0.50; 0.48
        # keeps near-peak F1 (0.931) at the sweep's best precision (0.968).
        # sface peaks narrowly near 0.45-0.55.
        self.default_cluster_threshold = 0.48 if embedder == "arcface" else 0.55
        # Model creation logs native "setPreferableTarget ... not supported"
        # WARNs from inside OpenCV on some builds; not actionable, so hold
        # cv2's native log level at ERROR just for the create calls.
        cv2_log = getattr(getattr(cv2, "utils", None), "logging", None)
        prev_level = None
        if cv2_log is not None:
            try:
                prev_level = cv2_log.getLogLevel()
                cv2_log.setLogLevel(cv2_log.LOG_LEVEL_ERROR)
            except Exception:  # log tuning must never block loading
                _logger.debug("handled a failure in __init__", exc_info=True)
                prev_level = None
        try:
            self._detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), score_threshold=min_det_score)
            if embedder == "arcface":
                self._recognizer = None
                self._arcface = cv2.dnn.readNetFromONNX(arcface_path)
            else:
                self._recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
                self._arcface = None
        except Exception as exc:
            raise BackendUnavailable(f"failed to load face models: {exc}") from exc
        finally:
            if prev_level is not None:
                cv2_log.setLogLevel(prev_level)
        self._min_det_score = min_det_score
        self._min_face_px = min_face_px
        self._detect_max_side = detect_max_side

    def detect(self, img: Image.Image) -> list:
        """Detect faces, embed each via SFace on the aligned crop, and return
        `FaceDetection`s with coordinates normalized to [0, 1]. Detection and
        alignment run on the (possibly downscaled) detect input; normalized
        coordinates are scale-free."""
        if self._detect_max_side and max(img.size) > self._detect_max_side:
            f = self._detect_max_side / max(img.size)
            img = img.resize(
                (max(1, round(img.size[0] * f)), max(1, round(img.size[1] * f))),
                Image.Resampling.LANCZOS,
            )
        bgr = _pil_to_bgr(img)
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return []
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None:
            return []

        detections = []
        # YuNet row layout: x, y, w, h, five landmark (x, y) pairs, confidence.
        for row in faces:
            score = float(row[14])
            if score < self._min_det_score:
                continue
            x, y, bw, bh = (float(v) for v in row[0:4])
            if min(bw, bh) < self._min_face_px:
                continue
            landmarks_px = row[4:14].reshape(5, 2)
            if self._arcface is not None:
                # ArcFace contract (insightface arcface_onnx.py get_feat):
                # canonical 112x112 norm_crop, then blob with mean/std
                # 127.5 and BGR->RGB swap; 512-d output, cosine-ready
                # after normalization downstream.
                aligned = _arcface_norm_crop(bgr, landmarks_px)
                blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                self._arcface.setInput(blob)
                feature = self._arcface.forward()
            else:
                aligned = self._recognizer.alignCrop(bgr, row)
                feature = self._recognizer.feature(aligned)
            embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
            bbox = (
                _clamp01(x / w),
                _clamp01(y / h),
                _clamp01(bw / w),
                _clamp01(bh / h),
            )
            landmarks = [(_clamp01(px / w), _clamp01(py / h)) for px, py in landmarks_px]
            detections.append(
                FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    det_score=score,
                    embedding=embedding,
                )
            )
        return detections


class InsightFaceBackend(FaceBackend):
    """insightface's own pipeline (FaceAnalysis over the provisioned
    antelopev2 pack): SCRFD-10GF joint 128+640 detection, upstream
    5-landmark alignment, glintr100 (ResNet100@Glint360K, 512-d)
    embedding. On the labeled A/B this is near-perfect
    (pairwise F1 0.999, P 1.000/R 0.998 at threshold 0.35-0.40 —
    benchmarks/results/face_embedder_ab.json); the gap over YuNet-based
    pipelines is detection + landmark-alignment quality, not the
    recognizer. Weights are non-commercial research license
    (deepinsight/insightface)."""

    model_id = "insightface/antelopev2"
    model_version = "scrfd10g+glintr100-v1"  # attributes fill in place; embeddings are version-stable
    # Pairwise F1 is 0.995-0.999 across 0.35-0.50 on the labeled A/B;
    # 0.40 keeps P 1.000 with F1 0.998.
    default_cluster_threshold = 0.40

    def __init__(self, models_dir: str, min_det_score: float = 0.5, min_face_px: int = 24):
        """`min_det_score` re-filters detections (FaceAnalysis is prepared
        at the same threshold); `min_face_px` drops noise-floor boxes by
        native-pixel side, same junk gate as the OpenCV backend."""
        self._app = get_insightface_app(models_dir)
        self._min_det_score = min_det_score
        self._min_face_px = min_face_px

    def detect(self, img: Image.Image) -> list:
        bgr = _pil_to_bgr(img)
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return []
        detections = []
        for face in self._app.get(bgr):
            score = float(face.det_score)
            if score < self._min_det_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            if min(x2 - x1, y2 - y1) < self._min_face_px:
                continue
            embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            bbox = (_clamp01(x1 / w), _clamp01(y1 / h), _clamp01((x2 - x1) / w), _clamp01((y2 - y1) / h))
            landmarks = (
                [(_clamp01(float(px) / w), _clamp01(float(py) / h)) for px, py in face.kps]
                if face.kps is not None
                else []
            )
            attributes: dict = {}
            if face.gender is not None and face.age is not None:
                attributes["age"] = int(face.age)
                attributes["sex"] = face.sex
            lmk106 = face.get("landmark_2d_106")
            if lmk106 is not None:
                attributes["landmark_2d_106"] = [
                    [round(_clamp01(float(px) / w), 5), round(_clamp01(float(py) / h), 5)] for px, py in lmk106
                ]
            lmk68 = face.get("landmark_3d_68")
            if lmk68 is not None:
                # x/y normalized like every other coordinate; z stays in
                # the model's pixel-scaled depth units (no image norm
                # exists for depth) — recorded as-is.
                attributes["landmark_3d_68"] = [
                    [round(_clamp01(float(px) / w), 5), round(_clamp01(float(py) / h), 5), round(float(pz), 2)]
                    for px, py, pz in lmk68
                ]
            pose = face.get("pose")
            if pose is not None:
                attributes["pose"] = {
                    "pitch": round(float(pose[0]), 2),
                    "yaw": round(float(pose[1]), 2),
                    "roll": round(float(pose[2]), 2),
                }
            detections.append(
                FaceDetection(
                    bbox=bbox, landmarks=landmarks, det_score=score, embedding=embedding, attributes=attributes or None
                )
            )
        return detections


def _ort_providers() -> list:
    """Execution providers for the insightface ORT sessions, in ORT's
    priority-list form (docs/python/api_summary.rst: kernels are chosen
    in the order given; anything a provider lacks runs on CPU).
    AI_DAM_ORT_PROVIDERS: 'auto' (default -- CUDA first when the
    installed onnxruntime build offers it, which means installing
    onnxruntime-gpu yourself; nothing swaps it in), 'cpu', or an explicit
    comma list."""
    value = os.environ.get("AI_DAM_ORT_PROVIDERS", "auto").strip()
    if value.lower() == "cpu":
        return ["CPUExecutionProvider"]
    if value and value.lower() != "auto":
        return [p.strip() for p in value.split(",") if p.strip()]
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        _logger.debug("ignored a failure in _ort_providers", exc_info=True)
    return ["CPUExecutionProvider"]


_insightface_apps: dict = {}  # models_dir -> FaceAnalysis (cached; models stay loaded)


def get_insightface_app(models_dir: str):
    """insightface's own pipeline (FaceAnalysis, detection + recognition)
    over the provisioned antelopev2 pack, cached per models_dir. Raises
    `BackendUnavailable` when the package or the pack is missing."""
    if models_dir in _insightface_apps:
        return _insightface_apps[models_dir]
    pack_dir = os.path.join(models_dir, _INSIGHTFACE_ROOT, "models", "antelopev2")
    if not os.path.isdir(pack_dir):
        raise BackendUnavailable(f"antelopev2 pack not found at {pack_dir}")
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise BackendUnavailable(f"insightface unavailable: {exc}") from exc
    try:
        # Every pack head loads: genderage (age/sex), 2d106det (dense
        # 106-pt 2D landmarks), 1k3d68 (3D 68-pt + pitch/yaw/roll pose,
        # a 143MB session — the cost of keeping the pack's data
        # first-class). All of it persists per face in
        # FaceDetection.attributes.
        #
        # Providers are PER STAGE, from measurement on the dev box:
        # detection runs dynamic input shapes (SCRFD '?' dims), where the
        # CUDA EP re-tunes conv algos per shape and loses to CPU (205ms
        # CPU vs 280-440ms CUDA per image); recognition is a heavy
        # ResNet100 at a fixed 112x112, where CUDA wins 4.4x (14.6ms vs
        # 64.5ms per face). So detection + genderage stay on CPU and the
        # recognition session gets _ort_providers() (CUDA when the
        # installed build offers it; AI_DAM_ORT_PROVIDERS overrides).
        app = FaceAnalysis(
            name="antelopev2",
            root=os.path.join(models_dir, _INSIGHTFACE_ROOT),
            allowed_modules=["detection", "recognition", "genderage", "landmark_2d_106", "landmark_3d_68"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0)  # Auto det-size: joint 128x128 + 640x640
        rec_providers = _ort_providers()
        if rec_providers != ["CPUExecutionProvider"]:
            from insightface.model_zoo import model_zoo

            rec = model_zoo.get_model(os.path.join(models_dir, _ARCFACE_FILENAME), providers=rec_providers)
            rec.prepare(ctx_id=0)
            app.models["recognition"] = rec
    except Exception as exc:
        raise BackendUnavailable(f"FaceAnalysis failed to load: {exc}") from exc
    _insightface_apps[models_dir] = app
    return app


def installed_pipelines(config: AIConfig) -> list:
    """Inventory of every face pipeline this install can run: identity,
    whether its weights are on disk, and whether it is the pipeline the
    `auto`/configured selector resolves to right now."""
    models_dir = config.models_dir
    active = None
    try:
        backend = get_face_backend(config)
        if backend is not None:
            active = (backend.model_id, backend.model_version)
    except (BackendUnavailable, ValueError):
        pass

    def _entry(name, model_id, model_version, weight_paths, selector):
        present = all(os.path.isfile(os.path.join(models_dir, p)) for p in weight_paths)
        return {
            "name": name,
            "model_id": model_id,
            "model_version": model_version,
            "weights_present": present,
            "selector": selector,
            "active": active == (model_id, model_version),
        }

    return [
        _entry(
            "scrfd+glintr100",
            "insightface/antelopev2",
            "scrfd10g+glintr100-v1",
            [os.path.join(_INSIGHTFACE_ROOT, "models", "antelopev2", "scrfd_10g_bnkps.onnx"), _ARCFACE_FILENAME],
            "AI_DAM_FACE_BACKEND=insightface",
        ),
        _entry(
            "yunet+arcface",
            "opencv/yunet+arcface",
            "yunet-2023mar+arcface-glintr100-ms1600",
            [_YUNET_FILENAME, _ARCFACE_FILENAME],
            "AI_DAM_FACE_BACKEND=opencv AI_DAM_FACE_EMBEDDER=arcface",
        ),
        _entry(
            "yunet+sface",
            "opencv/yunet+sface",
            "yunet-2023mar+sface-2021dec-v2-ms1600",
            [_YUNET_FILENAME, _SFACE_FILENAME],
            "AI_DAM_FACE_BACKEND=opencv AI_DAM_FACE_EMBEDDER=sface",
        ),
    ]


# Lane name -> the backend registry kind that supplies it. Lanes are the
# distinct DETECTION stacks (the two opencv embedder variants share YuNet
# boxes, so one opencv lane runs with the configured embedder). Both kinds
# construct their pipeline explicitly rather than through the `auto`
# selector, so the comparison always shows every lane whichever one
# production uses.
COMPARE_LANES = {"yunet": "faces_opencv", "scrfd": "faces_insightface"}


def compare_detectors(img: Image.Image, config: AIConfig, registry) -> dict:
    """Run every installed face pipeline on one image and report raw
    detections side by side — a diagnostic, never persisted. Returns
    {"lanes": {name: {model, elapsed_ms, faces|error}},
     "installed": installed_pipelines(...)} with normalized coords and
    per-face landmarks.

    `registry` is `smartgallery_ai.backends`, passed in rather than
    imported: that module imports this one to resolve face backends, so the
    dependency can only run one way. It supplies the loaded instances —
    re-reading the YuNet and ArcFace ONNX graphs on every call to this
    diagnostic is what passing it in avoids — and holds each lane
    exclusively while it runs, since no face backend is safe to call from
    two threads.
    """

    def _lane(lane: str, kind: str) -> dict:
        """One lane's timings and detections, or why it could not run.

        A lane whose weights are missing reports that and the comparison
        still shows the others.
        """
        with registry.lease(kind, config) as backend:
            if backend is None:
                return {"model": lane, "error": registry.why_unavailable(kind, config) or "unavailable"}
            t0 = time.perf_counter()
            dets = backend.detect(img)
            return {
                "model": f"{backend.model_id} ({backend.model_version})",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                "faces": [
                    {
                        "bbox": list(d.bbox),
                        "landmarks": d.landmarks,
                        "det_score": d.det_score,
                        "attributes": d.attributes,
                    }
                    for d in dets
                ],
            }

    lanes = {lane: _lane(lane, kind) for lane, kind in COMPARE_LANES.items()}
    return {"lanes": lanes, "installed": installed_pipelines(config)}


def get_face_backend(config: AIConfig) -> FaceBackend | None:
    """Resolve `config.face_backend` to a backend instance, or None.

    'none' -> None. 'stub' -> StubFaceBackend, sourced from
    `config.extra["face_stub_source"]` (defaults to always-empty, since a
    real per-test source can't be expressed through AIConfig alone).
    'insightface' -> InsightFaceBackend, raising if unavailable.
    'opencv' -> OpenCVFaceBackend, raising if unavailable.
    'auto' -> InsightFaceBackend when available (best measured pipeline,
    benchmarks/results/face_embedder_ab.json), else OpenCVFaceBackend,
    else None (never the stub).
    """
    name = config.face_backend
    if name == "none":
        return None
    if name == "stub":
        source = config.extra.get("face_stub_source", lambda _img: [])
        return StubFaceBackend(source)
    if name == "insightface":
        return InsightFaceBackend(
            config.models_dir,
            config.face_min_det_score,
            config.face_min_px,
        )
    if name == "opencv":
        return OpenCVFaceBackend(
            config.models_dir,
            config.face_min_det_score,
            config.face_min_px,
            config.face_detect_max_side,
            config.face_embedder,
        )
    if name == "auto":
        try:
            return InsightFaceBackend(
                config.models_dir,
                config.face_min_det_score,
                config.face_min_px,
            )
        except BackendUnavailable:
            pass
        try:
            return OpenCVFaceBackend(
                config.models_dir,
                config.face_min_det_score,
                config.face_min_px,
                config.face_detect_max_side,
                config.face_embedder,
            )
        except BackendUnavailable:
            return None
    raise ValueError(f"unknown face_backend: {name!r}")


def replace_faces_for_file(
    conn: sqlite3.Connection,
    file_id: str,
    detections: Sequence[FaceDetection],
    model_id: str,
    model_version: str,
    source_mtime: float,
    now: float,
) -> list:
    """Transactionally replace THIS MODEL's `ai_face_instances` rows for
    `file_id`.

    Rows are provenance-scoped: deleting is limited to (file_id, model_id,
    model_version), so switching `AI_DAM_FACE_BACKEND` (or running a second
    pipeline) never destroys another model's stored faces and embeddings —
    each pipeline owns its own rows and clusters. Re-running the same model
    with a different detection count leaves exactly that many rows behind.
    Returns the new `face_id`s in insertion order.
    """
    try:
        conn.execute(
            "DELETE FROM ai_face_instances WHERE file_id = ? AND model_id = ? AND model_version = ?",
            (file_id, model_id, model_version),
        )
        cur = conn.cursor()
        face_ids = []
        for det in detections:
            embedding_blob = None
            dim = det.dim
            if det.embedding is not None:
                embedding_blob = det.embedding.tobytes()
                dim = int(det.embedding.shape[0])
            landmarks_json = json.dumps([[x, y] for x, y in det.landmarks])
            # Scalars land in their own typed columns (comparable and
            # aggregatable in SQL); the structured geometry stays as
            # normalized JSON arrays in `attributes`.
            attrs = dict(det.attributes or {})
            age = attrs.pop("age", None)
            sex = attrs.pop("sex", None)
            pose = attrs.pop("pose", None) or {}
            attributes_json = json.dumps(attrs) if attrs else None
            cur.execute(
                """
                INSERT INTO ai_face_instances
                    (file_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, det_score,
                     embedding, dim, attributes, age, sex,
                     pose_pitch, pose_yaw, pose_roll,
                     model_id, model_version, source_mtime, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[2],
                    det.bbox[3],
                    landmarks_json,
                    det.det_score,
                    embedding_blob,
                    dim,
                    attributes_json,
                    age,
                    sex,
                    pose.get("pitch"),
                    pose.get("yaw"),
                    pose.get("roll"),
                    model_id,
                    model_version,
                    source_mtime,
                    now,
                ),
            )
            face_ids.append(cur.lastrowid)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    else:
        return face_ids


def _csr_from_edges(rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, n: int) -> tuple:
    """Assemble row-grouped edge arrays into CSR (indptr, cols, weights).

    `rows` must be non-decreasing (every backend emits edges row-major), so
    grouping is a bincount + cumsum, never a sort.
    """
    counts = np.bincount(rows, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, cols.astype(np.int64, copy=False), weights.astype(np.float32, copy=False)


def _neighbor_graph_torch_cuda(normed: np.ndarray, threshold: float) -> tuple:
    """Cosine-threshold graph via blocked CUDA matmul (torch) -> CSR.

    Faiss GPU indexes expose only k-NN `search`, never `range_search`
    (faiss/gpu/GpuIndex.h), but the *operation* — an exhaustive threshold
    graph — is a plain tiled matrix multiply, which any CUDA tensor runtime
    performs. TF32 is disabled (docs/source/notes/cuda.md: allow_tf32=False
    forces IEEE float32 matmul) so the edge set matches the CPU backends
    instead of drifting at the threshold boundary. Self-edges are masked on
    device; no per-edge Python loop anywhere.

    Raises (ImportError / RuntimeError) when torch or a CUDA device is
    absent; the dispatcher decides whether that is a hard error.

    torch is imported here rather than at module scope because this is the
    only function in the package that uses it, and importing it costs 1.7
    seconds and several hundred megabytes -- paid by every process that
    touched the gallery, including installs with the AI layer opted out,
    which never reach this line. It also made the ImportError promised
    above unreachable: a missing torch failed the whole package at import
    rather than falling back to the faiss and numpy backends.

    `import torch` alone binds both attributes used below -- torch/
    __init__.py:2304,2306 import `backends` and `cuda` eagerly.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch is installed but no CUDA device is available")
    torch.backends.cuda.matmul.allow_tf32 = False
    n = normed.shape[0]
    x = torch.from_numpy(normed).cuda()
    row_parts, col_parts, w_parts = [], [], []
    with torch.no_grad():
        for start in range(0, n, 1024):
            end = min(start + 1024, n)
            sims = x[start:end] @ x.T  # (b, n) tile, ~90MB at n~22k
            mask = sims >= threshold
            b = end - start
            mask[torch.arange(b, device=mask.device), torch.arange(start, end, device=mask.device)] = False
            idx = mask.nonzero()
            row_parts.append((idx[:, 0] + start).cpu().numpy())
            col_parts.append(idx[:, 1].cpu().numpy())
            w_parts.append(sims[mask].cpu().numpy())
    rows = np.concatenate(row_parts) if row_parts else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(col_parts) if col_parts else np.zeros(0, dtype=np.int64)
    weights = np.concatenate(w_parts) if w_parts else np.zeros(0, dtype=np.float32)
    return _csr_from_edges(rows, cols, weights, n)


def _neighbor_graph_faiss(normed: np.ndarray, threshold: float) -> tuple:
    """Cosine-threshold graph via FAISS CPU range_search -> CSR.

    IndexFlatIP is exact inner product, which equals cosine on the unit
    vectors passed in (facebookresearch/faiss README); range_search's
    (lims, D, I) result already IS a CSR triple — only self-edges are
    filtered out, vectorized. Raises ImportError when faiss is not
    installed.

    The radius is nudged one float32 step BELOW the threshold because this
    backend's comparison is strict where the others' are not. For a
    similarity metric, range_search keeps `dis > radius`:
    IndexFlat.cpp:73 -> distances.cpp:909-923 -> ResultHandler.h:760 picks
    CMin for METRIC_INNER_PRODUCT (MetricType.h:56-58), and
    ordered_key_value.h:46 defines CMin::cmp(a, b) as `a < b`. A face
    sitting exactly on the threshold would therefore be dropped here and
    kept by the numpy and torch paths, which use `>=` — the same library,
    the same threshold, two different cluster memberships. Stepping the
    radius down by one ULP makes `dis > radius` mean `dis >= threshold`.
    """
    faiss = import_faiss()

    n = normed.shape[0]
    index = faiss.IndexFlatIP(int(normed.shape[1]))
    index.add(normed)
    inclusive = float(np.nextafter(np.float32(threshold), np.float32("-inf")))
    lims, sims, ids = index.range_search(normed, inclusive)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(lims).astype(np.int64))
    keep = ids != rows
    return _csr_from_edges(rows[keep], ids[keep], sims[keep], n)


def _neighbor_graph_numpy(normed: np.ndarray, threshold: float) -> tuple:
    """Cosine-threshold graph via chunked NumPy matmul -> CSR; always
    available, memory bounded at O(chunk * n)."""
    n = normed.shape[0]
    row_parts, col_parts, w_parts = [], [], []
    for start in range(0, n, _SIM_CHUNK_SIZE):
        end = min(start + _SIM_CHUNK_SIZE, n)
        sims = normed[start:end] @ normed.T  # (b, n)
        mask = sims >= threshold
        mask[np.arange(end - start), np.arange(start, end)] = False
        bi, j = np.nonzero(mask)
        row_parts.append(bi.astype(np.int64) + start)
        col_parts.append(j)
        w_parts.append(sims[mask])
    rows = np.concatenate(row_parts) if row_parts else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(col_parts) if col_parts else np.zeros(0, dtype=np.int64)
    weights = np.concatenate(w_parts) if w_parts else np.zeros(0, dtype=np.float32)
    return _csr_from_edges(rows, cols, weights, n)


def _neighbor_graph(normed: np.ndarray, threshold: float) -> tuple:
    """CSR cosine graph (edge iff sim >= threshold, self excluded).

    Returns ((indptr, cols, weights), backend): indptr has n+1 entries and
    row i's neighbors live at cols[indptr[i]:indptr[i+1]] with matching
    weights; backend names the implementation that actually ran
    ("torch-cuda", "faiss-cpu", or "numpy"). All three backends compute the
    same exhaustive edge set — they differ only in where the multiply runs.

    Selection honors AI_DAM_FACE_GRAPH_BACKEND (auto | torch-cuda | faiss |
    numpy). A specific request that cannot be satisfied raises instead of
    silently falling back; "auto" tries torch-cuda, then faiss, then numpy.
    """
    requested = os.environ.get("AI_DAM_FACE_GRAPH_BACKEND", "auto").strip().lower()
    n = normed.shape[0]
    if n == 0:
        empty = (np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
        return empty, "numpy"

    if requested == "torch-cuda":
        return _neighbor_graph_torch_cuda(normed, threshold), "torch-cuda"
    if requested == "faiss":
        return _neighbor_graph_faiss(normed, threshold), "faiss-cpu"
    if requested == "numpy":
        return _neighbor_graph_numpy(normed, threshold), "numpy"
    if requested != "auto":
        raise ValueError(f"AI_DAM_FACE_GRAPH_BACKEND={requested!r} is not one of auto/torch-cuda/faiss/numpy")

    try:
        return _neighbor_graph_torch_cuda(normed, threshold), "torch-cuda"
    except Exception:
        _logger.debug("ignored a failure in _neighbor_graph", exc_info=True)
    try:
        return _neighbor_graph_faiss(normed, threshold), "faiss-cpu"
    except ImportError:
        pass
    return _neighbor_graph_numpy(normed, threshold), "numpy"


def _chinese_whispers(graph: tuple, sweeps: int = 20) -> list:
    """Deterministic chinese-whispers label propagation over the CSR
    neighbor graph (dlib/clustering/chinese_whispers.h is the canonical
    form: each node adopts the label with the highest summed edge weight
    among its neighbors). dlib visits nodes randomly; this variant sweeps
    nodes in ascending index order with ties broken toward the LOWEST label
    id (np.unique returns labels ascending and np.argmax takes the first
    maximum), so the result is a pure function of the graph.

    Single-linkage connected components are unsuitable here: transitive
    chaining merges dense look-alike sets into one cluster.
    """
    indptr, cols, weights = graph
    n = len(indptr) - 1
    labels = np.arange(n, dtype=np.int64)
    for _ in range(sweeps):
        changed = 0
        for i in range(n):
            s, e = int(indptr[i]), int(indptr[i + 1])
            if s == e:
                continue
            uniq, inverse = np.unique(labels[cols[s:e]], return_inverse=True)
            best = int(uniq[int(np.argmax(np.bincount(inverse, weights=weights[s:e])))])
            if best != labels[i]:
                labels[i] = best
                changed += 1
        if changed == 0:
            break
    return [int(v) for v in labels]


def _match_preserved_labels(new_centroids: np.ndarray, old_rows: list) -> dict:
    """Greedy best-first match of new-cluster-index -> preserved label.

    `old_rows` are prior labeled clusters as (cluster_id, label, centroid
    blob, dim), ordered by cluster_id for determinism. A pair is eligible
    iff centroid cosine similarity > `_LABEL_MATCH_THRESHOLD`; matching is
    injective in both directions (each old cluster feeds at most one new
    cluster and vice versa), resolved highest-similarity-first with
    (new_index, old_index) as a deterministic tiebreak.
    """
    if not old_rows or new_centroids.shape[0] == 0:
        return {}
    old_dim = old_rows[0][3]
    if new_centroids.shape[1] != old_dim:
        return {}
    old_labels = [r[1] for r in old_rows]
    old_matrix = np.zeros((len(old_rows), old_dim), dtype=np.float32)
    for i, r in enumerate(old_rows):
        old_matrix[i] = np.frombuffer(r[2], dtype="<f4")

    sims = new_centroids @ old_matrix.T  # (k_new, m_old)
    candidates = [
        (float(sims[ni, oi]), ni, oi)
        for ni in range(sims.shape[0])
        for oi in range(sims.shape[1])
        if sims[ni, oi] > _LABEL_MATCH_THRESHOLD
    ]
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    assigned: dict = {}
    used_old: set = set()
    used_new: set = set()
    for _sim, ni, oi in candidates:
        if ni in used_new or oi in used_old:
            continue
        assigned[ni] = old_labels[oi]
        used_new.add(ni)
        used_old.add(oi)
    return assigned


def _single_dim(rows, model_version: str) -> int:
    """The one embedding width every row shares.

    Rows of two widths cannot go in one matrix, and the mixture means two
    embedder versions wrote under the same name.
    """
    dims = {r[3] for r in rows}
    if len(dims) > 1:
        raise ValueError(f"inconsistent embedding dims for model_version={model_version!r}: {dims}")
    return next(iter(dims))


def cluster_faces(
    conn: sqlite3.Connection,
    model_id: str,
    model_version: str,
    threshold: float,
    min_cluster_size: int = 2,
    params_note: str | None = None,
) -> list:
    """Recompute `ai_face_clusters` for one (model_id, model_version).

    Loads every `ai_face_instances` row for that model/version with a
    non-null embedding, builds a cosine-similarity graph (edge iff cosine
    >= `threshold`; backend per AI_DAM_FACE_GRAPH_BACKEND, recorded in
    cluster params as `graph_backend`), and groups it with deterministic
    chinese-whispers label propagation. See docs/FACE_CLUSTERING.md for
    backends, thresholds, and measured behavior.
    Groups with >= `min_cluster_size` members
    become cluster rows (centroid = L2-normalized mean of member
    embeddings); every other instance's `cluster_id` is left/set NULL. A
    face is clustered independently per instance, so a file with two faces
    in two different groups correctly ends up represented in both clusters.

    Re-running is idempotent: prior clusters for this model/version are
    replaced, but a new cluster whose centroid is > 0.9 cosine-similar to a
    previously *labeled* cluster inherits that label (greedy 1:1 match), so
    human-assigned nicknames survive re-clustering.

    Clusters are recurring GENERATED identities for browsing only -- see
    the module docstring. This is not real-world identity recognition.
    """
    rows = conn.execute(
        """
        SELECT face_id, file_id, embedding, dim
        FROM ai_face_instances
        WHERE model_id = ? AND model_version = ? AND embedding IS NOT NULL
        ORDER BY face_id
        """,
        (model_id, model_version),
    ).fetchall()

    try:
        conn.execute(
            "UPDATE ai_face_instances SET cluster_id = NULL WHERE model_id = ? AND model_version = ?",
            (model_id, model_version),
        )
        old_rows = conn.execute(
            """
            SELECT cluster_id, label, centroid, dim FROM ai_face_clusters
            WHERE model_id = ? AND model_version = ? AND label IS NOT NULL
            ORDER BY cluster_id
            """,
            (model_id, model_version),
        ).fetchall()
        conn.execute(
            "DELETE FROM ai_face_clusters WHERE model_id = ? AND model_version = ?",
            (model_id, model_version),
        )

        if not rows:
            conn.commit()
            return []

        dim = _single_dim(rows, model_version)
        face_ids = [r[0] for r in rows]
        matrix = np.zeros((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            matrix[i] = np.frombuffer(r[2], dtype="<f4")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normed = (matrix / norms).astype(np.float32)

        graph, graph_backend = _neighbor_graph(normed, threshold)
        labels = _chinese_whispers(graph)
        components: dict = {}
        for idx, label in enumerate(labels):
            components.setdefault(label, []).append(idx)

        cluster_components = [members for members in components.values() if len(members) >= min_cluster_size]
        cluster_components.sort(key=min)

        centroids = np.zeros((len(cluster_components), dim), dtype=np.float32)
        for ci, members in enumerate(cluster_components):
            mean_vec = normed[members].mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            centroids[ci] = mean_vec / norm if norm > 0.0 else mean_vec

        label_by_new_index = _match_preserved_labels(centroids, old_rows)

        params = {
            "threshold": threshold,
            "algo": "cosine-chinese-whispers",
            "graph_backend": graph_backend,
            "min_cluster_size": min_cluster_size,
        }
        if params_note:
            params["note"] = params_note
        params_json = json.dumps(params, sort_keys=True)

        now = time.time()
        new_cluster_ids = []
        cur = conn.cursor()
        for ci, members in enumerate(cluster_components):
            cur.execute(
                """
                INSERT INTO ai_face_clusters
                    (label, centroid, dim, size, params, model_id, model_version, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label_by_new_index.get(ci),
                    centroids[ci].tobytes(),
                    dim,
                    len(members),
                    params_json,
                    model_id,
                    model_version,
                    now,
                ),
            )
            cluster_id = cur.lastrowid
            new_cluster_ids.append(cluster_id)
            cur.executemany(
                "UPDATE ai_face_instances SET cluster_id = ? WHERE face_id = ?",
                [(cluster_id, face_ids[m]) for m in members],
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    else:
        return new_cluster_ids
