"""Face detection, embedding, and generated-identity clustering (WI-31).

IMPORTANT: `ai_face_clusters` groups faces that recur across a *generated*
image collection because their embeddings are close in a similarity space
learned by a face-recognition network. This is clustering for browsing
convenience only -- it is NOT real-world identity recognition, verification,
or attribution, and a cluster's `label` is a free-text nickname a human
attaches to a bucket of similar-looking generated faces, never a claim about
who (if anyone real) a face resembles.

`StubFaceBackend` is a TEST/DEV stub: it returns pre-programmed detections
and does not look at pixels at all. Two real pipelines ship, chosen by
`backend_for` from the `face_backend` setting: `InsightFaceBackend`
(upstream FaceAnalysis over the antelopev2 pack -- SCRFD detection,
glintr100 embedding, genderage attributes; what `auto` means) and
`OpenCVFaceBackend` (YuNet detection plus ArcFace-glintr100-via-cv2.dnn
or SFace embedding; `auto`'s fallback when the insightface runtime is
absent). Weights resolve through vision/weights.py: the run's models_dir,
then the machine's shared cache, then -- for a job, never a request --
the registry that owns them. Backends self-report `BackendUnavailable`
instead of raising when runtime or weights are missing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import override

import cv2
import numpy as np
from PIL import Image

from vision import weights as weights_module

_logger = logging.getLogger(__name__)


class BackendUnavailable(LookupError):
    """A real backend's runtime or weights are not present locally. A
    LookupError, so the job runner records it on the item by name
    (db/runner.py ITEM_FAILURES) instead of dying and retrying forever."""


# insightface 1.0.1 aligns faces through skimage's pre-2.2 estimate() API, and
# skimage 0.26 deprecates it with a FutureWarning that fires on every alignment.
# Exactly that warning is silenced at its source module; every other stays.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"insightface\.utils\.face_align", message=r".*`estimate` is deprecated.*"
)

__all__ = [
    "BackendUnavailable",
    "FaceBackend",
    "FaceDetection",
    "InsightFaceBackend",
    "OpenCVFaceBackend",
    "StubFaceBackend",
    "backend_for",
    "image_key",
]

#: The `face_backend` setting's vocabulary (db/settings.py).
BACKEND_CHOICES = ("auto", "insightface", "opencv")

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
            self.dim = arr.shape[0]


#: The shared operating point both backends construct with -- one edit moves it.
#: `DEFAULT_MIN_DET_SCORE` is the detector-confidence floor; `DEFAULT_MIN_FACE_PX`
#: is the noise floor, below which boxes chain unrelated clusters together.
DEFAULT_MIN_DET_SCORE = 0.5
DEFAULT_MIN_FACE_PX = 24


class FaceBackend(ABC):
    """Face detector + per-face embedder over a single image."""

    model_id: str  # provenance recorded on every derived_face_instance row
    model_version: str  # scopes stored instances and clusters; versions never mix
    default_cluster_threshold: float = 0.55  # per-embedder operating point
    # No face backend declares itself callable from two threads at once:
    # cv2.FaceDetectorYN and insightface's FaceAnalysis both carry per-call state
    # inside the native object, so a shared instance is held exclusively.
    thread_safe: bool = False

    @abstractmethod
    def detect(self, img: Image.Image) -> list:
        """Detect faces in `img`. Returns a list of `FaceDetection`."""


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

    @override
    def detect(self, img: Image.Image) -> list:
        """Replay the pre-programmed detections for `img`; unknown images
        detect as no faces."""
        # Narrowed on Mapping rather than on `callable`: asking whether something
        # is callable narrows to "some callable" and loses the signature, so the
        # call below could not be checked. A Mapping has a runtime-checkable ABC.
        if isinstance(self._source, Mapping):
            return list(self._source.get(image_key(img), []))
        return list(self._source(img))


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
    """YuNet detector + a per-face recognizer, all through OpenCV, over
    the files vision/weights.py resolves (models_dir, the shared Hub
    cache, or -- provisioning -- the opencv org's own Hub repos).

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
        min_det_score: float = DEFAULT_MIN_DET_SCORE,
        min_face_px: int = DEFAULT_MIN_FACE_PX,
        detect_max_side: int = 1600,
        embedder: str = "auto",
        *,
        provision: bool = False,
    ):
        """Load the detector and the selected recognizer. `min_det_score`
        is the minimum detector confidence for a face to be reported;
        `min_face_px` is the minimum face box side in detect-input pixels —
        YuNet detects down to ~10px, and detections near that floor are
        featureless, embed into one generic region, and chain unrelated
        clusters together. `detect_max_side` caps the detection input:
        images larger than N px on their longest side are downscaled first,
        keeping large faces inside YuNet's ~10-300px training band. 0
        disables the cap. A forced `embedder` whose weights are missing
        raises instead of silently falling back. `provision` lets a missing
        file be fetched from its registry -- a job's right, never a
        request's.

        The cap's effect, read off the two recorded runs over the same 824
        ground-truth faces -- benchmarks/results/face_detection_recall_native.json
        (policy_max_side 0) against face_detection_recall_ms1600.json
        (policy_max_side 1600):

            >=300px band recall     0.5534  ->  0.9709
            false positives            336  ->  49
            ms per image, detect      61.5  ->  32.9

        Those files are the evidence; the numbers above are read from them
        rather than restated from memory."""
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise BackendUnavailable("this OpenCV build lacks FaceDetectorYN/FaceRecognizerSF")
        try:
            held = weights_module.opencv_weights(models_dir, provision=provision)
        except weights_module.Unprovisioned as exc:
            raise BackendUnavailable(str(exc)) from exc
        except Exception as exc:  # the registry refused or the network did
            raise BackendUnavailable(f"fetching the OpenCV face weights failed: {exc}") from exc
        detector_path = held.yunet
        if embedder == "auto":
            embedder = "arcface" if held.arcface is not None else "sface"
        # the one recognizer this backend will load, proven present here
        arcface_path: str | None = None
        recognizer_path: str | None = None
        if embedder == "arcface":
            if held.arcface is None:
                raise BackendUnavailable(f"ArcFace model (the {weights_module.PACK} pack) is not under {models_dir}")
            arcface_path = held.arcface
        elif embedder == "sface":
            if held.sface is None:
                raise BackendUnavailable(f"SFace model is not under {models_dir} or the shared HF cache")
            recognizer_path = held.sface
        else:
            raise ValueError(f"unknown face embedder: {embedder!r}")
        self._embedder = embedder
        self.model_id = f"opencv/yunet+{embedder}"
        # The detection cap is part of the version because it changes the
        # vectors, and invalidation.is_stale compares version strings exactly.
        # A cap left out of the string lets one cosine graph span two regimes.
        base = "yunet-2023mar+arcface-glintr100" if embedder == "arcface" else "yunet-2023mar+sface-2021dec-v2"
        self.model_version = f"{base}-ms{detect_max_side}"
        # 0.48 is what `just bench faces-validate` selects on its own, recording
        # `chosen` and `labels_best` as chinese-whispers at 0.48
        # (benchmarks/results/face_pipeline_validation.json).

        # The band there is wide: 0.40, 0.48 and 0.55 all reach pair_f1 1.0 and
        # 0.60 breaks a cluster, so 0.48 sits inside it rather than on an edge.
        # sface takes 0.55, a different embedder whose sweep is not that file.
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
            if arcface_path is not None:
                self._recognizer = None
                self._arcface = cv2.dnn.readNetFromONNX(arcface_path)
            elif recognizer_path is not None:
                self._recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
                self._arcface = None
        except Exception as exc:
            raise BackendUnavailable(f"failed to load face models: {exc}") from exc
        finally:
            if prev_level is not None and cv2_log is not None:
                cv2_log.setLogLevel(prev_level)
        self._min_det_score = min_det_score
        self._min_face_px = min_face_px
        self._detect_max_side = detect_max_side

    @override
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
                # ArcFace contract (insightface arcface_onnx.py get_feat): canonical
                # 112x112 norm_crop, then blob with mean/std 127.5 and BGR->RGB
                # swap; 512-d output, cosine-ready after normalization downstream.
                aligned = _arcface_norm_crop(bgr, landmarks_px)
                blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                self._arcface.setInput(blob)
                feature = self._arcface.forward()
            else:
                recognizer = self._recognizer
                if recognizer is None:
                    raise BackendUnavailable("constructor loaded neither arcface nor sface")
                aligned = recognizer.alignCrop(bgr, row)
                feature = recognizer.feature(aligned)
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


#: insightface's genderage head answers a letter (Face.sex: 'M' / 'F',
#: insightface/app/common.py); the schema stores the word
#: (db/schema.sql derived_face_instance.sex CHECK).
_SEX_WORDS = {"M": "male", "F": "female"}


def sex_word(code) -> str:
    return _SEX_WORDS.get(code, "unknown")


#: Detection sizes, descending, first hit wins -- the rule every declared
#: consumer of a stored face follows, and the sizes their own ladders sweep.
#: `first_hit_descending` says why 640 alone is not enough.
DET_SIZES: tuple[int, ...] = tuple(range(640, 128, -64))

#: The OTHER recovery a consumer may use when the first size finds nothing:
#: pad the frame and retry at the same size. Stored beside the descending
#: hit, because the two land 8.86 px apart and one record cannot be both.
PAD_RECOVERY_SCALE: float = 1.25


def first_hit_descending(app, bgr):
    """Detect down DET_SIZES, returning the first size that finds anything.

    640 alone is not enough: IP-Adapter's own `ai_face.png` holds a face
    this detector finds at 448 and not at 640, and every consumer that
    misses at 640 descends the same way (`range(640, 256, -64)` for
    IP-Adapter, InstantID and PuLID; 640/320/160 for InfiniteYou).

    The detector is re-prepared per size rather than merged across sizes.
    insightface 1.0.1's auto det-size runs 128 and 640 and NMS-merges them,
    which moved keypoints up to 8.06 px away from the 640-only ones on the
    same weights -- enough, through `norm_crop`'s similarity warp, to change
    almost every pixel of the aligned crop and the embedding taken from it.
    """
    detector = getattr(app, "det_model", None)
    if detector is None:
        # A replay double supplies `get` and nothing else, returning a fixed
        # detection. There is no size to sweep, and demanding a detector here
        # would make the storage lane depend on one it deliberately has not got.
        return app.get(bgr)

    for size in DET_SIZES:
        # `SCRFD._resolve_input_sizes` reads `input_sizes`, the list `prepare`
        # fills; assigning the singular `input_size` moves nothing and every
        # size would silently run the same detection.
        detector.prepare(-1, input_size=(size, size))
        if tuple(detector.input_sizes) != ((size, size),):
            raise BackendUnavailable(f"asked the detector for {size} and it holds {detector.input_sizes}")
        found = app.get(bgr)
        if found:
            return found
    return []


def padded_recovery(app, bgr):
    """The face found by padding and retrying at the first size, or None.

    UniPortrait recovers from an empty detection this way while IP-Adapter,
    InstantID and PuLID descend the det-size, and on the same photograph the
    two rules land 8.86 px apart -- far enough through `norm_crop` to change
    every pixel of the aligned crop. Neither is wrong, so the record keeps
    both and a consumer is served the one its own code would have computed.

    None when the first size already found a face: no recovery ran, so there
    is no second detection to store.
    """
    import cv2

    detector = getattr(app, "det_model", None)
    if detector is None:
        return None
    size = DET_SIZES[0]
    detector.prepare(-1, input_size=(size, size))
    if app.get(bgr):
        return None

    pad = PAD_RECOVERY_SCALE - 1.0
    height, width = bgr.shape[:2]
    top, left = int(height * pad), int(width * pad)
    padded = cv2.copyMakeBorder(bgr, top, top, left, left, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    detector.prepare(-1, input_size=(size, size))
    found = app.get(padded)
    if not found:
        return None

    # Back into the original frame's coordinates, the way upstream moves them.
    best = max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
    offset = np.array([left, top], dtype=np.float32)
    best.kps = np.asarray(best.kps, dtype=np.float32) - offset
    best.bbox = np.asarray(best.bbox, dtype=np.float32) - np.array([left, top, left, top], dtype=np.float32)
    return best


class InsightFaceBackend(FaceBackend):
    """insightface's own pipeline (FaceAnalysis over the provisioned
    antelopev2 pack): SCRFD-10GF detection at 640x640, upstream
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

    def __init__(
        self,
        models_dir: str,
        min_det_score: float = DEFAULT_MIN_DET_SCORE,
        min_face_px: int = DEFAULT_MIN_FACE_PX,
        providers: str = "auto",
        *,
        provision: bool = False,
    ):
        """`min_det_score` re-filters detections (FaceAnalysis is prepared
        at the same threshold); `min_face_px` drops noise-floor boxes by
        native-pixel side, same junk gate as the OpenCV backend.
        `providers` is the `ort_providers` spec for the recognition
        session (see `_ort_providers`); `provision` lets a missing pack
        be fetched by insightface itself -- a job's right, never a
        request's."""
        self._app = get_insightface_app(models_dir, providers=providers, provision=provision)
        self._min_det_score = min_det_score
        self._min_face_px = min_face_px

    @override
    def detect(self, img: Image.Image) -> list:
        bgr = _pil_to_bgr(img)
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return []
        detections = []
        for face in first_hit_descending(self._app, bgr):
            # insightface leaves both optional on its Face record. A detection
            # without a box or a confidence cannot be placed or ranked, so it is
            # dropped here rather than raising four frames later.
            if face.det_score is None or face.bbox is None:
                continue
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
                attributes["sex"] = sex_word(face.sex)
            # Not rounded: 5 decimals on a normalized coordinate discards up to
            # 0.033 source pixels on a 6528 px photograph, measured at up to
            # 0.0322 px and 0.0034 degrees (compat/consumers/gallery_storage.py).

            # A normalized float64 round trip is exact for a float32 source, so
            # the full value costs some JSON width and conserves the producer's
            # measurement outright.
            lmk106 = face.get("landmark_2d_106")
            if lmk106 is not None:
                attributes["landmark_2d_106"] = [
                    [_clamp01(float(px) / w), _clamp01(float(py) / h)] for px, py in lmk106
                ]
            lmk68 = face.get("landmark_3d_68")
            if lmk68 is not None:
                # x/y normalized like every other coordinate; z stays in
                # the model's pixel-scaled depth units (no image norm
                # exists for depth) — recorded as-is.
                attributes["landmark_3d_68"] = [
                    [_clamp01(float(px) / w), _clamp01(float(py) / h), float(pz)] for px, py, pz in lmk68
                ]
            pose = face.get("pose")
            if pose is not None:
                attributes["pose"] = {
                    "pitch": float(pose[0]),
                    "yaw": float(pose[1]),
                    "roll": float(pose[2]),
                }
            detections.append(
                FaceDetection(
                    bbox=bbox, landmarks=landmarks, det_score=score, embedding=embedding, attributes=attributes or None
                )
            )
        return detections


def _ort_providers(spec: str = "auto") -> list:
    """Execution providers for the insightface ORT sessions, in ORT's
    priority-list form (docs/python/api_summary.rst: kernels are chosen
    in the order given; anything a provider lacks runs on CPU).
    `spec`: 'auto' (CUDA first when the installed onnxruntime build
    offers it, which means installing onnxruntime-gpu yourself; nothing
    swaps it in), 'cpu', or an explicit comma list. The application's
    choice is the `ort_providers` setting (db/settings.py), passed down."""
    value = spec.strip()
    if value.lower() == "cpu":
        return ["CPUExecutionProvider"]
    if value and value.lower() != "auto":
        return [p.strip() for p in value.split(",") if p.strip()]
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except (ImportError, OSError, RuntimeError, AttributeError) as why:
        # no onnxruntime, a DLL that will not load, a build without the call
        _logger.warning("faces run on the CPU: onnxruntime named no providers: %s: %s", type(why).__name__, why)
    return ["CPUExecutionProvider"]


_insightface_apps: dict = {}  # (root, providers) -> FaceAnalysis (cached; models stay loaded)


def _prepared_recognition(root: str, providers: list):
    """The GPU recognition model, loaded and prepared, or a named refusal."""
    from insightface.model_zoo import model_zoo

    path = os.path.join(root, "models", weights_module.PACK, "glintr100.onnx")
    rec = model_zoo.get_model(path, providers=providers)
    if rec is None:
        raise BackendUnavailable(f"model_zoo could not load {path}")
    ready = getattr(rec, "prepare", None)
    if ready is None:
        raise BackendUnavailable("loaded recognition model has no prepare()")
    ready(ctx_id=0)
    return rec


def get_insightface_app(models_dir: str, providers: str = "auto", *, provision: bool = False):
    """insightface's own pipeline (FaceAnalysis, detection + recognition)
    over the antelopev2 pack -- the run's copy, the shared ~/.insightface,
    or (provisioning) insightface's own download into the run's copy --
    cached per (root, providers). Raises `BackendUnavailable` when the
    package or the pack is missing.

    Every pack head loads: genderage (age/sex), 2d106det (dense 106-pt 2D
    landmarks), 1k3d68 (3D 68-pt + pitch/yaw/roll pose). 1k3d68 is the expensive
    one to keep first-class: 143,607,619 bytes on disk against 5,030,888 for
    2d106det and 1,322,532 for genderage (antelopev2 pack). Those are file sizes,
    not the resident size of the ORT sessions they become.

    Providers are per stage, and the split is structural: detection runs dynamic
    input shapes (SCRFD '?' dims), where the CUDA EP re-tunes conv algorithms per
    distinct shape; recognition is a ResNet100 at a fixed 112x112, which gives the
    EP one shape to tune once. So detection and genderage stay on CPU and the
    recognition session gets _ort_providers(providers) -- CUDA when the installed
    build offers it, with the ort_providers setting overriding.
    """
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise BackendUnavailable(f"insightface unavailable: {exc}") from exc
    try:
        root = weights_module.insightface_root(models_dir, provision=provision)
    except Exception as exc:  # the registry refused or the network did
        raise BackendUnavailable(f"fetching the {weights_module.PACK} pack failed: {exc}") from exc
    if root is None:
        raise BackendUnavailable(
            f"{weights_module.PACK} pack not under {models_dir}/{weights_module.INSIGHTFACE_SUBDIR}"
            f" or {weights_module.INSIGHTFACE_HOME}; run /jobs/faces once to fetch it"
        )
    cache_key = (root, providers)
    if cache_key in _insightface_apps:
        return _insightface_apps[cache_key]
    try:
        # Every pack head loads, and all of it persists per face in
        # FaceDetection.attributes -- see the docstring for the cost of
        # 1k3d68 and for why the providers are split per stage.
        app = FaceAnalysis(
            name=weights_module.PACK,
            root=root,
            allowed_modules=["detection", "recognition", "genderage", "landmark_2d_106", "landmark_3d_68"],
            providers=["CPUExecutionProvider"],
        )
        # DET_SIZES, not insightface 1.0.1's auto default: that one detects
        # at 128 as well and NMS-merges, and no consumer of a stored face
        # computes the merged keypoints. `detect` walks the ladder itself.
        app.prepare(ctx_id=0, det_size=(DET_SIZES[0], DET_SIZES[0]))
        rec_providers = _ort_providers(providers)
        if rec_providers != ["CPUExecutionProvider"]:
            app.models["recognition"] = _prepared_recognition(root, rec_providers)
    except Exception as exc:
        raise BackendUnavailable(f"FaceAnalysis failed to load: {exc}") from exc
    _insightface_apps[cache_key] = app
    return app


def backend_for(models_dir: str, *, choice: str = "auto", providers: str = "auto", provision: bool = False):
    """The backend the `face_backend` setting names. 'insightface' and
    'opencv' are exactly that; 'auto' is insightface, and the OpenCV
    stack only when the insightface RUNTIME is absent -- a pack that
    cannot be fetched is a refusal, not a silent downgrade to a different
    embedding space."""
    if choice not in BACKEND_CHOICES:
        raise ValueError(f"face_backend must be one of {', '.join(BACKEND_CHOICES)}, not {choice!r}")
    if choice == "opencv":
        return OpenCVFaceBackend(models_dir, provision=provision)
    if choice == "auto":
        try:
            import insightface.app as _runtime  # the import itself, not find_spec: a broken install is absent too
        except (ImportError, OSError) as why:  # OSError: a native dependency (onnxruntime's DLLs) failed to load
            _logger.warning(
                "face_backend=auto: insightface does not import in this interpreter (%s: %s),"
                " so the OpenCV stack runs instead -- CPU, SFace unless the antelopev2 pack is present",
                sys.executable,
                why,
            )
            return OpenCVFaceBackend(models_dir, provision=provision)
        del _runtime
    return InsightFaceBackend(models_dir, providers=providers, provision=provision)
