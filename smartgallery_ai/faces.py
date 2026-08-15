"""Face detection, embedding, and generated-identity clustering (WI-31).

IMPORTANT: `ai_face_clusters` groups faces that recur across a *generated*
image collection because their embeddings are close in a similarity space
learned by a face-recognition network. This is clustering for browsing
convenience only -- it is NOT real-world identity recognition, verification,
or attribution, and a cluster's `label` is a free-text nickname a human
attaches to a bucket of similar-looking generated faces, never a claim about
who (if anyone real) a face resembles.

`StubFaceBackend` is a TEST/DEV stub: it returns pre-programmed detections
and does not look at pixels at all. `OpenCVFaceBackend` is the only real
backend wired up here, built on OpenCV's bundled YuNet detector and SFace
recognizer (both ONNX, loaded only from local files under `models_dir`,
never downloaded). It self-reports `BackendUnavailable` instead of raising
when the runtime or weights are missing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai.embedders import BackendUnavailable

__all__ = [
    "BackendUnavailable",
    "FaceDetection",
    "FaceBackend",
    "StubFaceBackend",
    "OpenCVFaceBackend",
    "get_face_backend",
    "replace_faces_for_file",
    "cluster_faces",
]

_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"  # detector ONNX, expected directly under models_dir
_SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"  # recognizer ONNX, expected directly under models_dir
_SIM_CHUNK_SIZE = 256  # rows per similarity block; caps clustering memory at O(chunk * n)
_LABEL_MATCH_THRESHOLD = 0.9  # min centroid cosine for a recomputed cluster to inherit an old label


@dataclass
class FaceDetection:
    """One detected face. Coordinates are normalized [0, 1] per schema.py."""

    bbox: tuple  # (x, y, w, h), normalized
    landmarks: list  # list[(x, y)], normalized, may be empty
    det_score: float  # detector confidence; higher is more face-like
    embedding: Optional[np.ndarray]  # float32 1-D, or None
    dim: Optional[int] = None  # embedding length; derived from `embedding` when one is present

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

    def __init__(self, source: Union[Callable[[Image.Image], list], Mapping]):
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
    import hashlib

    digest = hashlib.sha256(img.tobytes()).hexdigest()
    return f"{img.mode}:{img.size[0]}x{img.size[1]}:{digest}"


def _clamp01(value: float) -> float:
    """Clamp to [0, 1], the normalized-coordinate range stored in the DB."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert to the BGR uint8 array layout OpenCV expects."""
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class OpenCVFaceBackend(FaceBackend):
    """YuNet detector + SFace recognizer, both from OpenCV's `objdetect`
    module, loaded only from local ONNX files under `models_dir`.

    Raises `BackendUnavailable` (never crashes) when the cv2 build lacks
    these APIs or the model files are not present.
    """

    model_id = "opencv/yunet+sface"
    model_version = "yunet-2023mar+sface-2021dec-v1"

    def __init__(self, models_dir: str, min_det_score: float = 0.5):
        """Load both ONNX models; `min_det_score` is the minimum detector
        confidence for a face to be reported at all."""
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise BackendUnavailable(
                "this OpenCV build lacks FaceDetectorYN/FaceRecognizerSF"
            )
        detector_path = os.path.join(models_dir, _YUNET_FILENAME)
        recognizer_path = os.path.join(models_dir, _SFACE_FILENAME)
        if not os.path.isfile(detector_path):
            raise BackendUnavailable(f"YuNet model not found at {detector_path}")
        if not os.path.isfile(recognizer_path):
            raise BackendUnavailable(f"SFace model not found at {recognizer_path}")
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
                prev_level = None
        try:
            self._detector = cv2.FaceDetectorYN.create(
                detector_path, "", (320, 320), score_threshold=min_det_score
            )
            self._recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        except Exception as exc:
            raise BackendUnavailable(f"failed to load face models: {exc}") from exc
        finally:
            if prev_level is not None:
                cv2_log.setLogLevel(prev_level)
        self._min_det_score = min_det_score

    def detect(self, img: Image.Image) -> list:
        """Detect faces, embed each via SFace on the aligned crop, and return
        `FaceDetection`s with coordinates normalized to [0, 1]."""
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
            landmarks_px = row[4:14].reshape(5, 2)
            aligned = self._recognizer.alignCrop(bgr, row)
            feature = self._recognizer.feature(aligned)
            embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
            bbox = (
                _clamp01(x / w),
                _clamp01(y / h),
                _clamp01(bw / w),
                _clamp01(bh / h),
            )
            landmarks = [
                (_clamp01(px / w), _clamp01(py / h)) for px, py in landmarks_px
            ]
            detections.append(
                FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    det_score=score,
                    embedding=embedding,
                )
            )
        return detections


def get_face_backend(config: AIConfig) -> Optional[FaceBackend]:
    """Resolve `config.face_backend` to a backend instance, or None.

    'none' -> None. 'stub' -> StubFaceBackend, sourced from
    `config.extra["face_stub_source"]` (defaults to always-empty, since a
    real per-test source can't be expressed through AIConfig alone).
    'opencv' -> OpenCVFaceBackend, raising if unavailable (explicit ask).
    'auto' -> OpenCVFaceBackend if available, else None (never the stub).
    """
    name = config.face_backend
    if name == "none":
        return None
    if name == "stub":
        source = config.extra.get("face_stub_source", lambda _img: [])
        return StubFaceBackend(source)
    if name == "opencv":
        return OpenCVFaceBackend(config.models_dir, config.face_min_det_score)
    if name == "auto":
        try:
            return OpenCVFaceBackend(config.models_dir, config.face_min_det_score)
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
    """Transactionally replace all `ai_face_instances` rows for `file_id`.

    Deletes every existing row for this file (regardless of the model that
    produced it) then inserts one row per detection, so a multi-face asset
    yields multiple rows and re-running with a different detection count
    leaves exactly that many rows behind. Returns the new `face_id`s in
    insertion order.
    """
    try:
        conn.execute("DELETE FROM ai_face_instances WHERE file_id = ?", (file_id,))
        cur = conn.cursor()
        face_ids = []
        for det in detections:
            embedding_blob = None
            dim = det.dim
            if det.embedding is not None:
                embedding_blob = det.embedding.tobytes()
                dim = int(det.embedding.shape[0])
            landmarks_json = json.dumps([[x, y] for x, y in det.landmarks])
            cur.execute(
                """
                INSERT INTO ai_face_instances
                    (file_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, det_score,
                     embedding, dim, model_id, model_version, source_mtime, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    model_id,
                    model_version,
                    source_mtime,
                    now,
                ),
            )
            face_ids.append(cur.lastrowid)
        conn.commit()
        return face_ids
    except Exception:
        conn.rollback()
        raise


def _connected_components(normed: np.ndarray, threshold: float) -> list:
    """Union-find over a cosine-similarity graph (edge iff sim >= threshold).

    `normed` rows must already be unit vectors. Similarities are computed in
    row chunks so memory stays O(chunk_size * n) rather than O(n^2). Returns
    a root-index per row (deterministic: a component's root is always its
    lowest original row index, since union always attaches the higher root
    under the lower one).
    """
    n = normed.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        """Root of `x`, with path halving."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        """Merge components; the lower root survives, keeping roots deterministic."""
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for start in range(0, n, _SIM_CHUNK_SIZE):
        end = min(start + _SIM_CHUNK_SIZE, n)
        block = normed[start:end]  # (b, dim)
        sims = block @ normed[start:].T  # (b, n - start)
        for bi in range(end - start):
            i = start + bi
            hits = np.nonzero(sims[bi, bi + 1 :] >= threshold)[0]
            for h in hits:
                union(i, start + bi + 1 + int(h))

    return [find(i) for i in range(n)]


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


def cluster_faces(
    conn: sqlite3.Connection,
    model_id: str,
    model_version: str,
    threshold: float,
    min_cluster_size: int = 2,
    params_note: Optional[str] = None,
) -> list:
    """Recompute `ai_face_clusters` for one (model_id, model_version).

    Loads every `ai_face_instances` row for that model/version with a
    non-null embedding, builds a cosine-similarity graph (edge iff cosine
    >= `threshold`), and takes its connected components via union-find.
    Components with >= `min_cluster_size` members become cluster rows
    (centroid = L2-normalized mean of member embeddings); every other
    instance's `cluster_id` is left/set NULL. A face is clustered
    independently per instance, so a file with two faces in two different
    groups correctly ends up represented in both clusters.

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
            "UPDATE ai_face_instances SET cluster_id = NULL "
            "WHERE model_id = ? AND model_version = ?",
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

        dims = {r[3] for r in rows}
        if len(dims) > 1:
            raise ValueError(
                f"inconsistent embedding dims for model_version={model_version!r}: {dims}"
            )
        dim = next(iter(dims))
        face_ids = [r[0] for r in rows]
        matrix = np.zeros((len(rows), dim), dtype=np.float32)
        for i, r in enumerate(rows):
            matrix[i] = np.frombuffer(r[2], dtype="<f4")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normed = (matrix / norms).astype(np.float32)

        roots = _connected_components(normed, threshold)
        components: dict = {}
        for idx, root in enumerate(roots):
            components.setdefault(root, []).append(idx)

        cluster_components = [
            members for members in components.values() if len(members) >= min_cluster_size
        ]
        cluster_components.sort(key=lambda members: min(members))

        centroids = np.zeros((len(cluster_components), dim), dtype=np.float32)
        for ci, members in enumerate(cluster_components):
            mean_vec = normed[members].mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            centroids[ci] = mean_vec / norm if norm > 0.0 else mean_vec

        label_by_new_index = _match_preserved_labels(centroids, old_rows)

        params = {
            "threshold": threshold,
            "algo": "cosine-union-find",
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
        return new_cluster_ids
    except Exception:
        conn.rollback()
        raise
