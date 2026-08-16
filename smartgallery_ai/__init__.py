"""SmartGallery AI DAM extension package.

Local-first derived-AI layer for SmartGallery (WI-31):
  - exact + perceptual duplicate detection (no GPU required)
  - semantic and visual embedding spaces (separate, model-versioned)
  - face instances + generated-face clustering (similarity, not identity)
  - generation review (typed findings, quality/prompt-alignment scores, masks
    only for localizable findings)
  - human-feedback capture/export

Design invariants:
  - SQLite remains the authoritative metadata store. Every table this package
    creates is DERIVED, REBUILDABLE state keyed to source file mtime and model
    versions. Deleting all ai_* tables and re-indexing must reproduce them.
  - All inference is local. This package performs no network I/O at runtime;
    model weights are provisioned separately into the models directory.
  - Source media is opened read-only and never modified.
  - Heavy model backends are optional: when a backend is unavailable the
    feature reports itself as disabled instead of raising.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_bool(name: str, default: str = "false") -> bool:
    """Truthy-string environment flag: "1"/"true"/"yes"/"on" (any case) is True."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AIConfig:
    """Runtime configuration for the AI DAM layer.

    Constructed explicitly (tests) or from environment (app startup).
    `enabled` gates the worker and all API surfaces; individual backends can
    additionally be missing/disabled without breaking anything else. The
    layer is opt-OUT: `from_env` enables it unless ENABLE_AI_DAM=false
    (this dataclass default stays False so explicit test construction
    activates nothing by accident).
    """

    enabled: bool = False
    base_path: str = ""            # BASE_SMARTGALLERY_PATH of the host app
    db_path: str = ""              # main SmartGallery sqlite file (shared DB)
    models_dir: str = ""           # where provisioned model files live
    cache_dir: str = ""            # derived caches (vector index files, masks)
    ephemeral_index: bool = False  # True: never persist vector index to disk

    # Backend selectors; "none" disables, "auto" picks the best available.
    semantic_backend: str = "auto"
    visual_backend: str = "auto"
    face_backend: str = "auto"
    critic_backend: str = "auto"
    segmenter_backend: str = "auto"

    # True: the background worker installs missing runtime packages and
    # downloads missing model weights once, asynchronously, on startup
    # (the request path never downloads; network failure degrades to
    # backends-unavailable). False: strict no-egress. Like `enabled`, the
    # dataclass default is the INERT value so explicitly constructed
    # configs (tests) never touch the network; `from_env` flips it on
    # unless AI_DAM_AUTO_PROVISION=false (opt-out).
    auto_provision: bool = False

    # Tunables (documented in docs/AI_MODELS.md; override via env)
    near_dup_max_distance: int = 8      # max Hamming distance on phash64
    face_cluster_threshold: Optional[float] = None  # cosine similarity threshold;
                                        #   None = the face backend's per-embedder
                                        #   default (faces.resolve_cluster_threshold)
    face_min_det_score: float = 0.5     # min detection confidence [0,1] to keep a face
    face_min_px: int = 24               # min face box side in detect-input pixels; smaller
                                        #   detections sit at YuNet's ~10px noise floor and
                                        #   embed as junk (docs/FACE_CLUSTERING.md)
    face_detect_max_side: int = 1600    # cap detection input; keeps large faces inside
                                        #   YuNet's ~10-300px band (0 disables)
    face_embedder: str = "auto"         # 'arcface' (buffalo_l w600k_r50, 512-d) |
                                        #   'sface' (128-d) | 'auto' (arcface when
                                        #   its weights are present)
    similar_default_k: int = 24         # default neighbor count for similarity queries

    extra: dict = field(default_factory=dict)  # free-form backend-specific options

    @classmethod
    def from_env(cls, base_path: str, db_path: str) -> "AIConfig":
        """Build a config from AI_DAM_* environment variables, defaulting the
        cache and models directories to hidden folders under `base_path`."""
        # A blank is a fallback, not a value: `set "AI_DAM_MODELS_DIR="`
        # and an empty Docker/Unraid template field both define the
        # variable as "", which os.environ.get returns AS the value and
        # would scatter the cache and multi-GB weights into the working
        # directory.
        def _dir(name: str, default: str) -> str:
            value = os.environ.get(name)
            return default if value is None or not value.strip() else value.strip()

        cache_dir = _dir("AI_DAM_CACHE_DIR", os.path.join(base_path, ".ai_cache"))
        models_dir = _dir("AI_DAM_MODELS_DIR", os.path.join(base_path, ".AImodels"))
        return cls(
            enabled=_env_bool("ENABLE_AI_DAM", "true"),
            base_path=base_path,
            db_path=db_path,
            models_dir=models_dir,
            cache_dir=cache_dir,
            ephemeral_index=_env_bool("AI_DAM_EPHEMERAL_INDEX"),
            auto_provision=_env_bool("AI_DAM_AUTO_PROVISION", "true"),
            semantic_backend=os.environ.get("AI_DAM_SEMANTIC_BACKEND", "auto"),
            visual_backend=os.environ.get("AI_DAM_VISUAL_BACKEND", "auto"),
            face_backend=os.environ.get("AI_DAM_FACE_BACKEND", "auto"),
            critic_backend=os.environ.get("AI_DAM_CRITIC_BACKEND", "auto"),
            segmenter_backend=os.environ.get("AI_DAM_SEGMENTER_BACKEND", "auto"),
            near_dup_max_distance=int(os.environ.get("AI_DAM_NEAR_DUP_DISTANCE", "8")),
            face_cluster_threshold=(
                float(os.environ["AI_DAM_FACE_CLUSTER_THRESHOLD"])
                if "AI_DAM_FACE_CLUSTER_THRESHOLD" in os.environ else None
            ),
            face_min_px=int(os.environ.get("AI_DAM_FACE_MIN_PX", "24")),
            face_detect_max_side=int(
                os.environ.get("AI_DAM_FACE_DETECT_MAX_SIDE", "1600")
            ),
            face_embedder=os.environ.get("AI_DAM_FACE_EMBEDDER", "auto"),
            similar_default_k=int(os.environ.get("AI_DAM_SIMILAR_K", "24")),
        )


# Version stamps for derived state. Bumping one of these deterministically
# invalidates the corresponding derived rows (see invalidation.py).
HASH_ALGO_VERSION = "sha256+phash64/dhash64-v1"
RUBRIC_VERSION = "review-rubric-v2"

# Canonical space names. Semantic and visual are distinct spaces by design
# (WI-31): do not merge or cross-compare their scores.
SPACE_SEMANTIC = "semantic"
SPACE_VISUAL = "visual"
SPACE_FACE = "face"
