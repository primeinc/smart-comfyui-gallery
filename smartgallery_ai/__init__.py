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


def _env_bool(name: str, default: str = "false") -> bool:
    """Truthy-string environment flag: "1"/"true"/"yes"/"on" (any case) is True."""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AIConfig:
    """Runtime configuration for the AI DAM layer.

    Constructed explicitly (tests) or from environment (app startup).
    `enabled` gates the worker and all API surfaces; individual backends can
    additionally be missing/disabled without breaking anything else.
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

    # Tunables (documented in docs/AI_MODELS.md; override via env)
    near_dup_max_distance: int = 8      # max Hamming distance on phash64
    face_cluster_threshold: float = 0.55  # cosine similarity threshold
    face_min_det_score: float = 0.5     # min detection confidence [0,1] to keep a face
    similar_default_k: int = 24         # default neighbor count for similarity queries

    extra: dict = field(default_factory=dict)  # free-form backend-specific options

    @classmethod
    def from_env(cls, base_path: str, db_path: str) -> "AIConfig":
        """Build a config from AI_DAM_* environment variables, defaulting the
        cache and models directories to hidden folders under `base_path`."""
        cache_dir = os.environ.get(
            "AI_DAM_CACHE_DIR", os.path.join(base_path, ".ai_cache")
        )
        models_dir = os.environ.get(
            "AI_DAM_MODELS_DIR", os.path.join(base_path, ".AImodels")
        )
        return cls(
            enabled=_env_bool("ENABLE_AI_DAM"),
            base_path=base_path,
            db_path=db_path,
            models_dir=models_dir,
            cache_dir=cache_dir,
            ephemeral_index=_env_bool("AI_DAM_EPHEMERAL_INDEX"),
            semantic_backend=os.environ.get("AI_DAM_SEMANTIC_BACKEND", "auto"),
            visual_backend=os.environ.get("AI_DAM_VISUAL_BACKEND", "auto"),
            face_backend=os.environ.get("AI_DAM_FACE_BACKEND", "auto"),
            critic_backend=os.environ.get("AI_DAM_CRITIC_BACKEND", "auto"),
            segmenter_backend=os.environ.get("AI_DAM_SEGMENTER_BACKEND", "auto"),
            near_dup_max_distance=int(os.environ.get("AI_DAM_NEAR_DUP_DISTANCE", "8")),
            face_cluster_threshold=float(
                os.environ.get("AI_DAM_FACE_CLUSTER_THRESHOLD", "0.55")
            ),
            similar_default_k=int(os.environ.get("AI_DAM_SIMILAR_K", "24")),
        )


# Version stamps for derived state. Bumping one of these deterministically
# invalidates the corresponding derived rows (see invalidation.py).
HASH_ALGO_VERSION = "sha256+phash64/dhash64-v1"
RUBRIC_VERSION = "review-rubric-v1"

# Canonical space names. Semantic and visual are distinct spaces by design
# (WI-31): do not merge or cross-compare their scores.
SPACE_SEMANTIC = "semantic"
SPACE_VISUAL = "visual"
SPACE_FACE = "face"
