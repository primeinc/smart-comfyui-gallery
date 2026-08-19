"""Process-wide owner of loaded-model lifetime.

Constructing a real backend loads weights: `OpenClipSemanticEmbedder` and
`Dinov2VisualEmbedder` pull a torch model into memory, `InsightFaceBackend`
builds a FaceAnalysis pack, `OpenCVFaceBackend` reads its ONNX files through
cv2.dnn, `MobileSamSegmenter` loads a torch checkpoint. Nothing frees them. So
how long an instance lives, and who may call it concurrently, are decided here
rather than at each call site.

Sharing is a property of the backend, not of the caller. A class declares
`thread_safe` when it keeps no per-call state or guards its own forwards, and
those instances are handed to any number of threads at once. Everything else
-- cv2 FaceDetectorYN, insightface FaceAnalysis, Dinov2VisualEmbedder,
MobileSAM -- is handed out under an exclusive lease, so a second caller waits
for the first instead of racing it or loading a second copy of the weights.

Why a backend could NOT load is kept too (`why_unavailable`): it is the only
place that knows, and both the per-file walkthrough and the detector
comparison report it to the operator instead of a bare "unavailable".

Stubs are never cached. `StubFaceBackend` is built from
`AIConfig.extra["face_stub_source"]`, which differs per test and does not
belong in a key; they cost nothing to construct.

The background worker keeps its own cache (`AIWorker._backend`) with
retry-on-unavailable semantics tied to its poll cycle, and lends instances out
through `AIWorker.semantic_embedder_for_search`. Both it and this module ask
`serializes_internally` the same question, so the two agree on what is safe to
share.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NamedTuple

from smartgallery_ai import AIConfig, embedders, faces, review

_logger = logging.getLogger(__name__)


class _Kind(NamedTuple):
    """How one backend kind is resolved, keyed, and bypassed."""

    resolve: Any  # (AIConfig) -> backend | None
    identity: tuple  # AIConfig fields that select a distinct instance
    selector: str | None  # the field whose "stub" value bypasses the cache


# The config-selected kinds, plus the two explicit face lanes the detector
# comparison needs. `compare_detectors` deliberately bypasses the `auto`
# selector so its report always shows every lane, whichever one production
# uses -- but it still wants the loaded instance, not a fresh ONNX read.
_KINDS = {
    "semantic": _Kind(embedders.get_semantic_backend, ("semantic_backend",), "semantic_backend"),
    "visual": _Kind(embedders.get_visual_backend, ("visual_backend",), "visual_backend"),
    "segmenter": _Kind(review.get_segmenter_backend, ("segmenter_backend",), "segmenter_backend"),
    "faces": _Kind(
        faces.get_face_backend,
        ("face_backend", "face_embedder", "face_min_det_score", "face_min_px", "face_detect_max_side"),
        "face_backend",
    ),
    "faces_opencv": _Kind(
        lambda config: faces.OpenCVFaceBackend(
            config.models_dir,
            config.face_min_det_score,
            config.face_min_px,
            config.face_detect_max_side,
            config.face_embedder,
        ),
        ("face_embedder", "face_min_det_score", "face_min_px", "face_detect_max_side"),
        None,
    ),
    "faces_insightface": _Kind(
        lambda config: faces.InsightFaceBackend(config.models_dir, config.face_min_det_score, config.face_min_px),
        ("face_min_det_score", "face_min_px"),
        None,
    ),
}

KINDS = tuple(_KINDS)


class _Entry(NamedTuple):
    backend: Any
    use_lock: threading.Lock
    reason: str | None  # why `backend` is None, verbatim from the resolver


_REGISTRY_LOCK = threading.Lock()
_INSTANCES: dict = {}  # identity -> _Entry
_LOADING: dict = {}  # identity -> lock held while one thread resolves


def serializes_internally(backend) -> bool:
    """True when any number of threads may call this instance concurrently.

    The backend class declares it (`thread_safe`), so the answer travels with
    the implementation instead of being re-derived per call site: the lending
    rule in AIWorker and the lease rule here both ask this, and cannot
    disagree. An object that declares nothing is treated as unsafe.
    """
    return bool(getattr(backend, "thread_safe", False))


def _identity(kind: str, config: AIConfig) -> tuple:
    spec = _KINDS[kind]
    return (kind, config.models_dir, *(getattr(config, name) for name in spec.identity))


def _resolve(kind: str, config: AIConfig) -> tuple:
    """(instance, reason). Unavailability is not an error here -- the caller
    degrades to "no result" -- but the reason is kept so the operator can be
    told which file was missing rather than just "unavailable"."""
    try:
        backend = _KINDS[kind].resolve(config)
    except embedders.BackendUnavailable as exc:
        return None, str(exc)
    except Exception as exc:  # a broken backend must not fail the request
        _logger.debug("handled a failure in _resolve", exc_info=True)
        _logger.warning("[AIBackends] backend %s unavailable: %s", kind, exc)
        return None, str(exc)
    if backend is None:
        return None, f"{kind} backend is disabled or its weights are not provisioned"
    return backend, None


def _acquire(kind: str, config: AIConfig) -> _Entry:
    """The cached entry for this kind and config, resolving it if needed."""
    spec = _KINDS[kind]
    if spec.selector is not None and getattr(config, spec.selector) == "stub":
        backend, reason = _resolve(kind, config)
        return _Entry(backend, threading.Lock(), reason)

    key = _identity(kind, config)
    with _REGISTRY_LOCK:
        entry = _INSTANCES.get(key)
        if entry is not None:
            return entry
        loading = _LOADING.setdefault(key, threading.Lock())

    # Resolution stays outside _REGISTRY_LOCK -- it can take minutes -- but
    # under a per-key lock, so a burst of requests loads the weights once and
    # the rest wait rather than each starting their own copy.
    with loading:
        with _REGISTRY_LOCK:
            entry = _INSTANCES.get(key)
            if entry is not None:
                return entry
        backend, reason = _resolve(kind, config)
        entry = _Entry(backend, threading.Lock(), reason)
        with _REGISTRY_LOCK:
            _INSTANCES[key] = entry
            return entry


@contextmanager
def lease(kind: str, config: AIConfig) -> Generator[Any, None, None]:
    """Yield the backend for `kind`, or None when it cannot load.

    Held exclusively for the duration of the block unless the instance
    declares `thread_safe`, in which case the block runs concurrently with any
    other caller. Use this whenever the backend is about to be called;
    `shared` is for handing an instance to code that outlives the block.
    """
    entry = _acquire(kind, config)
    if not serializes_internally(entry.backend):
        with entry.use_lock:
            yield entry.backend
        return
    yield entry.backend


def shared(kind: str, config: AIConfig) -> Any:
    """The process-wide instance when any thread may call it, else None.

    Callers that hold the result past a `with` block get one only when it is
    safe unguarded; a stateful backend answers None rather than escaping
    without its lock.
    """
    entry = _acquire(kind, config)
    return entry.backend if serializes_internally(entry.backend) else None


def why_unavailable(kind: str, config: AIConfig) -> str | None:
    """Why `kind` could not load, or None when it did.

    Resolves if it has not been tried yet, so a caller can ask without first
    taking a lease. The string is the backend's own message -- which weights
    file was missing, which runtime failed to import.
    """
    return _acquire(kind, config).reason


def availability(config: AIConfig) -> dict:
    """`{kind: {"available": bool, "reason": str | None}}` for every kind.

    Resolves each one, so this loads whatever is provisioned. The status probe
    deliberately does NOT use it (service.py builds a probe that must never
    construct backends); the per-file walkthrough does, because by then the
    operator has asked a question only a real load can answer.
    """
    report = {}
    for kind in _KINDS:
        entry = _acquire(kind, config)
        report[kind] = {"available": entry.backend is not None, "reason": entry.reason}
    return report


def forget_unavailable() -> None:
    """Drop the entries that resolved to None so the next caller re-resolves.

    Provisioning lands weights while the process runs; a cached None would
    otherwise report the backend missing until restart. Loaded instances are
    kept -- they are still valid, and reloading them costs the weights again.
    """
    with _REGISTRY_LOCK:
        for key in [key for key, entry in _INSTANCES.items() if entry.backend is None]:
            _INSTANCES.pop(key, None)
            _LOADING.pop(key, None)


def reset() -> None:
    """Forget every instance. For tests: the registry is process-global and
    tests build configs pointing at throwaway directories."""
    with _REGISTRY_LOCK:
        _INSTANCES.clear()
        _LOADING.clear()
