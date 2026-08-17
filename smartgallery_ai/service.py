"""Flask blueprint for the AI DAM layer.

`create_ai_blueprint` builds a self-contained Blueprint the host app mounts
at `/galleryout/api/aidam`; every route opens its own SQLite connection to
`config.db_path` (never one shared with the background worker or another
request) and closes it before returning. When the AI layer is disabled
every route except `/status` responds immediately with
`{"enabled": False}` -- no DB access, no backend probing -- so the feature
is fully inert until an operator turns it on.

`create_ai_resolvers` builds the `omniquery` engine's `ai_resolvers` map
(`near_dup_of`, `similar_to_semantic`, `similar_to_visual`), each entry a
plain function resolving a field's raw AST value to a list of file ids.
"""

from __future__ import annotations

import itertools
import json
import os
import sqlite3
import threading
import time
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np
from flask import Blueprint, Response, abort, jsonify, request, send_file, url_for

from smartgallery_ai import (
    AIConfig,
    HASH_ALGO_VERSION,
    SPACE_SEMANTIC,
    SPACE_VISUAL,
)
from smartgallery_ai import embedders, faces, feedback, hashing, invalidation, review, runner, schema, vectors
from smartgallery_ai import provision as provisioning
from smartgallery_ai.worker import (
    _MTIME_EPSILON,
    app_git_ref,
    indexing_totals,
    load_source_image,
    mark_faces_cluster_pending,
    record_scan,
)

__all__ = ["create_ai_blueprint", "create_ai_resolvers", "set_worker", "get_worker"]

# Registered by the host app's startup code via `set_worker()` so `/status`
# can report on the running background worker without the blueprint factory
# needing a reference to it at creation time (the worker is only started
# after the blueprint has already been registered).
_worker_ref: dict = {"worker": None}


def set_worker(worker) -> None:
    """Register the running `AIWorker` (or None) so `/status` can report it."""
    _worker_ref["worker"] = worker


def get_worker():
    """Return the registered background `AIWorker`, or None when none is running."""
    return _worker_ref.get("worker")


# Every blueprint's backend-availability cache, so the worker can flush them
# after auto-provisioning lands new weights (a cached False would otherwise
# misreport the backend as unavailable until process restart).
_PROBE_CACHES: list = []


def invalidate_backend_probe_cache() -> None:
    """Clear every blueprint's cached backend-availability probe; the next
    `/status` re-probes against the current models_dir contents."""
    for cache in _PROBE_CACHES:
        cache.clear()


# File types the embedding/faces/review stages can render a frame from --
# the same membership the worker's stages use. Other types (audio, text)
# are never "pending": no stage will ever produce results for them.
_RENDERABLE_TYPES = tuple(hashing.IMAGE_FILE_TYPES | hashing.VIDEO_FILE_TYPES)


def _connect(config: AIConfig) -> sqlite3.Connection:
    """Open a fresh SQLite connection to the gallery DB with name-addressable rows."""
    conn = schema.connect(config.db_path)
    return conn


def _renderable(conn: sqlite3.Connection, file_id: str) -> bool:
    """Whether the file exists and is a type the AI stages can process --
    the panel shows 'queued for indexing' only when results can actually
    arrive, never for types no stage will touch."""
    row = conn.execute("SELECT type FROM files WHERE id = ?", (file_id,)).fetchone()
    return row is not None and row["type"] in _RENDERABLE_TYPES


def _was_scanned(conn: sqlite3.Connection, file_id: str, kind: str):
    """The file's ai_scan_log row for `kind` AT ITS CURRENT MTIME, or None.
    Row presence separates 'scanned, nothing found' from 'not reached yet';
    the mtime predicate keeps a modified file honest -- its stale scan row
    must read as pending again, because the worker WILL rescan it. (A
    model-version bump also re-queues; that rarer staleness is accepted
    here since checking it would mean constructing the backend.)"""
    return conn.execute(
        """
        SELECT sl.result_count FROM ai_scan_log sl
        JOIN files f ON f.id = sl.file_id
        WHERE sl.file_id = ? AND sl.kind = ?
          AND ABS(sl.source_mtime - f.mtime) <= ?
        """,
        (file_id, kind, _MTIME_EPSILON)).fetchone()


def _disabled_response():
    """Uniform body every route except `/status` answers with while the layer is off."""
    return jsonify({"enabled": False}), 200


def _extract_file_id(value: Any) -> Optional[str]:
    """File id from a resolver AST value -- bare string or `{"file_id": ...}` dict; None otherwise."""
    if isinstance(value, dict):
        return value.get("file_id")
    if isinstance(value, str):
        return value
    return None


def _extract_file_id_and_k(value: Any, default_k: int) -> tuple:
    """(file_id, k) from a resolver AST value; only the dict form can override `default_k`."""
    if isinstance(value, dict):
        return value.get("file_id"), int(value.get("k", default_k))
    return value, default_k


def create_ai_resolvers(config: AIConfig) -> dict:
    """Build the `omniquery` `ai_resolvers` map for `near_dup_of`,
    `similar_to_semantic`, and `similar_to_visual`. Each resolver opens its
    own connection and returns `[]` for an unknown/unembedded file."""

    def near_dup_of(value: Any) -> list:
        """Ids of files within the configured perceptual-hash distance of `value`'s file."""
        file_id = _extract_file_id(value)
        if not file_id:
            return []
        conn = _connect(config)
        try:
            pairs = hashing.find_near_duplicates(conn, file_id, config.near_dup_max_distance)
        finally:
            conn.close()
        return [fid for fid, _distance in pairs]

    def _similar(value: Any, space: str) -> list:
        """Ids of `value`'s file's k nearest neighbors in `space`, same model version only."""
        file_id, k = _extract_file_id_and_k(value, config.similar_default_k)
        if not file_id:
            return []
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT vector, model_version FROM ai_embeddings "
                "WHERE file_id = ? AND space = ?",
                (file_id, space),
            ).fetchone()
            if row is None:
                return []
            query_vec = np.frombuffer(row["vector"], dtype="<f4")
            store = vectors.VectorStore(cache_dir=config.cache_dir, ephemeral=config.ephemeral_index)
            # Pin to the query row's model version (see /similar).
            neighbors = store.topk(conn, space, query_vec, k, exclude=[file_id],
                                   model_version=row["model_version"])
        finally:
            conn.close()
        return [fid for fid, _score in neighbors]

    return {
        "near_dup_of": near_dup_of,
        "similar_to_semantic": lambda value: _similar(value, SPACE_SEMANTIC),
        "similar_to_visual": lambda value: _similar(value, SPACE_VISUAL),
    }


def _index_one_file(conn: sqlite3.Connection, config: AIConfig, file_row: sqlite3.Row,
                     force: bool) -> dict:
    """Synchronously bring one file's derived state up to date. Used by the
    `POST /index/<file_id>` endpoint -- the request thread blocks on this,
    unlike the background worker, so it is only ever one file at a time."""
    file_id, path, mtime, file_type = (
        file_row["id"], file_row["path"], file_row["mtime"], file_row["type"]
    )
    now = time.time()
    result: dict = {"file_id": file_id, "hashed": False, "embedded": [], "faces": False, "reviewed": False}

    def _backend_for(factory):
        """Fresh backend resolve for the no-worker inline path. NEVER hands
        out the worker's cached instances: the worker thread runs inference
        on those concurrently, and the detectors/predictors are stateful
        (a shared cv2 FaceDetectorYN or SamPredictor is a data race). With
        a running worker the endpoint defers to its priority queue instead
        of calling this at all."""
        return factory(config)

    existing_hash = conn.execute(
        "SELECT source_mtime, algo_version FROM ai_file_hashes WHERE file_id = ?", (file_id,)
    ).fetchone()
    needs_hash = (
        force or existing_hash is None
        or invalidation.is_stale(existing_hash["source_mtime"], mtime,
                                  existing_hash["algo_version"], HASH_ALGO_VERSION)
    )
    if needs_hash:
        hash_result = hashing.compute_hashes_for_file(path, file_type)
        hashing.upsert_hashes(conn, file_id, hash_result, mtime, HASH_ALGO_VERSION, now)
        result["hashed"] = True

    img = load_source_image(path, file_type)

    for space, get_backend in (
        (SPACE_SEMANTIC, embedders.get_semantic_backend),
        (SPACE_VISUAL, embedders.get_visual_backend),
    ):
        backend = _backend_for(get_backend)
        if backend is None or img is None:
            continue
        existing = conn.execute(
            "SELECT source_mtime, model_id, model_version FROM ai_embeddings "
            "WHERE file_id = ? AND space = ?",
            (file_id, space),
        ).fetchone()
        model_key = f"{backend.model_id}::{backend.model_version}"
        existing_key = (
            f"{existing['model_id']}::{existing['model_version']}" if existing is not None else None
        )
        needs = (
            force or existing is None
            or invalidation.is_stale(existing["source_mtime"], mtime, existing_key, model_key)
        )
        if needs:
            vec = backend.embed_image(img)
            store = vectors.VectorStore(cache_dir=config.cache_dir, ephemeral=config.ephemeral_index)
            store.add(conn, file_id, space, backend.model_id, backend.model_version, vec, mtime)
            result["embedded"].append(space)

    face_backend = _backend_for(faces.get_face_backend)
    if face_backend is not None and img is not None:
        detections = face_backend.detect(img)
        faces.replace_faces_for_file(
            conn, file_id, detections, face_backend.model_id, face_backend.model_version, mtime, now
        )
        # Mark the scan like the worker does: without this row the worker
        # re-detects this file next cycle and the panel cannot tell
        # "scanned, zero faces" apart from "not scanned yet".
        record_scan(conn, file_id, "faces", face_backend, mtime, now, len(detections))
        # And queue a recluster: these faces were stored outside the
        # worker's scan loop, which otherwise only clusters after its own
        # scans -- they would stay unclustered indefinitely.
        mark_faces_cluster_pending(conn, face_backend)
        result["faces"] = True

    # Reviews are NEVER run synchronously here: constructing the critic can
    # load a multi-GB VLM and one review takes minutes — that does not
    # belong in a Flask request thread. Worse, re-running store_review here
    # would wipe the previous findings' masks without regenerating them
    # (mask generation lives in the worker). Instead, clear this file's
    # review scan-log entry so the background worker re-reviews it (and
    # regenerates masks) on its next cycle.
    if force:
        conn.execute(
            "DELETE FROM ai_scan_log WHERE file_id = ? AND kind = 'review'",
            (file_id,))
        conn.commit()
        result["review_rescheduled"] = True

    return result


def create_ai_blueprint(config: AIConfig, guard: Optional[Callable] = None,
                        file_access_check: Optional[Callable[[str], bool]] = None,
                        generation_metadata_visible: Optional[Callable[[], bool]] = None,
                        ) -> Blueprint:
    """Build the AI DAM Flask blueprint.

    `guard`, if given, is applied to the mutating endpoints (recluster,
    feedback POST, index POST) AND to cross-file enumeration endpoints
    (cluster listings, feedback export) -- the caller passes its own auth
    decorator (e.g. `management_api_only`).

    `file_access_check(file_id) -> bool`, if given, gates every per-file
    read route (similar/duplicates/faces/review, and masks via their
    finding's file) with the host app's per-file visibility policy (e.g.
    `is_file_accessible`), so restricted-mode viewers cannot read derived
    AI metadata for files the normal gallery routes would refuse to serve.
    Inaccessible files answer 404, indistinguishable from nonexistent.

    `generation_metadata_visible() -> bool`, if given, says whether THIS
    caller may see how a picture was made. Seeing a picture and being told
    the prompt behind it are different permissions: a review's alignment
    elements are verbatim slices of the positive prompt, so a host that
    hides prompts from its visitors would otherwise publish them here, one
    element at a time, for every file those visitors are allowed to see.
    Absent, everything is visible."""
    bp = Blueprint("aidam", __name__)

    def _may_see_generation_metadata() -> bool:
        return generation_metadata_visible is None or bool(generation_metadata_visible())

    def _requires_enabled(view_func: Callable) -> Callable:
        """Decorator: short-circuit to the disabled response while the layer is off."""
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            """Answer the disabled body instead of running the view while the layer is off."""
            if not config.enabled:
                return _disabled_response()
            return view_func(*args, **kwargs)
        return wrapper

    def _wrap(view_func: Callable, guarded: bool = False) -> Callable:
        """Apply the caller's guard (guarded routes only), then the enabled check outermost."""
        if guarded and guard is not None:
            view_func = guard(view_func)
        return _requires_enabled(view_func)

    def _visible(file_id: str) -> bool:
        """Per-file visibility under the host policy; no policy means everything is visible."""
        return file_access_check is None or bool(file_access_check(file_id))

    def _check_file_access(file_id: str) -> None:
        """Abort 404 for policy-hidden files, indistinguishable from nonexistent ones."""
        if not _visible(file_id):
            abort(404)

    # -- GET /status : always reports, even when disabled -----------------------

    # Probing availability constructs real backends (may load model weights
    # and import torch), so it must never run while the layer is disabled,
    # and at most once per process while enabled.
    backend_probe_cache: dict = {}
    _PROBE_CACHES.append(backend_probe_cache)
    # Torch device each probed backend landed on (None for backends that
    # have no device concept, e.g. OpenCV faces); same lifecycle as the
    # probe cache so a post-provision invalidation refreshes both.
    backend_device_cache: dict = {}
    _PROBE_CACHES.append(backend_device_cache)

    # backend key -> (config selector attr, values that mean "real model",
    # provisioning group). The probe decides availability from weights on
    # disk + importable runtime — it must NEVER construct backends: doing so
    # loads multi-GB models just to compute a boolean and doubles every
    # model load at startup.
    _PROBE_MAP = {
        "semantic": ("semantic_backend", ("auto", "open_clip"), "semantic"),
        "visual": ("visual_backend", ("auto", "dinov2"), "visual"),
        "face": ("face_backend", ("auto", "opencv"), "faces"),
        "segmenter": ("segmenter_backend", ("auto", "mobilesam"), "segmenter"),
        "critic": ("critic_backend", ("auto", "vlm"), "critic"),
    }

    def _cheap_available(key: str) -> bool:
        """Weights present + runtime importable for `key`, without loading
        any model. Selector 'none' is unavailable; 'stub' is available."""
        attr, real_values, group_name = _PROBE_MAP[key]
        selector = getattr(config, attr)
        if selector == "none":
            return False
        if selector == "stub":
            return True
        if selector not in real_values:
            return False
        try:
            group = provisioning.resolve_groups([group_name])[0]
        except Exception:
            return False
        weights_ok = all(
            provisioning.artifact_present(config.models_dir, a)
            for a in group.artifacts
        )
        return weights_ok and not provisioning.runtime_missing(group)

    def _probe_backends() -> dict:
        """Availability flag per backend: all-False while disabled. Uses the
        running worker's already-loaded instances when it has them (ground
        truth), falling back to the cheap weights+runtime check."""
        if not config.enabled:
            return {"semantic": False, "visual": False, "face": False,
                    "critic": False, "segmenter": False}
        if not backend_probe_cache:
            worker = get_worker()
            loaded = dict(getattr(worker, "_backend_cache", {}) or {}) if worker else {}
            for key in _PROBE_MAP:
                inst = loaded.get(key)
                if inst is not None:
                    backend_probe_cache[key] = True
                    backend_device_cache[key] = getattr(inst, "_device", None)
                else:
                    backend_probe_cache[key] = _cheap_available(key)
                    backend_device_cache[key] = None
        return dict(backend_probe_cache)

    def _backend_devices() -> dict:
        """Cached torch device per backend, filled by the same probe pass."""
        if not config.enabled:
            return {}
        _probe_backends()
        return dict(backend_device_cache)

    def status():
        """Backend availability, per-table counts, and worker state; answers even while disabled."""
        conn = _connect(config)
        try:
            backends = _probe_backends()
            counts = {
                "hashed": conn.execute("SELECT COUNT(*) FROM ai_file_hashes").fetchone()[0],
                "embeddings_semantic": conn.execute(
                    "SELECT COUNT(*) FROM ai_embeddings WHERE space = ?", (SPACE_SEMANTIC,)
                ).fetchone()[0],
                "embeddings_visual": conn.execute(
                    "SELECT COUNT(*) FROM ai_embeddings WHERE space = ?", (SPACE_VISUAL,)
                ).fetchone()[0],
                # Faces are provenance-scoped per pipeline; count the active
                # model's rows so a backend switch doesn't double-count.
                "face_instances": (
                    conn.execute("SELECT COUNT(*) FROM ai_face_instances "
                                 "WHERE model_id = ?", (face_model,)).fetchone()[0]
                    if (face_model := _active_face_model()) else
                    conn.execute("SELECT COUNT(*) FROM ai_face_instances").fetchone()[0]),
                "face_clusters": (
                    conn.execute("SELECT COUNT(*) FROM ai_face_clusters "
                                 "WHERE model_id = ?", (face_model,)).fetchone()[0]
                    if face_model else
                    conn.execute("SELECT COUNT(*) FROM ai_face_clusters").fetchone()[0]),
                "reviews": conn.execute("SELECT COUNT(*) FROM ai_reviews").fetchone()[0],
            }
            indexing = indexing_totals(conn)
        finally:
            conn.close()

        worker = get_worker()
        worker_info = (
            {"running": bool(worker.is_running), "stats": dict(worker.stats),
             "provisioning": dict(getattr(worker, "provision_state", {}) or {}),
             "priority_queued": len(getattr(worker, "_priority_ids", []) or []),
             "recent_errors": list(getattr(worker, "recent_errors", []) or []),
             "review_seconds": getattr(worker, "_last_review_seconds", None),
             "stage_pace": {k: round(v, 4) for k, v in
                            (getattr(worker, "_stage_pace", {}) or {}).items()}}
            if worker is not None
            else {"running": False, "stats": {}, "provisioning": {},
                  "priority_queued": 0, "recent_errors": [],
                  "review_seconds": None, "stage_pace": {}}
        )
        try:
            gpu = provisioning.cuda_summary()
        except Exception:  # inventory is best-effort
            gpu = None
        return jsonify({
            "enabled": config.enabled,
            "app_git_ref": app_git_ref(),
            "backends": backends,
            "devices": _backend_devices(),
            "gpu": gpu,
            "counts": counts,
            "indexing": indexing,
            "worker": worker_info,
        })

    # -- GET /duplicates/<file_id> -----------------------------------------------

    def duplicates(file_id: str):
        """Exact and near duplicates of one file, both filtered through the visibility policy."""
        _check_file_access(file_id)
        max_distance = request.args.get("max_distance", config.near_dup_max_distance, type=int)
        conn = _connect(config)
        try:
            groups = hashing.find_exact_duplicates(conn)
            own_group = next((group for group in groups if file_id in group), [])
            # The visibility policy applies to every RETURNED id, not just
            # the anchor: a visible file must not reveal hidden relatives.
            exact = [fid for fid in own_group if fid != file_id and _visible(fid)]
            near_pairs = hashing.find_near_duplicates(conn, file_id, max_distance)
            # Duplicate detection needs this file's hashes; until they exist
            # an empty result means "not indexed yet", not "no duplicates" --
            # but only for files that actually exist to be hashed.
            exists = conn.execute(
                "SELECT 1 FROM files WHERE id = ?", (file_id,)
            ).fetchone() is not None
            hashed = conn.execute(
                "SELECT 1 FROM ai_file_hashes WHERE file_id = ?", (file_id,)
            ).fetchone() is not None
        finally:
            conn.close()
        near = [{"file_id": fid, "distance": distance}
                for fid, distance in near_pairs if _visible(fid)]
        return jsonify({"enabled": True, "exact": exact, "near": near,
                        "pending": exists and not hashed})

    # -- GET /similar/<file_id> --------------------------------------------------

    def similar(file_id: str):
        """The file's k nearest embedding neighbors in the requested space."""
        _check_file_access(file_id)
        space = request.args.get("space", SPACE_SEMANTIC)
        if space not in (SPACE_SEMANTIC, SPACE_VISUAL):
            return jsonify({"enabled": True, "error": f"invalid space: {space!r}"}), 400
        k = request.args.get("k", config.similar_default_k, type=int)
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT vector, model_version FROM ai_embeddings "
                "WHERE file_id = ? AND space = ?", (file_id, space)
            ).fetchone()
            if row is None:
                # pending separates "the worker has not reached this file"
                # (results will arrive) from "no stage will ever embed it".
                pending = _renderable(conn, file_id)
                return jsonify({
                    "enabled": True, "space": space, "neighbors": [],
                    "pending": pending,
                    "note": ("not indexed yet" if pending
                             else "no embedding for this file"),
                })
            query_vec = np.frombuffer(row["vector"], dtype="<f4")
            store = vectors.VectorStore(cache_dir=config.cache_dir, ephemeral=config.ephemeral_index)
            # Pin candidates to the query row's OWN model version: mid-
            # migration, the space's most recent version may differ, and
            # cross-version cosine is meaningless (or a dim mismatch).
            neighbors = store.topk(conn, space, query_vec, k, exclude=[file_id],
                                   model_version=row["model_version"])
        finally:
            conn.close()
        # Hidden neighbors are dropped, not backfilled: the response may
        # carry fewer than k entries rather than leak hidden file ids.
        return jsonify({
            "enabled": True, "space": space,
            "neighbors": [{"file_id": fid, "score": score}
                          for fid, score in neighbors if _visible(fid)],
        })

    # -- faces read scoping ---------------------------------------------------------

    def _active_face_model():
        """model_id of the configured face pipeline, or None when no backend
        resolves. Face rows, clusters, and scan logs are provenance-scoped
        per model; read surfaces serve the ACTIVE pipeline's rows, so
        switching AI_DAM_FACE_BACKEND changes what you see — never what any
        other pipeline stored."""
        try:
            backend = faces.get_face_backend(config)
        except Exception:
            return None
        return backend.model_id if backend is not None else None

    # -- GET /faces/<file_id> -----------------------------------------------------

    def faces_for_file(file_id: str):
        """Detected faces for one file: bounding boxes, landmarks, cluster assignments."""
        _check_file_access(file_id)
        model_id = _active_face_model()
        conn = _connect(config)
        try:
            rows = conn.execute(
                "SELECT face_id, model_id, bbox_x, bbox_y, bbox_w, bbox_h, "
                "landmarks, det_score, "
                "attributes, age, sex, pose_pitch, pose_yaw, pose_roll, cluster_id "
                "FROM ai_face_instances WHERE file_id = ? "
                + ("AND model_id = ? " if model_id else "")
                + "ORDER BY face_id",
                (file_id, model_id) if model_id else (file_id,),
            ).fetchall()
            # Zero faces means two very different things depending on whether
            # a detector has actually looked at this file yet.
            pending = (not rows and _was_scanned(conn, file_id, "faces") is None
                       and _renderable(conn, file_id))
        finally:
            conn.close()
        result = [
            {
                "face_id": row["face_id"],
                "model_id": row["model_id"],
                "bbox": [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]],
                "landmarks": json.loads(row["landmarks"]) if row["landmarks"] else [],
                "det_score": row["det_score"],
                "attributes": json.loads(row["attributes"]) if row["attributes"] else None,
                "age": row["age"],
                "sex": row["sex"],
                "pose": ({"pitch": row["pose_pitch"], "yaw": row["pose_yaw"],
                          "roll": row["pose_roll"]}
                         if row["pose_yaw"] is not None else None),
                "cluster_id": row["cluster_id"],
            }
            for row in rows
        ]
        return jsonify({"enabled": True, "faces": result, "pending": pending})

    # -- GET /faces/clusters -------------------------------------------------------

    def faces_clusters():
        """The active pipeline's face clusters, each with up to four sample
        file ids (all models' clusters when no backend resolves)."""
        model_id = _active_face_model()
        conn = _connect(config)
        try:
            cluster_rows = conn.execute(
                "SELECT cluster_id, label, size FROM ai_face_clusters "
                + ("WHERE model_id = ? " if model_id else "")
                + "ORDER BY cluster_id",
                (model_id,) if model_id else (),
            ).fetchall()
            clusters = []
            for crow in cluster_rows:
                sample_rows = conn.execute(
                    "SELECT DISTINCT file_id FROM ai_face_instances "
                    "WHERE cluster_id = ? ORDER BY file_id LIMIT 4",
                    (crow["cluster_id"],),
                ).fetchall()
                clusters.append({
                    "cluster_id": crow["cluster_id"],
                    "label": crow["label"],
                    "size": crow["size"],
                    "sample_file_ids": [s["file_id"] for s in sample_rows],
                })
        finally:
            conn.close()
        return jsonify({"enabled": True, "clusters": clusters})

    # -- GET /faces/clusters/<int:cluster_id> ---------------------------------------

    def faces_cluster_detail(cluster_id: int):
        """One cluster's metadata and every member face; unknown ids answer 404."""
        conn = _connect(config)
        try:
            crow = conn.execute(
                "SELECT cluster_id, label, size FROM ai_face_clusters WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            if crow is None:
                return jsonify({"enabled": True, "cluster": None, "members": []}), 404
            member_rows = conn.execute(
                "SELECT face_id, file_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, "
                "age, sex, pose_yaw "
                "FROM ai_face_instances WHERE cluster_id = ? ORDER BY face_id",
                (cluster_id,),
            ).fetchall()
            # Attribute aggregates over the typed per-face columns, so the
            # cluster UI can say who this bucket looks like without the
            # client re-deriving it.
            agg = conn.execute(
                "SELECT COUNT(age) AS with_age, MIN(age) AS age_min, "
                "MAX(age) AS age_max, ROUND(AVG(age), 1) AS age_avg, "
                "SUM(CASE WHEN sex = 'M' THEN 1 ELSE 0 END) AS male, "
                "SUM(CASE WHEN sex = 'F' THEN 1 ELSE 0 END) AS female, "
                "ROUND(AVG(ABS(pose_yaw)), 1) AS yaw_abs_avg "
                "FROM ai_face_instances WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
        finally:
            conn.close()
        members = [
            {
                "face_id": row["face_id"],
                "file_id": row["file_id"],
                "bbox": [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]],
                "det_score": row["det_score"],
                "age": row["age"],
                "sex": row["sex"],
                "pose_yaw": row["pose_yaw"],
            }
            for row in member_rows
        ]
        cluster = {"cluster_id": crow["cluster_id"], "label": crow["label"], "size": crow["size"]}
        return jsonify({"enabled": True, "cluster": cluster, "members": members,
                        "attributes": dict(agg) if agg else None})

    # -- POST /faces/recluster (guarded) --------------------------------------------

    def faces_recluster():
        """Re-run face clustering across the whole library; reports the resulting cluster count."""
        backend = faces.get_face_backend(config)
        if backend is None:
            return jsonify({"enabled": True, "clusters": 0, "note": "no face backend configured"})
        conn = _connect(config)
        try:
            new_cluster_ids = faces.cluster_faces(
                conn, backend.model_id, backend.model_version,
                faces.resolve_cluster_threshold(config, backend),
            )
        finally:
            conn.close()
        return jsonify({"enabled": True, "clusters": len(new_cluster_ids)})

    # -- GET /faces/recent (guarded) ------------------------------------------------

    def faces_recent():
        """The most recently scanned files that contain faces — the
        dashboard's picker for the detector-compare tool."""
        model_id = _active_face_model()
        conn = _connect(config)
        try:
            rows = conn.execute(
                "SELECT file_id, MAX(face_id) AS latest, COUNT(*) AS faces "
                "FROM ai_face_instances "
                + ("WHERE model_id = ? " if model_id else "")
                + "GROUP BY file_id ORDER BY latest DESC LIMIT 24",
                (model_id,) if model_id else ()).fetchall()
        finally:
            conn.close()
        return jsonify({"enabled": True, "files": [
            {"file_id": r["file_id"], "faces": r["faces"]}
            for r in rows if _visible(r["file_id"])]})

    # -- GET /faces/compare/<file_id> (guarded) -------------------------------------

    def faces_compare(file_id: str):
        """Run BOTH deployed face pipelines (opencv YuNet lane and
        insightface SCRFD lane) live on one file and answer their raw
        detections side by side, with per-pipeline model identity and
        timing. Pure diagnostic: nothing is persisted, stored instances
        are untouched. Compute-heavy, hence guarded."""
        _check_file_access(file_id)
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT path, type FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            abort(404)
        from smartgallery_ai.worker import load_source_image
        img = load_source_image(row["path"], row["type"])
        if img is None:
            return jsonify({"enabled": True,
                            "error": "file has no renderable frame"}), 422
        try:
            result = faces.compare_detectors(img, config)
        finally:
            img.close()
        return jsonify({"enabled": True, "file_id": file_id, **result})

    # -- GET /review/<file_id> -----------------------------------------------------

    def review_for_file(file_id: str):
        """The file's most recent review plus findings; mask URLs only where a mask exists."""
        _check_file_access(file_id)
        conn = _connect(config)
        try:
            review_row = conn.execute(
                "SELECT review_id, rubric_version, model_id, model_version, quality_score, "
                "prompt_alignment_score, summary, computed_at FROM ai_reviews "
                "WHERE file_id = ? ORDER BY computed_at DESC LIMIT 1",
                (file_id,),
            ).fetchone()
            # Whether this file HAS a generation prompt to align against --
            # a null alignment score means "no prompt to compare with" far
            # more often than "scoring failed", and the panel says which.
            # Same resolver the critic scored against, so the panel can
            # never claim "no prompt" about a file the critic did score.
            prompt_available = review.resolve_prompt_texts(conn, file_id)[0] is not None
            if review_row is None:
                # scan-log row present + no review row = the one attempt
                # failed (result_count -1); absent = not reached yet.
                scan = _was_scanned(conn, file_id, "review")
                pending = scan is None and _renderable(conn, file_id)
                failed = scan is not None and scan["result_count"] == -1
                return jsonify({"enabled": True, "review": None, "findings": [],
                                "pending": pending, "scan_failed": failed})

            finding_rows = conn.execute(
                "SELECT finding_id, type, severity, confidence, localizable, "
                "bbox_x, bbox_y, bbox_w, bbox_h, points, description, mask_path "
                "FROM ai_review_findings WHERE review_id = ? ORDER BY finding_id",
                (review_row["review_id"],),
            ).fetchall()
            alignment_rows = conn.execute(
                "SELECT element_id, ordinal, text, satisfied, confidence, "
                "bbox_x, bbox_y, bbox_w, bbox_h, mask_path "
                "FROM ai_review_alignment WHERE review_id = ? ORDER BY ordinal",
                (review_row["review_id"],),
            ).fetchall()
        finally:
            conn.close()

        findings = []
        for frow in finding_rows:
            localizable = bool(frow["localizable"])
            entry = {
                "finding_id": frow["finding_id"],
                "type": frow["type"],
                "severity": frow["severity"],
                "confidence": frow["confidence"],
                "localizable": localizable,
                "description": frow["description"],
                "bbox": None,
                "points": None,
            }
            if localizable:
                if frow["bbox_x"] is not None:
                    entry["bbox"] = [frow["bbox_x"], frow["bbox_y"], frow["bbox_w"], frow["bbox_h"]]
                if frow["points"]:
                    entry["points"] = json.loads(frow["points"])
                if frow["mask_path"]:
                    entry["mask_url"] = url_for(
                        "aidam.review_mask", finding_id=frow["finding_id"]
                    )
            findings.append(entry)

        # Each element's text is a verbatim slice of the positive prompt, so
        # a caller who may not read prompts may not read these -- the scores
        # and the findings, which describe the picture rather than quote the
        # request, still go out.
        show_prompt = _may_see_generation_metadata()

        alignment = []
        for arow in alignment_rows if show_prompt else []:
            satisfied = bool(arow["satisfied"])
            entry = {
                "element_id": arow["element_id"],
                "ordinal": arow["ordinal"],
                "text": arow["text"],
                "satisfied": satisfied,
                "confidence": arow["confidence"],
                "bbox": None,
            }
            if satisfied and arow["bbox_x"] is not None:
                entry["bbox"] = [arow["bbox_x"], arow["bbox_y"],
                                 arow["bbox_w"], arow["bbox_h"]]
                if arow["mask_path"]:
                    entry["mask_url"] = url_for(
                        "aidam.review_alignment_mask", element_id=arow["element_id"])
            alignment.append(entry)

        review_dict = {
            "scores": {
                "quality": review_row["quality_score"],
                # 0..1; the panel renders it as a percentage
                "prompt_alignment": review_row["prompt_alignment_score"],
            },
            "alignment": alignment,
            # The critic writes the summary about the picture, but it is free
            # prose and quotes the prompt when explaining a mismatch.
            "summary": review_row["summary"] if show_prompt else None,
            "model": {
                "id": review_row["model_id"],
                "version": review_row["model_version"],
                "rubric_version": review_row["rubric_version"],
            },
            "computed_at": review_row["computed_at"],
        }
        return jsonify({"enabled": True, "review": review_dict, "findings": findings,
                        "prompt_available": prompt_available})

    # -- GET /review/mask/<int:finding_id> -------------------------------------------

    def _serve_mask(table: str, key_column: str, key: int):
        """Serve one stored mask PNG from `table`, enforcing file-level
        access and path containment. The stored path must resolve inside
        the masks cache -- a row pointing anywhere else is a 404, not a
        file read."""
        conn = _connect(config)
        try:
            row = conn.execute(
                f"SELECT mask_path, file_id FROM {table} WHERE {key_column} = ?",
                (key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None or not row["mask_path"]:
            abort(404)
        # A mask belongs to a file: the caller must be allowed to see it.
        _check_file_access(row["file_id"])

        # Containment is checked against the masks/ subdirectory (the only
        # place the writer puts masks), not the whole cache dir.
        masks_root = os.path.realpath(os.path.join(config.cache_dir, "masks"))
        resolved = os.path.realpath(row["mask_path"])
        try:
            inside = os.path.commonpath([masks_root, resolved]) == masks_root
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(resolved):
            abort(404)
        return send_file(resolved, mimetype="image/png")

    def review_mask(finding_id: int):
        """Serve one finding's mask PNG."""
        return _serve_mask("ai_review_findings", "finding_id", finding_id)

    def review_alignment_mask(element_id: int):
        """Serve one satisfied prompt element's highlight mask PNG."""
        return _serve_mask("ai_review_alignment", "element_id", element_id)

    # -- GET /review/run/<file_id> (SSE) -------------------------------------------

    def review_run(file_id: str):
        """Stream an interactive review of one file as server-sent events.

        Answers the question background indexing cannot: what is the critic
        doing RIGHT NOW on THIS image. Each pipeline step and each VLM
        protocol stage arrives as its own event, so a ~200s review is
        legible while it runs instead of only after it ends.

        `?steps=resolve,load,critic` runs a prefix -- inspect a payload
        without writing anything. 409 when another run holds the critic:
        an interactive action silently queued behind a 200s job is a worse
        answer than saying so.
        """
        _check_file_access(file_id)
        try:
            events = runner.run_review(config, file_id,
                                       steps=request.args.get("steps"))
            first = next(events)
        except runner.RunnerBusy as exc:
            return jsonify({"enabled": True, "error": str(exc)}), 409

        def stream():
            # `first` is already drawn; re-emit it before draining the rest.
            try:
                for event in itertools.chain([first], events):
                    yield f"data: {json.dumps(event)}\n\n"
            except GeneratorExit:
                # Client hung up. Closing the generator releases the run
                # lock via its finally block; without this the lock would
                # be held until process exit and every later run would 409.
                events.close()
                raise
            except Exception as exc:  # a stream must end, not hang
                yield f"data: {json.dumps({'step': 'run', 'status': 'error', 'detail': {'error': str(exc)}})}\n\n"

        return Response(stream(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            # Long-lived streams die behind proxies that buffer; this is the
            # conventional opt-out and is inert when no proxy is present.
            "X-Accel-Buffering": "no",
        })

    # -- POST /review/feedback (guarded) / GET /review/feedback/export --------------

    def review_feedback_post():
        """Record one human feedback entry; validation failures answer 400."""
        data = request.get_json(silent=True) or {}
        conn = _connect(config)
        try:
            feedback_id = feedback.record_feedback(
                conn,
                target_kind=data.get("target_kind"),
                target_id=data.get("target_id"),
                verdict=data.get("verdict"),
                file_id=data.get("file_id"),
                rating=data.get("rating"),
                note=data.get("note"),
                created_by=data.get("created_by"),
            )
        except feedback.FeedbackValidationError as exc:
            return jsonify({"enabled": True, "error": str(exc)}), 400
        finally:
            conn.close()
        return jsonify({"enabled": True, "feedback_id": feedback_id}), 201

    def review_feedback_export():
        """Full feedback log as a downloadable NDJSON attachment."""
        conn = _connect(config)
        try:
            text = feedback.export_feedback(conn)
        finally:
            conn.close()
        return Response(
            text,
            mimetype="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=ai_feedback_export.jsonl"},
        )

    # -- POST /index/<file_id> (guarded) --------------------------------------------

    def index_file(file_id: str):
        """(Re-)index one file NOW. With a running worker the file jumps
        its priority queue and the worker (whose thread owns every loaded
        model) runs all stages immediately -- sharing its live backend
        instances with this request thread would be a data race. Only
        without a worker does the request run the fast stages inline,
        with backends it constructs itself. `force` additionally
        reschedules the file's background review."""
        data = request.get_json(silent=True) or {}
        force = bool(data.get("force", False))
        worker = get_worker()
        defer = worker is not None and worker.is_running
        conn = _connect(config)
        try:
            file_row = conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            if file_row is None:
                return jsonify({"enabled": True, "error": f"unknown file_id: {file_id!r}"}), 404
            if defer:
                if force:
                    conn.execute(
                        "DELETE FROM ai_scan_log WHERE file_id = ? AND kind = 'review'",
                        (file_id,))
                    conn.commit()
                result = {"file_id": file_id,
                          "review_rescheduled": force,
                          "worker_queued": worker.request_priority_index(file_id)}
            else:
                result = _index_one_file(conn, config, file_row, force=force)
                result["worker_queued"] = False
        finally:
            conn.close()
        return jsonify({"enabled": True, **result})

    # -- GET /search/semantic?q= ----------------------------------------------------

    # Fallback text encoder for deployments without a running worker;
    # request threads share one instance (the real backend serializes its
    # forwards internally). Registered for post-provision invalidation so
    # a freshly landed backend replaces a cached None.
    search_embedder_cache: dict = {}
    _PROBE_CACHES.append(search_embedder_cache)
    search_embedder_lock = threading.Lock()

    def _search_embedder():
        """The worker's loaded semantic embedder when available (cheapest,
        already in memory), else one lazily constructed for this blueprint."""
        worker = get_worker()
        if worker is not None:
            borrowed = worker.semantic_embedder_for_search()
            if borrowed is not None:
                return borrowed
        with search_embedder_lock:
            if "semantic" not in search_embedder_cache:
                try:
                    search_embedder_cache["semantic"] = embedders.get_semantic_backend(config)
                except Exception:  # unavailable, not fatal
                    search_embedder_cache["semantic"] = None
            return search_embedder_cache["semantic"]

    def search_semantic():
        """Free-text semantic image search: the query text goes through the
        CLIP text tower into the SAME space as the image embeddings, so
        'a red car at night' finds images by meaning, not filename. Every
        returned id passes the visibility policy."""
        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify({"enabled": True, "error": "missing query ?q="}), 400
        k = request.args.get("k", config.similar_default_k, type=int)
        embedder = _search_embedder()
        if embedder is None:
            return jsonify({"enabled": True, "query": query, "results": [],
                            "note": "semantic backend not available yet"})
        query_vec = embedder.embed_text(query)
        conn = _connect(config)
        try:
            store = vectors.VectorStore(cache_dir=config.cache_dir,
                                        ephemeral=config.ephemeral_index)
            neighbors = store.topk(conn, SPACE_SEMANTIC, query_vec, k,
                                   model_version=embedder.model_version)
        finally:
            conn.close()
        return jsonify({
            "enabled": True, "query": query,
            "results": [{"file_id": fid, "score": score}
                        for fid, score in neighbors if _visible(fid)],
        })

    # -- GET /reviews (guarded): gallery-wide review browser ------------------------

    _REVIEW_SORTS = {
        # SQLite sorts NULLs first on ASC; push score-less rows last instead.
        "quality_asc": "r.quality_score IS NULL, r.quality_score ASC",
        "quality_desc": "r.quality_score IS NULL, r.quality_score DESC",
        "alignment_asc": "r.prompt_alignment_score IS NULL, r.prompt_alignment_score ASC",
        "alignment_desc": "r.prompt_alignment_score IS NULL, r.prompt_alignment_score DESC",
        "newest": "r.computed_at DESC",
    }

    def reviews_list():
        """Newest review per file across the gallery, sortable by quality /
        prompt alignment / recency — the 'show me my worst generations'
        browser. Carries a findings count per row."""
        sort = request.args.get("sort", "quality_asc")
        order = _REVIEW_SORTS.get(sort)
        if order is None:
            return jsonify({"enabled": True,
                            "error": f"invalid sort: {sort!r}"}), 400
        limit = min(max(request.args.get("limit", 60, type=int), 1), 200)
        offset = max(request.args.get("offset", 0, type=int), 0)
        conn = _connect(config)
        try:
            total = conn.execute(
                "SELECT COUNT(DISTINCT file_id) FROM ai_reviews").fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT r.file_id, r.quality_score, r.prompt_alignment_score,
                       r.summary, r.computed_at,
                       (SELECT COUNT(*) FROM ai_review_findings f
                        WHERE f.review_id = r.review_id) AS finding_count
                FROM ai_reviews r
                WHERE r.review_id IN
                      (SELECT MAX(review_id) FROM ai_reviews GROUP BY file_id)
                ORDER BY {order}
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        finally:
            conn.close()
        return jsonify({
            "enabled": True, "sort": sort, "total": total, "offset": offset,
            "reviews": [
                {"file_id": row["file_id"],
                 "quality": row["quality_score"],
                 "prompt_alignment": row["prompt_alignment_score"],
                 "summary": row["summary"],
                 "computed_at": row["computed_at"],
                 "finding_count": row["finding_count"]}
                for row in rows
            ],
        })

    # -- GET /duplicates (guarded): gallery-wide duplicate sweep --------------------

    def duplicates_overview():
        """Every exact-duplicate group in the gallery with the bytes you
        would reclaim by keeping one copy of each, largest waste first."""
        conn = _connect(config)
        try:
            groups = hashing.find_exact_duplicates(conn)
            out = []
            total_reclaimable = 0
            for group in groups:
                placeholders = ",".join("?" for _ in group)
                size_rows = conn.execute(
                    f"SELECT id, COALESCE(size, 0) AS size FROM files "
                    f"WHERE id IN ({placeholders})", list(group)).fetchall()
                sizes = [row["size"] for row in size_rows]
                reclaimable = max(sum(sizes) - max(sizes), 0) if sizes else 0
                total_reclaimable += reclaimable
                out.append({"file_ids": list(group), "count": len(group),
                            "bytes_reclaimable": reclaimable})
        finally:
            conn.close()
        out.sort(key=lambda g: g["bytes_reclaimable"], reverse=True)
        return jsonify({
            "enabled": True,
            "group_count": len(out),
            "redundant_files": sum(g["count"] - 1 for g in out),
            "total_bytes_reclaimable": total_reclaimable,
            "groups": out,
        })

    # -- POST /faces/clusters/<id>/label (guarded) ----------------------------------

    def faces_cluster_label(cluster_id: int):
        """Name a face cluster ('Sarah', 'the knight character'); empty
        label clears back to the numbered default."""
        data = request.get_json(silent=True) or {}
        label = (data.get("label") or "").strip()[:80] or None
        conn = _connect(config)
        try:
            cur = conn.execute(
                "UPDATE ai_face_clusters SET label = ? WHERE cluster_id = ?",
                (label, cluster_id))
            conn.commit()
        finally:
            conn.close()
        if cur.rowcount == 0:
            return jsonify({"enabled": True,
                            "error": f"unknown cluster_id: {cluster_id}"}), 404
        return jsonify({"enabled": True, "cluster_id": cluster_id, "label": label})

    # -- route table -----------------------------------------------------------------

    # Status carries the management guard but NOT _requires_enabled: it has to
    # keep reporting while the layer is off, which is how the panel knows to
    # say so. It reports across the whole library -- worker errors quote the
    # path of the file that failed -- so it belongs with the other cross-file
    # routes rather than open to anyone who can reach the port.
    bp.add_url_rule("/status", "status",
                    guard(status) if guard is not None else status, methods=["GET"])
    bp.add_url_rule("/duplicates/<file_id>", "duplicates", _wrap(duplicates), methods=["GET"])
    bp.add_url_rule("/similar/<file_id>", "similar", _wrap(similar), methods=["GET"])
    bp.add_url_rule("/faces/<file_id>", "faces_for_file", _wrap(faces_for_file), methods=["GET"])
    # Cluster listings enumerate metadata ACROSS files, so per-file
    # visibility checks cannot scope them; they carry the management guard
    # (open in local mode, privileged-only in restricted modes).
    bp.add_url_rule("/faces/clusters", "faces_clusters",
                    _wrap(faces_clusters, guarded=True), methods=["GET"])
    bp.add_url_rule(
        "/faces/clusters/<int:cluster_id>", "faces_cluster_detail",
        _wrap(faces_cluster_detail, guarded=True), methods=["GET"],
    )
    bp.add_url_rule(
        "/faces/recent", "faces_recent",
        _wrap(faces_recent, guarded=True), methods=["GET"])
    bp.add_url_rule(
        "/faces/compare/<file_id>", "faces_compare",
        _wrap(faces_compare, guarded=True), methods=["GET"])
    bp.add_url_rule(
        "/faces/recluster", "faces_recluster",
        _wrap(faces_recluster, guarded=True), methods=["POST"],
    )
    bp.add_url_rule("/review/<file_id>", "review_for_file", _wrap(review_for_file), methods=["GET"])
    bp.add_url_rule(
        "/review/mask/<int:finding_id>", "review_mask", _wrap(review_mask), methods=["GET"]
    )
    bp.add_url_rule(
        "/review/alignment/mask/<int:element_id>", "review_alignment_mask",
        _wrap(review_alignment_mask), methods=["GET"]
    )
    # Guarded despite being a GET. The guard was applied to the "mutating"
    # endpoints, and this one reads as a read -- but it runs the critic for
    # minutes, holds the run lock while it does, and writes a review at the
    # end. Left open it is both a way to make the machine work on demand and
    # a way to watch the prompt arrive element by element in the stream.
    bp.add_url_rule(
        "/review/run/<file_id>", "review_run", _wrap(review_run, guarded=True),
        methods=["GET"]
    )
    bp.add_url_rule(
        "/review/feedback", "review_feedback_post",
        _wrap(review_feedback_post, guarded=True), methods=["POST"],
    )
    bp.add_url_rule(
        "/review/feedback/export", "review_feedback_export",
        _wrap(review_feedback_export, guarded=True), methods=["GET"],
    )
    bp.add_url_rule("/index/<file_id>", "index_file", _wrap(index_file, guarded=True), methods=["POST"])
    bp.add_url_rule("/search/semantic", "search_semantic",
                    _wrap(search_semantic), methods=["GET"])
    # Cross-file listings carry the management guard, like cluster listings.
    bp.add_url_rule("/reviews", "reviews_list",
                    _wrap(reviews_list, guarded=True), methods=["GET"])
    bp.add_url_rule("/duplicates", "duplicates_overview",
                    _wrap(duplicates_overview, guarded=True), methods=["GET"])
    bp.add_url_rule("/faces/clusters/<int:cluster_id>/label", "faces_cluster_label",
                    _wrap(faces_cluster_label, guarded=True), methods=["POST"])

    return bp
