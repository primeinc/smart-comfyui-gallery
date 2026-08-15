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

import json
import os
import sqlite3
import time
from functools import wraps
from typing import Any, Callable, Optional

import numpy as np
from flask import Blueprint, Response, abort, jsonify, request, send_file, url_for

from smartgallery_ai import (
    AIConfig,
    HASH_ALGO_VERSION,
    RUBRIC_VERSION,
    SPACE_SEMANTIC,
    SPACE_VISUAL,
)
from smartgallery_ai import embedders, faces, feedback, hashing, invalidation, review, vectors
from smartgallery_ai.worker import (
    _MTIME_EPSILON,
    _has_column,
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
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
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

    def _backend_for(key: str, factory):
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

    for key, space, get_backend in (
        ("semantic", SPACE_SEMANTIC, embedders.get_semantic_backend),
        ("visual", SPACE_VISUAL, embedders.get_visual_backend),
    ):
        backend = _backend_for(key, get_backend)
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

    face_backend = _backend_for("face", faces.get_face_backend)
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
    Inaccessible files answer 404, indistinguishable from nonexistent."""
    bp = Blueprint("aidam", __name__)

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

    def _probe_backends() -> dict:
        """Availability flag per backend: all-False while disabled, else probed once and cached."""
        if not config.enabled:
            return {"semantic": False, "visual": False, "face": False,
                    "critic": False, "segmenter": False}
        if not backend_probe_cache:
            instances = {
                "semantic": embedders.get_semantic_backend(config),
                "visual": embedders.get_visual_backend(config),
                "face": faces.get_face_backend(config),
                "critic": review.get_critic_backend(config),
            }
            try:
                instances["segmenter"] = review.get_segmenter_backend(config)
            except Exception:  # noqa: BLE001 - availability probe must not raise
                instances["segmenter"] = None
            backend_probe_cache.update(
                {key: inst is not None for key, inst in instances.items()})
            backend_device_cache.update(
                {key: getattr(inst, "_device", None) for key, inst in instances.items()})
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
                "face_instances": conn.execute("SELECT COUNT(*) FROM ai_face_instances").fetchone()[0],
                "face_clusters": conn.execute("SELECT COUNT(*) FROM ai_face_clusters").fetchone()[0],
                "reviews": conn.execute("SELECT COUNT(*) FROM ai_reviews").fetchone()[0],
            }
            indexing = indexing_totals(conn)
        finally:
            conn.close()

        worker = get_worker()
        worker_info = (
            {"running": bool(worker.is_running), "stats": dict(worker.stats),
             "provisioning": dict(getattr(worker, "provision_state", {}) or {})}
            if worker is not None
            else {"running": False, "stats": {}, "provisioning": {}}
        )
        return jsonify({
            "enabled": config.enabled,
            "backends": backends,
            "devices": _backend_devices(),
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

    # -- GET /faces/<file_id> -----------------------------------------------------

    def faces_for_file(file_id: str):
        """Detected faces for one file: bounding boxes, landmarks, cluster assignments."""
        _check_file_access(file_id)
        conn = _connect(config)
        try:
            rows = conn.execute(
                "SELECT face_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, det_score, cluster_id "
                "FROM ai_face_instances WHERE file_id = ? ORDER BY face_id",
                (file_id,),
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
                "bbox": [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]],
                "landmarks": json.loads(row["landmarks"]) if row["landmarks"] else [],
                "det_score": row["det_score"],
                "cluster_id": row["cluster_id"],
            }
            for row in rows
        ]
        return jsonify({"enabled": True, "faces": result, "pending": pending})

    # -- GET /faces/clusters -------------------------------------------------------

    def faces_clusters():
        """All face clusters, each with up to four sample file ids."""
        conn = _connect(config)
        try:
            cluster_rows = conn.execute(
                "SELECT cluster_id, label, size FROM ai_face_clusters ORDER BY cluster_id"
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
                "SELECT face_id, file_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score "
                "FROM ai_face_instances WHERE cluster_id = ? ORDER BY face_id",
                (cluster_id,),
            ).fetchall()
        finally:
            conn.close()
        members = [
            {
                "face_id": row["face_id"],
                "file_id": row["file_id"],
                "bbox": [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]],
                "det_score": row["det_score"],
            }
            for row in member_rows
        ]
        cluster = {"cluster_id": crow["cluster_id"], "label": crow["label"], "size": crow["size"]}
        return jsonify({"enabled": True, "cluster": cluster, "members": members})

    # -- POST /faces/recluster (guarded) --------------------------------------------

    def faces_recluster():
        """Re-run face clustering across the whole library; reports the resulting cluster count."""
        backend = faces.get_face_backend(config)
        if backend is None:
            return jsonify({"enabled": True, "clusters": 0, "note": "no face backend configured"})
        conn = _connect(config)
        try:
            new_cluster_ids = faces.cluster_faces(
                conn, backend.model_id, backend.model_version, config.face_cluster_threshold
            )
        finally:
            conn.close()
        return jsonify({"enabled": True, "clusters": len(new_cluster_ids)})

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
            prompt_available = False
            if _has_column(conn, "files", "workflow_prompt"):
                prow = conn.execute(
                    "SELECT workflow_prompt FROM files WHERE id = ?", (file_id,)
                ).fetchone()
                prompt_available = bool(prow and (prow["workflow_prompt"] or "").strip())
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

        review_dict = {
            "scores": {
                "quality": review_row["quality_score"],
                "prompt_alignment": review_row["prompt_alignment_score"],
            },
            "summary": review_row["summary"],
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

    def review_mask(finding_id: int):
        """Serve one finding's mask PNG; the stored path must resolve inside the masks cache."""
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT mask_path, file_id FROM ai_review_findings "
                "WHERE finding_id = ?", (finding_id,)
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

    # -- route table -----------------------------------------------------------------

    bp.add_url_rule("/status", "status", status, methods=["GET"])
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
        "/faces/recluster", "faces_recluster",
        _wrap(faces_recluster, guarded=True), methods=["POST"],
    )
    bp.add_url_rule("/review/<file_id>", "review_for_file", _wrap(review_for_file), methods=["GET"])
    bp.add_url_rule(
        "/review/mask/<int:finding_id>", "review_mask", _wrap(review_mask), methods=["GET"]
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

    return bp
