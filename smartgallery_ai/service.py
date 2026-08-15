"""Flask blueprint for the AI DAM layer (WI-31 wave 2).

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
from smartgallery_ai.worker import load_source_image

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
    return _worker_ref.get("worker")


def _connect(config: AIConfig) -> sqlite3.Connection:
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _disabled_response():
    return jsonify({"enabled": False}), 200


def _segmenter_available(config: AIConfig) -> bool:
    """No 'auto'/real segmenter resolver exists yet (see review.py); only
    the explicit test/dev stub is reachable."""
    return config.segmenter_backend == "stub"


def _extract_file_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        return value.get("file_id")
    if isinstance(value, str):
        return value
    return None


def _extract_file_id_and_k(value: Any, default_k: int) -> tuple:
    if isinstance(value, dict):
        return value.get("file_id"), int(value.get("k", default_k))
    return value, default_k


def create_ai_resolvers(config: AIConfig) -> dict:
    """Build the `omniquery` `ai_resolvers` map for `near_dup_of`,
    `similar_to_semantic`, and `similar_to_visual`. Each resolver opens its
    own connection and returns `[]` for an unknown/unembedded file."""

    def near_dup_of(value: Any) -> list:
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
        file_id, k = _extract_file_id_and_k(value, config.similar_default_k)
        if not file_id:
            return []
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT vector FROM ai_embeddings WHERE file_id = ? AND space = ?",
                (file_id, space),
            ).fetchone()
            if row is None:
                return []
            query_vec = np.frombuffer(row["vector"], dtype="<f4")
            store = vectors.VectorStore(cache_dir=config.cache_dir, ephemeral=config.ephemeral_index)
            neighbors = store.topk(conn, space, query_vec, k, exclude=[file_id])
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
        backend = get_backend(config)
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

    face_backend = faces.get_face_backend(config)
    if face_backend is not None and img is not None:
        detections = face_backend.detect(img)
        faces.replace_faces_for_file(
            conn, file_id, detections, face_backend.model_id, face_backend.model_version, mtime, now
        )
        result["faces"] = True

    critic_backend = review.get_critic_backend(config)
    if critic_backend is not None and img is not None:
        prompt_text = file_row["workflow_prompt"] if "workflow_prompt" in file_row.keys() else None
        payload = critic_backend.review(img, prompt_text, RUBRIC_VERSION)
        review_result = review.validate_review_payload(payload)
        review.store_review(
            conn, file_id, review_result, critic_backend.model_id, critic_backend.model_version,
            RUBRIC_VERSION, json.dumps(payload), mtime, now,
        )
        result["reviewed"] = True

    return result


def create_ai_blueprint(config: AIConfig, guard: Optional[Callable] = None) -> Blueprint:
    """Build the AI DAM Flask blueprint. `guard`, if given, is applied to
    the mutating endpoints (recluster, feedback POST, index POST) -- the
    caller passes its own auth decorator (e.g. `management_api_only`)."""
    bp = Blueprint("aidam", __name__)

    def _requires_enabled(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not config.enabled:
                return _disabled_response()
            return view_func(*args, **kwargs)
        return wrapper

    def _wrap(view_func: Callable, guarded: bool = False) -> Callable:
        if guarded and guard is not None:
            view_func = guard(view_func)
        return _requires_enabled(view_func)

    # -- GET /status : always reports, even when disabled -----------------------

    def status():
        conn = _connect(config)
        try:
            backends = {
                "semantic": embedders.get_semantic_backend(config) is not None,
                "visual": embedders.get_visual_backend(config) is not None,
                "face": faces.get_face_backend(config) is not None,
                "critic": review.get_critic_backend(config) is not None,
                "segmenter": _segmenter_available(config),
            }
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
        finally:
            conn.close()

        worker = get_worker()
        worker_info = (
            {"running": bool(worker.is_running), "stats": dict(worker.stats)}
            if worker is not None else {"running": False, "stats": {}}
        )
        return jsonify({
            "enabled": config.enabled,
            "backends": backends,
            "counts": counts,
            "worker": worker_info,
        })

    # -- GET /duplicates/<file_id> -----------------------------------------------

    def duplicates(file_id: str):
        max_distance = request.args.get("max_distance", config.near_dup_max_distance, type=int)
        conn = _connect(config)
        try:
            groups = hashing.find_exact_duplicates(conn)
            own_group = next((group for group in groups if file_id in group), [])
            exact = [fid for fid in own_group if fid != file_id]
            near_pairs = hashing.find_near_duplicates(conn, file_id, max_distance)
        finally:
            conn.close()
        near = [{"file_id": fid, "distance": distance} for fid, distance in near_pairs]
        return jsonify({"enabled": True, "exact": exact, "near": near})

    # -- GET /similar/<file_id> --------------------------------------------------

    def similar(file_id: str):
        space = request.args.get("space", SPACE_SEMANTIC)
        if space not in (SPACE_SEMANTIC, SPACE_VISUAL):
            return jsonify({"enabled": True, "error": f"invalid space: {space!r}"}), 400
        k = request.args.get("k", config.similar_default_k, type=int)
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT vector FROM ai_embeddings WHERE file_id = ? AND space = ?", (file_id, space)
            ).fetchone()
            if row is None:
                return jsonify({
                    "enabled": True, "space": space, "neighbors": [],
                    "note": "no embedding for this file",
                })
            query_vec = np.frombuffer(row["vector"], dtype="<f4")
            store = vectors.VectorStore(cache_dir=config.cache_dir, ephemeral=config.ephemeral_index)
            neighbors = store.topk(conn, space, query_vec, k, exclude=[file_id])
        finally:
            conn.close()
        return jsonify({
            "enabled": True, "space": space,
            "neighbors": [{"file_id": fid, "score": score} for fid, score in neighbors],
        })

    # -- GET /faces/<file_id> -----------------------------------------------------

    def faces_for_file(file_id: str):
        conn = _connect(config)
        try:
            rows = conn.execute(
                "SELECT face_id, bbox_x, bbox_y, bbox_w, bbox_h, landmarks, det_score, cluster_id "
                "FROM ai_face_instances WHERE file_id = ? ORDER BY face_id",
                (file_id,),
            ).fetchall()
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
        return jsonify({"enabled": True, "faces": result})

    # -- GET /faces/clusters -------------------------------------------------------

    def faces_clusters():
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
        conn = _connect(config)
        try:
            review_row = conn.execute(
                "SELECT review_id, rubric_version, model_id, model_version, quality_score, "
                "prompt_alignment_score, summary, computed_at FROM ai_reviews "
                "WHERE file_id = ? ORDER BY computed_at DESC LIMIT 1",
                (file_id,),
            ).fetchone()
            if review_row is None:
                return jsonify({"enabled": True, "review": None, "findings": []})

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
        return jsonify({"enabled": True, "review": review_dict, "findings": findings})

    # -- GET /review/mask/<int:finding_id> -------------------------------------------

    def review_mask(finding_id: int):
        conn = _connect(config)
        try:
            row = conn.execute(
                "SELECT mask_path FROM ai_review_findings WHERE finding_id = ?", (finding_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None or not row["mask_path"]:
            abort(404)

        cache_root = os.path.realpath(config.cache_dir)
        resolved = os.path.realpath(row["mask_path"])
        try:
            inside = os.path.commonpath([cache_root, resolved]) == cache_root
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(resolved):
            abort(404)
        return send_file(resolved, mimetype="image/png")

    # -- POST /review/feedback (guarded) / GET /review/feedback/export --------------

    def review_feedback_post():
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
        data = request.get_json(silent=True) or {}
        force = bool(data.get("force", False))
        conn = _connect(config)
        try:
            file_row = conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            if file_row is None:
                return jsonify({"enabled": True, "error": f"unknown file_id: {file_id!r}"}), 404
            result = _index_one_file(conn, config, file_row, force=force)
        finally:
            conn.close()
        return jsonify({"enabled": True, **result})

    # -- route table -----------------------------------------------------------------

    bp.add_url_rule("/status", "status", status, methods=["GET"])
    bp.add_url_rule("/duplicates/<file_id>", "duplicates", _wrap(duplicates), methods=["GET"])
    bp.add_url_rule("/similar/<file_id>", "similar", _wrap(similar), methods=["GET"])
    bp.add_url_rule("/faces/<file_id>", "faces_for_file", _wrap(faces_for_file), methods=["GET"])
    bp.add_url_rule("/faces/clusters", "faces_clusters", _wrap(faces_clusters), methods=["GET"])
    bp.add_url_rule(
        "/faces/clusters/<int:cluster_id>", "faces_cluster_detail",
        _wrap(faces_cluster_detail), methods=["GET"],
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
