"""Background indexing worker (WI-31 wave 2).

`AIWorker` is a daemon thread, separate from Flask request handling, that
periodically catches derived AI DAM state up with the source-of-truth
`files` table: it computes content hashes, embeddings, faces, and reviews
for files that are missing them or whose derived rows are stale (per
`invalidation.py`). Every cycle opens its own SQLite connection -- never
shared with the Flask request handlers in `service.py` or with any other
cycle -- and is capped to a bounded batch of files so a cycle never runs
long enough to make the worker unresponsive to `stop()`.

The worker never raises out of its thread loop: every per-file failure
(unreadable media, a backend raising) is caught, counted in `.stats`, and
logged once per file so a bad file cannot spam the log every cycle.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from typing import Optional

import cv2
from PIL import Image

from smartgallery_ai import (
    AIConfig,
    HASH_ALGO_VERSION,
    RUBRIC_VERSION,
    SPACE_SEMANTIC,
    SPACE_VISUAL,
)
from smartgallery_ai import embedders, faces, hashing, invalidation, review, schema, vectors

__all__ = ["AIWorker", "load_source_image"]

_logger = logging.getLogger(__name__)

# Mirrors invalidation._MTIME_EPSILON: mtimes round-trip through SQLite REAL
# columns, so an exact float comparison is too strict.
_MTIME_EPSILON = 1e-6

# File types a frame can actually be rendered from, for embeddings/faces/
# review -- audio and documents have no visual content, so we never queue
# them for those stages (they would never succeed and would just churn the
# per-cycle budget forever).
_VISUAL_TYPES = tuple(hashing.IMAGE_FILE_TYPES | hashing.VIDEO_FILE_TYPES)

_MAX_VIDEO_FRAME_ATTEMPTS = 60


def _first_video_frame(path: str) -> Optional[Image.Image]:
    """First decodable video frame as a PIL image, or None. Never raises."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        for _ in range(_MAX_VIDEO_FRAME_ATTEMPTS):
            ok, frame = cap.read()
            if not ok:
                return None
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
        return None
    finally:
        cap.release()


def load_source_image(path: str, file_type: str) -> Optional[Image.Image]:
    """Load a read-only PIL frame for embedding/face/review backends.

    Images are opened directly via PIL; video uses its first decodable frame
    (mirrors `hashing.compute_hashes_for_file`'s frame-selection rule).
    Returns None -- never raises -- for missing/unreadable files or types
    with no visual frame (audio/document); callers treat None as "skip".
    Read-only: the source path is only ever opened, never written to.
    """
    if file_type in hashing.IMAGE_FILE_TYPES:
        try:
            with Image.open(path) as img:
                return img.copy()
        except Exception:
            return None
    if file_type in hashing.VIDEO_FILE_TYPES:
        try:
            return _first_video_frame(path)
        except Exception:
            return None
    return None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def _fetch_candidates(
    conn: sqlite3.Connection,
    file_ids,
    limit: int,
    allowed_types: Optional[tuple] = None,
) -> list:
    """`files` rows for `file_ids`, newest-mtime first, capped at `limit`.

    Optionally restricted to `allowed_types` (used by stages that need a
    renderable frame, to avoid endlessly re-queuing audio/document files).
    """
    ids = list(file_ids)
    if not ids or limit <= 0:
        return []
    # Chunk the IN list: on a first-time index of a large gallery the
    # staleness helpers can return every file id, and one bound variable
    # per id blows SQLite's variable limit (999 on older builds). Query in
    # bounded chunks, then merge-sort and re-apply the limit.
    CHUNK = 500
    rows: list = []
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start:start + CHUNK]
        id_placeholders = ",".join("?" for _ in chunk)
        query = f"SELECT id, path, mtime, type FROM files WHERE id IN ({id_placeholders})"
        params: list = list(chunk)
        if allowed_types is not None:
            type_placeholders = ",".join("?" for _ in allowed_types)
            query += f" AND type IN ({type_placeholders})"
            params.extend(allowed_types)
        query += " ORDER BY mtime DESC, id ASC LIMIT ?"
        params.append(limit)
        rows.extend(conn.execute(query, params).fetchall())
    rows.sort(key=lambda r: (-r["mtime"], r["id"]))
    return rows[:limit]


class AIWorker:
    """Daemon-thread background indexer for the AI DAM layer.

    Each wake cycle opens its own SQLite connection, ensures the AI DAM
    schema exists, and works through hashing -> embeddings (semantic,
    visual) -> faces -> review, each stage drawing from a shared per-cycle
    file budget (`batch_size`) so a cycle stays short even on a large,
    mostly-stale library. Stages whose backend is unavailable (`config`
    resolves it to None) are skipped entirely for that cycle.
    """

    def __init__(self, config: AIConfig, db_path: str, poll_interval: float = 20.0,
                 batch_size: int = 50):
        self.config = config
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.batch_size = batch_size

        self.stats = {
            "cycles": 0,
            "hashed": 0,
            "embedded": 0,
            "faces_indexed": 0,
            "reviewed": 0,
            "errors": 0,
        }

        self._lock = threading.Lock()
        self._logged_errors: set = set()
        self._backend_cache: dict = {}
        # monotonic time of the last failed resolution attempt per key;
        # unavailable backends are re-probed after _backend_retry_seconds
        # so provisioning weights later activates them without a restart.
        self._backend_failed_at: dict = {}
        self._backend_retry_seconds = 300.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background thread. No-op if already running."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="AIWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the thread to stop and join it. Safe to call repeatedly."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if not thread.is_alive():
                self._thread = None

    # -- main loop -------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception:
                _logger.exception("[AIWorker] cycle failed")
                with self._lock:
                    self.stats["errors"] += 1
            self._stop_event.wait(self.poll_interval)

    def _run_cycle(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            schema.init_schema(conn)

            budget = self.batch_size
            budget -= self._process_hashes(conn, budget)

            if budget > 0:
                semantic_backend = self._backend("semantic",
                                                 embedders.get_semantic_backend)
                if semantic_backend is not None:
                    budget -= self._process_embedding_space(
                        conn, semantic_backend, SPACE_SEMANTIC, budget
                    )

            if budget > 0:
                visual_backend = self._backend("visual",
                                               embedders.get_visual_backend)
                if visual_backend is not None:
                    budget -= self._process_embedding_space(
                        conn, visual_backend, SPACE_VISUAL, budget
                    )

            if budget > 0:
                face_backend = self._backend("face", faces.get_face_backend)
                if face_backend is not None:
                    budget -= self._process_faces(conn, face_backend, budget)

            if budget > 0:
                critic_backend = self._backend("critic",
                                               review.get_critic_backend)
                if critic_backend is not None:
                    budget -= self._process_reviews(conn, critic_backend, budget)

            if budget > 0:
                segmenter = self._backend("segmenter",
                                          review.get_segmenter_backend)
                if segmenter is not None:
                    budget -= self._process_masks(conn, segmenter, budget)

            self._sweep_orphaned_masks(conn)
        finally:
            conn.close()

        with self._lock:
            self.stats["cycles"] += 1

    def _sweep_orphaned_masks(self, conn: sqlite3.Connection) -> None:
        """Deleting a `files` row cascades away its findings rows but not
        their mask PNGs on disk. Sweep the derived masks cache each cycle:
        any per-file mask directory whose file id no longer exists in the
        DB is removed. Cheap (one dir listing), keeps the derived cache
        from leaking on file deletions."""
        masks_root = os.path.join(self.config.cache_dir, "masks")
        try:
            entries = os.listdir(masks_root)
        except OSError:
            return
        if not entries:
            return
        existing = {r[0] for r in conn.execute("SELECT id FROM files")}
        for entry in entries:
            if entry in existing:
                continue
            target = os.path.join(masks_root, entry)
            try:
                shutil.rmtree(target)
            except OSError as exc:
                self._note_error(f"sweep:{entry}",
                                 f"mask sweep: could not remove {target}: {exc}")

    # -- backend caching ---------------------------------------------------------

    def _backend(self, key: str, resolver):
        """Resolve a backend and reuse the instance. Constructing real
        backends can load multi-GB models; doing that per poll cycle (the
        resolver's natural behavior) is unaffordable, so a successful
        instance is kept for the worker's lifetime. An UNAVAILABLE result
        (None or a raising resolver) is cached only for
        `_backend_retry_seconds`: weights provisioned while the worker runs
        must eventually activate the backend without a process restart.

        Resolution must stay OUTSIDE self._lock: _note_error acquires the
        same non-reentrant lock. Only the worker thread resolves backends,
        so the unlocked window cannot double-load."""
        now = time.monotonic()
        with self._lock:
            if key in self._backend_cache:
                cached = self._backend_cache[key]
                if cached is not None:
                    return cached
                if now - self._backend_failed_at.get(key, 0.0) < self._backend_retry_seconds:
                    return None
        try:
            backend = resolver(self.config)
        except Exception as exc:  # noqa: BLE001 - resolution must not kill the cycle
            self._note_error(f"backend:{key}", f"backend {key}: {exc}")
            backend = None
        with self._lock:
            self._backend_cache[key] = backend
            if backend is None:
                self._backend_failed_at[key] = now
            else:
                self._backend_failed_at.pop(key, None)
            return backend

    # -- error bookkeeping -------------------------------------------------------

    def _note_error(self, key: str, message: str) -> None:
        with self._lock:
            self.stats["errors"] += 1
        if key not in self._logged_errors:
            self._logged_errors.add(key)
            _logger.warning("[AIWorker] %s", message)

    # -- stages ------------------------------------------------------------------

    def _process_hashes(self, conn: sqlite3.Connection, limit: int) -> int:
        if limit <= 0:
            return 0
        missing = invalidation.find_missing(conn, "ai_file_hashes")
        stale = invalidation.find_stale_hashes(conn, HASH_ALGO_VERSION)
        candidates = _fetch_candidates(conn, set(missing) | set(stale), limit)
        now = time.time()
        for row in candidates:
            file_id, path, mtime, file_type = row["id"], row["path"], row["mtime"], row["type"]
            try:
                result = hashing.compute_hashes_for_file(path, file_type)
            except Exception as exc:
                self._note_error(f"hash:{file_id}", f"hash: could not read {path}: {exc}")
                continue
            hashing.upsert_hashes(conn, file_id, result, mtime, HASH_ALGO_VERSION, now)
            with self._lock:
                self.stats["hashed"] += 1
        return len(candidates)

    def _process_embedding_space(
        self, conn: sqlite3.Connection, backend, space: str, limit: int
    ) -> int:
        if limit <= 0:
            return 0
        missing = invalidation.find_missing(conn, "ai_embeddings", space=space)
        stale = invalidation.find_stale_embeddings(conn, space, backend.model_id, backend.model_version)
        candidates = _fetch_candidates(
            conn, set(missing) | set(stale), limit, allowed_types=_VISUAL_TYPES
        )
        store = vectors.VectorStore(cache_dir=self.config.cache_dir, ephemeral=self.config.ephemeral_index)
        for row in candidates:
            file_id, path, mtime, file_type = row["id"], row["path"], row["mtime"], row["type"]
            img = load_source_image(path, file_type)
            if img is None:
                self._note_error(f"embed:{space}:{file_id}", f"embed[{space}]: could not read {path}")
                continue
            try:
                vec = backend.embed_image(img)
                store.add(conn, file_id, space, backend.model_id, backend.model_version, vec, mtime)
            except Exception as exc:
                self._note_error(
                    f"embed:{space}:{file_id}", f"embed[{space}]: failed for {path}: {exc}"
                )
                continue
            with self._lock:
                self.stats["embedded"] += 1
        return len(candidates)

    def _scan_candidates(self, conn: sqlite3.Connection, kind: str, backend,
                         limit: int, extra_cols: str = "") -> list:
        """Files needing a (re-)scan for `kind`: no ai_scan_log row for the
        current model at the current source mtime. Zero-result scans are
        logged too, so a file with no faces is scanned exactly once per
        (model, mtime) instead of every cycle."""
        type_placeholders = ",".join("?" for _ in _VISUAL_TYPES)
        return conn.execute(
            f"""
            SELECT f.id, f.path, f.mtime, f.type{extra_cols} FROM files f
            WHERE f.type IN ({type_placeholders})
              AND NOT EXISTS (
                SELECT 1 FROM ai_scan_log sl
                WHERE sl.file_id = f.id AND sl.kind = ?
                  AND sl.model_id = ? AND sl.model_version = ?
                  AND ABS(sl.source_mtime - f.mtime) <= ?
              )
            ORDER BY f.mtime DESC, f.id ASC
            LIMIT ?
            """,
            (*_VISUAL_TYPES, kind, backend.model_id, backend.model_version,
             _MTIME_EPSILON, limit),
        ).fetchall()

    @staticmethod
    def _log_scan(conn: sqlite3.Connection, file_id: str, kind: str, backend,
                  source_mtime: float, now: float, result_count: int) -> None:
        conn.execute(
            """
            INSERT INTO ai_scan_log
                (file_id, kind, model_id, model_version, source_mtime,
                 scanned_at, result_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (file_id, kind) DO UPDATE SET
                model_id = excluded.model_id,
                model_version = excluded.model_version,
                source_mtime = excluded.source_mtime,
                scanned_at = excluded.scanned_at,
                result_count = excluded.result_count
            """,
            (file_id, kind, backend.model_id, backend.model_version,
             source_mtime, now, result_count),
        )
        conn.commit()

    def _process_faces(self, conn: sqlite3.Connection, backend, limit: int) -> int:
        if limit <= 0:
            return 0
        rows = self._scan_candidates(conn, "faces", backend, limit)
        now = time.time()
        for row in rows:
            file_id, path, mtime, file_type = row["id"], row["path"], row["mtime"], row["type"]
            img = load_source_image(path, file_type)
            if img is None:
                self._note_error(f"faces:{file_id}", f"faces: could not read {path}")
                continue
            try:
                detections = backend.detect(img)
                faces.replace_faces_for_file(
                    conn, file_id, detections, backend.model_id, backend.model_version, mtime, now
                )
                self._log_scan(conn, file_id, "faces", backend, mtime, now, len(detections))
            except Exception as exc:
                self._note_error(f"faces:{file_id}", f"faces: failed for {path}: {exc}")
                continue
            with self._lock:
                self.stats["faces_indexed"] += 1
        # Recluster when this cycle indexed faces OR an earlier clustering
        # attempt is still pending. The pending marker is persisted BEFORE
        # the attempt (face scans are already committed by this point, so
        # without it a single clustering failure would never be retried:
        # the next cycle would see no face candidates and skip this block).
        pending_key = f"faces_cluster_pending:{backend.model_id}:{backend.model_version}"
        if rows:
            self._set_state(conn, pending_key, "1")
        if rows or self._get_state(conn, pending_key) is not None:
            try:
                faces.cluster_faces(conn, backend.model_id, backend.model_version,
                                    self.config.face_cluster_threshold)
                self._clear_state(conn, pending_key)
            except Exception as exc:  # noqa: BLE001
                self._note_error("faces:cluster", f"face clustering failed: {exc}")
        return len(rows)

    # -- small persistent state (ai_dam_state) -----------------------------------

    @staticmethod
    def _set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO ai_dam_state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, time.time()))
        conn.commit()

    @staticmethod
    def _get_state(conn: sqlite3.Connection, key: str):
        row = conn.execute(
            "SELECT value FROM ai_dam_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row is not None else None

    @staticmethod
    def _clear_state(conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM ai_dam_state WHERE key = ?", (key,))
        conn.commit()

    def _process_reviews(self, conn: sqlite3.Connection, backend, limit: int) -> int:
        if limit <= 0:
            return 0
        prompt_expr = ", f.workflow_prompt" if _has_column(conn, "files", "workflow_prompt") else ", NULL AS workflow_prompt"
        rows = self._scan_candidates(conn, "review", backend, limit,
                                     extra_cols=prompt_expr)
        segmenter = self._backend("segmenter", review.get_segmenter_backend)
        now = time.time()
        for row in rows:
            file_id, path, mtime, file_type = row["id"], row["path"], row["mtime"], row["type"]
            img = load_source_image(path, file_type)
            if img is None:
                self._note_error(f"review:{file_id}", f"review: could not read {path}")
                continue
            try:
                payload = backend.review(img, row["workflow_prompt"], RUBRIC_VERSION)
                result = review.validate_review_payload(payload)
                review_id = review.store_review(
                    conn, file_id, result, backend.model_id, backend.model_version,
                    RUBRIC_VERSION, json.dumps(payload), mtime, now,
                )
                if segmenter is not None:
                    generated = self._generate_masks(conn, img, file_id,
                                                     review_id, segmenter)
                    self._log_masks_if_complete(conn, file_id, review_id,
                                                segmenter, mtime, now, generated)
                self._log_scan(conn, file_id, "review", backend, mtime, now,
                               len(result.findings))
            except Exception as exc:
                self._note_error(f"review:{file_id}", f"review: failed for {path}: {exc}")
                # Record the FAILED attempt too (result_count = -1): a
                # grounding rejection or malformed generation must not put a
                # ~200s VLM inference on infinite retry every cycle. The
                # file re-enters the queue when its mtime or the model
                # changes (normal staleness), or on rebuild.
                try:
                    self._log_scan(conn, file_id, "review", backend, mtime,
                                   now, -1)
                except Exception:  # noqa: BLE001
                    pass
                continue
            with self._lock:
                self.stats["reviewed"] += 1
        return len(rows)

    def _generate_masks(self, conn: sqlite3.Connection, img, file_id: str,
                        review_id: int, segmenter) -> int:
        """Segment every localizable finding of a review. Global findings
        never reach the segmenter (generate_finding_mask enforces it); a
        per-finding failure is logged, never fatal. Returns the number of
        masks successfully generated."""
        finding_ids = [r[0] for r in conn.execute(
            "SELECT finding_id FROM ai_review_findings "
            "WHERE review_id = ? AND localizable = 1 AND mask_path IS NULL",
            (review_id,)).fetchall()]
        generated = 0
        for finding_id in finding_ids:
            try:
                review.generate_finding_mask(
                    conn, self.config.cache_dir, img, file_id, finding_id, segmenter)
                generated += 1
            except Exception as exc:  # noqa: BLE001
                self._note_error(f"mask:{finding_id}",
                                 f"mask: failed for finding {finding_id}: {exc}")
        return generated

    def _process_masks(self, conn: sqlite3.Connection, segmenter, limit: int) -> int:
        """Standalone mask stage: covers reviews whose localizable findings
        still lack masks — because the segmenter was provisioned AFTER the
        review ran, or an earlier mask attempt failed. Its own
        segmenter-keyed ai_scan_log unit ('masks') makes the attempt
        recorded and retryable independently of the review row."""
        if limit <= 0:
            return 0
        rows = conn.execute(
            """
            SELECT DISTINCT f.id, f.path, f.mtime, f.type,
                   rv.review_id
            FROM files f
            JOIN ai_reviews rv ON rv.file_id = f.id
            JOIN ai_review_findings rf ON rf.review_id = rv.review_id
            WHERE rf.localizable = 1 AND rf.mask_path IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM ai_scan_log sl
                WHERE sl.file_id = f.id AND sl.kind = 'masks'
                  AND sl.model_id = ? AND sl.model_version = ?
                  AND ABS(sl.source_mtime - f.mtime) <= ?
              )
            ORDER BY f.mtime DESC, f.id ASC
            LIMIT ?
            """,
            (segmenter.model_id, segmenter.model_version, _MTIME_EPSILON, limit),
        ).fetchall()
        now = time.time()
        for row in rows:
            file_id, path, mtime = row["id"], row["path"], row["mtime"]
            img = load_source_image(path, row["type"])
            if img is None:
                self._note_error(f"mask:{file_id}", f"mask: could not read {path}")
                continue
            generated = self._generate_masks(conn, img, file_id,
                                             row["review_id"], segmenter)
            self._log_masks_if_complete(conn, file_id, row["review_id"],
                                        segmenter, mtime, now, generated)
        return len(rows)

    def _log_masks_if_complete(self, conn: sqlite3.Connection, file_id: str,
                               review_id: int, segmenter, mtime: float,
                               now: float, generated: int) -> None:
        """Record the mask scan ONLY when every localizable finding of the
        review has a mask. A partial failure (transient segmenter or
        filesystem error) leaves the file selectable so the next cycle
        retries the remaining findings, instead of a completion row
        freezing them mask-less until the next mtime/model change."""
        remaining = conn.execute(
            "SELECT COUNT(*) FROM ai_review_findings "
            "WHERE review_id = ? AND localizable = 1 AND mask_path IS NULL",
            (review_id,)).fetchone()[0]
        if remaining == 0:
            self._log_scan(conn, file_id, "masks", segmenter, mtime, now, generated)
