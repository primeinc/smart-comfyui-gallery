"""Background indexing worker.

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
from collections import deque
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
from smartgallery_ai import provision as provisioning

__all__ = ["AIWorker", "load_source_image", "provision_groups_for"]

_logger = logging.getLogger(__name__)

# Mirrors invalidation._MTIME_EPSILON: mtimes round-trip through SQLite REAL
# columns, so an exact float comparison is too strict.
_MTIME_EPSILON = 1e-6

# File types a frame can actually be rendered from, for embeddings/faces/
# review -- audio and documents have no visual content, so we never queue
# them for those stages (they would never succeed and would just churn the
# per-cycle budget forever).
_VISUAL_TYPES = tuple(hashing.IMAGE_FILE_TYPES | hashing.VIDEO_FILE_TYPES)

# Upper bound on frames decoded while hunting for the first usable one, so
# a corrupt or all-empty video cannot stall a worker cycle indefinitely.
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


# Backend selector -> (accepted selector values, provision group). A backend
# participates in auto-provisioning only when its selector would actually
# load the real model ("auto" or the explicit real-backend name).
_PROVISION_MAP = (
    ("semantic_backend", ("auto", "open_clip"), "semantic"),
    ("visual_backend", ("auto", "dinov2"), "visual"),
    ("face_backend", ("auto", "opencv"), "faces"),
    ("segmenter_backend", ("auto", "mobilesam"), "segmenter"),
    ("critic_backend", ("auto", "qwen-vl"), "critic"),
)


def provision_groups_for(config: AIConfig) -> list:
    """Provision groups the configured backends would use but which cannot
    load right now — weights missing from `config.models_dir`, runtime
    packages not importable, or a CPU-build torch that CUDA hardware wants
    swapped (auto-provisioning fixes all three). The qwen-vl critic
    additionally needs the semantic (grounding-gate) stack."""
    wanted: list = []
    for attr, accepted, group in _PROVISION_MAP:
        if getattr(config, attr) in accepted:
            wanted.append(group)
            if group == "critic" and "semantic" not in wanted:
                wanted.append("semantic")
    if not wanted:
        return []
    missing = []
    for group in provisioning.resolve_groups(wanted):
        weights_missing = any(
            not provisioning.artifact_present(config.models_dir, a)
            for a in group.artifacts)
        needs_cuda_swap = (
            any(req == "torch" for _, req in group.runtime)
            and provisioning.torch_cuda_reinstall_needed())
        if weights_missing or needs_cuda_swap or provisioning.runtime_missing(group):
            missing.append(group.name)
    return missing


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """True when `table` has a column named `column`; a missing table reads as no columns."""
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def mark_faces_cluster_pending(conn: sqlite3.Connection, backend) -> None:
    """Set the persistent marker that makes the worker's next faces stage
    re-cluster even when it has no new scan candidates. The synchronous
    /index path stores faces OUTSIDE the worker's scan loop; without this
    marker those faces would stay unclustered until some other file's
    face scan happened to trigger clustering."""
    AIWorker._set_state(
        conn,
        f"faces_cluster_pending:{backend.model_id}:{backend.model_version}",
        "1")


def record_scan(conn: sqlite3.Connection, file_id: str, kind: str, backend,
                source_mtime: float, now: float, result_count: int) -> None:
    """Upsert the single (file, kind) scan-log row recording an attempt at
    (model, mtime); `result_count` of -1 marks a failed attempt. Shared by
    the worker stages and the synchronous /index path so both mark work the
    same way and neither re-scans the other's results."""
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


def indexing_totals(conn: sqlite3.Connection) -> dict:
    """Backlog progress snapshot: how many files exist vs. how many each
    stage has covered so far. Approximate BY DESIGN (a model-version bump
    re-queues files without resetting these counters) — meant for progress
    display in /status, the panel, and the cycle log, never scheduling."""
    type_placeholders = ",".join("?" for _ in _VISUAL_TYPES)
    def one(sql, params=()):
        return conn.execute(sql, params).fetchone()[0]
    return {
        "files_total": one("SELECT COUNT(*) FROM files"),
        "visual_files_total": one(
            f"SELECT COUNT(*) FROM files WHERE type IN ({type_placeholders})",
            _VISUAL_TYPES),
        "hashed": one("SELECT COUNT(*) FROM ai_file_hashes"),
        "embeddings_semantic": one(
            "SELECT COUNT(*) FROM ai_embeddings WHERE space = ?", (SPACE_SEMANTIC,)),
        "embeddings_visual": one(
            "SELECT COUNT(*) FROM ai_embeddings WHERE space = ?", (SPACE_VISUAL,)),
        "faces_scanned": one(
            "SELECT COUNT(*) FROM ai_scan_log WHERE kind = 'faces'"),
        "reviews_scanned": one(
            "SELECT COUNT(*) FROM ai_scan_log WHERE kind = 'review'"),
    }


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
        """`poll_interval` is the sleep between wake cycles in seconds; `batch_size`
        is the per-cycle file budget shared across all stages combined."""
        self.config = config
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.batch_size = batch_size

        # Cumulative counters since construction; exposed verbatim by /status.
        self.stats = {
            "cycles": 0,
            "hashed": 0,
            "embedded": 0,
            "faces_indexed": 0,
            "reviewed": 0,
            "errors": 0,
        }
        # First-occurrence error messages (newest last), bounded; the
        # status page shows these so failures are visible without shell
        # access to the server log.
        self.recent_errors: deque = deque(maxlen=20)

        self._lock = threading.Lock()
        self._logged_errors: set = set()  # error keys already logged (log-once dedup)
        self._backend_cache: dict = {}  # key -> backend instance, or None for a cached miss
        # monotonic time of the last failed resolution attempt per key;
        # unavailable backends are re-probed after _backend_retry_seconds
        # so provisioning weights later activates them without a restart.
        self._backend_failed_at: dict = {}
        self._backend_retry_seconds = 300.0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # On-demand indexing: file ids a user is actively looking at (the
        # AI panel requests them) jump the queue. Deduped FIFO, bounded so
        # a request flood cannot grow it without limit; _wake_event breaks
        # the between-cycle sleep so a request is picked up immediately.
        self._priority_ids: list = []
        self._priority_max = 100
        self._wake_event = threading.Event()
        # True after a cycle that exhausted a budget: the loop skips its
        # between-cycle sleep and continues the crawl immediately.
        self._backlog_remaining = False
        # True while auto-provisioning intends to (or is about to) swap a
        # CPU-build torch for CUDA wheels: torch-dependent backends must
        # not import torch in the meantime -- an imported torch pins its
        # files and would force a second restart to finish the swap.
        self._hold_torch_backends = False

        # Background weight-provisioning: one attempt per worker lifetime,
        # in its own daemon thread so cycles are never blocked by
        # downloads. state: 'idle' | 'downloading' | 'done' | 'failed: ...'
        # | 'disabled'; groups: the missing groups being (or last) fetched.
        self.provision_state: dict = {"state": "idle", "groups": []}
        self._provision_thread: Optional[threading.Thread] = None
        self._provision_next_log_pct = 10  # byte-progress log throttle

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background thread (and, when enabled, one async
        weight-provisioning attempt). No-op if already running."""
        if self.is_running:
            return
        # The host app never configures the logging module, which leaves
        # INFO invisible (no handler, WARNING-level last resort). Attach one
        # plain stream handler to the package logger so worker/provisioning
        # progress reaches the console — unless the app or user configured
        # logging themselves, which always wins.
        pkg_logger = logging.getLogger("smartgallery_ai")
        if not pkg_logger.handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            pkg_logger.addHandler(handler)
            pkg_logger.setLevel(logging.INFO)
        self._stop_event.clear()
        # Boot GPU inventory: what the machine has and which wheels it
        # gets, before any provisioning/backends run. Silence = no GPU.
        try:
            gpu = provisioning.cuda_summary()
        except Exception:  # noqa: BLE001 - inventory is best-effort
            gpu = None
        if gpu is not None:
            for idx, card in enumerate(gpu.get("gpus") or []):
                _logger.info(
                    "[AI] GPU%d: %s (compute capability %s, %s)",
                    idx, card.get("name") or "unknown NVIDIA device",
                    card.get("compute_capability"), card.get("vram") or "?")
            _logger.info(
                "[AI] driver %s (CUDA %s) -> torch wheels %s; device rule: "
                "most VRAM, newest generation on ties (AI_DAM_DEVICE=cuda:N "
                "overrides)",
                gpu.get("driver") or "?", gpu.get("driver_cuda") or "?",
                (gpu.get("torch_index") or "").rsplit("/", 1)[-1] or "?")
        else:
            _logger.info("[AI] no NVIDIA GPU detected — CPU wheels/devices")
        self._maybe_start_auto_provision()
        self._thread = threading.Thread(target=self._run_loop, name="AIWorker", daemon=True)
        self._thread.start()

    # -- background weight provisioning --------------------------------------

    def _maybe_start_auto_provision(self) -> None:
        """Spawn the one-shot provisioning thread when auto-provisioning is
        enabled and any configured backend's weights are missing. Never
        blocks: downloads run in a daemon thread, and cycles proceed with
        whatever backends already resolve."""
        if not self.config.auto_provision:
            self.provision_state = {"state": "disabled", "groups": []}
            return
        if self._provision_thread is not None:
            return
        try:
            missing = provision_groups_for(self.config)
        except Exception as exc:  # noqa: BLE001 - startup must not fail on this
            self._note_error("provision:plan", f"auto-provision planning failed: {exc}")
            return
        if not missing:
            self.provision_state = {"state": "done", "groups": []}
            return
        # A planned CUDA swap uninstalls torch: hold every torch-dependent
        # backend un-imported until provisioning finishes, or the crawl
        # would import (and pin) the CPU build first and the swap would
        # need a second restart. Set BEFORE the cycle thread exists.
        try:
            self._hold_torch_backends = provisioning.torch_cuda_reinstall_needed()
        except Exception:  # noqa: BLE001 - detection is best-effort
            self._hold_torch_backends = False
        self.provision_state = {"state": "downloading", "groups": list(missing)}
        self._provision_thread = threading.Thread(
            target=self._provision_worker, args=(list(missing),),
            name="AIWorkerProvision", daemon=True)
        self._provision_thread.start()

    def _on_provision_event(self, event: dict) -> None:
        """Fold one provisioning progress event into `provision_state`
        (served live by /status) and into the visible log — byte events
        throttled to every 10% so a 5 GB file logs ~10 lines, not 5000.
        The state dict is replaced, never mutated: /status snapshots it
        from another thread."""
        state = dict(self.provision_state)
        state["done"] = list(state.get("done", []))
        item = event.get("item", "")
        if event["phase"] == "start":
            state["current"] = item
            state["detail"] = ("installing package" if event["kind"] == "runtime"
                              else f"downloading ({event.get('size', '?')})")
            self._provision_next_log_pct = 10
            _logger.info("[AIWorker] provisioning %s: %s", state["detail"], item)
        elif event["phase"] == "bytes":
            done, total = event["bytes_done"], event.get("bytes_total")
            if total:
                pct = int(done * 100 / total)
                state["detail"] = (f"{done / 1e6:.1f} MB / {total / 1e6:.1f} MB "
                                   f"({pct}%)")
                if pct >= self._provision_next_log_pct:
                    self._provision_next_log_pct = pct + 10
                    _logger.info("[AIWorker] %s: %s", item, state["detail"])
            else:
                state["detail"] = f"{done / 1e6:.1f} MB"
        elif event["phase"] == "done":
            state["done"].append(item)
            state["current"] = None
            state["detail"] = None
            _logger.info("[AIWorker] provisioned: %s", item)
        self.provision_state = state

    def _provision_worker(self, groups: list) -> None:
        """Thread body: install the missing groups' runtime packages and
        download their weights, then force an immediate backend re-probe so
        everything activates in this process without a restart. Network
        failure (e.g. an egress-denied host) leaves the layer degraded
        exactly as if nothing had been provisioned."""
        _logger.info("[AIWorker] auto-provisioning missing capability "
                     "group(s): %s (set AI_DAM_AUTO_PROVISION=false to opt out)",
                     ", ".join(groups))
        try:
            result = provisioning.provision(
                self.config.models_dir, groups,
                log=lambda msg: _logger.info("[AIWorker] provision %s", msg),
                progress=self._on_provision_event)
            self.provision_state = {
                "state": "done", "groups": list(groups),
                "done": self.provision_state.get("done", []),
            }
            _logger.info("[AIWorker] provisioning complete: %d installed, "
                         "%d downloaded, %d already present",
                         len(result["installed"]), len(result["downloaded"]),
                         len(result["skipped"]))
        except Exception as exc:  # noqa: BLE001 - downloads may fail; never fatal
            self.provision_state = {
                "state": f"failed: {exc}", "groups": list(groups),
                "done": self.provision_state.get("done", []),
            }
            self._note_error("provision:download", f"auto-provision failed: {exc}")
            return
        finally:
            # Success or failure, backends may resolve again (a held torch
            # import either finds the CUDA build now or fails cleanly into
            # the bounded re-probe).
            self._hold_torch_backends = False
        with self._lock:
            # Drop cached misses and their retry timestamps: the next cycle
            # re-resolves every backend against the freshly landed weights.
            for key in [k for k, v in self._backend_cache.items() if v is None]:
                self._backend_cache.pop(key, None)
            self._backend_failed_at.clear()
        try:
            from smartgallery_ai import service as _service
            _service.invalidate_backend_probe_cache()
        except Exception:  # noqa: BLE001 - status cache refresh is best-effort
            pass

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the thread to stop and join it. Safe to call repeatedly."""
        self._stop_event.set()
        self._wake_event.set()  # break the between-cycle sleep immediately
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if not thread.is_alive():
                self._thread = None

    # -- on-demand indexing ------------------------------------------------------

    def request_priority_index(self, file_id: str) -> bool:
        """Queue one file for immediate indexing ahead of the backlog crawl
        and wake the worker. Duplicate requests collapse; returns False
        (not queued) only when the bounded queue is full."""
        with self._lock:
            if file_id in self._priority_ids:
                self._wake_event.set()
                return True
            if len(self._priority_ids) >= self._priority_max:
                return False
            self._priority_ids.append(file_id)
        self._wake_event.set()
        return True

    def _drain_priority(self) -> list:
        """Take (and clear) the queued priority file ids, oldest request first."""
        with self._lock:
            ids, self._priority_ids = self._priority_ids, []
        return ids

    # -- main loop -------------------------------------------------------------

    def _run_loop(self) -> None:
        """Thread body: run cycles until stopped; a failing cycle is counted, never fatal."""
        while not self._stop_event.is_set():
            self._backlog_remaining = False
            try:
                self._run_cycle()
            except Exception:
                _logger.exception("[AIWorker] cycle failed")
                with self._lock:
                    self.stats["errors"] += 1
            # A cycle that exhausted its budget almost certainly left work
            # behind: start the next one immediately. Sleeping the full
            # poll interval between full batches turns a first index of a
            # large gallery into hours of idle waiting. Otherwise sleep
            # until the interval elapses OR a priority-index request
            # arrives (request_priority_index sets the event).
            if self._backlog_remaining:
                continue
            self._wake_event.wait(self.poll_interval)
            self._wake_event.clear()

    def _run_cycle(self) -> None:
        """One wake: fresh connection, schema ensured, user-requested files
        first, then stages run in fixed order — hashing against its own
        budget (it costs milliseconds per file; sharing a budget with the
        model stages would starve them for hours on a large hash backlog),
        the model stages against the shared file budget — then the
        orphaned-mask sweep."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        with self._lock:
            stats_before = dict(self.stats)
        try:
            schema.init_schema(conn)

            self._process_priority_requests(conn)

            hashed = self._process_hashes(conn, self.batch_size)

            budget = self.batch_size

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
            self._log_cycle_progress(conn, stats_before)
            # Exhausted hash budget or exhausted stage budget both mean the
            # backlog continues; the loop skips its sleep and keeps going.
            self._backlog_remaining = (
                self.batch_size > 0
                and (hashed >= self.batch_size or budget <= 0))
        finally:
            conn.close()

        with self._lock:
            self.stats["cycles"] += 1

    def _process_priority_requests(self, conn: sqlite3.Connection) -> None:
        """Fully index the user-requested files NOW, every stage whose
        backend is up, outside the shared cycle budget: the AI panel is
        open on these files and waiting for results."""
        for file_id in self._drain_priority():
            self._process_hashes(conn, 1, only_file_id=file_id)
            semantic = self._backend("semantic", embedders.get_semantic_backend)
            if semantic is not None:
                self._process_embedding_space(conn, semantic, SPACE_SEMANTIC, 1,
                                              only_file_id=file_id)
            visual = self._backend("visual", embedders.get_visual_backend)
            if visual is not None:
                self._process_embedding_space(conn, visual, SPACE_VISUAL, 1,
                                              only_file_id=file_id)
            face_backend = self._backend("face", faces.get_face_backend)
            if face_backend is not None:
                self._process_faces(conn, face_backend, 1, only_file_id=file_id)
            critic = self._backend("critic", review.get_critic_backend)
            if critic is not None:
                self._process_reviews(conn, critic, 1, only_file_id=file_id)

    def _log_cycle_progress(self, conn: sqlite3.Connection, stats_before: dict) -> None:
        """One INFO line per cycle that did work — what was indexed and how
        far the gallery backlog has progressed — so a long first index is
        visibly alive in the console. Idle cycles stay silent."""
        with self._lock:
            deltas = {key: self.stats[key] - stats_before.get(key, 0)
                      for key in ("hashed", "embedded", "faces_indexed", "reviewed")}
        if not any(deltas.values()):
            return
        totals = indexing_totals(conn)
        _logger.info(
            "[AIWorker] indexed: +%d hashed, +%d embedded, +%d faces, +%d reviews "
            "(gallery: %d/%d hashed, %d/%d semantic, %d/%d visual, %d/%d faces)",
            deltas["hashed"], deltas["embedded"], deltas["faces_indexed"],
            deltas["reviewed"],
            totals["hashed"], totals["files_total"],
            totals["embeddings_semantic"], totals["visual_files_total"],
            totals["embeddings_visual"], totals["visual_files_total"],
            totals["faces_scanned"], totals["visual_files_total"],
        )

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
        # While a CUDA swap is pending, torch-dependent backends stay
        # unresolved (importing torch now would pin the CPU build mid-swap
        # and force a restart). Faces are cv2-only and keep working.
        if self._hold_torch_backends and key != "face":
            return None
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
        """Count an error in stats; log and remember the message only on
        `key`'s first occurrence (recent_errors feeds the status page)."""
        with self._lock:
            self.stats["errors"] += 1
        if key not in self._logged_errors:
            self._logged_errors.add(key)
            self.recent_errors.append({"at": time.time(), "message": message})
            _logger.warning("[AIWorker] %s", message)

    # -- stages ------------------------------------------------------------------

    def _process_hashes(self, conn: sqlite3.Connection, limit: int,
                        only_file_id: Optional[str] = None) -> int:
        """Hash stage: (re)compute content hashes for missing/stale files. Returns
        candidates consumed -- charged against the budget even when hashing fails.
        `only_file_id` restricts the stage to that file (priority requests)."""
        if limit <= 0:
            return 0
        missing = invalidation.find_missing(conn, "ai_file_hashes")
        stale = invalidation.find_stale_hashes(conn, HASH_ALGO_VERSION)
        wanted = set(missing) | set(stale)
        if only_file_id is not None:
            wanted &= {only_file_id}
        candidates = _fetch_candidates(conn, wanted, limit)
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
        self, conn: sqlite3.Connection, backend, space: str, limit: int,
        only_file_id: Optional[str] = None,
    ) -> int:
        """Embedding stage for one space: embed missing/stale renderable files.
        Returns candidates consumed, successful or not. `only_file_id`
        restricts the stage to that file (priority requests)."""
        if limit <= 0:
            return 0
        missing = invalidation.find_missing(conn, "ai_embeddings", space=space)
        stale = invalidation.find_stale_embeddings(conn, space, backend.model_id, backend.model_version)
        wanted = set(missing) | set(stale)
        if only_file_id is not None:
            wanted &= {only_file_id}
        candidates = _fetch_candidates(
            conn, wanted, limit, allowed_types=_VISUAL_TYPES
        )
        store = vectors.VectorStore(cache_dir=self.config.cache_dir, ephemeral=self.config.ephemeral_index)
        for row in candidates:
            # A user is waiting on priority files; serve them between items
            # so panel requests never queue behind a long crawl batch.
            if only_file_id is None:
                self._process_priority_requests(conn)
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
                         limit: int, extra_cols: str = "",
                         only_file_id: Optional[str] = None) -> list:
        """Files needing a (re-)scan for `kind`: no ai_scan_log row for the
        current model at the current source mtime. Zero-result scans are
        logged too, so a file with no faces is scanned exactly once per
        (model, mtime) instead of every cycle. `only_file_id` restricts the
        scan to that file (priority requests)."""
        type_placeholders = ",".join("?" for _ in _VISUAL_TYPES)
        only_clause = "AND f.id = ?" if only_file_id is not None else ""
        only_params = (only_file_id,) if only_file_id is not None else ()
        return conn.execute(
            f"""
            SELECT f.id, f.path, f.mtime, f.type{extra_cols} FROM files f
            WHERE f.type IN ({type_placeholders})
              {only_clause}
              AND NOT EXISTS (
                SELECT 1 FROM ai_scan_log sl
                WHERE sl.file_id = f.id AND sl.kind = ?
                  AND sl.model_id = ? AND sl.model_version = ?
                  AND ABS(sl.source_mtime - f.mtime) <= ?
              )
            ORDER BY f.mtime DESC, f.id ASC
            LIMIT ?
            """,
            (*_VISUAL_TYPES, *only_params, kind, backend.model_id,
             backend.model_version, _MTIME_EPSILON, limit),
        ).fetchall()

    @staticmethod
    def _log_scan(conn: sqlite3.Connection, file_id: str, kind: str, backend,
                  source_mtime: float, now: float, result_count: int) -> None:
        """Upsert the single (file, kind) scan-log row recording an attempt at
        (model, mtime); `result_count` of -1 marks a failed attempt."""
        record_scan(conn, file_id, kind, backend, source_mtime, now, result_count)

    def _process_faces(self, conn: sqlite3.Connection, backend, limit: int,
                       only_file_id: Optional[str] = None) -> int:
        """Face stage: detect and store faces per candidate, then recluster when
        faces were indexed or a clustering attempt is still pending. Returns
        candidates consumed, successful or not."""
        if limit <= 0:
            return 0
        rows = self._scan_candidates(conn, "faces", backend, limit,
                                     only_file_id=only_file_id)
        now = time.time()
        for row in rows:
            if only_file_id is None:
                self._process_priority_requests(conn)
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
        """Upsert one key of the persistent worker state, committed immediately."""
        conn.execute(
            "INSERT INTO ai_dam_state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, time.time()))
        conn.commit()

    @staticmethod
    def _get_state(conn: sqlite3.Connection, key: str):
        """Stored value for `key`, or None when unset."""
        row = conn.execute(
            "SELECT value FROM ai_dam_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row is not None else None

    @staticmethod
    def _clear_state(conn: sqlite3.Connection, key: str) -> None:
        """Remove `key` from the persistent state; absent keys are a no-op."""
        conn.execute("DELETE FROM ai_dam_state WHERE key = ?", (key,))
        conn.commit()

    def _process_reviews(self, conn: sqlite3.Connection, backend, limit: int,
                         only_file_id: Optional[str] = None) -> int:
        """Review stage: run the critic per candidate, store the review, and
        generate finding masks when a segmenter is available. Returns candidates
        consumed, successful or not."""
        if limit <= 0:
            return 0
        prompt_expr = ", f.workflow_prompt" if _has_column(conn, "files", "workflow_prompt") else ", NULL AS workflow_prompt"
        rows = self._scan_candidates(conn, "review", backend, limit,
                                     extra_cols=prompt_expr,
                                     only_file_id=only_file_id)
        segmenter = self._backend("segmenter", review.get_segmenter_backend)
        now = time.time()
        for row in rows:
            if only_file_id is None:
                self._process_priority_requests(conn)
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
