"""What the pipeline has and has not done to ONE file, and why.

Every AI stage answers the same four questions here: did it run, what did it
produce, if it did not run then why not, and what would make it run. The
per-file panel answers the first two for stages that succeeded; the parts an
operator actually gets stuck on -- a backend whose weights never landed, a
review frozen because one scan-log row says the file is current, a file whose
type no stage will ever touch -- were previously only visible by reading the
tables by hand.

The reason a stage is BLOCKED comes from `backends.why_unavailable`, which is
the backend's own message (which weights file was missing, which runtime
failed to import), not a generic flag. That is the whole point: "unavailable"
is not an answer an operator can act on.

This module only READS. Acting on what it reports is the existing
`POST /index/<file_id>` and `GET /review/run/<file_id>`; each stage names
which one applies so the page can offer it.
"""

from __future__ import annotations

import logging
import sqlite3

from smartgallery_ai import (
    HASH_ALGO_VERSION,
    RUBRIC_VERSION,
    SPACE_SEMANTIC,
    SPACE_VISUAL,
    AIConfig,
    backends,
    hashing,
    invalidation,
)

_logger = logging.getLogger(__name__)

# One vocabulary for every row on the page.
DONE = "done"  # ran, result is current
PARTIAL = "partial"  # ran, but only some of the unit completed
STALE = "stale"  # ran, but the file or the model has moved since
PENDING = "pending"  # has not run and nothing prevents it
BLOCKED = "blocked"  # cannot run until something changes
NA = "n/a"  # will never run for this file, and that is correct

# File types any pixel stage can render a frame from. Types outside this set
# are N/A rather than pending -- no stage will ever produce a result for them.
RENDERABLE_TYPES = tuple(hashing.IMAGE_FILE_TYPES | hashing.VIDEO_FILE_TYPES)

# stage key -> (registry kind, provisioning group). Stages absent from this
# map need no model at all and can never be BLOCKED on one. The group names
# match service.py's status probe, which passes them to
# `provisioning.resolve_groups`.
_STAGE_BACKEND = {
    "semantic": ("semantic", "semantic"),
    "visual": ("visual", "visual"),
    "faces": ("faces", "faces"),
    "masks": ("segmenter", "segmenter"),
}


def _provision_hint(group: str) -> str:
    return f"python -m smartgallery_ai provision {group}"


def _row(key, group, label, state, detail, **extra) -> dict:
    stage = {"key": key, "group": group, "label": label, "state": state, "detail": detail}
    stage.update(extra)
    return stage


class _Ctx:
    """Everything the stage probes share, resolved once per walkthrough."""

    def __init__(self, conn: sqlite3.Connection, config: AIConfig, file_row: sqlite3.Row, worker):
        self.conn = conn
        self.config = config
        self.file = file_row
        self.file_id = file_row["id"]
        self.mtime = file_row["mtime"]
        self.renderable = file_row["type"] in RENDERABLE_TYPES
        self.worker = worker
        self._reasons: dict = {}

    def unavailable(self, stage_key: str) -> str | None:
        """Why `stage_key`'s backend could not load, or None when it did.

        Resolving loads whatever is provisioned, which is why the cheap
        `/status` probe does not do this -- but an operator asking about one
        specific file has asked a question only a real load can answer.
        """
        if stage_key not in _STAGE_BACKEND:
            return None
        if stage_key not in self._reasons:
            kind, _group = _STAGE_BACKEND[stage_key]
            self._reasons[stage_key] = backends.why_unavailable(kind, self.config)
        return self._reasons[stage_key]

    def guard(self, stage_key: str, label: str, group: str) -> dict | None:
        """The row to report INSTEAD of running this stage's probe, or None
        when nothing stands in the way. Covers the three answers that are the
        same for every stage: the layer is off, the file has no renderable
        frame, or the stage's backend could not load."""
        if not self.config.enabled:
            return _row(
                stage_key,
                group,
                label,
                BLOCKED,
                "the AI layer is switched off",
                blocked_reason="ENABLE_AI_DAM is false",
                fix="set ENABLE_AI_DAM=true and restart",
            )
        if not self.renderable:
            return _row(
                stage_key,
                group,
                label,
                NA,
                f"no stage can render a frame from a '{self.file['type']}' file",
            )
        reason = self.unavailable(stage_key)
        if reason is not None:
            _kind, provision_group = _STAGE_BACKEND[stage_key]
            return _row(
                stage_key,
                group,
                label,
                BLOCKED,
                "the backend could not load",
                blocked_reason=reason,
                fix=_provision_hint(provision_group),
            )
        return None

    def scan(self, kind: str):
        """The most recent ai_scan_log row for this file and stage kind."""
        return self.conn.execute(
            "SELECT model_id, model_version, source_mtime, scanned_at, result_count, input_key "
            "FROM ai_scan_log WHERE file_id = ? AND kind = ? ORDER BY scanned_at DESC LIMIT 1",
            (self.file_id, kind),
        ).fetchone()


# --- ingest ---------------------------------------------------------------------


def _stage_indexed(ctx: _Ctx) -> dict:
    """The file is in the gallery's own table. Everything else depends on it."""
    return _row(
        "indexed",
        "ingest",
        "Indexed",
        DONE,
        f"{ctx.file['type'] or 'unknown type'}, mtime {ctx.mtime}",
        evidence={"path": ctx.file["path"], "size": ctx.file["size"]},
    )


def _stage_metadata(ctx: _Ctx) -> dict:
    """The generation prompt, which prompt-alignment and the review both need."""
    prompt = (ctx.file["workflow_prompt"] or "").strip()
    if not prompt:
        return _row(
            "metadata",
            "ingest",
            "Generation metadata",
            NA,
            "no workflow prompt was traced from this file",
            fix="re-scan the folder if the file should carry ComfyUI metadata",
        )
    return _row(
        "metadata",
        "ingest",
        "Generation metadata",
        DONE,
        f"workflow prompt traced, {len(prompt)} chars",
        evidence={"prompt_chars": len(prompt)},
    )


# --- ai stages ------------------------------------------------------------------


def _stage_hashes(ctx: _Ctx) -> dict:
    """Perceptual + content hashes. Pure compute -- no model, so the only
    thing that can stop it is the layer being off or the type having no
    renderable frame."""
    guard = ctx.guard("hashes", "Hashes", "ai")
    if guard is not None:
        return guard
    row = ctx.conn.execute(
        "SELECT sha256, phash64, dhash64, algo_version, source_mtime, computed_at "
        "FROM ai_file_hashes WHERE file_id = ?",
        (ctx.file_id,),
    ).fetchone()
    if row is None:
        return _row("hashes", "ai", "Hashes", PENDING, "not hashed yet", action="index")
    if invalidation.is_stale(row["source_mtime"], ctx.mtime, row["algo_version"], HASH_ALGO_VERSION):
        return _row(
            "hashes",
            "ai",
            "Hashes",
            STALE,
            f"hashed at algo {row['algo_version']}; current is {HASH_ALGO_VERSION}",
            action="index",
        )
    return _row(
        "hashes",
        "ai",
        "Hashes",
        DONE,
        "sha256 + phash64 + dhash64",
        evidence={"phash64": row["phash64"], "dhash64": row["dhash64"], "algo_version": row["algo_version"]},
    )


def _embedding_stage(ctx: _Ctx, key: str, space: str, label: str) -> dict:
    """One embedding space: present, stale against its model, or missing."""
    guard = ctx.guard(key, label, "ai")
    if guard is not None:
        return guard
    row = ctx.conn.execute(
        "SELECT model_id, model_version, dim, source_mtime, computed_at FROM ai_embeddings "
        "WHERE file_id = ? AND space = ?",
        (ctx.file_id, space),
    ).fetchone()
    if row is None:
        return _row(key, "ai", label, PENDING, "not embedded yet", action="index")
    stored = f"{row['model_id']}::{row['model_version']}"
    if invalidation.is_stale(row["source_mtime"], ctx.mtime, stored, stored):
        return _row(key, "ai", label, STALE, "the file changed after it was embedded", action="index")
    return _row(
        key,
        "ai",
        label,
        DONE,
        f"{row['dim']}-d vector from {row['model_id']}",
        evidence={"model_id": row["model_id"], "model_version": row["model_version"], "dim": row["dim"]},
    )


def _stage_semantic(ctx: _Ctx) -> dict:
    return _embedding_stage(ctx, "semantic", SPACE_SEMANTIC, "Semantic embedding")


def _stage_visual(ctx: _Ctx) -> dict:
    return _embedding_stage(ctx, "visual", SPACE_VISUAL, "Visual embedding")


def _stage_faces(ctx: _Ctx) -> dict:
    """Detection. A scan row with zero faces is DONE, not pending -- that
    distinction is the whole reason ai_scan_log exists."""
    guard = ctx.guard("faces", "Face detection", "ai")
    if guard is not None:
        return guard
    count = ctx.conn.execute("SELECT COUNT(*) FROM ai_face_instances WHERE file_id = ?", (ctx.file_id,)).fetchone()[0]
    scan = ctx.scan("faces")
    if scan is None:
        return _row("faces", "ai", "Face detection", PENDING, "no detector has looked at this file", action="index")
    if invalidation.is_stale(scan["source_mtime"], ctx.mtime, scan["model_version"], scan["model_version"]):
        return _row("faces", "ai", "Face detection", STALE, "the file changed after it was scanned", action="index")
    detail = f"{count} face{'' if count == 1 else 's'} detected" if count else "scanned, no faces found"
    return _row(
        "faces",
        "ai",
        "Face detection",
        DONE,
        detail,
        evidence={"faces": count, "model_id": scan["model_id"], "scanned_at": scan["scanned_at"]},
    )


def _stage_clustering(ctx: _Ctx) -> dict:
    """Whether this file's faces were assigned to identity clusters."""
    rows = ctx.conn.execute("SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (ctx.file_id,)).fetchall()
    if not rows:
        return _row("clustering", "ai", "Face clustering", NA, "this file has no detected faces")
    clustered = sum(1 for r in rows if r["cluster_id"] is not None)
    if clustered == 0:
        return _row(
            "clustering",
            "ai",
            "Face clustering",
            PENDING,
            f"0 of {len(rows)} faces clustered",
            fix="the worker clusters after its next faces scan, or POST /faces/recluster",
        )
    if clustered < len(rows):
        return _row(
            "clustering",
            "ai",
            "Face clustering",
            PARTIAL,
            f"{clustered} of {len(rows)} faces clustered; the rest sit below the similarity threshold",
            evidence={"clustered": clustered, "total": len(rows)},
        )
    return _row(
        "clustering",
        "ai",
        "Face clustering",
        DONE,
        f"all {clustered} faces clustered",
        evidence={"clustered": clustered, "total": len(rows)},
    )


def _stage_review(ctx: _Ctx) -> dict:
    """The generation review. Its backend is the critic; resolving that would
    load a multi-GB VLM, so availability is left to /status and this stage
    reports what is stored plus the one state the panel cannot express: a
    scan recorded with nothing to show for it."""
    guard = ctx.guard("review", "Review", "ai")
    if guard is not None:
        return guard
    row = ctx.conn.execute(
        "SELECT review_id, model_id, quality_score, prompt_alignment_score, summary, computed_at "
        "FROM ai_reviews WHERE file_id = ? AND rubric_version = ? ORDER BY computed_at DESC LIMIT 1",
        (ctx.file_id, RUBRIC_VERSION),
    ).fetchone()
    if row is None:
        if ctx.scan("review") is not None:
            return _row(
                "review",
                "ai",
                "Review",
                BLOCKED,
                "a scan was recorded but no review was stored",
                blocked_reason="the scan-log row marks this file current, so the worker will not retry it",
                fix="run the review directly, or index with force",
                action="review",
            )
        return _row("review", "ai", "Review", PENDING, "the worker has not reached this file", action="review")
    findings = ctx.conn.execute(
        "SELECT COUNT(*) FROM ai_review_findings WHERE review_id = ?", (row["review_id"],)
    ).fetchone()[0]
    return _row(
        "review",
        "ai",
        "Review",
        DONE,
        f"quality {row['quality_score']}, {findings} finding{'' if findings == 1 else 's'}",
        evidence={
            "review_id": row["review_id"],
            "model_id": row["model_id"],
            "quality_score": row["quality_score"],
            "findings": findings,
        },
    )


def _stage_alignment(ctx: _Ctx) -> dict:
    """Prompt alignment. N/A only when the file genuinely has no prompt --
    a null score on a file that HAS one is a stuck stage, not an absence."""
    prompt = (ctx.file["workflow_prompt"] or "").strip()
    if not prompt:
        return _row("alignment", "ai", "Prompt alignment", NA, "this file carries no generation prompt to compare")
    review = ctx.conn.execute(
        "SELECT review_id, prompt_alignment_score FROM ai_reviews WHERE file_id = ? ORDER BY computed_at DESC LIMIT 1",
        (ctx.file_id,),
    ).fetchone()
    if review is None:
        return _row("alignment", "ai", "Prompt alignment", PENDING, "no review yet; alignment is part of it")
    if review["prompt_alignment_score"] is None:
        return _row(
            "alignment",
            "ai",
            "Prompt alignment",
            BLOCKED,
            "the file has a prompt but its alignment score is null",
            blocked_reason="the critic's ALIGN reply did not parse, and the scan-log row stops a retry",
            fix="re-run the review for this file",
            action="review",
        )
    elements = ctx.conn.execute(
        "SELECT COUNT(*) FROM ai_review_alignment WHERE review_id = ?", (review["review_id"],)
    ).fetchone()[0]
    return _row(
        "alignment",
        "ai",
        "Prompt alignment",
        DONE,
        f"score {review['prompt_alignment_score']}, {elements} prompt element{'' if elements == 1 else 's'}",
        evidence={"score": review["prompt_alignment_score"], "elements": elements},
    )


def _stage_masks(ctx: _Ctx) -> dict:
    """Segmentation masks over localizable findings."""
    review = ctx.conn.execute(
        "SELECT review_id FROM ai_reviews WHERE file_id = ? ORDER BY computed_at DESC LIMIT 1",
        (ctx.file_id,),
    ).fetchone()
    if review is None:
        return _row("masks", "ai", "Finding masks", NA, "no review, so nothing to segment")
    rows = ctx.conn.execute(
        "SELECT mask_path FROM ai_review_findings WHERE review_id = ? AND localizable = 1",
        (review["review_id"],),
    ).fetchall()
    if not rows:
        return _row("masks", "ai", "Finding masks", NA, "this review has no localizable findings")
    guard = ctx.guard("masks", "Finding masks", "ai")
    if guard is not None:
        return guard
    have = sum(1 for r in rows if r["mask_path"])
    if have == 0:
        return _row("masks", "ai", "Finding masks", PENDING, f"0 of {len(rows)} findings segmented", action="review")
    if have < len(rows):
        return _row("masks", "ai", "Finding masks", PARTIAL, f"{have} of {len(rows)} findings segmented")
    return _row("masks", "ai", "Finding masks", DONE, f"all {have} localizable findings segmented")


# --- derived capabilities -------------------------------------------------------


def _stage_near_dup(ctx: _Ctx) -> dict:
    """Near-duplicate search needs this file's perceptual hash, and others to compare it against."""
    row = ctx.conn.execute("SELECT phash64 FROM ai_file_hashes WHERE file_id = ?", (ctx.file_id,)).fetchone()
    if row is None or row["phash64"] is None:
        return _row(
            "near_dup",
            "query",
            "Near-duplicate search",
            BLOCKED,
            "this file has no perceptual hash yet",
            blocked_reason="hashes have not been computed for it",
            fix="index this file",
            action="index",
        )
    corpus = ctx.conn.execute("SELECT COUNT(*) FROM ai_file_hashes WHERE phash64 IS NOT NULL").fetchone()[0]
    return _row(
        "near_dup",
        "query",
        "Near-duplicate search",
        DONE,
        f"answerable against {corpus} hashed file{'' if corpus == 1 else 's'}",
        evidence={"corpus": corpus, "max_distance": ctx.config.near_dup_max_distance},
    )


def _stage_similar(ctx: _Ctx) -> dict:
    """Similarity search only answers within the model version the file was embedded at."""
    rows = ctx.conn.execute(
        "SELECT space, model_version FROM ai_embeddings WHERE file_id = ?", (ctx.file_id,)
    ).fetchall()
    if not rows:
        return _row(
            "similar",
            "query",
            "Similarity search",
            BLOCKED,
            "this file has no embeddings",
            blocked_reason="neither embedding space has a vector for it",
            fix="index this file",
            action="index",
        )
    spaces = {}
    for r in rows:
        spaces[r["space"]] = ctx.conn.execute(
            "SELECT COUNT(*) FROM ai_embeddings WHERE space = ? AND model_version = ?",
            (r["space"], r["model_version"]),
        ).fetchone()[0]
    detail = ", ".join(
        f"{space}: {n} neighbour candidate{'' if n == 1 else 's'}" for space, n in sorted(spaces.items())
    )
    return _row("similar", "query", "Similarity search", DONE, detail, evidence=spaces)


_STAGES = (
    _stage_indexed,
    _stage_metadata,
    _stage_hashes,
    _stage_semantic,
    _stage_visual,
    _stage_faces,
    _stage_clustering,
    _stage_review,
    _stage_alignment,
    _stage_masks,
    _stage_near_dup,
    _stage_similar,
)


def walk(conn: sqlite3.Connection, config: AIConfig, file_id: str, worker=None, extra_stages=None) -> dict | None:
    """Every stage's state for one file, or None when the file is unknown.

    Resolves the backends the pixel stages need, so a BLOCKED row can carry
    the loader's own message rather than a bare flag.

    `extra_stages(file_id) -> list[row]` lets the host app contribute the
    ingest facts only it knows -- whether a thumbnail was built, whether the
    file's fields reached the query engine. This package cannot compute those
    without reaching into the gallery's caches, and a walkthrough that stops
    at the AI boundary is not the file's pipeline. A probe that raises is
    dropped rather than failing the whole page: a broken extra must not cost
    the operator the stages that DID resolve.
    """
    file_row = conn.execute(
        "SELECT id, path, name, type, mtime, size, workflow_prompt FROM files WHERE id = ?", (file_id,)
    ).fetchone()
    if file_row is None:
        return None

    ctx = _Ctx(conn, config, file_row, worker)
    stages = [probe(ctx) for probe in _STAGES]

    if extra_stages is not None:
        try:
            extra = list(extra_stages(file_id) or [])
        except Exception:
            _logger.debug("handled a failure in walk", exc_info=True)
            _logger.warning("[AIWalkthrough] host ingest probe failed for %s", file_id)
            extra = []
        # Host rows are ingest facts; keep them with the others rather than
        # trailing the AI stages they come before in reality.
        after = max((i for i, stage in enumerate(stages) if stage["group"] == "ingest"), default=-1) + 1
        stages[after:after] = extra

    counts: dict = {}
    for stage in stages:
        counts[stage["state"]] = counts.get(stage["state"], 0) + 1

    return {
        "enabled": bool(config.enabled),
        "file": {
            "file_id": file_row["id"],
            "name": file_row["name"],
            "path": file_row["path"],
            "type": file_row["type"],
            "size": file_row["size"],
            "mtime": file_row["mtime"],
            "renderable": ctx.renderable,
        },
        "worker": {
            "running": bool(worker is not None and getattr(worker, "is_running", False)),
            "note": (
                "a worker is running; queued work happens on its thread"
                if worker is not None
                else "no worker is running; requested stages run inline on the request thread"
            ),
        },
        "stages": stages,
        "counts": counts,
    }
