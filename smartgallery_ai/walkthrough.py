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

import numpy as np

from smartgallery_ai import (
    HASH_ALGO_VERSION,
    RUBRIC_VERSION,
    SPACE_SEMANTIC,
    SPACE_VISUAL,
    AIConfig,
    backends,
    hashing,
    invalidation,
    vectors,
)
from sqlbind import with_id_placeholders

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


# What each stage actually computes, in the terms someone would need to
# decide whether its answer is the one they want. Static per stage: the
# state and the evidence change per file, the mechanism does not. Every
# number here is read off the implementation, not the docs -- see the
# module each line names.
_STAGE_DOES = {
    "indexed": (
        "The gallery's own scan wrote this file's row: path, mtime, size, type. Everything below keys off "
        "that row, and a changed mtime is what makes all of it stale at once."
    ),
    "metadata": (
        "Pulls the ComfyUI workflow out of the PNG and traces the positive prompt through it. That text is "
        "what prompt alignment compares the picture against, and what a review is allowed to quote."
    ),
    "thumbnail": (
        "A downscaled JPEG cached under a digest of path+mtime, so editing a file misses the old cache entry "
        "rather than serving a stale picture."
    ),
    "metadata_visibility": (
        "Whether this viewer may see how a picture was made. Seeing the image and reading its prompt are "
        "separate permissions: alignment elements are verbatim slices of the prompt, so a gallery that hides "
        "prompts must withhold them here too."
    ),
    "hashes": (
        "sha256 over the raw bytes gives exact identity. phash64 is perceptual: grayscale, resize to 32x32, "
        "2D DCT-II, keep the top-left 8x8 low-frequency block, and set each bit where the coefficient beats "
        "the median of the 63 AC terms. dhash64 is a 9x8 horizontal gradient, one bit per left>right "
        "comparison. Both survive re-encoding and mild edits, which sha256 does not."
    ),
    "semantic": (
        "open_clip ViT-B-32 (laion2b_s34b_b79k) maps the image to a 512-d L2-normalized vector in a space it "
        "SHARES with its own text tower. That shared space is the whole trick: 'a red car at night' goes "
        "through the text side of the same model and lands near matching pixels, so cosine similarity is "
        "meaningful between a sentence and a picture."
    ),
    "visual": (
        "facebook/dinov2-small, a self-supervised ViT, takes its CLS token as a 384-d L2-normalized "
        "descriptor. No text side, no labels, trained only to be consistent with itself under augmentation -- "
        "so it matches on appearance and composition and will hand you a different neighbour set than CLIP "
        "does for the same picture. That is why both spaces exist."
    ),
    "faces": (
        "insightface antelopev2: SCRFD-10GF finds each box and its 5 landmarks, those landmarks drive a "
        "similarity-transform crop to a canonical layout, and glintr100 (ResNet100 trained on Glint360K) "
        "embeds the aligned crop to a 512-d ArcFace vector; genderage adds age and sex. The OpenCV lane "
        "swaps in YuNet detection with SFace (128-d) or the same ArcFace weights through cv2.dnn."
    ),
    "clustering": (
        "Builds a cosine-similarity graph over the face vectors -- an edge wherever two faces clear the "
        "threshold -- then runs Chinese Whispers label propagation (dlib's algorithm) over it. Nobody tells "
        "it how many people are in the library; the clusters fall out of the graph. Faces below the "
        "threshold stay unassigned rather than being forced into a bucket. The graph itself is built by "
        "torch-CUDA, FAISS, or numpy, whichever is available."
    ),
    "review": (
        "A vision-language critic looks at the picture and returns typed findings -- severity, confidence, "
        "and whether it can actually point at the defect -- plus a quality score, validated against a strict "
        "schema before a row is written. A finding that claims a location must also survive a crop check."
    ),
    "alignment": (
        "Splits the positive prompt into its elements and asks, per element, whether the image satisfies it "
        "and where. The answer is 'cat: yes, here / neon sign: no' rather than one number for the whole "
        "prompt."
    ),
    "masks": (
        "MobileSAM segments the region a localizable finding pointed at, turning its bounding rectangle into "
        "a mask that follows the actual shape of the defect."
    ),
    "near_dup": (
        "FAISS IndexBinaryFlat over the 64-bit phash values: exact Hamming search using popcount "
        "instructions, with a chunked numpy XOR+popcount sweep as the fallback when FAISS is absent. Both "
        "paths return the same pairs. Two files count as near-duplicates when their distance is at or under "
        "AI_DAM_NEAR_DUP_DISTANCE."
    ),
    "similar": (
        "FAISS IndexFlatIP over the L2-normalized vectors. Inner product on unit vectors IS cosine "
        "similarity, and Flat means exhaustive -- every candidate is scored, no approximation, so the "
        "neighbours are exact. Results are pinned to the model version the query file was embedded at, so "
        "vectors produced by two different models are never compared."
    ),
}


def _provision_hint(group: str) -> str:
    return f"python -m smartgallery_ai provision {group}"


def _neighbours(ctx, space: str, model_version: str, k: int) -> list:
    """This file's k nearest neighbours in `space`, as (file_id, cosine).

    Empty when anything goes wrong: an example is worth having, never worth
    failing the page for. Shared by the embedding stages and the similarity
    row so both report the same ranking.
    """
    try:
        row = ctx.conn.execute(
            "SELECT vector FROM ai_embeddings WHERE file_id = ? AND space = ?", (ctx.file_id, space)
        ).fetchone()
        if row is None:
            return []
        query = np.frombuffer(row["vector"], dtype="<f4")
        store = vectors.VectorStore(cache_dir=ctx.config.cache_dir, ephemeral=ctx.config.ephemeral_index)
        return store.topk(ctx.conn, space, query, k, exclude=[ctx.file_id], model_version=model_version)
    except Exception:
        _logger.debug("handled a failure in _neighbours", exc_info=True)
        return []


def _example(caption: str, rows: list, tiles: list | None = None, boxes: list | None = None) -> dict:
    """A worked example computed from THIS file, not a description of one.

    `rows` are (label, value) pairs the page renders as a small table, so a
    reader can check the mechanism against the numbers it actually produced
    rather than take the prose on trust.

    `tiles` are {file_id, caption} -- the page draws each as a thumbnail, so
    a neighbour list is pictures rather than filenames. This is a gallery:
    "these two images scored 0.24" is an assertion, the two images side by
    side is the evidence.

    `boxes` are {x, y, w, h, label} in image fractions, drawn over this
    file's own thumbnail. A face box is only checkable by looking at it.
    """
    example: dict = {"caption": caption, "rows": [[str(a), str(b)] for a, b in rows]}
    if tiles:
        example["tiles"] = tiles
    if boxes:
        example["boxes"] = boxes
    return example


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

    def names(self, file_ids) -> dict:
        """file_id -> display name, for turning ids in an example into
        something a person can recognise on the page."""
        ids = list(file_ids)
        if not ids:
            return {}
        rows = self.conn.execute(
            with_id_placeholders("SELECT id, name FROM files WHERE id IN ({ids})", ids), ids
        ).fetchall()
        return {r["id"]: r["name"] for r in rows}

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
    # The bits themselves, plus what they measure against a real neighbour:
    # a Hamming distance is only meaningful next to the threshold it is
    # compared with.
    phash = hashing.to_unsigned64(row["phash64"]) if row["phash64"] is not None else None
    dhash = hashing.to_unsigned64(row["dhash64"]) if row["dhash64"] is not None else None
    example_rows = [
        ("sha256", row["sha256"][:32] + "…"),
        ("phash64 (hex)", f"{phash:016x}" if phash is not None else "—"),
        ("phash64 (bits)", f"{phash:064b}"[:32] + "…" if phash is not None else "—"),
        ("dhash64 (hex)", f"{dhash:016x}" if dhash is not None else "—"),
    ]
    others = ctx.conn.execute(
        "SELECT file_id, phash64 FROM ai_file_hashes WHERE file_id != ? AND phash64 IS NOT NULL LIMIT 4",
        (ctx.file_id,),
    ).fetchall()
    names = ctx.names(r["file_id"] for r in others)
    for other in others:
        distance = hashing.hamming64(row["phash64"], other["phash64"])
        verdict = "near-duplicate" if distance <= ctx.config.near_dup_max_distance else "different image"
        example_rows.append(
            (f"vs {names.get(other['file_id'], other['file_id'])}", f"{distance} bits apart → {verdict}")
        )
    return _row(
        "hashes",
        "ai",
        "Hashes",
        DONE,
        "sha256 + phash64 + dhash64",
        evidence={"phash64": row["phash64"], "dhash64": row["dhash64"], "algo_version": row["algo_version"]},
        example=_example(
            f"this file's hashes, and the distance to every other hashed file "
            f"(threshold {ctx.config.near_dup_max_distance})",
            example_rows,
        ),
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
    # The neighbours this space actually returns for this file. Semantic and
    # visual disagreeing on the same picture is the point of having both, and
    # a list of names with scores shows that in a way prose cannot.
    neighbours = _neighbours(ctx, space, row["model_version"], 5)
    names = ctx.names(fid for fid, _score in neighbours)
    example_rows = [(names.get(fid, fid), f"cosine {score:.4f}") for fid, score in neighbours]

    example = (
        _example(
            f"nearest neighbours in the {space} space, scored by cosine similarity",
            example_rows,
            tiles=[
                {"file_id": fid, "caption": f"{score:.4f}", "title": names.get(fid, fid)} for fid, score in neighbours
            ],
        )
        if example_rows
        else _example(
            "no other file is embedded in this space yet, so there is nothing to compare against",
            [("corpus", "1 file (this one)")],
        )
    )
    return _row(
        key,
        "ai",
        label,
        DONE,
        f"{row['dim']}-d vector from {row['model_id']}",
        evidence={"model_id": row["model_id"], "model_version": row["model_version"], "dim": row["dim"]},
        example=example,
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
    # The actual detections: box, confidence, embedding width, cluster. A
    # reader can check a low det_score against a box they think is wrong.
    detections = ctx.conn.execute(
        "SELECT face_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, dim, cluster_id, age, sex "
        "FROM ai_face_instances WHERE file_id = ? ORDER BY det_score DESC LIMIT 6",
        (ctx.file_id,),
    ).fetchall()
    rows = [
        (
            f"face {d['face_id']}",
            f"box ({d['bbox_x']:.3f}, {d['bbox_y']:.3f}, {d['bbox_w']:.3f}, {d['bbox_h']:.3f}) · "
            f"det_score {d['det_score']:.3f} · {d['dim']}-d vector · "
            + (f"cluster {d['cluster_id']}" if d["cluster_id"] is not None else "unclustered")
            + (f" · {d['sex']} ~{d['age']}" if d["sex"] and d["age"] is not None else ""),
        )
        for d in detections
    ]
    # Every box, not just the six listed: the overlay is the only place the
    # detector's actual behaviour is checkable, and a missed face is visible
    # there and nowhere else.
    all_boxes = ctx.conn.execute(
        "SELECT face_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, cluster_id "
        "FROM ai_face_instances WHERE file_id = ? ORDER BY face_id",
        (ctx.file_id,),
    ).fetchall()
    boxes = [
        {
            "x": b["bbox_x"],
            "y": b["bbox_y"],
            "w": b["bbox_w"],
            "h": b["bbox_h"],
            "label": f"{b['det_score']:.2f}",
            "clustered": b["cluster_id"] is not None,
        }
        for b in all_boxes
    ]
    example = (
        _example(
            f"every detection drawn over the image, and the {len(rows)} most confident in full; "
            f"kept only above det_score {ctx.config.face_min_det_score}",
            rows,
            boxes=boxes,
        )
        if rows
        else _example("the detector ran and returned nothing", [("faces", "0")])
    )
    return _row(
        "faces",
        "ai",
        "Face detection",
        DONE,
        detail,
        evidence={"faces": count, "model_id": scan["model_id"], "scanned_at": scan["scanned_at"]},
        example=example,
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

    # Which bucket each face landed in, how big that bucket is library-wide,
    # and the threshold that decided it -- the three numbers needed to judge
    # whether a cluster is too greedy or too shy.
    threshold = ctx.conn.execute(
        "SELECT params FROM ai_face_clusters WHERE params IS NOT NULL ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    members = ctx.conn.execute(
        "SELECT i.cluster_id, c.label, c.size, COUNT(*) AS here FROM ai_face_instances i "
        "JOIN ai_face_clusters c ON c.cluster_id = i.cluster_id "
        "WHERE i.file_id = ? GROUP BY i.cluster_id ORDER BY here DESC",
        (ctx.file_id,),
    ).fetchall()
    example_rows = [
        (
            f"cluster {m['cluster_id']}" + (f" ({m['label']})" if m["label"] else ""),
            f"{m['here']} of this file's faces · {m['size']} faces library-wide",
        )
        for m in members
    ]
    if clustered < len(rows):
        example_rows.append(("unclustered", f"{len(rows) - clustered} faces below the similarity threshold"))
    if threshold is not None:
        example_rows.append(("params", str(threshold["params"])[:120]))

    # The other files whose faces landed in the same bucket -- the claim
    # "these are the same person" is only judgeable by seeing them.
    cluster_ids = [m["cluster_id"] for m in members]
    siblings = []
    if cluster_ids:
        siblings = ctx.conn.execute(
            with_id_placeholders(
                "SELECT DISTINCT i.file_id, i.cluster_id FROM ai_face_instances i "
                "WHERE i.cluster_id IN ({ids}) AND i.file_id != ? LIMIT 12",
                cluster_ids,
            ),
            [*cluster_ids, ctx.file_id],
        ).fetchall()
    sibling_names = ctx.names(s["file_id"] for s in siblings)
    example = _example(
        "the buckets this file's faces landed in",
        example_rows,
        tiles=[
            {
                "file_id": s["file_id"],
                "caption": f"cluster {s['cluster_id']}",
                "title": sibling_names.get(s["file_id"], s["file_id"]),
            }
            for s in siblings
        ],
    )

    if clustered < len(rows):
        return _row(
            "clustering",
            "ai",
            "Face clustering",
            PARTIAL,
            f"{clustered} of {len(rows)} faces clustered; the rest sit below the similarity threshold",
            evidence={"clustered": clustered, "total": len(rows)},
            example=example,
        )
    return _row(
        "clustering",
        "ai",
        "Face clustering",
        DONE,
        f"all {clustered} faces clustered",
        evidence={"clustered": clustered, "total": len(rows)},
        example=example,
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
    hits = hashing.find_near_duplicates(ctx.conn, ctx.file_id, ctx.config.near_dup_max_distance)
    names = ctx.names(fid for fid, _distance in hits)
    example_rows = [(names.get(fid, fid), f"{distance} bits apart") for fid, distance in hits[:6]]
    example = _example(
        f"what this query returns right now, at threshold {ctx.config.near_dup_max_distance}",
        example_rows or [("result", f"no file within {ctx.config.near_dup_max_distance} bits of this one")],
        tiles=[
            {"file_id": fid, "caption": f"{distance} bits", "title": names.get(fid, fid)} for fid, distance in hits[:6]
        ],
    )
    return _row(
        "near_dup",
        "query",
        "Near-duplicate search",
        DONE,
        f"answerable against {corpus} hashed file{'' if corpus == 1 else 's'}",
        evidence={"corpus": corpus, "max_distance": ctx.config.near_dup_max_distance},
        example=example,
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

    # Both rankings, interleaved by position. Where rank 1 differs between the
    # spaces you can see directly that "similar" is two questions rather than
    # one -- which a single merged list would hide.
    ranked = {r["space"]: _neighbours(ctx, r["space"], r["model_version"], 3) for r in rows}
    lookup = ctx.names(fid for hits in ranked.values() for fid, _score in hits)
    example_rows = []
    for position in range(max((len(hits) for hits in ranked.values()), default=0)):
        for space in sorted(ranked):
            hits = ranked[space]
            if position < len(hits):
                fid, score = hits[position]
                example_rows.append((f"{space} #{position + 1}", f"{lookup.get(fid, fid)} — cosine {score:.4f}"))
    # Tiles carry the space in the caption, so a file appearing under one
    # space and not the other is visible rather than inferred.
    tiles = [
        {"file_id": fid, "caption": f"{space} {score:.3f}", "title": lookup.get(fid, fid)}
        for space in sorted(ranked)
        for fid, score in ranked[space][:3]
    ]
    example = _example(
        "what each space returns for this file, best first",
        example_rows or [("result", "no other file is embedded yet, so both spaces return nothing")],
        tiles=tiles,
    )
    return _row("similar", "query", "Similarity search", DONE, detail, evidence=spaces, example=example)


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

    # Attached here rather than inside each probe: what a stage computes does
    # not depend on what it found, and repeating it down every branch is how
    # two descriptions of one mechanism start to disagree.
    for stage in stages:
        does = _STAGE_DOES.get(stage["key"])
        if does:
            stage["does"] = does

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
