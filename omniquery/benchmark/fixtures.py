"""Deterministic fixture SQLite database for OmniQuery v2 tests.

The schema is copied inline (core SmartGallery tables mirror
smartgallery.py's init_db(); AI tables come straight from
smartgallery_ai.schema, the authoritative DDL for that layer) rather than
built by importing and running the monolith. Content is generated from a
seeded RNG, so ``build_fixture_db(path, seed=42)`` always produces the same
rows. ``FIXTURE_EXPECTATIONS`` is computed directly from those in-memory
records -- not derived from SQL -- so it can cross-check the compiler.
"""

from __future__ import annotations

import hashlib
import os
import random
import sqlite3
import struct
from collections import defaultdict
from typing import Any

from omniquery.fields import REVIEW_ISSUE_VALUES
from smartgallery_ai import HASH_ALGO_VERSION, RUBRIC_VERSION, SPACE_SEMANTIC, SPACE_VISUAL
from smartgallery_ai.schema import DDL as _AI_DDL

FIXTURE_BASE_PATH = "/gallery"  # virtual gallery root prefixed onto every generated file path
# A fixed instant (2025-01-01T00:00:00 local), not wall-clock "now" -- every
# generated mtime and every FIXTURE_EXPECTATIONS entry is independent of
# when the test suite happens to run.
ANCHOR_EPOCH = 1735689600.0

# Relative folder pool; files are assigned round-robin by index.
FOLDERS = ["portraits", "landscapes/2024", "landscapes/2025", "renders/batch_a", "misc"]

# One media-type slot per generated file; the shuffled pool fixes each
# file's type while keeping the overall mix (30/10/6/8/6) constant.
_TYPE_POOL = ["image"] * 30 + ["video"] * 10 + ["animated_image"] * 6 + ["audio"] * 8 + ["document"] * 6
# Canonical file extension per media type.
_EXT = {"image": ".png", "video": ".mp4", "animated_image": ".gif", "audio": ".mp3", "document": ".pdf"}
# "WIDTHxHEIGHT" pixel strings, as stored in files.dimensions.
_DIMENSIONS_POOL = ["512x512", "768x1024", "1024x1024", "1920x1080", "1280x720", "2048x1536"]

# Workflow prompts; "cyberpunk" appears in exactly one entry, so the
# prompt-substring expectation selects a single well-defined file set.
_PROMPT_POOL = [
    "cyberpunk city skyline at night, neon reflections",
    "portrait of a warrior queen, dramatic lighting",
    "serene mountain landscape at dawn, misty valleys",
    "steampunk mechanical dragon, intricate gears",
    "underwater bioluminescent forest, glowing coral",
]
_CAPTION_POOL = [
    "A detailed digital painting of a futuristic cityscape.",
    "A close-up portrait with dramatic studio lighting.",
    "A wide landscape shot of mountains under a pastel sky.",
    "A mechanical creature rendered in a steampunk style.",
    "An underwater scene with glowing plant life.",
]
# Two entries contain "amazing" (differing case), giving the
# case-insensitive comment-substring expectation real matches.
_COMMENT_POOL = [
    "This is amazing work, love the lighting.",
    "The composition feels a bit off to me.",
    "Amazing color palette, very cohesive.",
    "Needs another pass on the hands.",
    "Great mood, very atmospheric.",
]

# (user_id, username, full_name, role) rows for the users table; one
# account per role tier exercised by role-scoped queries.
USERS: list[tuple[int, str, str, str]] = [
    (1, "alice", "Alice Anders", "USER"),
    (2, "bob", "Bob Baker", "GUEST"),
    (3, "carol", "Carol Chen", "STAFF"),
    (4, "dave", "Dave Diaz", "MANAGER"),
    (5, "erin", "Erin Evans", "ADMIN"),
]

# (collection name, collection type, hex color) rows for system flags.
STATUS_FLAGS: list[tuple[str, str, str]] = [
    ("Approved", "system_flag", "#28a745"),
    ("Review", "system_flag", "#ffc107"),
    ("To Edit", "system_flag", "#17a2b8"),
    ("Rejected", "system_flag", "#dc3545"),
    ("Select", "system_flag", "#6f42c1"),
]
_FLAG_SIZES = [8, 6, 5, 4, 3]  # disjoint chunk sizes, matched to STATUS_FLAGS order

USER_ALBUMS = ["Portfolio", "Client Picks", "WIP"]

# client_uuid pools: opaque anonymous browser ids plus digit strings that
# alias logged-in users (stringified user_id -- e.g. "3" is carol).
_RATING_CLIENT_POOL = ["anon-aaa", "anon-bbb", "anon-ccc", "anon-ddd", "1", "2", "3", "4", "5"]
_COMMENT_CLIENT_POOL = ["anon-aaa", "anon-bbb", "3", "4"]
_USERNAME_BY_UUID = {
    "1": "alice",
    "2": "bob",
    "3": "carol",
    "4": "dave",
    "5": "erin",
}  # stringified user_id -> username, mirroring USERS

# --- core SmartGallery tables (mirrors smartgallery.py's init_db()) --------

_CORE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        mtime REAL NOT NULL,
        name TEXT NOT NULL,
        type TEXT,
        duration TEXT,
        dimensions TEXT,
        has_workflow INTEGER,
        is_favorite INTEGER DEFAULT 0,
        size INTEGER DEFAULT 0,
        last_scanned REAL DEFAULT 0,
        workflow_files TEXT DEFAULT '',
        workflow_prompt TEXT DEFAULT '',
        ai_last_scanned REAL DEFAULT 0,
        ai_caption TEXT,
        ai_embedding BLOB,
        ai_error TEXT,
        workflow_hash TEXT,
        prompt_hash TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        color TEXT,
        is_public INTEGER DEFAULT 0,
        parent_id INTEGER DEFAULT NULL,
        created_at REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS collection_files (
        collection_id INTEGER,
        file_id TEXT,
        added_at REAL,
        PRIMARY KEY (collection_id, file_id),
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE ON UPDATE CASCADE,
        FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS file_ratings (
        file_id TEXT,
        client_uuid TEXT,
        rating INTEGER CHECK(rating >= 1 AND rating <= 5),
        created_at REAL,
        PRIMARY KEY (file_id, client_uuid),
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS file_comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT,
        client_uuid TEXT,
        author_name TEXT,
        comment_text TEXT,
        target_audience TEXT DEFAULT 'public',
        created_at REAL,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS generation_params (
        file_id TEXT PRIMARY KEY REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        tool TEXT NOT NULL,
        detection TEXT NOT NULL,
        positive_prompt TEXT NOT NULL DEFAULT '',
        negative_prompt TEXT NOT NULL DEFAULT '',
        model TEXT,
        model_hash TEXT,
        sampler TEXT,
        scheduler TEXT,
        seed INTEGER,
        steps INTEGER,
        cfg REAL,
        width INTEGER,
        height INTEGER,
        denoise REAL,
        clip_skip INTEGER,
        version TEXT,
        loras TEXT,
        extra TEXT,
        parsed_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT,
        phone_number TEXT,
        role TEXT CHECK(role IN
            ('USER','STAFF','MANAGER','CUSTOMER','FRIEND','GUEST','ADMIN')) DEFAULT 'GUEST',
        start_date DATE DEFAULT CURRENT_DATE,
        expiry_date DATE,
        is_active BOOLEAN DEFAULT 1,
        last_login REAL
    );
    """,
]


# ---------------------------------------------------------------------------
# Deterministic content generation
# ---------------------------------------------------------------------------


def _make_id(idx: int) -> str:
    """Fixture file id for a 1-based index, zero-padded to three digits ("f001")."""
    return f"f{idx:03d}"


def _generate(seed: int) -> dict[str, Any]:
    """Build the complete in-memory fixture record set from `seed` alone.

    Every random draw comes from one seeded RNG consumed in a fixed order,
    so equal seeds yield identical records; both the database rows and the
    ground-truth expectations derive from this single output.
    """
    rng = random.Random(seed)

    types = list(_TYPE_POOL)
    rng.shuffle(types)

    files: list[dict[str, Any]] = []
    for i, ftype in enumerate(types):
        idx = i + 1
        fid = _make_id(idx)
        folder = FOLDERS[idx % len(FOLDERS)]
        name = f"{ftype}_{idx:03d}{_EXT[ftype]}"
        path = f"{FIXTURE_BASE_PATH}/{folder}/{name}"

        day_offset = rng.randint(0, 400)
        hour_jitter = rng.randint(0, 86399)
        mtime = ANCHOR_EPOCH - day_offset * 86400.0 - hour_jitter

        size = rng.randint(15_000, 45_000_000)

        duration = ""
        duration_seconds = None
        if ftype in ("video", "audio"):
            secs = rng.randint(3, 3599)
            duration = f"{secs // 60:02d}:{secs % 60:02d}"
            duration_seconds = secs

        dimensions = ""
        width = height = megapixels = None
        if ftype in ("image", "video", "animated_image"):
            dimensions = rng.choice(_DIMENSIONS_POOL)
            width, height = (int(p) for p in dimensions.split("x"))
            megapixels = width * height / 1_000_000.0

        has_workflow = 0
        workflow_prompt = ""
        workflow_files = ""
        if ftype in ("image", "animated_image") and rng.random() < 0.6:
            has_workflow = 1
            workflow_prompt = rng.choice(_PROMPT_POOL)
            workflow_files = f"workflow_{fid}.json"

        ai_caption = rng.choice(_CAPTION_POOL) if rng.random() < 0.5 else None
        is_favorite = 1 if rng.random() < 0.25 else 0

        files.append(
            {
                "id": fid,
                "path": path,
                "folder": folder,
                "name": name,
                "type": ftype,
                "mtime": mtime,
                "duration": duration,
                "duration_seconds": duration_seconds,
                "dimensions": dimensions,
                "width": width,
                "height": height,
                "megapixels": megapixels,
                "has_workflow": has_workflow,
                "is_favorite": is_favorite,
                "size": size,
                "workflow_files": workflow_files,
                "workflow_prompt": workflow_prompt,
                "ai_caption": ai_caption,
            }
        )

    ratings: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []
    for f in files:
        if rng.random() < 0.55:
            n = rng.randint(1, 3)
            for cu in rng.sample(_RATING_CLIENT_POOL, k=n):
                ratings.append(
                    {
                        "file_id": f["id"],
                        "client_uuid": cu,
                        "rating": rng.randint(1, 5),
                        "created_at": f["mtime"] + 10.0,
                    }
                )
        if rng.random() < 0.4:
            n = rng.randint(1, 2)
            for cu in rng.sample(_COMMENT_CLIENT_POOL, k=n):
                comments.append(
                    {
                        "file_id": f["id"],
                        "client_uuid": cu,
                        "author_name": _USERNAME_BY_UUID.get(cu, cu),
                        "comment_text": rng.choice(_COMMENT_POOL),
                        "created_at": f["mtime"] + 20.0,
                    }
                )

    all_ids = [f["id"] for f in files]
    shuffled = all_ids[:]
    rng.shuffle(shuffled)

    membership: dict[tuple[str, str], list[str]] = {}
    cursor = 0
    for (flag_name, flag_type, _color), chunk_size in zip(STATUS_FLAGS, _FLAG_SIZES, strict=False):
        membership[(flag_name, flag_type)] = shuffled[cursor : cursor + chunk_size]
        cursor += chunk_size
    for album_name in USER_ALBUMS:
        membership[(album_name, "user_album")] = rng.sample(all_ids, 15)

    hashes: list[dict[str, Any]] = []
    for f in files:
        phash = dhash = None
        if f["type"] in ("image", "video", "animated_image"):
            phash = _signed64(rng.getrandbits(64))
            dhash = _signed64(rng.getrandbits(64))
        hashes.append(
            {
                "file_id": f["id"],
                "sha256": hashlib.sha256(f["id"].encode()).hexdigest(),
                "phash64": phash,
                "dhash64": dhash,
                "algo_version": HASH_ALGO_VERSION,
                "source_mtime": f["mtime"],
                "computed_at": f["mtime"] + 1.0,
            }
        )

    embeddings: list[dict[str, Any]] = []
    embed_candidates = [f for f in files if f["type"] in ("image", "animated_image")][:20]
    for f in embed_candidates:
        for space in (SPACE_SEMANTIC, SPACE_VISUAL):
            vector = struct.pack("<8f", *(rng.uniform(-1, 1) for _ in range(8)))
            embeddings.append(
                {
                    "file_id": f["id"],
                    "space": space,
                    "model_id": "stub-embedder",
                    "model_version": "v1",
                    "dim": 8,
                    "vector": vector,
                    "source_mtime": f["mtime"],
                    "computed_at": f["mtime"] + 1.0,
                }
            )

    face_clusters = [
        {"cluster_id": 1, "label": "Character A", "model_id": "stub-facenet", "model_version": "v1"},
        {"cluster_id": 2, "label": "Character B", "model_id": "stub-facenet", "model_version": "v1"},
    ]
    face_instances: list[dict[str, Any]] = []
    face_candidates = [
        f for f in files if f["folder"] == "portraits" and f["type"] in ("image", "video", "animated_image")
    ]
    for i, f in enumerate(face_candidates):
        n_faces = 2 if i % 3 == 0 else 1
        for _ in range(n_faces):
            cluster_id = None if i % 4 == 3 else (1 if i % 2 == 0 else 2)
            face_instances.append(
                {
                    "file_id": f["id"],
                    "bbox_x": round(rng.uniform(0.05, 0.5), 3),
                    "bbox_y": round(rng.uniform(0.05, 0.5), 3),
                    "bbox_w": round(rng.uniform(0.1, 0.4), 3),
                    "bbox_h": round(rng.uniform(0.1, 0.4), 3),
                    "det_score": round(rng.uniform(0.5, 0.99), 3),
                    "embedding": struct.pack("<8f", *(rng.uniform(-1, 1) for _ in range(8))),
                    "dim": 8,
                    "model_id": "stub-facenet",
                    "model_version": "v1",
                    "source_mtime": f["mtime"],
                    "computed_at": f["mtime"] + 1.0,
                    "cluster_id": cluster_id,
                }
            )

    # Typed generation parameters for every workflow-bearing file. The
    # FIRST row is pinned (seed/steps/cfg/model/lora) so corpus entries can
    # target known values; the "girlnextdoor" LoRA also lands on every
    # third gp row, giving bare-term text searches a real multi-file match.
    genparams: list[dict[str, Any]] = []
    for f in files:
        if not f["has_workflow"]:
            continue
        i = len(genparams)
        genparams.append(
            {
                "file_id": f["id"],
                "tool": "comfyui",
                "detection": "workflow",
                "positive_prompt": f["workflow_prompt"],
                "negative_prompt": "blurry, lowres, watermark",
                "model": ["flux1-dev-fp8", "sdxl_base_1.0", "dreamshaper_8"][i % 3],
                "sampler": ["euler", "dpmpp_2m"][i % 2],
                "scheduler": "normal",
                "seed": rng.randint(1, 2**31),
                "steps": [20, 25, 30][i % 3],
                "cfg": [4.0, 7.0, 7.5][i % 3],
                "width": f["width"],
                "height": f["height"],
                "loras": (
                    '[{"name": "girlnextdoor", "weight": 0.8}]'
                    if i % 3 == 0
                    else '[{"name": "detail-tweaker", "weight": 0.6}]'
                    if i % 3 == 1
                    else "[]"
                ),
                "parsed_at": f["mtime"] + 3.0,
            }
        )
    if genparams:
        genparams[0].update(seed=424242, steps=30, cfg=7.5, model="flux1-dev-fp8")

    review_candidates = files[::4][:15]
    reviews: list[dict[str, Any]] = []
    for f in review_candidates:
        reviews.append(
            {
                "file_id": f["id"],
                "rubric_version": RUBRIC_VERSION,
                "model_id": "stub-critic",
                "model_version": "v1",
                "quality_score": round(rng.uniform(1, 10), 2),
                "prompt_alignment_score": round(rng.uniform(0, 1), 3),
                "summary": "",
                "raw_response": "{}",
                "source_mtime": f["mtime"],
                "computed_at": f["mtime"] + 2.0,
            }
        )

    issue_values = sorted(REVIEW_ISSUE_VALUES)
    findings: list[dict[str, Any]] = []
    for i, rv in enumerate(reviews[:10]):
        issue_type = issue_values[i % len(issue_values)]
        findings.append(
            {
                "file_id": rv["file_id"],
                "review_index": i,
                "type": issue_type,
                "severity": rng.choice(["low", "medium", "high"]),
                "confidence": round(rng.uniform(0.5, 0.99), 2),
                "description": f"{issue_type} issue detected",
            }
        )

    return {
        "files": files,
        "ratings": ratings,
        "comments": comments,
        "membership": membership,
        "hashes": hashes,
        "embeddings": embeddings,
        "face_clusters": face_clusters,
        "face_instances": face_instances,
        "reviews": reviews,
        "findings": findings,
        "genparams": genparams,
    }


def _signed64(unsigned: int) -> int:
    """Reinterpret an unsigned 64-bit value as two's-complement signed,
    the form SQLite INTEGER columns store 64-bit perceptual hashes in."""
    return unsigned - (1 << 64) if unsigned >= (1 << 63) else unsigned


# ---------------------------------------------------------------------------
# DB construction
# ---------------------------------------------------------------------------


def build_fixture_db(path: str, seed: int = 42) -> None:
    """Write a deterministic fixture SQLite database to `path`, replacing
    any existing file; equal (path, seed) always yields identical content."""
    data = _generate(seed)
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for stmt in _CORE_DDL:
            conn.execute(stmt)
        for stmt in _AI_DDL:
            conn.execute(stmt)

        conn.executemany(
            "INSERT INTO files (id, path, mtime, name, type, duration, dimensions, "
            "has_workflow, is_favorite, size, workflow_files, workflow_prompt, ai_caption) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    f["id"],
                    f["path"],
                    f["mtime"],
                    f["name"],
                    f["type"],
                    f["duration"],
                    f["dimensions"],
                    f["has_workflow"],
                    f["is_favorite"],
                    f["size"],
                    f["workflow_files"],
                    f["workflow_prompt"],
                    f["ai_caption"],
                )
                for f in data["files"]
            ],
        )

        conn.executemany(
            "INSERT INTO generation_params (file_id, tool, detection, "
            "positive_prompt, negative_prompt, model, sampler, scheduler, "
            "seed, steps, cfg, width, height, loras, parsed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    g["file_id"],
                    g["tool"],
                    g["detection"],
                    g["positive_prompt"],
                    g["negative_prompt"],
                    g["model"],
                    g["sampler"],
                    g["scheduler"],
                    g["seed"],
                    g["steps"],
                    g["cfg"],
                    g["width"],
                    g["height"],
                    g["loras"],
                    g["parsed_at"],
                )
                for g in data["genparams"]
            ],
        )

        collection_ids: dict[tuple[str, str], int] = {}
        all_collections = STATUS_FLAGS + [(n, "user_album", "#888888") for n in USER_ALBUMS]
        for name, ctype, color in all_collections:
            cur = conn.execute(
                "INSERT INTO collections (name, type, color, is_public, created_at) VALUES (?,?,?,0,?)",
                (name, ctype, color, ANCHOR_EPOCH),
            )
            collection_ids[(name, ctype)] = cur.lastrowid

        for key, file_ids in data["membership"].items():
            cid = collection_ids[key]
            conn.executemany(
                "INSERT INTO collection_files (collection_id, file_id, added_at) VALUES (?,?,?)",
                [(cid, fid, ANCHOR_EPOCH) for fid in file_ids],
            )

        conn.executemany(
            "INSERT INTO users (user_id, username, password, full_name, role, is_active) VALUES (?,?,?,?,?,1)",
            [(uid, uname, "x", full_name, role) for uid, uname, full_name, role in USERS],
        )

        conn.executemany(
            "INSERT INTO file_ratings (file_id, client_uuid, rating, created_at) VALUES (?,?,?,?)",
            [(r["file_id"], r["client_uuid"], r["rating"], r["created_at"]) for r in data["ratings"]],
        )

        conn.executemany(
            "INSERT INTO file_comments "
            "(file_id, client_uuid, author_name, comment_text, target_audience, created_at) "
            "VALUES (?,?,?,?,'public',?)",
            [
                (c["file_id"], c["client_uuid"], c["author_name"], c["comment_text"], c["created_at"])
                for c in data["comments"]
            ],
        )

        conn.executemany(
            "INSERT INTO ai_file_hashes "
            "(file_id, sha256, phash64, dhash64, algo_version, source_mtime, computed_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    h["file_id"],
                    h["sha256"],
                    h["phash64"],
                    h["dhash64"],
                    h["algo_version"],
                    h["source_mtime"],
                    h["computed_at"],
                )
                for h in data["hashes"]
            ],
        )

        conn.executemany(
            "INSERT INTO ai_embeddings "
            "(file_id, space, model_id, model_version, dim, vector, source_mtime, computed_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    e["file_id"],
                    e["space"],
                    e["model_id"],
                    e["model_version"],
                    e["dim"],
                    e["vector"],
                    e["source_mtime"],
                    e["computed_at"],
                )
                for e in data["embeddings"]
            ],
        )

        conn.executemany(
            "INSERT INTO ai_face_clusters (cluster_id, label, model_id, model_version, updated_at) VALUES (?,?,?,?,?)",
            [
                (c["cluster_id"], c["label"], c["model_id"], c["model_version"], ANCHOR_EPOCH)
                for c in data["face_clusters"]
            ],
        )

        conn.executemany(
            "INSERT INTO ai_face_instances "
            "(file_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding, dim, "
            "model_id, model_version, source_mtime, computed_at, cluster_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    fi["file_id"],
                    fi["bbox_x"],
                    fi["bbox_y"],
                    fi["bbox_w"],
                    fi["bbox_h"],
                    fi["det_score"],
                    fi["embedding"],
                    fi["dim"],
                    fi["model_id"],
                    fi["model_version"],
                    fi["source_mtime"],
                    fi["computed_at"],
                    fi["cluster_id"],
                )
                for fi in data["face_instances"]
            ],
        )

        review_ids: list[int] = []
        for rv in data["reviews"]:
            cur = conn.execute(
                "INSERT INTO ai_reviews "
                "(file_id, rubric_version, model_id, model_version, quality_score, "
                "prompt_alignment_score, summary, raw_response, source_mtime, computed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    rv["file_id"],
                    rv["rubric_version"],
                    rv["model_id"],
                    rv["model_version"],
                    rv["quality_score"],
                    rv["prompt_alignment_score"],
                    rv["summary"],
                    rv["raw_response"],
                    rv["source_mtime"],
                    rv["computed_at"],
                ),
            )
            review_ids.append(cur.lastrowid)

        conn.executemany(
            "INSERT INTO ai_review_findings "
            "(review_id, file_id, type, severity, confidence, localizable, description) "
            "VALUES (?,?,?,?,?,0,?)",
            [
                (
                    review_ids[finding["review_index"]],
                    finding["file_id"],
                    finding["type"],
                    finding["severity"],
                    finding["confidence"],
                    finding["description"],
                )
                for finding in data["findings"]
            ],
        )

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ground-truth expectations, computed straight from the generated records
# ---------------------------------------------------------------------------


def _compute_expectations(data: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Ground-truth answer sets, keyed by scenario name, each the frozenset
    of file ids satisfying that predicate. Computed from the in-memory
    records -- never via SQL -- so they can cross-check the compiler."""
    files = data["files"]

    rating_totals: dict[str, list[int]] = defaultdict(list)
    for r in data["ratings"]:
        rating_totals[r["file_id"]].append(r["rating"])

    return {
        "type_image": frozenset(f["id"] for f in files if f["type"] == "image"),
        "type_video_or_audio": frozenset(f["id"] for f in files if f["type"] in ("video", "audio")),
        "is_favorite_true": frozenset(f["id"] for f in files if f["is_favorite"] == 1),
        "has_workflow_true": frozenset(f["id"] for f in files if f["has_workflow"] == 1),
        "ai_caption_not_null": frozenset(f["id"] for f in files if f["ai_caption"] is not None),
        "ai_caption_null": frozenset(f["id"] for f in files if f["ai_caption"] is None),
        "folder_landscapes_2024": frozenset(f["id"] for f in files if f["folder"] == "landscapes/2024"),
        "folder_contains_landscapes": frozenset(f["id"] for f in files if "landscapes" in f["folder"]),
        "size_gt_20mb": frozenset(f["id"] for f in files if f["size"] > 20 * 1024 * 1024),
        "duration_seconds_ge_300": frozenset(
            f["id"] for f in files if f["duration_seconds"] is not None and f["duration_seconds"] >= 300
        ),
        "workflow_prompt_contains_cyberpunk": frozenset(
            f["id"] for f in files if "cyberpunk" in (f["workflow_prompt"] or "").lower()
        ),
        "status_flag_approved": frozenset(data["membership"][("Approved", "system_flag")]),
        "collection_portfolio": frozenset(data["membership"][("Portfolio", "user_album")]),
        "comment_contains_amazing": frozenset(
            c["file_id"] for c in data["comments"] if "amazing" in c["comment_text"].lower()
        ),
        "has_faces_true": frozenset(fi["file_id"] for fi in data["face_instances"]),
        "rating_avg_ge_4": frozenset(fid for fid, vals in rating_totals.items() if sum(vals) / len(vals) >= 4.0),
        "rated_by_carol": frozenset(r["file_id"] for r in data["ratings"] if r["client_uuid"] == "3"),
    }


# The FIXTURE_* views below describe exactly the database that
# build_fixture_db writes with its default seed.
_DEFAULT_SEED = 42
_DEFAULT_DATA = _generate(_DEFAULT_SEED)

FIXTURE_FILES: list[dict[str, Any]] = _DEFAULT_DATA["files"]
FIXTURE_RATINGS: list[dict[str, Any]] = _DEFAULT_DATA["ratings"]
FIXTURE_COMMENTS: list[dict[str, Any]] = _DEFAULT_DATA["comments"]
FIXTURE_MEMBERSHIP: dict[tuple[str, str], list[str]] = _DEFAULT_DATA["membership"]
FIXTURE_EXPECTATIONS: dict[str, frozenset[str]] = _compute_expectations(_DEFAULT_DATA)
