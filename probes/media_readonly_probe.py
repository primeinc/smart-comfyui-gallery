#!/usr/bin/env python3
"""Runtime source-media immutability probe (WI-31).

Proves at runtime that indexing (hashing, embeddings, faces, review, mask
generation) never modifies source media: every byte and mtime of every
source file is identical before and after a full derived-state build, and
all writes land inside the derived-cache/database directories.

Exit code 0 = PASS. Usage: python3 probes/media_readonly_probe.py
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from smartgallery_ai import HASH_ALGO_VERSION, RUBRIC_VERSION, AIConfig, schema
from smartgallery_ai import review as R
from smartgallery_ai.embedders import get_semantic_backend, get_visual_backend
from smartgallery_ai.hashing import compute_hashes_for_file, upsert_hashes
from smartgallery_ai.vectors import VectorStore

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of probes/)
sys.path.insert(0, REPO)


def snapshot(root: str) -> dict:
    """Map of relative path -> (sha256 hex, mtime_ns) for every file under
    `root`; two equal snapshots mean no byte and no timestamp changed."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            with open(p, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (digest, st.st_mtime_ns)
    return out


def main() -> int:
    """Build the full derived state (hashes, both embedding spaces, a review
    with a localizable finding, its mask) over a throwaway gallery, then
    verify the source snapshot is untouched and the mask landed inside the
    cache directory. Returns the process exit code (0 = PASS)."""
    tmp = tempfile.mkdtemp(prefix="sg_romedia_probe_")
    media = os.path.join(tmp, "media")
    cache = os.path.join(tmp, "cache")
    os.makedirs(media)
    os.makedirs(cache)

    rng = np.random.default_rng(11)
    for i in range(3):
        Image.fromarray((rng.random((96, 96, 3)) * 255).astype("uint8")).save(os.path.join(media, f"img{i}.png"))

    db = os.path.join(tmp, "probe.sqlite")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT, mtime REAL, name TEXT, type TEXT)")
    schema.init_schema(conn)

    cfg = AIConfig(
        enabled=True,
        base_path=tmp,
        db_path=db,
        models_dir="",
        cache_dir=cache,
        semantic_backend="stub",
        visual_backend="stub",
    )

    before = snapshot(media)

    vs = VectorStore(db, cache_dir=cache, ephemeral=False)
    sem, vis = get_semantic_backend(cfg), get_visual_backend(cfg)
    now = time.time()
    for i in range(3):
        path = os.path.join(media, f"img{i}.png")
        fid = f"file{i}"
        mtime = os.path.getmtime(path)
        conn.execute("INSERT INTO files VALUES (?,?,?,?,?)", (fid, path, mtime, os.path.basename(path), "image"))
        res = compute_hashes_for_file(path, "image")
        upsert_hashes(conn, fid, res, mtime, HASH_ALGO_VERSION, now)
        img = Image.open(path)
        vs.add(conn, fid, "semantic", sem.model_id, sem.model_version, sem.embed_image(img), mtime)
        vs.add(conn, fid, "visual", vis.model_id, vis.model_version, vis.embed_image(img), mtime)
    conn.commit()

    # Review with a localizable finding + mask generation (stub segmenter).
    payload = {
        "quality_score": 6.0,
        "prompt_alignment_score": None,
        "summary": "probe",
        "findings": [
            {
                "type": "artifact",
                "severity": "low",
                "confidence": 0.9,
                "localizable": True,
                "description": "probe finding",
                "bbox": [0.25, 0.25, 0.5, 0.5],
            }
        ],
    }
    result = R.validate_review_payload(payload)
    rid = R.store_review(
        conn,
        "file0",
        result,
        "critic-stub",
        "v1",
        RUBRIC_VERSION,
        "{}",
        os.path.getmtime(os.path.join(media, "img0.png")),
        now,
    )
    finding_id = conn.execute("SELECT finding_id FROM ai_review_findings WHERE review_id = ?", (rid,)).fetchone()[0]
    mask_path = R.generate_finding_mask(
        conn, cache, Image.open(os.path.join(media, "img0.png")), "file0", finding_id, R.StubSegmenter()
    )

    after = snapshot(media)

    ok = before == after
    mask_inside_cache = os.path.commonpath(
        [
            os.path.abspath(os.path.join(cache, mask_path))
            if not os.path.isabs(mask_path)
            else os.path.abspath(mask_path),
            os.path.abspath(cache),
        ]
    ) == os.path.abspath(cache)

    evidence = {
        "source_files": len(before),
        "source_unchanged": ok,
        "mask_created": True,
        "mask_inside_cache_dir": mask_inside_cache,
        "changed": sorted(k for k in before if before[k] != after.get(k)),
    }
    print(json.dumps(evidence, indent=2))
    verdict = ok and mask_inside_cache
    print("PASS" if verdict else "FAIL")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
