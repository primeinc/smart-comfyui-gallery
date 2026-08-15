"""Derived-AI database schema.

All tables live in the main SmartGallery SQLite database so that foreign keys
to files(id) cascade on delete/rename. Every table here is DERIVED state:
it can be dropped and rebuilt from source media + provisioned models.
`files` and the other core tables remain the authoritative system of record.

Conventions:
  - vectors are float32 little-endian BLOBs, dimension recorded per row
  - bounding boxes are normalized [0,1] floats relative to image width/height
  - perceptual hashes are stored as signed 64-bit integers (two's complement)
  - every derived row records source_mtime + model/algo version so staleness
    is decidable without opening the media file (see invalidation.py)
"""

AI_SCHEMA_VERSION = 1

DDL = [
    # --- content identity / near-duplicates (no GPU required) ---
    """
    CREATE TABLE IF NOT EXISTS ai_file_hashes (
        file_id TEXT PRIMARY KEY REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        sha256 TEXT NOT NULL,
        phash64 INTEGER,
        dhash64 INTEGER,
        algo_version TEXT NOT NULL,
        source_mtime REAL NOT NULL,
        computed_at REAL NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_hashes_sha256 ON ai_file_hashes(sha256);",
    "CREATE INDEX IF NOT EXISTS idx_ai_hashes_phash ON ai_file_hashes(phash64);",

    # --- embedding spaces (semantic and visual are SEPARATE spaces) ---
    """
    CREATE TABLE IF NOT EXISTS ai_embeddings (
        file_id TEXT NOT NULL REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        space TEXT NOT NULL,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL,
        source_mtime REAL NOT NULL,
        computed_at REAL NOT NULL,
        PRIMARY KEY (file_id, space)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_emb_space "
    "ON ai_embeddings(space, model_id, model_version);",

    # --- face instances (one row per detected face; multi-face assets OK) ---
    """
    CREATE TABLE IF NOT EXISTS ai_face_clusters (
        cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        centroid BLOB,
        dim INTEGER,
        size INTEGER NOT NULL DEFAULT 0,
        params TEXT,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_face_instances (
        face_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT NOT NULL REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        bbox_x REAL NOT NULL,
        bbox_y REAL NOT NULL,
        bbox_w REAL NOT NULL,
        bbox_h REAL NOT NULL,
        landmarks TEXT,
        det_score REAL,
        embedding BLOB,
        dim INTEGER,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        source_mtime REAL NOT NULL,
        computed_at REAL NOT NULL,
        cluster_id INTEGER REFERENCES ai_face_clusters(cluster_id)
            ON DELETE SET NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_faces_file ON ai_face_instances(file_id);",
    "CREATE INDEX IF NOT EXISTS idx_ai_faces_cluster "
    "ON ai_face_instances(cluster_id);",

    # --- generation review (typed findings; masks only when localizable) ---
    """
    CREATE TABLE IF NOT EXISTS ai_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT NOT NULL REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        rubric_version TEXT NOT NULL,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        quality_score REAL,
        prompt_alignment_score REAL,
        summary TEXT,
        raw_response TEXT,
        source_mtime REAL NOT NULL,
        computed_at REAL NOT NULL,
        UNIQUE (file_id, rubric_version, model_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_review_findings (
        finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_id INTEGER NOT NULL REFERENCES ai_reviews(review_id)
            ON DELETE CASCADE,
        file_id TEXT NOT NULL,
        type TEXT NOT NULL,
        severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        localizable INTEGER NOT NULL CHECK (localizable IN (0, 1)),
        bbox_x REAL,
        bbox_y REAL,
        bbox_w REAL,
        bbox_h REAL,
        points TEXT,
        description TEXT NOT NULL,
        mask_path TEXT,
        mask_model_id TEXT,
        mask_model_version TEXT,
        -- masks and grounding geometry are only meaningful for localizable
        -- findings; global findings must keep ALL of these columns NULL.
        -- (Existing databases created before this CHECK was widened keep
        -- the narrower constraint until rebuilt; validate_review_payload
        -- enforces the full invariant in code regardless.)
        CHECK (localizable = 1 OR (bbox_x IS NULL AND bbox_y IS NULL
               AND bbox_w IS NULL AND bbox_h IS NULL
               AND points IS NULL AND mask_path IS NULL))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_findings_file "
    "ON ai_review_findings(file_id, type);",

    # --- human feedback (exportable for reviewer tuning / LoRA work) ---
    """
    CREATE TABLE IF NOT EXISTS ai_feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_kind TEXT NOT NULL CHECK (target_kind IN
            ('review', 'finding', 'similarity', 'face_cluster', 'duplicate')),
        target_id TEXT NOT NULL,
        file_id TEXT,
        verdict TEXT NOT NULL CHECK (verdict IN
            ('accept', 'reject', 'false_positive', 'rating')),
        rating INTEGER CHECK (rating IS NULL OR (rating >= 1 AND rating <= 5)),
        note TEXT,
        created_by TEXT,
        created_at REAL NOT NULL,
        exported_at REAL
    );
    """,

    # --- scan bookkeeping: records that a (file, kind) was scanned with a
    # given model at a given source mtime, INCLUDING zero-result scans
    # (a file with no faces must not be re-scanned every cycle) ---
    """
    CREATE TABLE IF NOT EXISTS ai_scan_log (
        file_id TEXT NOT NULL REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('faces', 'review')),
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        source_mtime REAL NOT NULL,
        scanned_at REAL NOT NULL,
        result_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (file_id, kind)
    );
    """,

    # --- small key/value state (active model versions, measured thresholds) ---
    """
    CREATE TABLE IF NOT EXISTS ai_dam_state (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at REAL NOT NULL
    );
    """,
]

# Tables owned by this package, in drop order (children before parents).
DERIVED_TABLES = [
    "ai_review_findings",
    "ai_reviews",
    "ai_face_instances",
    "ai_face_clusters",
    "ai_embeddings",
    "ai_file_hashes",
    "ai_scan_log",
    "ai_feedback",
    "ai_dam_state",
]

# ai_feedback is human-authored, not derived: never drop it on rebuild.
REBUILDABLE_TABLES = [t for t in DERIVED_TABLES if t != "ai_feedback"]


def init_schema(conn) -> None:
    """Create AI DAM tables if missing. Idempotent."""
    for stmt in DDL:
        conn.execute(stmt)
    conn.commit()


def drop_derived_state(conn, keep_feedback: bool = True) -> None:
    """Drop rebuildable derived tables (rebuild path).

    Human feedback is preserved by default because it cannot be recomputed.
    """
    tables = REBUILDABLE_TABLES if keep_feedback else DERIVED_TABLES
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
