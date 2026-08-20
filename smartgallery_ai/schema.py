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

import sqlite3
import time

AI_SCHEMA_VERSION = 1  # structural version of the ai_* tables

# How long a connection waits for another writer before giving up.
#
# The gallery and the AI worker write to the same file. In WAL a reader
# never waits, but a writer waits for a writer, and how long it waits is
# this. Python's default is five seconds, which a scan's bulk insert
# passes without trying -- measured against a write held for eight
# seconds, a default connection raised "database is locked" after 5.6s
# while runner.py's timeout=30 connection waited 2.4s and wrote.
#
# The worker's failures land in broad `except Exception` handlers, so
# what a lock costs is not an error anybody sees: it is files that
# quietly never get indexed, on exactly the libraries big enough for the
# scan to hold the lock that long.
#
# Sixty seconds, the same as the gallery's own connection, because there
# is no reason for two writers to the one database to disagree about it.
DB_TIMEOUT_SECONDS = 60


def connect(db_path, row_factory=sqlite3.Row, **kwargs):
    """Open the gallery database the way everything else opens it.

    `PRAGMA foreign_keys` is per-connection and OFF by default in SQLite,
    so every declaration in DDL below -- `REFERENCES files(id) ON DELETE
    CASCADE`, `ON UPDATE CASCADE`, and ai_face_instances' `ON DELETE SET
    NULL` -- was inert on every connection this package opens. The cascade
    held only for deletes performed through the host app's own connection,
    which does enable it; a row removed through one of ours orphaned its
    children silently. The invariant survived by accident, because
    cluster_faces NULLs cluster_id by hand and store_review deletes its own
    children first. Turn it on and the schema means what it says.
    """
    kwargs.setdefault("timeout", DB_TIMEOUT_SECONDS)
    conn = sqlite3.connect(db_path, **kwargs)
    if row_factory is not None:
        conn.row_factory = row_factory
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Idempotent statements executed in order by init_schema().
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
    ("CREATE INDEX IF NOT EXISTS idx_ai_emb_space ON ai_embeddings(space, model_id, model_version);"),
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
        attributes TEXT,
        age INTEGER,
        sex TEXT,
        pose_pitch REAL,
        pose_yaw REAL,
        pose_roll REAL,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        source_mtime REAL NOT NULL,
        computed_at REAL NOT NULL,
        cluster_id INTEGER REFERENCES ai_face_clusters(cluster_id)
            ON DELETE SET NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_faces_file ON ai_face_instances(file_id);",
    ("CREATE INDEX IF NOT EXISTS idx_ai_faces_cluster ON ai_face_instances(cluster_id);"),
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
    ("CREATE INDEX IF NOT EXISTS idx_ai_findings_file ON ai_review_findings(file_id, type);"),
    """
    CREATE TABLE IF NOT EXISTS ai_review_alignment (
        element_id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_id INTEGER NOT NULL REFERENCES ai_reviews(review_id)
            ON DELETE CASCADE,
        file_id TEXT NOT NULL,
        -- position in the generation prompt, so the panel can render the
        -- elements in the order the user actually wrote them
        ordinal INTEGER NOT NULL,
        -- verbatim slice of the positive prompt (never model-authored text)
        text TEXT NOT NULL,
        satisfied INTEGER NOT NULL CHECK (satisfied IN (0, 1)),
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        -- where the element was found. Present for satisfied elements the
        -- model could locate; NULL for whole-image judgements (style,
        -- lighting, "cinematic") and for absent elements, which have no
        -- locus by definition.
        bbox_x REAL,
        bbox_y REAL,
        bbox_w REAL,
        bbox_h REAL,
        mask_path TEXT,
        mask_model_id TEXT,
        mask_model_version TEXT,
        UNIQUE (review_id, ordinal),
        -- an absent element cannot have been located
        CHECK (satisfied = 1 OR (bbox_x IS NULL AND bbox_y IS NULL
               AND bbox_w IS NULL AND bbox_h IS NULL AND mask_path IS NULL))
    );
    """,
    ("CREATE INDEX IF NOT EXISTS idx_ai_alignment_review ON ai_review_alignment(review_id, ordinal);"),
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
    # --- scan bookkeeping: records that a (file, kind, model) was scanned
    # at a given source mtime, INCLUDING zero-result scans (a file with no
    # faces must not be re-scanned every cycle). `scanned_at` is the
    # last-run stamp. The model is part of the key: each pipeline keeps its
    # own run history, so switching backends never erases another model's
    # bookkeeping. kind 'masks' records the segmentation pass over a
    # review's localizable findings as its own segmenter-keyed unit of
    # work, so masks are retried when a segmenter is provisioned later ---
    """
    CREATE TABLE IF NOT EXISTS ai_scan_log (
        file_id TEXT NOT NULL REFERENCES files(id)
            ON DELETE CASCADE ON UPDATE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('faces', 'review', 'masks')),
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        source_mtime REAL NOT NULL,
        scanned_at REAL NOT NULL,
        result_count INTEGER NOT NULL DEFAULT 0,
        -- Digest of the stage's NON-FILE, NON-MODEL inputs (worker.stage_input_key).
        -- source_mtime covers the pixels and model_id/model_version cover the
        -- model, but a review also depends on the generation prompt and the
        -- rubric. Those used to sit outside the key entirely, so a prompt that
        -- was traced AFTER a file was reviewed changed nothing the staleness
        -- check could see, and that review stayed frozen -- with a null
        -- alignment score -- forever. Stages with no such inputs record ''.
        input_key TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (file_id, kind, model_id, model_version)
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
    "ai_review_alignment",
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


def schema_version(conn) -> int:
    """The AI schema version this database was last reconciled to, or 0.

    Stored in ai_dam_state rather than `PRAGMA user_version`, which belongs
    to the host app and is already used by it -- two owners writing one
    integer would be worse than no version at all.
    """
    try:
        row = conn.execute("SELECT value FROM ai_dam_state WHERE key = 'ai_schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0  # ai_dam_state itself predates this, or does not exist yet
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return 0


def init_schema(conn) -> None:
    """Create AI DAM tables if missing and reconcile any table whose stored
    DDL diverges from the DDL here. Idempotent.

    AI_SCHEMA_VERSION is recorded after a successful reconcile. It is the
    declared answer to "has this database been through the current
    migrations", which until now nothing asked: the constant existed and was
    referenced nowhere, leaving each migration to re-derive that for itself
    from an ad-hoc column probe or, in one case, a substring search of the
    stored DDL. Those probes still run -- they are what actually repairs an
    old database, and they are idempotent -- but a change with neither a new
    column nor a matching substring now has somewhere to hang a migration.
    """
    for stmt in DDL:
        conn.execute(stmt)
    _migrate_scan_log_kinds(conn)
    _migrate_scan_log_input_key(conn)
    _migrate_face_attributes(conn)
    conn.execute(
        "INSERT INTO ai_dam_state (key, value, updated_at) VALUES ('ai_schema_version', ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (str(AI_SCHEMA_VERSION), time.time()),
    )
    conn.commit()


def _migrate_scan_log_input_key(conn) -> None:
    """Add `input_key` to databases whose ai_scan_log predates it.

    Existing rows keep '' while the current code computes a real digest, so
    every previously-scanned file mismatches once and re-enters the queue.
    That is the intended effect, not a side effect: those rows were recorded
    under a key that could not see the prompt, so none of them can be
    trusted to be current."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ai_scan_log)").fetchall()}
    if "input_key" not in cols:
        conn.execute("ALTER TABLE ai_scan_log ADD COLUMN input_key TEXT NOT NULL DEFAULT ''")


def _migrate_face_attributes(conn) -> None:
    """Add the per-face attribute columns to databases created before
    they existed. Scalars are real columns so they stay comparable and
    aggregatable in SQL (age/sex from genderage, pitch/yaw/roll from the
    3D landmark head); `attributes` JSON carries the structured geometry
    (normalized landmark arrays). All nullable: existing rows stay valid."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(ai_face_instances)").fetchall()}
    for name, decl in (
        ("attributes", "TEXT"),
        ("age", "INTEGER"),
        ("sex", "TEXT"),
        ("pose_pitch", "REAL"),
        ("pose_yaw", "REAL"),
        ("pose_roll", "REAL"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE ai_face_instances ADD COLUMN {name} {decl}")


def _migrate_scan_log_kinds(conn) -> None:
    """Rebuild ai_scan_log in place, preserving rows, when its stored DDL
    predates either the 'masks' kind or the model-scoped primary key
    (file_id, kind, model_id, model_version). SQLite cannot ALTER a CHECK
    or a PK, but the table is derived bookkeeping, so a rename/copy/drop
    rebuild is safe. Detection reads the stored DDL — deterministic, no
    probe writes."""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='ai_scan_log'").fetchone()
    if row is None:
        return
    current = "'masks'" in row[0] and "kind, model_id, model_version" in row[0]
    if current:
        return
    target_ddl = next(s for s in DDL if "ai_scan_log" in s)
    # One transaction for the whole swap. Python's sqlite3 runs DDL in
    # autocommit and only opens an implicit transaction at the INSERT, so
    # the rename and the CREATE were already durable before the copy began:
    # an interruption in that window left an EMPTY ai_scan_log beside an
    # orphaned ai_scan_log_old, and the DDL sniff above would then see a
    # current table and return early -- so the orphan stayed forever and the
    # rows in it were never copied. BEGIN IMMEDIATE takes the write lock up
    # front and makes the swap all-or-nothing.
    conn.execute("DROP TABLE IF EXISTS ai_scan_log_old")
    # init_schema's own DDL may have left an implicit transaction open, and
    # SQLite has no nested transactions. Settle it before taking the write
    # lock: everything before this point is idempotent CREATE IF NOT EXISTS,
    # so committing it is a no-op on an already-current database.
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("ALTER TABLE ai_scan_log RENAME TO ai_scan_log_old")
    conn.execute(target_ddl.replace("IF NOT EXISTS ", ""))
    # Columns named explicitly, never SELECT *: the old table has whatever
    # column set its vintage had, and a positional copy breaks the moment
    # the target gains one (it did, with input_key). Carried columns are
    # the ones every vintage shares; input_key intentionally defaults to ''
    # so migrated rows read as stale and re-scan under a real key.
    conn.execute(
        """
        INSERT INTO ai_scan_log
            (file_id, kind, model_id, model_version, source_mtime,
             scanned_at, result_count)
        SELECT file_id, kind, model_id, model_version, source_mtime,
               scanned_at, result_count
        FROM ai_scan_log_old
        """
    )
    conn.execute("DROP TABLE ai_scan_log_old")
    conn.commit()


def drop_derived_state(conn, keep_feedback: bool = True) -> None:
    """Drop rebuildable derived tables (rebuild path).

    Human feedback is preserved by default because it cannot be recomputed.
    """
    tables = REBUILDABLE_TABLES if keep_feedback else DERIVED_TABLES
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
