"""Moving a database forward without destroying what a person put in it.

`check_version` refuses a database this build does not recognise, and until
now the only way past that refusal was `db.build --force`, which deletes every
rating, comment, album and name in the file. That is a data-loss upgrade path
wearing a version check.

Forward only. Rolling a schema change back is the case the Well-Architected
guidance singles out as hardest -- "Rolling back changes, especially database,
schema, or other stateful component changes, can be complex. Your SDP
guidelines should provide clear instructions on how to deal with data changes"
(refs/MicrosoftDocs/well-architected/well-architected/operational-excellence/
safe-deployments.md:67). This is that instruction: there is no down step, a
downgrade is refused by name, and the way back is the backup taken on the way
out (:108, "Preserve state before removal. Take a snapshot, export, or backup")
which is named for the version it holds (:71, versioning across artifacts).

Three things about the driver decide the shape of the runner, all of them
from Doc/library/sqlite3.rst in the CPython tree:

`isolation_level=None` is the only mode where nothing is implicitly opened:
"If isolation_level is set to None, no transactions are implicitly opened at
all ... but also allows the user to perform their own transaction handling
using explicit SQL statements" (:2722-2728). The runner needs that, because a
pragma it depends on does not apply inside a transaction.

`PRAGMA foreign_keys` inside a transaction is a silent no-op -- no error, it
simply does not take. Verified on SQLite 3.47.1 rather than assumed. So the
pragma is set before BEGIN, never within, and the default connection would
have opened one already: under the legacy default "a transaction is implicitly
opened before executing sql" for any INSERT/UPDATE/DELETE/REPLACE (:1504-1509).

`executescript` must never appear in a migration step. It "implicitly commits
any pending transaction before execution of the given SQL script, regardless
of the value of isolation_level" (:2730-2732) -- so one call inside a step
would commit the half-finished migration and take atomicity with it.
"""

from __future__ import annotations

import pathlib
import shutil
import sqlite3
from collections.abc import Callable

from .connect import APPLICATION_ID, USER_VERSION

#: from_version -> the step that takes a database to from_version + 1.
STEPS: dict[int, Callable[[sqlite3.Connection], None]] = {}


class Downgrade(Exception):
    """The file is newer than this build. There is no way back."""


class StepMissing(Exception):
    """No migration exists for a version this build is supposed to reach."""


class NotOurDatabase(Exception):
    """The file is not a gallery database."""


def step(from_version: int):
    """Register the step that moves a database off `from_version`.

    Steps are numbered by where they start, not where they end, so a gap is
    obvious: the runner asks for `STEPS[3]` and either has it or stops.
    """

    def register(fn):
        if from_version in STEPS:
            raise ValueError(f"two migrations claim v{from_version}")
        STEPS[from_version] = fn
        return fn

    return register


def version_of(conn: sqlite3.Connection) -> int:
    app = conn.execute("PRAGMA application_id").fetchone()[0]
    if app != APPLICATION_ID:
        raise NotOurDatabase(f"application_id={app:#x}, not a gallery database")
    return conn.execute("PRAGMA user_version").fetchone()[0]


def pending(path, *, target: int = USER_VERSION) -> list[int]:
    """The versions a migration would pass through, without touching anything."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        current = version_of(conn)
    finally:
        conn.close()
    if current > target:
        raise Downgrade(f"database is v{current}, this build is v{target}")
    return list(range(current + 1, target + 1))


def snapshot(path, version: int) -> pathlib.Path:
    """Copy the database aside before changing it.

    `Connection.backup` rather than a file copy: it "Works even if the
    database is being accessed by other clients or concurrently by the same
    connection" (sqlite3.rst:1130-1135), where copying the file while a WAL
    is live captures a torn database that looks fine until it is opened.

    Named for the version it holds, so the file itself says what it can be
    restored into.
    """
    path = pathlib.Path(path)
    target = path.with_suffix(f".v{version}.backup")
    if target.exists():
        target.unlink()
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    destination = sqlite3.connect(str(target))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def _nothing_dangling(conn, version: int) -> None:
    """Refuse to commit a step that left references pointing at nothing."""
    broken = conn.execute("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise sqlite3.IntegrityError(
            f"v{version} -> v{version + 1} left {len(broken)} dangling references: {broken[:5]}"
        )


def migrate(path, *, target: int = USER_VERSION, take_snapshot: bool = True) -> list[int]:
    """Bring a database up to `target`, one version at a time.

    Returns the versions applied. Each step is its own transaction, so a
    failure leaves the file at the last version that completed rather than
    somewhere between two.
    """
    path = pathlib.Path(path)
    # isolation_level=None: explicit transaction control, and pragmas that
    # actually apply. See the module docstring.
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        current = version_of(conn)
        if current > target:
            raise Downgrade(
                f"database is v{current}, this build is v{target}. Restore the "
                f"backup taken when it was upgraded; there is no down step."
            )
        if current == target:
            return []

        missing = [v for v in range(current, target) if v not in STEPS]
        if missing:
            raise StepMissing(
                f"no migration from v{missing[0]}: this build cannot open a v{current} database without one"
            )

        if take_snapshot:
            snapshot(path, current)

        conn.execute("PRAGMA busy_timeout=5000")
        # Before BEGIN, never inside: the pragma is silently ignored within a
        # transaction. A step that rebuilds a table needs the keys off, and
        # `foreign_key_check` below is what proves nothing was left dangling.
        conn.execute("PRAGMA foreign_keys=OFF")

        applied: list[int] = []
        try:
            for version in range(current, target):
                conn.execute("BEGIN IMMEDIATE")
                try:
                    STEPS[version](conn)
                    conn.execute(f"PRAGMA user_version = {version + 1}")
                    _nothing_dangling(conn, version)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                applied.append(version + 1)
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
        return applied
    finally:
        conn.close()


def restore(backup, path) -> None:
    """Put a snapshot back. The only way out of a failed upgrade."""
    path = pathlib.Path(path)
    for suffix in ("-wal", "-shm"):
        pathlib.Path(str(path) + suffix).unlink(missing_ok=True)
    shutil.copyfile(backup, path)


@step(1)
def _smart_collections_refuse_stored_members(conn: sqlite3.Connection) -> None:
    """v1 -> v2: the guards keeping smart collections rule-defined.

    One execute per statement, never executescript -- see the module
    docstring. NOT purely additive: a v1 library could file rows into a
    smart collection, because the guards did not exist -- that hole is
    why they do now -- and stamping those rows forward would produce a
    database whose data violates its own triggers' intent. Deleting them
    unasked is not this schema's way either, so the step refuses by
    name and the operator empties or re-kinds the collections first;
    the refusal rolls this step back and the file stays at v1.
    """
    offenders = conn.execute(
        "SELECT c.name, count(*) FROM collection c"
        " JOIN collection_file cf ON cf.collection_id = c.id"
        " WHERE c.kind = 'smart' GROUP BY c.id ORDER BY c.name"
    ).fetchall()
    if offenders:
        named = "; ".join(f"{name!r} holds {n} filed row(s)" for name, n in offenders)
        raise sqlite3.IntegrityError(
            f"a smart collection derives its members from its rule, and this library filed rows into: {named}."
            f" Empty or re-kind those collections, then migrate again."
        )
    conn.execute(
        "CREATE TRIGGER collection_file_not_into_smart BEFORE INSERT ON collection_file "
        "WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) = 'smart' BEGIN "
        "SELECT RAISE(ABORT,'a smart collection derives its members from its rule; nothing is filed into it'); "
        "END"
    )
    conn.execute(
        "CREATE TRIGGER collection_file_not_moved_into_smart BEFORE UPDATE OF collection_id ON collection_file "
        "WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) = 'smart' BEGIN "
        "SELECT RAISE(ABORT,'a smart collection derives its members from its rule; nothing is filed into it'); "
        "END"
    )
    conn.execute(
        "CREATE TRIGGER collection_with_members_stays_listed BEFORE UPDATE OF kind ON collection "
        "WHEN NEW.kind = 'smart' AND OLD.kind <> 'smart' "
        "AND EXISTS (SELECT 1 FROM collection_file WHERE collection_id = NEW.id) BEGIN "
        "SELECT RAISE(ABORT,'this collection holds filed members; empty it before making it smart'); "
        "END"
    )


@step(2)
def _near_duplicate_groups(conn: sqlite3.Connection) -> None:
    """v2 -> v3: where the dupes job writes its groups. Purely additive
    for real this time: a new empty table cannot disagree with anything.

    The DDL is schema.sql's block VERBATIM, comments included:
    sqlite_master stores the literal statement text, and the drift check
    compares it -- a migrated database must be indistinguishable from a
    fresh build down to the words a reader of `.schema` sees.
    """
    conn.execute(
        """CREATE TABLE derived_dupe_group (
    file_id     INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    -- the group's seed: its lowest member id. Deleting the seed file
    -- cascades the whole group away; the next job rebuilds what remains.
    group_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    -- hamming bits from this member's phash64 to the BEST member's -- the
    -- canonical image every member is a duplicate OF. Never to an arbitrary
    -- neighbour: A~B and B~C does not make A a duplicate of C, and a chain
    -- is exactly how a "duplicate group" collects two pictures its own
    -- verifier says are different.
    distance    INTEGER NOT NULL CHECK (distance BETWEEN 0 AND 64),
    threshold   INTEGER NOT NULL CHECK (threshold BETWEEN 0 AND 64),
    is_best     INTEGER NOT NULL DEFAULT 0 CHECK (is_best IN (0, 1)),
    -- 1: this member's dHash agreed with the best member's -- two
    -- independent fingerprints both said duplicate. 0: pHash alone said so
    -- (a dHash was missing, or verification was off). A verified duplicate
    -- and an unverified candidate are different claims, and a page that
    -- cannot tell them apart flattens the difference into false confidence.
    verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    computed_at REAL NOT NULL
) STRICT"""
    )
    conn.execute("CREATE INDEX derived_dupe_group_group ON derived_dupe_group(group_id)")
    conn.execute("CREATE UNIQUE INDEX derived_dupe_group_best ON derived_dupe_group(group_id) WHERE is_best = 1")


@step(3)
def _embeddings_get_immutable_ids(conn: sqlite3.Connection) -> None:
    """v3 -> v4: `derived_embedding` keyed by an immutable id of its own.

    Under v3 the table was keyed (file_id, space_id) and the resident FAISS
    index stored file ids -- a mutable identity: re-embedding a replaced
    file changed the vector UNDER the id, and a crash between commit and
    index sync left a snapshot answering queries with the old picture's
    vector. v4 gives every vector an AUTOINCREMENT id the index stores
    instead; a replacement is a NEW id and alignment sees exactly that.

    The rebuild copies every existing row -- the vectors cost GPU-seconds
    each and nothing about them is wrong, only their key -- and mints ids
    in deterministic (space_id, file_id) order. Resident index checkpoints
    keyed by the old ids simply fail their digest check and re-align.

    Version 3 drifted during development. A v3 file from before the
    similarity rework may lack `similarity_space` or `derived_embedding`
    entirely; both are created empty here, and the derived_* rebuild
    contract covers what jobs regenerate. A pre-space embedding shape
    (no space_id column) cannot name what produced its vectors, so those
    rows do not survive: an unattributable vector answering queries is
    worse than a re-embed.
    """
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "similarity_space" not in tables:
        conn.execute(
            """CREATE TABLE similarity_space (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    key                TEXT NOT NULL,
    representation     TEXT NOT NULL CHECK (representation IN ('binary','float32')),
    dimensions         INTEGER NOT NULL CHECK (dimensions > 0),
    metric             TEXT NOT NULL CHECK (metric IN ('hamming','cosine')),
    producer           TEXT NOT NULL,
    producer_version   TEXT NOT NULL,
    preprocess         TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    spec_hash          TEXT NOT NULL UNIQUE,
    created_at         REAL NOT NULL
) STRICT"""
        )
        conn.execute(
            """CREATE TRIGGER similarity_space_is_immutable
BEFORE UPDATE ON similarity_space
BEGIN
    SELECT RAISE(ABORT, 'similarity_space rows are immutable: a changed meaning is a new space');
END"""
        )

    survivors = False
    if "derived_embedding" in tables:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(derived_embedding)")}
        survivors = "space_id" in columns
        if survivors:
            conn.execute(
                "CREATE TABLE migrate_v3_embedding AS SELECT"
                " file_id, space_id, vector, source_sha256, computed_at FROM derived_embedding"
            )
        conn.execute("DROP TABLE derived_embedding")

    # schema.sql's block VERBATIM, comments included, same rationale as the
    # v2 -> v3 step: the drift check compares sqlite_master text.
    conn.execute(
        """CREATE TABLE derived_embedding (
    -- AUTOINCREMENT for the same reason as derived_face_instance: this id
    -- is what the resident index stores, and index alignment treats an id
    -- as an IMMUTABLE identity. A file's embedding legitimately changes
    -- (re-embed after an in-place byte replacement, a recompute), so the
    -- file id cannot be the vector's identity -- a crash between commit
    -- and index sync would leave a snapshot holding the OLD vector under
    -- an id alignment has no reason to doubt. A replacement row is a NEW
    -- id; the old id disappears; alignment sees exactly that.
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    space_id      INTEGER NOT NULL REFERENCES similarity_space(id) ON DELETE RESTRICT,
    -- Packed float32, as the encoder emits it. No dim column: the space
    -- row owns the dimensions, and the triggers below hold every vector
    -- to them -- a second copy of the same fact is a place for the two
    -- to disagree.
    vector        BLOB NOT NULL,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL,
    UNIQUE (file_id, space_id)
) STRICT"""
    )
    conn.execute("CREATE INDEX derived_embedding_space ON derived_embedding(space_id)")
    conn.execute(
        """CREATE TRIGGER derived_embedding_fits_its_space
BEFORE INSERT ON derived_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'embedding length disagrees with its space''s dimensions');
END"""
    )
    conn.execute(
        """CREATE TRIGGER derived_embedding_fits_its_space_update
BEFORE UPDATE ON derived_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'embedding length disagrees with its space''s dimensions');
END"""
    )
    if survivors:
        conn.execute(
            "INSERT INTO derived_embedding(file_id, space_id, vector, source_sha256, computed_at)"
            " SELECT file_id, space_id, vector, source_sha256, computed_at"
            " FROM migrate_v3_embedding ORDER BY space_id, file_id"
        )
        conn.execute("DROP TABLE migrate_v3_embedding")


def optimize(conn: sqlite3.Connection) -> None:
    """Let SQLite refresh the statistics the planner runs on.

    A table is analyzed only when an index has no `sqlite_stat1` entry, when
    the planner used those statistics on this connection, or when the table
    grew or shrank tenfold
    (refs/sqlite/sqlite/src/pragma.c:2485-2510). Cheap in the normal case --
    the usual outcome is that no ANALYZE runs at all.

    Worth calling when a connection closes and after a scan, because a
    library that has just gone from empty to 100k files has statistics that
    describe a database that no longer exists.
    """
    conn.execute("PRAGMA optimize")


def analyze(conn: sqlite3.Connection) -> None:
    """Force the full pass over every table, not only the queried ones.

    `PRAGMA optimize` alone will not look at a table the connection never
    queried; the 0x10000 bit is what makes it consider every table
    ("Look at tables to see if they need to be reanalyzed due to growth or
    shrinkage even if they have not been queried during the current
    connection. Off by default." -- pragma.c:2475-2478).

    Worth running after a first scan, when a library has just gone from empty
    to a hundred thousand files and the planner's statistics describe a
    database that no longer exists.

    Measured on 100k files, the people page: 17.7 ms with no statistics,
    5.4 ms with them. The plan changes from scanning `file` to driving from
    300 people through an index -- which is right, and which needs a page
    cache big enough to hold the random lookups it implies. At SQLite's 2 MiB
    default the same analyzed plan measures 60 ms, and the first reading of
    that number here was that ANALYZE had made things worse. It had not; see
    `connect.CACHE_KIB`. A plan is only as good as the cache under it, and a
    benchmark that leaves the cache at its default is measuring the default.
    """
    conn.execute("PRAGMA optimize=0x10012")
