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


@step(4)
def _sibling_names_ignore_case(conn: sqlite3.Connection) -> None:
    """v4 -> v5: sibling names are unique the way the platform spells them.

    The scanner has always matched directory names COLLATE NOCASE
    (db/scan.py ensure_folder) and files carry the same rule in
    file_in_folder -- but the folder uniqueness indexes were binary, so
    the schema permitted live siblings 'Vacation' and 'vacation' that
    the scanner treats as one directory. The indexes now say what the
    scanner does; NOCASE also lets the folder and collection child
    listings ride them for their name ordering instead of a temp B-tree
    (collection_parent widens for the same reason).

    A library that somehow holds case-equivalent live siblings cannot be
    stamped forward without merging identities nobody asked to merge, so
    the step refuses and NAMES them; the file stays at v4. Index DDL is
    schema.sql's text VERBATIM -- the drift check compares sqlite_master.
    """
    twins = conn.execute(
        "SELECT group_concat(name, ' / ') FROM folder"
        " WHERE parent_id IS NOT NULL AND missing_since IS NULL"
        " GROUP BY parent_id, name COLLATE NOCASE HAVING count(*) > 1"
    ).fetchall()
    twins += conn.execute(
        "SELECT group_concat(name, ' / ') FROM folder"
        " WHERE parent_id IS NULL AND missing_since IS NULL"
        " GROUP BY root_id, name COLLATE NOCASE HAVING count(*) > 1"
    ).fetchall()
    if twins:
        named = "; ".join(row[0] for row in twins)
        raise sqlite3.IntegrityError(
            f"these live sibling directories differ only by case, and the scanner treats each pair as one:"
            f" {named}. Rename them apart on disk and rescan, then migrate again."
        )
    conn.execute("DROP INDEX folder_root_unique")
    conn.execute("DROP INDEX folder_child_unique")
    conn.execute("DROP INDEX collection_parent")
    conn.execute(
        """CREATE UNIQUE INDEX folder_root_unique  ON folder(root_id, name COLLATE NOCASE)
    WHERE parent_id IS NULL AND missing_since IS NULL"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX folder_child_unique ON folder(parent_id, name COLLATE NOCASE)
    WHERE parent_id IS NOT NULL AND missing_since IS NULL"""
    )
    conn.execute("CREATE INDEX collection_parent ON collection(parent_id, name COLLATE NOCASE)")


@step(5)
def _the_time_index_carries_the_tiebreak(conn: sqlite3.Connection) -> None:
    """v5 -> v6: file_recent gains the id tiebreak, in the sort's own
    direction.

    The ResultSet's ordering contract has always been (mtime DESC,
    id DESC) and its reversal -- the same contract
    file_in_folder_by_time carries -- but file_recent held only
    (mtime DESC), so every time-sorted global membership walk fell back
    to a whole-membership sort at read time. The index now implements
    the contract; the contract did not bend to the index. Index DDL is
    schema.sql's text VERBATIM -- the drift check compares
    sqlite_master. No row is touched.
    """
    conn.execute("DROP INDEX file_recent")
    conn.execute("CREATE INDEX file_recent ON file(mtime DESC, id DESC) WHERE missing_since IS NULL")


@step(6)
def _smart_rules_get_their_own_table(conn: sqlite3.Connection) -> None:
    """v6 -> v7: a smart collection's rule moves to `collection_rule`.

    The collection row says what the entity IS; the rule row says how
    dynamic membership is derived -- typed, versioned, never executable.
    Existing smart rows move losslessly: nl_text becomes source_text,
    sql_text becomes legacy_sql_text, rule_json stays NULL -- explicitly
    UNEVALUATED until somebody authors a typed rule, because "execute
    arbitrary SQL somebody saved months ago" is not a capability.

    The column removal is the twelve-step rebuild ALTER TABLE cannot
    express, so every trigger on `collection` is recreated -- DDL is
    schema.sql's text VERBATIM; the drift check compares sqlite_master.
    """
    conn.execute(
        """CREATE TABLE collection_rule (
    collection_id INTEGER PRIMARY KEY REFERENCES collection(id) ON DELETE CASCADE,
    rule_version  INTEGER,
    rule_json     TEXT CHECK (rule_json IS NULL OR json_valid(rule_json)),
    -- WHOSE judgement the rule's authored facets (favorite, rating_min)
    -- mean -- pinned at creation, never the viewer. RESTRICT: nulling it
    -- would silently change what the rule answers.
    actor_id      INTEGER REFERENCES user(id) ON DELETE RESTRICT,
    source_text     TEXT,
    legacy_sql_text TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    CHECK ((rule_json IS NULL AND rule_version IS NULL) OR (rule_json IS NOT NULL AND rule_version IS NOT NULL))
) STRICT"""
    )
    conn.execute("CREATE INDEX collection_rule_actor ON collection_rule(actor_id)")
    conn.execute(
        "INSERT INTO collection_rule(collection_id, source_text, legacy_sql_text, created_at, updated_at)"
        " SELECT id, nl_text, sql_text, created_at, created_at FROM collection"
        " WHERE kind = 'smart' AND (nl_text IS NOT NULL OR sql_text IS NOT NULL)"
    )
    # The old table is renamed AWAY and the new one created under the
    # final name directly: renaming new->old would leave a QUOTED table
    # name in sqlite_master's stored DDL, which the drift check rightly
    # counts as a different schema. legacy_alter_table for the rename,
    # because the modern form validates the collection_file triggers
    # whose subject is mid-rebuild -- sqlite's own escape for the
    # 12-step dance, restored immediately after.
    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.execute("ALTER TABLE collection RENAME TO collection_old")
    conn.execute("PRAGMA legacy_alter_table=OFF")
    conn.execute(
        """CREATE TABLE collection (
    id          INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    parent_id   INTEGER REFERENCES collection(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('album','flag','smart')),
    color       TEXT,
    description TEXT,
    created_at  REAL NOT NULL
) STRICT"""
    )
    conn.execute(
        "INSERT INTO collection(id, parent_id, name, kind, color, description, created_at)"
        " SELECT id, parent_id, name, kind, color, description, created_at FROM collection_old"
    )
    conn.execute("DROP TABLE collection_old")
    conn.execute("CREATE INDEX collection_parent ON collection(parent_id, name COLLATE NOCASE)")
    conn.execute(
        """CREATE TRIGGER collection_no_self_parent BEFORE INSERT ON collection
WHEN NEW.parent_id IS NOT NULL AND NEW.parent_id = NEW.id BEGIN
  SELECT RAISE(ABORT,'collection parent cycle');
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_no_cycle BEFORE UPDATE OF parent_id ON collection
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'collection parent cycle') WHERE NEW.id IN (
    WITH RECURSIVE up(id) AS (
      SELECT NEW.parent_id
      UNION SELECT a.parent_id FROM collection a JOIN up ON a.id = up.id
        WHERE a.parent_id IS NOT NULL)
    SELECT id FROM up);
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_collection_ins AFTER INSERT ON collection
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_collection_upd AFTER UPDATE OF name ON collection BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_collection_del AFTER DELETE ON collection BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_kind_agrees BEFORE INSERT ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_kind_keeps_agreeing BEFORE UPDATE OF id ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_takes_its_entity AFTER DELETE ON collection BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_with_members_stays_listed BEFORE UPDATE OF kind ON collection
WHEN NEW.kind = 'smart' AND OLD.kind <> 'smart'
 AND EXISTS (SELECT 1 FROM collection_file WHERE collection_id = NEW.id) BEGIN
  SELECT RAISE(ABORT,'this collection holds filed members; empty it before making it smart');
END"""
    )


@step(7)
def _one_membership_definition_per_collection(conn: sqlite3.Connection) -> None:
    """v7 -> v8: the database itself refuses two membership definitions.

    v7 let a listed collection carry a dormant collection_rule row and
    let a rule-carrying smart collection become listed -- two authored
    answers waiting to disagree, exactly what the collection_file
    guards were built to make impossible in the other direction. A v7
    library that already holds a rule on a non-smart collection cannot
    be stamped forward without deciding which answer wins, so the step
    refuses and NAMES them; deleting or re-kinding is the operator's
    deliberate act. Trigger DDL is schema.sql's text VERBATIM.
    """
    strays = conn.execute(
        "SELECT c.name FROM collection_rule r JOIN collection c ON c.id = r.collection_id"
        " WHERE c.kind <> 'smart' ORDER BY c.name"
    ).fetchall()
    if strays:
        named = "; ".join(repr(name) for (name,) in strays)
        raise sqlite3.IntegrityError(
            f"these listed collections carry a dormant rule, a second membership definition: {named}."
            f" Delete the rule or make the collection smart, then migrate again."
        )
    conn.execute(
        """CREATE TRIGGER collection_rule_only_on_smart BEFORE INSERT ON collection_rule
WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) <> 'smart' BEGIN
  SELECT RAISE(ABORT,'only a smart collection carries a rule; a listed collection''s membership is its filed rows');
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_rule_stays_on_smart BEFORE UPDATE OF collection_id ON collection_rule
WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) <> 'smart' BEGIN
  SELECT RAISE(ABORT,'only a smart collection carries a rule; a listed collection''s membership is its filed rows');
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_with_rule_stays_smart BEFORE UPDATE OF kind ON collection
WHEN OLD.kind = 'smart' AND NEW.kind <> 'smart'
 AND EXISTS (SELECT 1 FROM collection_rule WHERE collection_id = NEW.id) BEGIN
  SELECT RAISE(ABORT,'this collection is rule-defined; delete its rule before making it listed');
END"""
    )


@step(8)
def _a_collection_carries_its_lifecycle(conn: sqlite3.Connection) -> None:
    """v8 -> v9: the collection row gains the facts its lifecycle needs.

    updated_at/created_by/updated_by say who last defined it,
    archived_at makes retirement a state instead of a deletion, and
    definition_rev is the optimistic-concurrency guard every definition
    write checks. parent_id turns ON DELETE RESTRICT: authored children
    have independent addresses, and deleting an organizer must never
    silently take a subtree with it. Existing rows stamp
    updated_at = created_at, no authorship, active, revision 1 -- no
    authored state changes. The FK change is the twelve-step rebuild;
    every trigger on `collection` is recreated with schema.sql's text
    VERBATIM -- the drift check compares sqlite_master.
    """
    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.execute("ALTER TABLE collection RENAME TO collection_old")
    conn.execute("PRAGMA legacy_alter_table=OFF")
    conn.execute(
        """CREATE TABLE collection (
    id          INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    -- RESTRICT: authored children have independent addresses, and deleting
    -- an organizer must never silently take a subtree with it.
    parent_id   INTEGER REFERENCES collection(id) ON DELETE RESTRICT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('album','flag','smart')),
    color       TEXT,
    description TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    created_by  INTEGER REFERENCES user(id) ON DELETE SET NULL,
    updated_by  INTEGER REFERENCES user(id) ON DELETE SET NULL,
    -- Lifecycle, never deletion: archiving keeps the address, the members,
    -- the children and the rule; it changes discoverability only. NULL
    -- means active.
    archived_at REAL,
    -- Optimistic concurrency over the DEFINITION -- name, kind, color,
    -- description, parent, archive state, rule. Membership never bumps it:
    -- filing a picture does not invalidate an open description editor.
    definition_rev INTEGER NOT NULL DEFAULT 1 CHECK (definition_rev > 0)
) STRICT"""
    )
    conn.execute(
        "INSERT INTO collection(id, parent_id, name, kind, color, description,"
        " created_at, updated_at, created_by, updated_by, archived_at, definition_rev)"
        " SELECT id, parent_id, name, kind, color, description,"
        " created_at, created_at, NULL, NULL, NULL, 1 FROM collection_old"
    )
    conn.execute("DROP TABLE collection_old")
    conn.execute("CREATE INDEX collection_parent ON collection(parent_id, name COLLATE NOCASE)")
    conn.execute("CREATE INDEX collection_created_by ON collection(created_by)")
    conn.execute("CREATE INDEX collection_updated_by ON collection(updated_by)")
    conn.execute(
        """CREATE TRIGGER collection_no_self_parent BEFORE INSERT ON collection
WHEN NEW.parent_id IS NOT NULL AND NEW.parent_id = NEW.id BEGIN
  SELECT RAISE(ABORT,'collection parent cycle');
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_no_cycle BEFORE UPDATE OF parent_id ON collection
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'collection parent cycle') WHERE NEW.id IN (
    WITH RECURSIVE up(id) AS (
      SELECT NEW.parent_id
      UNION SELECT a.parent_id FROM collection a JOIN up ON a.id = up.id
        WHERE a.parent_id IS NOT NULL)
    SELECT id FROM up);
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_collection_ins AFTER INSERT ON collection
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_collection_upd AFTER UPDATE OF name ON collection BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_collection_del AFTER DELETE ON collection BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_kind_agrees BEFORE INSERT ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_kind_keeps_agreeing BEFORE UPDATE OF id ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_takes_its_entity AFTER DELETE ON collection BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_with_members_stays_listed BEFORE UPDATE OF kind ON collection
WHEN NEW.kind = 'smart' AND OLD.kind <> 'smart'
 AND EXISTS (SELECT 1 FROM collection_file WHERE collection_id = NEW.id) BEGIN
  SELECT RAISE(ABORT,'this collection holds filed members; empty it before making it smart');
END"""
    )
    conn.execute(
        """CREATE TRIGGER collection_with_rule_stays_smart BEFORE UPDATE OF kind ON collection
WHEN OLD.kind = 'smart' AND NEW.kind <> 'smart'
 AND EXISTS (SELECT 1 FROM collection_rule WHERE collection_id = NEW.id) BEGIN
  SELECT RAISE(ABORT,'this collection is rule-defined; delete its rule before making it listed');
END"""
    )


@step(9)
def _media_gets_a_context(conn: sqlite3.Connection) -> None:
    """v9 -> v10: places become entities, media gets its ONE derived
    context, events get their grouping tables -- and three CHECK
    vocabularies widen, which SQLite can only say as a rebuild: entity
    and slug_history learn the 'place' kind, job learns 'context' and
    'events'.

    The entity rebuild carries the AUTOINCREMENT doctrine by hand:
    sqlite_sequence holds the largest id EVER issued, which can exceed
    max(id) after deletions, and losing it would let a new entity take a
    dead entity's id -- the exact reuse the column exists to forbid. The
    old sequence is read first and restored after, whichever is larger.
    All DDL is schema.sql's text VERBATIM; the drift check compares
    sqlite_master.
    """
    minted = conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'sqlite_sequence'").fetchone()[0]
    held = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'entity'").fetchone() if minted else None
    old_seq = held[0] if held else 0
    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.execute("ALTER TABLE entity RENAME TO entity_old")
    conn.execute("PRAGMA legacy_alter_table=OFF")
    conn.execute(
        """CREATE TABLE entity (
    -- ================= CONVENTIONS FOR THE WHOLE SCHEMA =================
    -- Deliberately inside a CREATE statement. SQLite keeps only the comments
    -- that sit within one; everything written above a table is discarded, so
    -- a rule stated there is invisible to anyone reading the built database
    -- and survives only in the source file. This is the first table, so this
    -- is the first thing `.schema` prints.
    --
    -- TIME   Every *_at column is UNIX EPOCH SECONDS IN UTC, as a REAL.
    --        The one exception is capture.captured_at, which may be a wall
    --        clock with no zone -- it says so itself, and capture.tz_offset_min
    --        is how you tell which it is.
    -- SIZE   Bytes.
    -- SCORES det_score and confidence are 0..1, never percentages.
    -- ANGLES Degrees.
    -- BOXES  Fractions of the frame, 0..1. See `region`.
    -- ====================================================================
    -- AUTOINCREMENT, which on a rowid table means "never hand out an id this
    -- table has ever used". Plain `INTEGER PRIMARY KEY` reuses the largest
    -- free rowid, and the minter compounded it by computing `max(id) + 1`
    -- itself: delete the newest entity and the next one created took its id
    -- with a different uuid, so anything holding an id outside this database
    -- -- a thumbnail cache key, an export, a bookmarked address -- silently
    -- resolved to a different picture. SQLite keeps the maximum ever used in
    -- `sqlite_sequence` (refs/sqlite/sqlite/src/insert.c:385-391).
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid BLOB NOT NULL UNIQUE CHECK (length(uuid) = 16),
    kind TEXT NOT NULL CHECK (kind IN
           ('file','folder','person','artifact','prompt','collection','place')),
    slug TEXT NOT NULL,
    UNIQUE (kind, slug)
) STRICT"""
    )
    conn.execute("INSERT INTO entity(id, uuid, kind, slug) SELECT id, uuid, kind, slug FROM entity_old")
    conn.execute("DROP TABLE entity_old")
    conn.execute(
        """CREATE TRIGGER entity_kind_is_permanent BEFORE UPDATE OF kind ON entity
WHEN NEW.kind <> OLD.kind BEGIN
  SELECT RAISE(ABORT,'an entity cannot change kind');
END"""
    )
    conn.execute(
        "INSERT INTO sqlite_sequence(name, seq) SELECT 'entity', 0"
        " WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'entity')"
    )
    conn.execute("UPDATE sqlite_sequence SET seq = max(seq, ?) WHERE name = 'entity'", (old_seq,))

    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.execute("ALTER TABLE slug_history RENAME TO slug_history_old")
    conn.execute("PRAGMA legacy_alter_table=OFF")
    conn.execute(
        """CREATE TABLE slug_history (
    -- The same list `entity.kind` is held to. Unconstrained, a retirement
    -- could name a kind no entity can ever be, and that address then
    -- resolves to nothing for the rest of the library's life.
    kind       TEXT    NOT NULL CHECK (kind IN
                 ('file','folder','person','artifact','prompt','collection','place')),
    slug       TEXT    NOT NULL,
    entity_id  INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    retired_at REAL    NOT NULL,
    -- retired_at is in the key: a slug released, reissued and released again
    -- must be recordable twice. Resolution order is fixed: a live
    -- entity.slug always wins, history answers only on a miss, most recent
    -- retirement first.
    PRIMARY KEY (kind, slug, retired_at)
) STRICT, WITHOUT ROWID"""
    )
    conn.execute(
        "INSERT INTO slug_history(kind, slug, entity_id, retired_at)"
        " SELECT kind, slug, entity_id, retired_at FROM slug_history_old"
    )
    conn.execute("DROP TABLE slug_history_old")
    conn.execute("CREATE INDEX slug_history_entity ON slug_history(entity_id)")

    conn.execute("PRAGMA legacy_alter_table=ON")
    conn.execute("ALTER TABLE job RENAME TO job_old")
    conn.execute("PRAGMA legacy_alter_table=OFF")
    conn.execute(
        """CREATE TABLE job (
    id               INTEGER PRIMARY KEY,
    -- Constrained like every other `kind` here. A typo is otherwise a job
    -- that queues successfully and no worker ever claims, because claim()
    -- filters on the kinds it knows -- so it waits forever and looks fine.
    kind             TEXT NOT NULL CHECK (kind IN
                       ('scan','hash','embed','detect_faces','cluster_faces',
                        'sample_frames','annotate','remix','zip','context','events')),
    target_id        INTEGER REFERENCES entity(id) ON DELETE SET NULL,
    state            TEXT NOT NULL CHECK (state IN
                       ('queued','running','done','failed','cancelled')),
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0,1)),
    payload          TEXT,
    total            INTEGER,
    done_count       INTEGER NOT NULL DEFAULT 0,
    checkpoint       TEXT,
    attempt          INTEGER NOT NULL DEFAULT 0,
    -- a lease nobody owns cannot fence anyone: the reclaiming worker must be
    -- able to prove it holds the job, and the evicted one must be rejected.
    owner            TEXT,
    fence            INTEGER NOT NULL DEFAULT 0,
    lease_until      REAL,
    heartbeat_at     REAL,
    error            TEXT,
    -- No external_ref here. `derivation_intent` already carries the
    -- generator's own id, UNIQUE, and having it on both meant two rows could
    -- claim the same external job and disagree about which one owned it.
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
) STRICT"""
    )
    conn.execute(
        "INSERT INTO job(id, kind, target_id, state, cancel_requested, payload, total,"
        " done_count, checkpoint, attempt, owner, fence, lease_until, heartbeat_at,"
        " error, created_at, started_at, finished_at)"
        " SELECT id, kind, target_id, state, cancel_requested, payload, total,"
        " done_count, checkpoint, attempt, owner, fence, lease_until, heartbeat_at,"
        " error, created_at, started_at, finished_at FROM job_old"
    )
    conn.execute("DROP TABLE job_old")
    conn.execute("CREATE INDEX job_state ON job(state)")
    conn.execute("CREATE INDEX job_target ON job(target_id)")

    conn.execute(
        """CREATE TABLE place (
    id           INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    -- RESTRICT: places nest, and deleting a region must never silently
    -- take its cities' identities with it.
    parent_id    INTEGER REFERENCES place(id) ON DELETE RESTRICT,
    kind         TEXT NOT NULL CHECK (kind IN
                   ('country','region','island','county','city','locality','neighborhood','poi')),
    name         TEXT NOT NULL,
    centroid_lat REAL,
    centroid_lon REAL,
    country_code TEXT,
    -- Which enrichment provider claimed this place, and its key there --
    -- provenance for refresh, never identity.
    provider     TEXT,
    provider_key TEXT,
    created_at   REAL NOT NULL
) STRICT"""
    )
    conn.execute("""CREATE INDEX place_parent ON place(parent_id)""")
    conn.execute(
        """CREATE TRIGGER place_kind_agrees BEFORE INSERT ON place BEGIN
  SELECT RAISE(ABORT,'entity kind does not match place')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'place';
END"""
    )
    conn.execute(
        """CREATE TRIGGER place_kind_keeps_agreeing BEFORE UPDATE OF id ON place BEGIN
  SELECT RAISE(ABORT,'entity kind does not match place')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'place';
END"""
    )
    conn.execute(
        """CREATE TRIGGER place_takes_its_entity AFTER DELETE ON place BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END"""
    )
    conn.execute(
        """CREATE TRIGGER place_no_self_parent BEFORE INSERT ON place
WHEN NEW.parent_id IS NOT NULL AND NEW.parent_id = NEW.id BEGIN
  SELECT RAISE(ABORT,'place parent cycle');
END"""
    )
    conn.execute(
        """CREATE TRIGGER place_no_cycle BEFORE UPDATE OF parent_id ON place
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'place parent cycle') WHERE NEW.id IN (
    WITH RECURSIVE up(id) AS (
      SELECT NEW.parent_id
      UNION SELECT a.parent_id FROM place a JOIN up ON a.id = up.id
        WHERE a.parent_id IS NOT NULL)
    SELECT id FROM up);
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_place_ins AFTER INSERT ON place
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_place_upd AFTER UPDATE OF name ON place BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END"""
    )
    conn.execute(
        """CREATE TRIGGER name_fts_place_del AFTER DELETE ON place BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END"""
    )
    conn.execute(
        """CREATE TABLE derived_media_context (
    file_id             INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    origin              TEXT NOT NULL CHECK (origin IN
                          ('captured','generated','imported','unknown')),
    -- TWO time concepts, never one column doing both jobs: `local_at`
    -- is what the human clock said (the wall time a camera claimed);
    -- `instant_at` is the actual UTC instant, present ONLY when
    -- knowable. An unzoned camera claim keeps its wall time and has no
    -- instant -- a known human clock is never replaced by a filesystem
    -- time just to make a column easier to sort.
    local_at            REAL,
    instant_at          REAL,
    tz_offset_min       INTEGER,
    time_basis          TEXT CHECK (time_basis IN
                          ('capture','embedded','btime','mtime','first_seen')),
    time_certainty      REAL CHECK (time_certainty BETWEEN 0 AND 1),
    gps_lat             REAL,
    gps_lon             REAL,
    place_id            INTEGER REFERENCES place(id) ON DELETE SET NULL,
    location_basis      TEXT CHECK (location_basis IN
                          ('gps','sidecar','inferred','authored')),
    location_certainty  REAL CHECK (location_certainty BETWEEN 0 AND 1),
    rebuilt_at          REAL NOT NULL,
    -- a time without a recorded basis is an unexplained date
    CHECK ((time_basis IS NULL) = (local_at IS NULL AND instant_at IS NULL)),
    -- an offset explains a wall clock; without one it explains nothing
    CHECK (tz_offset_min IS NULL OR local_at IS NOT NULL)
) STRICT"""
    )
    conn.execute("""CREATE INDEX media_context_when ON derived_media_context(instant_at)""")
    conn.execute("""CREATE INDEX media_context_local ON derived_media_context(local_at)""")
    conn.execute("""CREATE INDEX media_context_place ON derived_media_context(place_id)""")
    conn.execute("""CREATE INDEX media_context_origin_when ON derived_media_context(origin, instant_at)""")
    conn.execute(
        """CREATE TABLE derived_event_run (
    id              INTEGER PRIMARY KEY,
    grouper         TEXT NOT NULL,
    grouper_version TEXT NOT NULL,
    settings_hash   TEXT NOT NULL,
    created_at      REAL NOT NULL
) STRICT"""
    )
    conn.execute("""CREATE INDEX event_run_grouper ON derived_event_run(grouper, created_at)""")
    conn.execute(
        """CREATE TABLE derived_event (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES derived_event_run(id) ON DELETE CASCADE,
    parent_id   INTEGER REFERENCES derived_event(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('generation_session','capture_session')),
    start_at    REAL NOT NULL,
    end_at      REAL NOT NULL,
    place_id    INTEGER REFERENCES place(id) ON DELETE SET NULL,
    confidence  REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    member_hash TEXT NOT NULL,
    CHECK (start_at <= end_at)
) STRICT"""
    )
    conn.execute("""CREATE INDEX event_run ON derived_event(run_id)""")
    conn.execute("""CREATE INDEX event_parent ON derived_event(parent_id)""")
    conn.execute("""CREATE INDEX event_when ON derived_event(start_at)""")
    conn.execute("""CREATE INDEX event_place ON derived_event(place_id)""")
    conn.execute(
        """CREATE TABLE derived_event_file (
    event_id INTEGER NOT NULL REFERENCES derived_event(id) ON DELETE CASCADE,
    file_id  INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    score    REAL,
    PRIMARY KEY (event_id, file_id)
) STRICT, WITHOUT ROWID"""
    )
    conn.execute("""CREATE INDEX event_file_file ON derived_event_file(file_id)""")


@step(10)
def _time_gets_a_domain_and_context_gets_an_identity(conn: sqlite3.Connection) -> None:
    """v10 -> v11: truth over time. Event intervals carry their temporal
    DOMAIN (a wall-clock pair, an instant pair, or both -- never one
    ambiguous pair); every event run names the context generation and
    policy it was computed over; origin is DETERMINED from coexisting
    capture/generation facts by CHECK instead of asserted by precedence;
    and `derived_context_state` is the interpretation's identity.

    Every changed table is derived and rebuildable by contract, so the
    step drops and recreates -- children first, because with foreign
    keys on a child cannot even be dropped after its parent is gone.
    The explicit context and events jobs repopulate. DDL is schema.sql's
    text VERBATIM; the drift check compares sqlite_master.
    """
    for table in ("derived_event_file", "derived_event", "derived_event_run", "derived_media_context"):
        conn.execute(f"DROP TABLE {table}")
    conn.execute(
        """CREATE TABLE derived_media_context (
    file_id             INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    -- Coexistence is FACT, never precedence: a photograph that was also
    -- run through a generator has both claims, and `origin` is fully
    -- determined from them by CHECK -- a classification that could
    -- silently erase one fact is the lie this shape forbids.
    has_capture         INTEGER NOT NULL CHECK (has_capture IN (0, 1)),
    has_generation      INTEGER NOT NULL CHECK (has_generation IN (0, 1)),
    origin              TEXT NOT NULL CHECK (origin IN
                          ('captured','generated','mixed','imported')),
    -- TWO time concepts, never one column doing both jobs: `local_at`
    -- is what the human clock said (the wall time a camera or a
    -- generator claimed); `instant_at` is the actual UTC instant,
    -- present ONLY when knowable. An unzoned claim keeps its wall time
    -- and has no instant -- a known human clock is never replaced by a
    -- filesystem time to make a column easier to sort.
    local_at            REAL,
    instant_at          REAL,
    tz_offset_min       INTEGER,
    time_basis          TEXT CHECK (time_basis IN
                          ('capture','embedded','btime','mtime','first_seen')),
    time_certainty      REAL CHECK (time_certainty BETWEEN 0 AND 1),
    -- How FINE the claim is -- orthogonal to certainty: a day-resolution
    -- generator date can be almost certainly the right DAY while saying
    -- nothing about minutes, and a distrusted btime is subsecond-fine.
    -- Coarse claims are never promoted into fine-grained boundaries.
    time_precision      TEXT CHECK (time_precision IN
                          ('day','hour','minute','second','subsecond')),
    gps_lat             REAL,
    gps_lon             REAL,
    place_id            INTEGER REFERENCES place(id) ON DELETE SET NULL,
    location_basis      TEXT CHECK (location_basis IN
                          ('gps','sidecar','inferred','authored')),
    location_certainty  REAL CHECK (location_certainty BETWEEN 0 AND 1),
    -- WHICH MEANING produced this row: the interpretation ladder's own
    -- version, so a better ladder tomorrow visibly obsoletes today's
    -- rows instead of impersonating them.
    policy_version      INTEGER NOT NULL,
    rebuilt_at          REAL NOT NULL,
    -- a time without a recorded basis is an unexplained date
    CHECK ((time_basis IS NULL) = (local_at IS NULL AND instant_at IS NULL)),
    -- and one without a precision is an unexplained kind of date
    CHECK ((time_basis IS NULL) = (time_precision IS NULL)),
    -- an offset explains a wall clock; without one it explains nothing
    CHECK (tz_offset_min IS NULL OR local_at IS NOT NULL),
    -- origin is DETERMINED, never asserted
    CHECK (origin = CASE
             WHEN has_generation = 1 AND has_capture = 1 THEN 'mixed'
             WHEN has_generation = 1 THEN 'generated'
             WHEN has_capture = 1 THEN 'captured'
             ELSE 'imported' END)
) STRICT"""
    )
    conn.execute("""CREATE INDEX media_context_when ON derived_media_context(instant_at)""")
    conn.execute("""CREATE INDEX media_context_local ON derived_media_context(local_at)""")
    conn.execute("""CREATE INDEX media_context_place ON derived_media_context(place_id)""")
    conn.execute("""CREATE INDEX media_context_origin_when ON derived_media_context(origin, instant_at)""")
    conn.execute(
        """CREATE TABLE derived_context_state (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    generation     INTEGER NOT NULL,
    policy_version INTEGER NOT NULL
) STRICT"""
    )
    conn.execute(
        """CREATE TABLE derived_event_run (
    id                     INTEGER PRIMARY KEY,
    grouper                TEXT NOT NULL,
    grouper_version        TEXT NOT NULL,
    settings_hash          TEXT NOT NULL,
    context_generation     INTEGER NOT NULL,
    context_policy_version INTEGER NOT NULL,
    created_at             REAL NOT NULL
) STRICT"""
    )
    conn.execute("""CREATE INDEX event_run_grouper ON derived_event_run(grouper, created_at)""")
    conn.execute(
        """CREATE TABLE derived_event (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES derived_event_run(id) ON DELETE CASCADE,
    parent_id     INTEGER REFERENCES derived_event(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('generation_session','capture_session')),
    -- The interval carries its TEMPORAL DOMAIN: a wall-clock pair, an
    -- instant pair, or both when every member makes both knowable --
    -- never one ambiguous pair that is secretly sometimes each. Unlike
    -- domains are never subtracted from each other.
    local_start   REAL,
    local_end     REAL,
    instant_start REAL,
    instant_end   REAL,
    place_id      INTEGER REFERENCES place(id) ON DELETE SET NULL,
    confidence    REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    member_hash   TEXT NOT NULL,
    CHECK ((local_start IS NULL) = (local_end IS NULL)),
    CHECK ((instant_start IS NULL) = (instant_end IS NULL)),
    CHECK (local_start IS NULL OR local_start <= local_end),
    CHECK (instant_start IS NULL OR instant_start <= instant_end),
    CHECK (local_start IS NOT NULL OR instant_start IS NOT NULL)
) STRICT"""
    )
    conn.execute("""CREATE INDEX event_run ON derived_event(run_id)""")
    conn.execute("""CREATE INDEX event_parent ON derived_event(parent_id)""")
    conn.execute("""CREATE INDEX event_when_instant ON derived_event(instant_start)""")
    conn.execute("""CREATE INDEX event_when_local ON derived_event(local_start)""")
    conn.execute("""CREATE INDEX event_place ON derived_event(place_id)""")
    conn.execute(
        """CREATE TABLE derived_event_file (
    event_id INTEGER NOT NULL REFERENCES derived_event(id) ON DELETE CASCADE,
    file_id  INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    score    REAL,
    PRIMARY KEY (event_id, file_id)
) STRICT, WITHOUT ROWID"""
    )
    conn.execute("""CREATE INDEX event_file_file ON derived_event_file(file_id)""")


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
