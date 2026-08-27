"""Upgrading the app must not cost the user what they put in the library.

The version check refuses a database this build does not recognise, and the
only way past it was a rebuild that deletes every rating, comment, album and
name. These pin the way out: forward-only steps, each in its own transaction,
with a snapshot taken first and a downgrade refused by name.

The step used throughout is a table rebuild -- the twelve-step dance SQLite
requires for a change ALTER TABLE cannot express -- because that is the case
where a migration runner either holds or loses the data.
"""

import gc
import itertools
import pathlib
import shutil
import sqlite3
import sys
import uuid

import pytest

from db import authored, build, collections, connect, migrate, scan
from tests import schemas
from tests.staging import NOW, migrated


@pytest.fixture
def pinned_identity(monkeypatch):
    """Deterministic entity uuids while a test stages its fixture.

    `scan.mint` draws uuid4, so a staged database's logical content --
    and with it `migrated`'s cache key -- was different every run, and
    the cached replay could never be found again."""
    counter = itertools.count(1)
    monkeypatch.setattr(scan.uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))


@pytest.fixture
def steps():
    """A clean registry, restored afterwards, so tests cannot leak into each
    other or into the real migration list."""
    original = dict(migrate.STEPS)
    migrate.STEPS.clear()
    yield migrate.STEPS
    migrate.STEPS.clear()
    migrate.STEPS.update(original)


_TEMPLATE: list[pathlib.Path] = []


@pytest.fixture(scope="module", autouse=True)
def built(tmp_path_factory):
    """Today's build, made once; every database below starts as a copy of
    it -- the same file build.build writes, without executing the DDL
    on disk per test."""
    template = tmp_path_factory.mktemp("built") / "gallery.db"
    build.build(template)
    _TEMPLATE.append(template)
    yield template
    _TEMPLATE.clear()


def _built(path: pathlib.Path) -> None:
    shutil.copy(_TEMPLATE[0], path)


@pytest.fixture
def library(tmp_path):
    """A built database with a person's work already in it."""
    path = tmp_path / "gallery.db"
    _built(path)
    conn = connect.connect(path)
    root = int(
        conn.execute(
            "INSERT INTO root(path, kind, created_at) VALUES(?, 'library', ?)",
            (str(tmp_path / "pics"), NOW),
        ).lastrowid
        or 0
    )
    folder = scan.ensure_folder(conn, root, None, "pics")
    file_id = scan.mint(conn, "file", "dusk")
    conn.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'dusk.png', 'image', 10, 0, 'aa', ?, ?)",
        (file_id, folder, NOW, NOW),
    )
    user = authored.add_user(conn, "will", "hash", "ADMIN", NOW)
    person = authored.person(conn, "Ilse", NOW)
    album = collections.collection(conn, "Keepers", NOW)
    authored.rate(conn, file_id, user, 5, NOW)
    authored.comment(conn, file_id, user, "the good one", NOW)
    authored.favourite(conn, file_id, user, NOW)
    collections.set_membership(conn, album, file_id, True, NOW)
    authored.assert_person(conn, person, file_id, user, NOW)
    conn.commit()
    conn.close()
    return path


def authored_state(path):
    """Everything a person made, as one comparable value."""
    conn = connect.connect(path, read_only=True)
    try:
        return {
            "rating": conn.execute("SELECT rating FROM rating").fetchall(),
            "comments": conn.execute("SELECT body FROM comment").fetchall(),
            "favourites": conn.execute("SELECT count(*) FROM favorite").fetchone()[0],
            "albums": conn.execute("SELECT name FROM collection").fetchall(),
            "membership": conn.execute("SELECT count(*) FROM collection_file").fetchone()[0],
            "people": conn.execute("SELECT name FROM person").fetchall(),
            "assertions": conn.execute("SELECT count(*) FROM person_assertion").fetchone()[0],
            "slugs": conn.execute("SELECT kind, slug FROM entity ORDER BY id").fetchall(),
        }
    finally:
        conn.close()


def rebuild_comment_table(conn):
    """A migration of the kind ALTER TABLE cannot express.

    Adds a column and a CHECK by building the new table, copying, dropping and
    renaming -- the case a runner has to survive, and the reason foreign keys
    are off while it runs.
    """
    conn.execute("""
        CREATE TABLE comment_new (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
            body TEXT NOT NULL CHECK (length(body) > 0),
            pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0,1)),
            created_at REAL NOT NULL, edited_at REAL
        ) STRICT
    """)
    conn.execute(
        "INSERT INTO comment_new(id, file_id, user_id, body, created_at, edited_at)"
        " SELECT id, file_id, user_id, body, created_at, edited_at FROM comment"
    )
    conn.execute("DROP TABLE comment")
    conn.execute("ALTER TABLE comment_new RENAME TO comment")
    conn.execute("CREATE INDEX comment_file ON comment(file_id)")
    conn.execute("CREATE INDEX comment_user ON comment(user_id)")


# --- the contract ----------------------------------------------------------


def _backup_of(library: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(library).replace(".db", f".v{connect.USER_VERSION}.backup"))


def _user_version(path) -> int:
    conn = connect.connect(path, read_only=True)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _columns_of(path, table: str) -> list[str]:
    conn = connect.connect(path, read_only=True)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def test_an_upgrade_keeps_everything_a_person_made(library, steps):
    before = authored_state(library)
    steps[connect.USER_VERSION] = rebuild_comment_table

    applied = migrate.migrate(library, target=connect.USER_VERSION + 1)

    assert applied == [connect.USER_VERSION + 1]
    assert authored_state(library) == before, "the upgrade cost the user something"
    assert _user_version(library) == connect.USER_VERSION + 1
    # the rebuild really happened, so this is not passing by doing nothing
    assert "pinned" in _columns_of(library, "comment")


def test_the_snapshot_is_taken_before_anything_changes(library, steps):
    before = authored_state(library)
    steps[connect.USER_VERSION] = rebuild_comment_table

    migrate.migrate(library, target=connect.USER_VERSION + 1)

    backup = _backup_of(library)
    assert backup.exists(), "no way back was left"
    assert authored_state(backup) == before
    assert _user_version(backup) == connect.USER_VERSION
    assert "pinned" not in _columns_of(backup, "comment")


def test_restoring_the_snapshot_undoes_the_upgrade(library, steps):
    before = authored_state(library)
    steps[connect.USER_VERSION] = rebuild_comment_table
    migrate.migrate(library, target=connect.USER_VERSION + 1)
    assert "pinned" in _columns_of(library, "comment")

    migrate.restore(_backup_of(library), library)

    assert authored_state(library) == before
    assert _user_version(library) == connect.USER_VERSION
    assert "pinned" not in _columns_of(library, "comment")


@pytest.fixture
def newer_library(library):
    """A file stamped five versions ahead of this build."""
    conn = connect.connect(library, autocommit=True)
    conn.execute(f"PRAGMA user_version = {connect.USER_VERSION + 5}")
    conn.close()
    return library


def test_a_downgrade_is_refused_by_name(newer_library, steps):
    """A newer file opened by an older build. Guessing at it is how a
    database gets truncated to the columns this build happens to know."""
    with pytest.raises(migrate.Downgrade, match="no down step"):
        migrate.migrate(newer_library)


def test_pending_refuses_a_downgrade_too(newer_library, steps):
    with pytest.raises(migrate.Downgrade):
        migrate.pending(newer_library)


def test_a_missing_step_stops_before_touching_the_file(library, steps):
    """The registry is keyed on where a step starts, so a gap is visible
    rather than being discovered halfway through."""
    before = authored_state(library)
    with pytest.raises(migrate.StepMissing, match=f"v{connect.USER_VERSION}"):
        migrate.migrate(library, target=connect.USER_VERSION + 2)
    assert authored_state(library) == before
    conn = connect.connect(library, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
    finally:
        conn.close()


def test_a_step_that_fails_leaves_the_database_where_it_was(library, steps):
    before = authored_state(library)

    def half_way(conn):
        conn.execute("DELETE FROM rating")
        raise RuntimeError("the migration hit a problem")

    steps[connect.USER_VERSION] = half_way
    with pytest.raises(RuntimeError, match="hit a problem"):
        migrate.migrate(library, target=connect.USER_VERSION + 1)

    assert authored_state(library) == before, "a failed step took the ratings with it"
    conn = connect.connect(library, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
    finally:
        conn.close()


def test_a_step_that_orphans_a_row_is_rolled_back(library, steps):
    """Foreign keys are off while a step runs, which is what makes a table
    rebuild possible and also what lets a careless step leave a dangling
    reference. `PRAGMA foreign_key_check` before COMMIT is the net."""
    before = authored_state(library)

    def orphan_the_ratings(conn):
        conn.execute("DELETE FROM user")

    steps[connect.USER_VERSION] = orphan_the_ratings
    with pytest.raises(sqlite3.IntegrityError, match="dangling"):
        migrate.migrate(library, target=connect.USER_VERSION + 1)

    assert authored_state(library) == before
    conn = connect.connect(library, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
    finally:
        conn.close()


def test_each_step_commits_on_its_own(library, steps):
    """Two versions, the second failing. The first must stand, or a long
    upgrade is all-or-nothing and a failure at step nine repeats steps one
    to eight on the next attempt."""
    steps[connect.USER_VERSION] = rebuild_comment_table

    def fails(conn):
        raise RuntimeError("second step")

    steps[connect.USER_VERSION + 1] = fails
    with pytest.raises(RuntimeError, match="second step"):
        migrate.migrate(library, target=connect.USER_VERSION + 2)

    conn = connect.connect(library, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION + 1
        assert "pinned" in [r[1] for r in conn.execute("PRAGMA table_info(comment)")]
    finally:
        conn.close()


def test_a_current_database_has_nothing_pending(library, steps):
    assert migrate.pending(library) == []


def test_migrating_a_current_database_does_nothing(library, steps):
    assert migrate.migrate(library) == []
    assert not _backup_of(library).exists(), "a no-op upgrade left a snapshot behind"


def test_something_that_is_not_ours_is_not_migrated(tmp_path, steps):
    other = tmp_path / "someone-elses.db"
    conn = connect.connect(other)
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()
    with pytest.raises(migrate.NotOurDatabase):
        migrate.migrate(other)


def test_pending_says_what_would_happen_without_doing_it(library, steps):
    """Two versions behind, counted from the current one so this test says the
    same thing whatever USER_VERSION happens to be."""
    before = authored_state(library)
    behind = connect.USER_VERSION
    conn = connect.connect(library, autocommit=True)
    conn.execute(f"PRAGMA user_version = {behind}")
    conn.close()
    assert migrate.pending(library, target=behind + 2) == [behind + 1, behind + 2]
    assert authored_state(library) == before


def test_two_steps_cannot_claim_one_version(steps):
    """A duplicate is a silent skip otherwise: whichever registered last wins
    and the other migration never runs."""
    steps[7] = lambda conn: None
    with pytest.raises(ValueError, match="two migrations"):
        migrate.step(7)(lambda conn: None)


# --- the REAL registry, not synthetic steps ---------------------------------


def a_v1_smart_collection(conn, name: str, sql: str) -> int:
    """A smart row as v1 wrote it: rule text on the collection itself."""
    cid = scan.mint(conn, "collection", name)
    conn.execute(
        "INSERT INTO collection(id, name, kind, sql_text, created_at) VALUES(?, ?, 'smart', ?, ?)",
        (cid, name, sql, NOW),
    )
    return cid


def v1_database(tmp_path):
    """The schema that shipped as v1, executed.

    Not today's build with the steps inverted back down. That fixture
    could not fail the way a real v1 database fails, because it started
    from the answer: every object today's schema has, it had, whatever
    the steps did or did not do on the way. It is why the drift check
    below was green for thirty-five versions over a migration that
    leaves a real v1 library with no `derived_face_space`
    (KNOWN_DRIFT).
    """
    path = tmp_path / "gallery.db"
    schemas.seed(path, 1)
    return path


def a_file_row(conn):
    root = int(
        # The uuid stated, not left to the schema's randomblob(16) default:
        # the staged fixture is `migrated`'s cache key, and one random blob
        # made every replay unfindable on the next run.
        conn.execute(
            "INSERT INTO root(path, kind, created_at, uuid) VALUES('Z:/x', 'library', ?, ?)", (NOW, b"\x01" * 16)
        ).lastrowid
        or 0
    )
    # Straight INSERT, not `scan.ensure_folder`: this row goes into a
    # database that has been stepped BACK to an older shape, and the
    # scanner speaks today's column names.
    folder = scan.mint(conn, "folder", "x")
    conn.execute(
        "INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, NULL, 'x', 0)",
        (folder, root),
    )
    file_id = scan.mint(conn, "file", "dusk")
    conn.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'dusk.png', 'image', 10, 0, ?, ?)",
        (file_id, folder, NOW, NOW),
    )
    return file_id


@pytest.mark.slow
def test_the_shipped_steps_take_a_v1_database_to_the_current_build(tmp_path):
    """The real STEPS, executed: every other test here swaps in synthetic
    steps to prove the runner, which left the one migration that actually
    ships executed by nothing. A v1 file must migrate, and every trigger
    the steps install must FIRE afterwards -- including the moved-into
    one no other test reaches.

    Behaviour only. Whether the result matches a fresh build is
    `test_an_authentic_database_of_each_version_reaches_today`, and for
    v1 it does not: see KNOWN_DRIFT. This test asserted `drift == []`
    here for thirty-five versions and was believed, because the fixture
    it asserted it about was today's build wearing a lower number.
    """
    path = v1_database(tmp_path)
    assert migrated(path) == list(range(2, connect.USER_VERSION + 1))

    conn = connect.connect(path)
    try:
        file_id = a_file_row(conn)
        smart = collections.collection(conn, "Big seeds", NOW, kind="smart")
        album = collections.collection(conn, "Keepers", NOW)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (smart, file_id, NOW))
        conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (album, file_id, NOW))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE collection_file SET collection_id = ? WHERE collection_id = ?", (smart, album))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE collection SET kind = 'smart' WHERE id = ?", (album,))
    finally:
        conn.close()


def test_a_legacy_smart_membership_stops_the_migration_by_name(tmp_path):
    """The hole's own artifact: a v1 library that filed rows into a smart
    collection, because nothing refused it then. Migrating those rows
    forward would stamp a v2 database whose data violates the invariant
    v2 exists to establish -- and deleting them unasked is not this
    schema's way. The step refuses, names the collection, and leaves the
    file at v1 with its rows intact."""
    path = v1_database(tmp_path)
    conn = connect.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    file_id = a_file_row(conn)
    smart = a_v1_smart_collection(conn, "Big seeds", "SELECT 1")
    conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (smart, file_id, NOW))
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="Big seeds"):
        migrate.migrate(path)

    conn = connect.connect(path, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1, "a refused step must leave the file at v1"
        assert conn.execute("SELECT count(*) FROM collection_file").fetchone()[0] == 1, "the rows are the human's"
    finally:
        conn.close()


def test_case_twin_siblings_stop_the_migration_by_name(tmp_path):
    """The binary indexes' own artifact: a v4 library holding live
    siblings that differ only by case, which the scanner treats as one
    directory. Stamping them forward would merge identities nobody asked
    to merge; deleting one is worse. The step refuses, names both
    spellings, and leaves the file at v4 with its rows intact."""
    path = tmp_path / "gallery.db"
    # The schema that shipped as v4, which permits the twins because its
    # sibling indexes are binary -- no inversion needed to say so.
    schemas.seed(path, 4)
    conn = connect.connect(path, autocommit=True)
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = conn.execute("INSERT INTO root(path, kind, created_at) VALUES('Z:/x', 'library', 0)").lastrowid
    assert root_id is not None
    # Straight INSERT rather than `scan.ensure_folder`: the scanner writes
    # `fs_id`, a v31 column, and this database really is a v4 one. That is
    # the cost of an authentic fixture and it is the point of one -- the
    # inverted v4 accepted the scanner because it was today's schema.
    top = scan.mint(conn, "folder", "x")
    conn.execute("INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, NULL, 'x', 0)", (top, root_id))
    for name in ("Vacation", "vacation"):
        twin = scan.mint(conn, "folder", name)
        # Straight INSERT: ensure_folder itself matches NOCASE and would
        # hand back the first twin instead of creating the second.
        conn.execute(
            "INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, ?, ?, 0)",
            (twin, root_id, top, name),
        )
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="Vacation"):
        migrate.migrate(path)

    conn = connect.connect(path, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4, "a refused step must leave the file at v4"
        assert conn.execute("SELECT count(*) FROM folder WHERE parent_id IS NOT NULL").fetchone()[0] == 2
    finally:
        conn.close()


# --- planner statistics ----------------------------------------------------


def test_a_fresh_build_has_no_statistics_until_it_is_asked(library):
    """A planner with no stats guesses, and on a 100k-file library it guesses
    wrong. `analyze` is the pass that has to run after the first scan."""
    conn = connect.connect(library)
    try:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()[0] == 0
        migrate.analyze(conn)
        conn.commit()
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='sqlite_stat1'").fetchone()[0] == 1
    finally:
        conn.close()


def test_optimize_is_cheap_on_a_database_nothing_has_queried(library):
    """The usual outcome is that no ANALYZE runs at all, which is what makes
    it safe to call on every close."""
    conn = connect.connect(library)
    try:
        migrate.optimize(conn)
        conn.commit()
    finally:
        conn.close()


# --- against the real schema, not a table invented for the test -------------


def rebuild_the_file_table(conn):
    """The hardest migration this schema can be asked for.

    Twenty-three columns across twenty tables reference `file`, and it is the
    table every page reads. Rebuilding it is the twelve-step dance with the
    most at stake: foreign keys are off while it runs, so every one of those
    references is unprotected for the duration, and a copy that loses a
    column or drops a row detaches somebody's ratings from their pictures.

    `rebuild_comment_table` above proves the runner works, on a table two
    things point at. This proves it where it would actually hurt.
    """
    conn.execute(
        "CREATE TABLE file_new ("
        " id INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,"
        " folder_id INTEGER NOT NULL REFERENCES folder(id) ON DELETE CASCADE,"
        " name TEXT NOT NULL,"
        " kind TEXT NOT NULL CHECK (kind IN"
        "   ('image','animated_image','video','audio','document')),"
        " size INTEGER NOT NULL, mtime REAL NOT NULL, btime REAL,"
        " fs_id TEXT, content_sha256 TEXT,"
        " width INTEGER, height INTEGER, duration REAL,"
        # the change: a column that did not exist before
        " starred INTEGER NOT NULL DEFAULT 0 CHECK (starred IN (0,1)),"
        " first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL,"
        " missing_since REAL"
        ") STRICT"
    )
    conn.execute(
        "INSERT INTO file_new(id, folder_id, name, kind, size, mtime, btime, fs_id,"
        " content_sha256, width, height, duration, first_seen_at, last_seen_at,"
        " missing_since)"
        " SELECT id, folder_id, name, kind, size, mtime, btime, fs_id,"
        " content_sha256, width, height, duration, first_seen_at, last_seen_at,"
        " missing_since FROM file"
    )
    conn.execute("DROP TABLE file")
    conn.execute("ALTER TABLE file_new RENAME TO file")
    for statement in (
        ("CREATE UNIQUE INDEX file_in_folder ON file(folder_id, name COLLATE NOCASE) WHERE missing_since IS NULL"),
        "CREATE INDEX file_recent ON file(mtime DESC) WHERE missing_since IS NULL",
        ("CREATE INDEX file_in_folder_by_time ON file(folder_id, mtime, id) WHERE missing_since IS NULL"),
        "CREATE INDEX file_added ON file(first_seen_at DESC) WHERE missing_since IS NULL",
        "CREATE INDEX file_sha ON file(content_sha256)",
        "CREATE INDEX file_kind ON file(kind)",
    ):
        conn.execute(statement)


def a_whole_library(path, root):
    """A library built by the real producers, with work of every kind in it."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    from db import derived, ingest
    from db import library as library_module

    root.mkdir(parents=True, exist_ok=True)
    recipe = (
        "a castle <lora:filmGrain:0.4>\nNegative prompt: blur\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, "
        "Size: 512x512, Model: dreamshaper_8"
    )
    for i in range(4):
        info = PngInfo()
        info.add_text("parameters", recipe)
        Image.new("RGB", (16, 16), (20 + i, 40, 60)).save(root / f"p{i}.png", pnginfo=info)

    _built(path)
    conn = connect.connect(path)
    root_id = library_module.add_root(conn, root, "library", NOW)
    scan.scan(conn, root_id, root, NOW)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, NOW)

    user = authored.add_user(conn, "will", "hash", "ADMIN", NOW)
    person = authored.person(conn, "Ilse", NOW)
    album = collections.collection(conn, "Keepers", NOW)
    files = [r[0] for r in conn.execute("SELECT id FROM file ORDER BY id")]
    authored.rate(conn, files[0], user, 5, NOW)
    authored.comment(conn, files[0], user, "the good one", NOW)
    authored.favourite(conn, files[1], user, NOW)
    collections.set_membership(conn, album, files[2], True, NOW)
    authored.assert_person(conn, person, files[3], user, NOW)
    derived.annotate(conn, files[0], "caption", "a brass helmet", "qwen-vl", "2.5", "aa", NOW)
    derived.record_hash(conn, files[1], "aa", NOW, phash64=123)
    conn.commit()
    conn.close()
    return path


def everything_in(path):
    """The whole library, as one comparable value."""
    conn = connect.connect(path, read_only=True)
    try:
        return {
            "files": conn.execute("SELECT id, folder_id, name, content_sha256 FROM file ORDER BY id").fetchall(),
            "ratings": conn.execute("SELECT file_id, rating FROM rating").fetchall(),
            "comments": conn.execute("SELECT file_id, body FROM comment").fetchall(),
            "favourites": conn.execute("SELECT file_id FROM favorite").fetchall(),
            "membership": conn.execute("SELECT file_id FROM collection_file").fetchall(),
            "assertions": conn.execute("SELECT person_id, file_id FROM person_assertion").fetchall(),
            "params": conn.execute("SELECT count(*) FROM file_param").fetchone()[0],
            "artifacts": conn.execute("SELECT file_id, role FROM file_artifact ORDER BY file_id, role").fetchall(),
            "generation": conn.execute("SELECT file_id, seed FROM generation").fetchall(),
            "captions": conn.execute("SELECT file_id, text FROM derived_annotation").fetchall(),
            "hashes": conn.execute(
                "SELECT file_id, value FROM derived_file_hash ORDER BY file_id, space_id"
            ).fetchall(),
            "slugs": conn.execute("SELECT kind, slug FROM entity ORDER BY id").fetchall(),
            "caption_search": conn.execute(
                "SELECT count(*) FROM annotation_fts WHERE annotation_fts MATCH 'helmet'"
            ).fetchone()[0],
            "param_search": conn.execute(
                "SELECT count(*) FROM param_fts WHERE param_fts MATCH 'dreamshaper'"
            ).fetchone()[0],
        }
    finally:
        conn.close()


def test_rebuilding_the_table_everything_points_at_keeps_the_library(tmp_path, steps):
    """Claim 5 against the real schema.

    Twenty-three columns across twenty tables reference `file`. Foreign keys
    are off while a rebuild runs, so all of them are unprotected for its
    duration -- this is the migration where a lost column or a dropped row
    silently detaches every rating in the library from its picture.
    """
    path = a_whole_library(tmp_path / "gallery.db", tmp_path / "pics")
    before = everything_in(path)
    assert before["ratings"], "the fixture holds nothing, so surviving it proves nothing"
    assert before["params"], "the fixture holds nothing, so surviving it proves nothing"
    assert before["caption_search"], "the fixture holds nothing, so surviving it proves nothing"

    steps[connect.USER_VERSION] = rebuild_the_file_table
    assert migrate.migrate(path, target=connect.USER_VERSION + 1) == [connect.USER_VERSION + 1]

    assert everything_in(path) == before, "the migration cost the library something"

    conn = connect.connect(path, read_only=True)
    try:
        # the rebuild really happened, so this is not passing by doing nothing
        assert "starred" in [r[1] for r in conn.execute("PRAGMA table_info(file)")]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_a_migration_that_loses_a_column_is_caught(tmp_path, steps):
    """The control.

    Without it, `everything_in(...) == before` might be comparing two copies
    of nothing. A rebuild that forgets one column is the ordinary way this
    goes wrong -- the twelve-step dance restates the whole table by hand --
    and it has to be visible, not silently absorbed.
    """
    path = a_whole_library(tmp_path / "gallery.db", tmp_path / "pics")
    before = everything_in(path)

    def forget_the_hashes(conn):
        rebuild_the_file_table(conn)
        conn.execute("UPDATE file SET content_sha256 = NULL")

    steps[connect.USER_VERSION] = forget_the_hashes
    migrate.migrate(path, target=connect.USER_VERSION + 1)
    assert everything_in(path) != before, "a migration that wiped every content hash compared equal"


def test_the_snapshot_of_a_whole_library_can_be_restored(tmp_path, steps):
    """The way back, on a real library rather than a table of four rows."""
    path = a_whole_library(tmp_path / "gallery.db", tmp_path / "pics")
    before = everything_in(path)

    steps[connect.USER_VERSION] = rebuild_the_file_table
    migrate.migrate(path, target=connect.USER_VERSION + 1)
    backup = pathlib.Path(str(path).replace(".db", f".v{connect.USER_VERSION}.backup"))

    migrate.restore(backup, path)
    assert everything_in(path) == before
    conn = connect.connect(path, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
        assert "starred" not in [r[1] for r in conn.execute("PRAGMA table_info(file)")]
    finally:
        conn.close()


# --- the shipped v3 -> v4 step ---------------------------------------------


def v3_database_with_embeddings(tmp_path):
    """A database in the exact v3 shape semantic search first shipped
    with: embeddings keyed (file_id, space_id), no id of their own.

    The schema that shipped as v3, executed -- not a fresh build with
    that generation's DDL pasted back over it. The difference is not
    tidiness: an inverted fixture has every OTHER object at today's
    shape, so a step that forgets one is invisible to it.
    """
    import numpy as np

    from db import similarity
    from vision.faiss_index import SpaceSpec

    path = tmp_path / "gallery.db"
    schemas.seed(path, 3)
    conn = connect.connect(path)
    root = int(
        # The uuid stated, not the schema's randomblob(16): the fixture
        # feeds `migrated`'s cache key, which a random blob drifts.
        conn.execute(
            "INSERT INTO root(path, kind, created_at, uuid) VALUES('Z:/x', 'library', ?, ?)", (NOW, b"\x03" * 16)
        ).lastrowid
        or 0
    )
    # Straight INSERT: the scanner writes `fs_id`, a v31 column.
    folder = scan.mint(conn, "folder", "x")
    conn.execute("INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, NULL, 'x', 0)", (folder, root))
    files = []
    for name in ("dusk", "dawn"):
        fid = scan.mint(conn, "file", name)
        conn.execute(
            "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
            " first_seen_at, last_seen_at) VALUES(?, ?, ?, 'image', 10, 0, 'aa', ?, ?)",
            (fid, folder, f"{name}.png", NOW, NOW),
        )
        files.append(fid)
    spec = SpaceSpec(
        key="semantic.test.m.c",
        representation="float32",
        dimensions=4,
        metric="cosine",
        producer="test:m",
        producer_version="c",
        preprocess="t",
        preprocess_version="1",
    )
    sid = similarity.space_id(conn, spec, NOW)
    conn.commit()

    # The one object still written by hand, and the reason is in
    # `@step(3)`: "version 3 drifted during development". The commit that
    # LAST stamped v3 already carried the immutable-id rework that
    # shipped as v4, so the vendored v3 is the late one. This reverts
    # exactly that table to the shape semantic search first shipped with
    # -- one object, against a database authentic in every other.
    conn.execute("DROP TABLE derived_embedding")
    conn.execute("""CREATE TABLE derived_embedding (
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    space_id      INTEGER NOT NULL REFERENCES similarity_space(id) ON DELETE RESTRICT,
    vector        BLOB NOT NULL,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL,
    PRIMARY KEY (file_id, space_id)
) STRICT""")
    conn.execute("CREATE INDEX derived_embedding_space ON derived_embedding(space_id)")
    conn.execute("""CREATE TRIGGER derived_embedding_fits_its_space
BEFORE INSERT ON derived_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'embedding length disagrees with its space''s dimensions');
END""")
    conn.execute("""CREATE TRIGGER derived_embedding_fits_its_space_update
BEFORE UPDATE ON derived_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'embedding length disagrees with its space''s dimensions');
END""")
    for at, fid in enumerate(files):
        vec = np.zeros(4, dtype=np.float32)
        vec[at] = 1.0
        conn.execute(
            "INSERT INTO derived_embedding(file_id, space_id, vector, source_sha256, computed_at)"
            " VALUES(?, ?, ?, 'aa', ?)",
            (fid, sid, vec.tobytes(), NOW),
        )
    conn.commit()
    conn.close()
    return path, files, spec, sid


def test_a_v3_library_keeps_its_embeddings_and_they_still_answer(tmp_path, pinned_identity):
    """The hostile v3 -> v4 gate: a database built with the old
    derived_embedding shape migrates to one indistinguishable from a fresh
    build, every vector survives byte-for-byte under a freshly minted
    immutable id, and the migrated rows still retrieve through the real
    alignment path."""
    import numpy as np

    from db import retrieval, similarity
    from vision.faiss_index import IndexManager

    path, files, spec, sid = v3_database_with_embeddings(tmp_path)
    ro = connect.connect(path, read_only=True)
    try:
        before = ro.execute(
            "SELECT file_id, space_id, vector, source_sha256 FROM derived_embedding ORDER BY file_id"
        ).fetchall()
    finally:
        ro.close()
    assert len(before) == 2

    assert migrated(path) == list(range(4, connect.USER_VERSION + 1))
    assert build.drift(path) == [], "the migrated file differs from a fresh build"

    conn = connect.connect(path)
    try:
        rows = conn.execute(
            "SELECT id, file_id, space_id, vector, source_sha256 FROM derived_embedding ORDER BY file_id"
        ).fetchall()
        assert [(fid, s, vec, sha) for _, fid, s, vec, sha in rows] == before, "a row or a vector byte changed"
        minted = [row[0] for row in rows]
        assert all(isinstance(new_id, int) for new_id in minted)
        assert len(set(minted)) == 2

        current = retrieval.current_rows(conn, sid)
        assert sorted(fid for _, fid in current) == sorted(files)
        ids = [embedding_id for embedding_id, _ in current]
        manager = IndexManager(tmp_path / "spaces", gpu=False)
        key = similarity.align(conn, manager, spec, ids, lambda wanted: retrieval._vectors(conn, wanted), NOW)
        query = np.zeros(4, dtype=np.float32)
        query[0] = 1.0
        labels, _scores = manager.search(key, [query], 1)
        to_file = dict(current)
        assert to_file[int(labels[0][0])] == files[0], "the migrated vectors no longer answer a query"
    finally:
        conn.close()


def test_a_dormant_rule_on_a_listed_collection_stops_v8_by_name(tmp_path):
    """v7's own artifact: a listed collection carrying a rule row --
    two authored membership definitions. Stamping it forward would
    install guards the data already violates; deleting the rule unasked
    is not this schema's way. The step refuses, names the collection,
    and leaves the file at v7 with the rule intact."""
    path = tmp_path / "gallery.db"
    # The schema that shipped as v7, which has none of the guards v8
    # installs -- so the row below goes in without a trigger being
    # dropped to let it.
    schemas.seed(path, 7)
    conn = connect.connect(path, autocommit=True)
    # Straight INSERT rather than `collections.collection`: that writes
    # `updated_at`, which v7's collection table does not have. An
    # authentic fixture is the schema of its day, and today's writers
    # speak today's columns.
    album = scan.mint(conn, "collection", "Keepers")
    conn.execute(
        "INSERT INTO collection(id, parent_id, name, kind, created_at) VALUES(?, NULL, 'Keepers', 'album', ?)",
        (album, NOW),
    )
    conn.execute(
        "INSERT INTO collection_rule(collection_id, source_text, created_at, updated_at) VALUES(?, 'x', ?, ?)",
        (album, NOW, NOW),
    )
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="Keepers"):
        migrate.migrate(path)

    conn = connect.connect(path, read_only=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7, "a refused step must leave the file at v7"
        assert conn.execute("SELECT count(*) FROM collection_rule").fetchone()[0] == 1, "the rule is the human's"
    finally:
        conn.close()


def test_a_zoned_camera_time_keeps_its_wall_clock_and_its_instant_across_v21(tmp_path, pinned_identity):
    """Under v20 a capture row WITH an offset stored the instant (the
    reader folded the zone in); from v21 `captured_at` is the camera's
    wall clock with the zone beside it. A library upgraded without
    re-ingesting must read the same wall clock and the same instant as
    before -- not the zone applied twice -- and ISO 0 is refused by v23's
    CHECK (the fixture is built from the current DDL, so a legacy zero
    cannot be planted here; the step's NULLIF is its repair)."""
    import datetime as dt

    from db import scan, when

    path, _files, _spec, _sid = v3_database_with_embeddings(tmp_path)
    migrated(path, target=20)
    conn = connect.connect(path)
    try:
        root_id = conn.execute(
            "INSERT INTO root(path, kind, created_at, uuid) VALUES('C:/z', 'library', 0, ?)", (b"\x02" * 16,)
        ).lastrowid
        folder = scan.mint(conn, "folder", "z")
        conn.execute(
            "INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, NULL, 'z', 0)", (folder, root_id)
        )
        file_id = scan.mint(conn, "file", "zoned.jpg")
        conn.execute(
            "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
            " VALUES(?, ?, 'zoned.jpg', 'image', 1, 0, 'zz', 0, 0)",
            (file_id, folder),
        )
        # 2026-08-19 14:23:01 at +02:00, stored the v20 way: the INSTANT 12:23:01Z
        instant = dt.datetime(2026, 8, 19, 12, 23, 1, tzinfo=dt.UTC).timestamp()
        conn.execute(
            "INSERT INTO capture(file_id, captured_at, tz_offset_min, iso, parsed_at) VALUES(?, ?, 120, NULL, 0)",
            (file_id, instant),
        )
        conn.commit()
    finally:
        connect.close(conn)
    assert migrated(path) == list(range(21, connect.USER_VERSION + 1))
    conn = connect.connect(path)
    try:
        captured_at, tz, iso = conn.execute("SELECT captured_at, tz_offset_min, iso FROM capture").fetchone()
        assert dt.datetime.fromtimestamp(captured_at, dt.UTC).strftime("%H:%M:%S") == "14:23:01", "the wall clock"
        told = when.judge_capture(
            captured_at=captured_at, subsec_ms=None, tz_offset_min=tz, maker_tz_offset_min=None, mtime=None, btime=None
        )
        assert told is not None
        assert told.instant_at == instant, "and the same instant as before the upgrade"
        assert iso is None
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE capture SET iso = 0")
    finally:
        connect.close(conn)


def test_v26_backfills_a_pass_for_every_file_with_faces(tmp_path, pinned_identity):
    """A v25 library's whole-still face instances prove a detector looked
    at those bytes: v26 records the pass with its face count, so the
    next faces sweep leaves those files alone. A face-free file left no
    trace and is not invented."""
    import numpy as np

    from db import derived

    path = tmp_path / "gallery.db"
    schemas.seed(path, 25)  # the schema that shipped as v25: no derived_face_scan
    conn = connect.connect(path, autocommit=True)
    conn.execute("PRAGMA foreign_keys = ON")
    file_id = a_file_row(conn)
    conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("a" * 64, file_id))
    faces = [
        {"region": derived.region(conn, 0.1 * i, 0.1, 0.2, 0.2), "embedding": np.full(4, i, np.float32).tobytes()}
        for i in range(1, 3)
    ]
    derived.record_faces(conn, file_id, "m", "1", "a" * 64, NOW, faces)
    conn.close()

    assert migrated(path) == list(range(26, connect.USER_VERSION + 1))
    assert build.drift(path) == []
    ro = connect.connect(path, read_only=True)
    try:
        assert ro.execute(
            "SELECT file_id, model_id, model_version, source_sha256, faces FROM derived_face_scan"
        ).fetchall() == [(file_id, "m", "1", "a" * 64, 2)]
    finally:
        ro.close()


def test_v30_retires_everything_derived_from_a_portrait_raw(tmp_path):
    """A v29 library's portrait RAW -- orientation 8 -- was rendered
    sideways for every reader: its thumbnails, face scan, embedding,
    file hash and annotation all came from that frame. v30 retires all
    of it and queues the thumbnails' render; a landscape RAW and the
    JPEG sibling keep everything, and no file's sha moves."""
    import numpy as np
    from PIL import Image

    from db import derived
    from vision import thumbs

    path = tmp_path / "gallery.db"
    schemas.seed(path, 29)  # the schema that shipped as v29
    conn = connect.connect(path, autocommit=True)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        portrait = a_file_row(conn)
        conn.execute("UPDATE file SET name = '666A0273.CR2', content_sha256 = ? WHERE id = ?", ("b" * 64, portrait))
        folder = conn.execute("SELECT folder_id FROM file WHERE id = ?", (portrait,)).fetchone()[0]
        others = {}
        for slug, name, sha in (("wide", "666A0111.CR2", "d" * 64), ("sibling", "666A0273.JPG", "c" * 64)):
            others[name] = scan.mint(conn, "file", slug)
            conn.execute(
                "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
                " VALUES(?, ?, ?, 'image', 10, 0, ?, ?, ?)",
                (others[name], folder, name, sha, NOW, NOW),
            )
        for file_id, orientation in ((portrait, 8), (others["666A0111.CR2"], 1), (others["666A0273.JPG"], 8)):
            conn.execute("INSERT INTO capture(file_id, orientation, parsed_at) VALUES(?, ?, 0)", (file_id, orientation))
            sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
            faces = [
                {"region": derived.region(conn, 0.1, 0.1, 0.2, 0.2), "embedding": np.ones(4, np.float32).tobytes()}
            ]
            derived.record_faces(conn, file_id, "m", "1", sha, NOW, faces)
            derived.record_face_scan(conn, file_id, "m", "1", sha, NOW, 1)
            derived.record_hash(conn, file_id, sha, NOW, phash64=1)
    finally:
        conn.close()
    cache = tmp_path / thumbs.DIRNAME
    for sha in ("b" * 64, "c" * 64, "d" * 64):
        thumbs.put_all(cache, sha, Image.new("RGB", (40, 30), (9, 9, 9)))

    # Live, never `migrated`: the v30 step DELETES thumbnail files, and a
    # cached database replay cannot replay a filesystem side effect --
    # on a cache hit the staged thumbnails would survive their own test.
    assert migrate.migrate(path) == list(range(30, connect.USER_VERSION + 1))
    assert not any(thumbs.path_for(cache, "b" * 64, kind).exists() for kind in thumbs.EDGES), "the portrait RAW's go"
    for sha in ("c" * 64, "d" * 64):
        assert all(thumbs.path_for(cache, sha, kind).exists() for kind in thumbs.EDGES), "the others keep theirs"
    ro = connect.connect(path, read_only=True)
    try:
        shas = dict(ro.execute("SELECT id, content_sha256 FROM file"))
        assert shas[portrait] == "b" * 64, "the bytes' own sha never moves"
        for table in ("derived_face_scan", "derived_face_instance", "derived_file_hash"):
            held = {row[0] for row in ro.execute(f"SELECT file_id FROM {table}")}
            assert portrait not in held, f"{table} still derives the portrait RAW from a sideways frame"
            assert set(others.values()) <= held, f"{table} lost a file it should not have"
        queued = ro.execute("SELECT kind, payload FROM job ORDER BY id DESC LIMIT 1").fetchone()
        assert queued is not None
        assert queued[0] == "hash"
        assert '"derive": "thumbs"' in queued[1]
        assert ro.execute("SELECT count(*) FROM job_item WHERE job_id = (SELECT max(id) FROM job)").fetchone()[0] == 1
    finally:
        ro.close()


def test_the_app_brings_an_older_database_forward_at_boot(tmp_path):
    """A run over a file an older build wrote does not 500 on the first
    column it lacks: build_app migrates it forward first, snapshot beside
    it; a file from a newer build is refused at boot with the reason."""
    from litestar.testing import TestClient

    from sg_web.app import build_app
    from sg_web.home import db_path

    burrow = tmp_path / "run"
    burrow.mkdir()
    path = db_path(burrow)
    schemas.seed(path, 26)  # the schema that shipped as v26

    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        assert client.get("/g", headers={"accept": "application/json"}).status_code == 200
        assert client.get("/operations/overview").status_code == 200
    ro = connect.connect(path, read_only=True)
    try:
        assert ro.execute("PRAGMA user_version").fetchone()[0] == migrate.USER_VERSION
    finally:
        ro.close()
    assert path.with_suffix(".v26.backup").exists(), "the snapshot sits beside the file it was taken from"

    conn = connect.connect(path, autocommit=True)
    conn.execute(f"PRAGMA user_version = {migrate.USER_VERSION + 1}")
    conn.close()
    with pytest.raises(SystemExit, match="Restore the backup"):
        build_app(str(burrow), worker=False)


def test_a_connection_that_cannot_be_prepared_closes_its_handle(monkeypatch):
    """The one connection-lifetime fact no linter can see.

    Between sqlite3.connect returning and _prepared finishing, the handle
    is open and unnamed: sglint SG103 cannot reach it, because db/connect.py
    is the one file where raw sqlite3.connect is allowed. A pragma that
    raises in there -- WAL conversion timing out against another process,
    a read-only open refused a write pragma -- used to drop the handle,
    and it leaked exactly like a connection a caller forgot.

    The unraisable hook is the assertion: a ResourceWarning from a
    finalizer is not raised at the leak, so `pytest.warns` would not see
    it and `filterwarnings = error` would blame a later test.
    """
    caught: list = []
    monkeypatch.setattr(sys, "unraisablehook", caught.append)

    def refuses(conn, *, journal):
        raise RuntimeError("a pragma refused")

    monkeypatch.setattr(connect, "_prepared", refuses)

    for open_one in (connect.memory, lambda: connect.connect(":memory:")):
        caught.clear()
        with pytest.raises(RuntimeError):
            open_one()
        gc.collect()
        assert caught == [], [str(one.exc_value) for one in caught]
