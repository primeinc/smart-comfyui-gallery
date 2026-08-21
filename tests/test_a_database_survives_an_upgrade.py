"""Upgrading the app must not cost the user what they put in the library.

The version check refuses a database this build does not recognise, and the
only way past it was a rebuild that deletes every rating, comment, album and
name. These pin the way out: forward-only steps, each in its own transaction,
with a snapshot taken first and a downgrade refused by name.

The step used throughout is a table rebuild -- the twelve-step dance SQLite
requires for a change ALTER TABLE cannot express -- because that is the case
where a migration runner either holds or loses the data.
"""

import pathlib
import sqlite3

import pytest

from db import authored, build, connect, migrate, scan

NOW = 1_700_000_000.0


@pytest.fixture
def steps():
    """A clean registry, restored afterwards, so tests cannot leak into each
    other or into the real migration list."""
    original = dict(migrate.STEPS)
    migrate.STEPS.clear()
    yield migrate.STEPS
    migrate.STEPS.clear()
    migrate.STEPS.update(original)


@pytest.fixture
def library(tmp_path):
    """A built database with a person's work already in it."""
    path = tmp_path / "gallery.db"
    build.build(path)
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
    album = authored.collection(conn, "Keepers", NOW)
    authored.rate(conn, file_id, user, 5, NOW)
    authored.comment(conn, file_id, user, "the good one", NOW)
    authored.favourite(conn, file_id, user, NOW)
    authored.add_to_collection(conn, album, file_id, NOW)
    authored.assert_person(conn, person, file_id, user, NOW)
    conn.commit()
    conn.close()
    return path


def authored_state(path):
    """Everything a person made, as one comparable value."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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


def test_an_upgrade_keeps_everything_a_person_made(library, steps):
    before = authored_state(library)
    steps[connect.USER_VERSION] = rebuild_comment_table

    applied = migrate.migrate(library, target=connect.USER_VERSION + 1)

    assert applied == [connect.USER_VERSION + 1]
    after = authored_state(library)
    assert after == before, "the upgrade cost the user something"
    conn = sqlite3.connect(f"file:{library}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION + 1
        # the rebuild really happened, so this is not passing by doing nothing
        assert "pinned" in [r[1] for r in conn.execute("PRAGMA table_info(comment)")]
    finally:
        conn.close()


def test_the_snapshot_is_taken_before_anything_changes(library, steps):
    before = authored_state(library)
    steps[connect.USER_VERSION] = rebuild_comment_table
    migrate.migrate(library, target=connect.USER_VERSION + 1)

    backup = pathlib.Path(str(library).replace(".db", f".v{connect.USER_VERSION}.backup"))
    assert backup.exists(), "no way back was left"
    assert authored_state(backup) == before
    kept = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    try:
        assert kept.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
        assert "pinned" not in [r[1] for r in kept.execute("PRAGMA table_info(comment)")]
    finally:
        kept.close()


def test_restoring_the_snapshot_undoes_the_upgrade(library, steps):
    before = authored_state(library)
    steps[connect.USER_VERSION] = rebuild_comment_table
    migrate.migrate(library, target=connect.USER_VERSION + 1)

    backup = pathlib.Path(str(library).replace(".db", f".v{connect.USER_VERSION}.backup"))
    migrate.restore(backup, library)
    assert authored_state(library) == before
    conn = sqlite3.connect(f"file:{library}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
    finally:
        conn.close()


def test_a_downgrade_is_refused_by_name(library, steps):
    """A newer file opened by an older build. Guessing at it is how a
    database gets truncated to the columns this build happens to know."""
    conn = sqlite3.connect(str(library), isolation_level=None)
    conn.execute(f"PRAGMA user_version = {connect.USER_VERSION + 5}")
    conn.close()

    with pytest.raises(migrate.Downgrade, match="no down step"):
        migrate.migrate(library)
    with pytest.raises(migrate.Downgrade):
        migrate.pending(library)


def test_a_missing_step_stops_before_touching_the_file(library, steps):
    """The registry is keyed on where a step starts, so a gap is visible
    rather than being discovered halfway through."""
    before = authored_state(library)
    with pytest.raises(migrate.StepMissing, match=f"v{connect.USER_VERSION}"):
        migrate.migrate(library, target=connect.USER_VERSION + 2)
    assert authored_state(library) == before
    conn = sqlite3.connect(f"file:{library}?mode=ro", uri=True)
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
    conn = sqlite3.connect(f"file:{library}?mode=ro", uri=True)
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
    conn = sqlite3.connect(f"file:{library}?mode=ro", uri=True)
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

    conn = sqlite3.connect(f"file:{library}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION + 1
        assert "pinned" in [r[1] for r in conn.execute("PRAGMA table_info(comment)")]
    finally:
        conn.close()


def test_migrating_a_current_database_does_nothing(library, steps):
    assert migrate.pending(library) == []
    assert migrate.migrate(library) == []
    backup = pathlib.Path(str(library).replace(".db", f".v{connect.USER_VERSION}.backup"))
    assert not backup.exists(), "a no-op upgrade left a snapshot behind"


def test_something_that_is_not_ours_is_not_migrated(tmp_path, steps):
    other = tmp_path / "someone-elses.db"
    conn = sqlite3.connect(str(other))
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
    conn = sqlite3.connect(str(library), isolation_level=None)
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


def v1_database(tmp_path):
    """Today's build, taken back to v1 by inverting the shipped steps."""
    path = tmp_path / "gallery.db"
    build.build(path)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("DROP TABLE derived_dupe_group")  # v3's addition; indexes go with it
    for trigger in (
        "collection_file_not_into_smart",
        "collection_file_not_moved_into_smart",
        "collection_with_members_stays_listed",
    ):
        conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("PRAGMA user_version = 1")
    conn.close()
    return path


def a_file_row(conn):
    root = int(
        conn.execute("INSERT INTO root(path, kind, created_at) VALUES('Z:/x', 'library', ?)", (NOW,)).lastrowid or 0
    )
    folder = scan.ensure_folder(conn, root, None, "x")
    file_id = scan.mint(conn, "file", "dusk")
    conn.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'dusk.png', 'image', 10, 0, ?, ?)",
        (file_id, folder, NOW, NOW),
    )
    return file_id


def test_the_shipped_steps_take_a_v1_database_to_the_current_build(tmp_path):
    """The real STEPS, executed: every other test here swaps in synthetic
    steps to prove the runner, which left the one migration that actually
    ships executed by nothing. A v1 file must migrate to a database
    indistinguishable from a fresh build, and every trigger the step
    installs must fire -- including the moved-into one no other test
    reaches."""
    path = v1_database(tmp_path)
    assert migrate.migrate(path) == list(range(2, connect.USER_VERSION + 1))
    assert build.drift(path) == [], "the migrated file differs from a fresh build"

    conn = connect.connect(path)
    try:
        file_id = a_file_row(conn)
        smart = authored.collection(conn, "Big seeds", NOW, kind="smart", sql_text="SELECT 1")
        album = authored.collection(conn, "Keepers", NOW)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (smart, file_id, NOW))
        conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (album, file_id, NOW))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE collection_file SET collection_id = ? WHERE collection_id = ?", (smart, album))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE collection SET kind = 'smart', sql_text = 'SELECT 1' WHERE id = ?", (album,))
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
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    file_id = a_file_row(conn)
    smart = authored.collection(conn, "Big seeds", NOW, kind="smart", sql_text="SELECT 1")
    conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (smart, file_id, NOW))
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="Big seeds"):
        migrate.migrate(path)

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1, "a refused step must leave the file at v1"
        assert conn.execute("SELECT count(*) FROM collection_file").fetchone()[0] == 1, "the rows are the human's"
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
        " inode INTEGER, content_sha256 TEXT,"
        " width INTEGER, height INTEGER, duration REAL,"
        # the change: a column that did not exist before
        " starred INTEGER NOT NULL DEFAULT 0 CHECK (starred IN (0,1)),"
        " first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL,"
        " missing_since REAL"
        ") STRICT"
    )
    conn.execute(
        "INSERT INTO file_new(id, folder_id, name, kind, size, mtime, btime, inode,"
        " content_sha256, width, height, duration, first_seen_at, last_seen_at,"
        " missing_since)"
        " SELECT id, folder_id, name, kind, size, mtime, btime, inode,"
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

    build.build(path)
    conn = connect.connect(path)
    root_id = library_module.add_root(conn, root, "library", NOW)
    scan.scan(conn, root_id, root, NOW)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, NOW)

    user = authored.add_user(conn, "will", "hash", "ADMIN", NOW)
    person = authored.person(conn, "Ilse", NOW)
    album = authored.collection(conn, "Keepers", NOW)
    files = [r[0] for r in conn.execute("SELECT id FROM file ORDER BY id")]
    authored.rate(conn, files[0], user, 5, NOW)
    authored.comment(conn, files[0], user, "the good one", NOW)
    authored.favourite(conn, files[1], user, NOW)
    authored.add_to_collection(conn, album, files[2], NOW)
    authored.assert_person(conn, person, files[3], user, NOW)
    derived.annotate(conn, files[0], "caption", "a brass helmet", "qwen-vl", "2.5", "aa", NOW)
    derived.record_hash(conn, files[1], "aa", NOW, phash64=123)
    conn.commit()
    conn.close()
    return path


def everything_in(path):
    """The whole library, as one comparable value."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == connect.USER_VERSION
        assert "starred" not in [r[1] for r in conn.execute("PRAGMA table_info(file)")]
    finally:
        conn.close()
