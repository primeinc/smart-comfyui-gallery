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
    root = conn.execute(
        "INSERT INTO root(path, kind, created_at) VALUES(?, 'library', ?)",
        (str(tmp_path / "pics"), NOW),
    ).lastrowid
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


# --- planner statistics ----------------------------------------------------


def test_a_fresh_build_has_no_statistics_until_it_is_asked(library):
    """A planner with no stats guesses, and on a 100k-file library it guesses
    wrong. `analyze` is the pass that has to run after the first scan."""
    conn = connect.connect(library)
    try:
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='sqlite_stat1'"
        ).fetchone()[0] == 0
        migrate.analyze(conn)
        conn.commit()
        assert conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='sqlite_stat1'"
        ).fetchone()[0] == 1
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
