"""An answer is discarded when an answer could have changed -- not when
anything at all was written.

db/resultset.py caches the whole ordered answer and pages it by
slicing, valid for one (question, library state) pair. Library state was
`PRAGMA data_version`, which means "somebody committed something". That
is what FTS5 uses for its own structure cache (sqlite/sqlite
ext/fts5/fts5_index.c fts5IndexDataVersion) and it is fine there,
because the thing FTS5 re-reads is one small record. Here the cached
thing is the whole library:

    files    at rest    while a job commits    factor
    1,000    0.19 ms         0.64 ms            3.4x
   10,000    0.18 ms         4.29 ms           23.8x
   40,000    0.18 ms        19.04 ms          105.9x
   80,000    0.18 ms        38.26 ms          214.4x

Flat at rest, linear in motion, and jobs commit per item. Traced, the
job that runs for hours over a large library writes ONLY the ledger: a
thumbs pass over 12 files made 180 writes, every one of them `job`,
`job_item` or `job_event`.

So `answer_generation` moves for every table EXCEPT those three. The
direction matters more than the list: a table wrongly INCLUDED costs a
little speed, a table wrongly EXCLUDED serves an answer computed before
a commit under a state taken after one, which is the single failure the
currency contract exists to prevent. This module holds that line -- the
coverage check below fails on a table added later without its triggers,
rather than letting it quietly stop invalidating.
"""

from __future__ import annotations

import pathlib
import uuid

import pytest

from db import build, connect, resultset

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
NOW = 1_700_000_000.0

#: The only tables that may leave an answer alone. Written out so the
#: set is a decision somebody made, not whatever the code happens to do.
#:
#: Named for what they have in common, which is not where they came
#: from: nothing in them can change what a page would ANSWER. Most are
#: operational -- a job queued, an item finished, a ledger line, a
#: schedule saying when a collection next starts -- and invalidating
#: every cached answer because a worker picked up a thumbnail is how a
#: busy library never serves a warm one.
#:
#: `saved_view` is authored rather than operational and belongs here for
#: the same reason: it remembers a QUESTION. Asking one again returns
#: whatever the library says at the time, so writing one down changes no
#: answer -- and a trigger on it would be the schema claiming otherwise.
CHANGES_NO_ANSWER = {"job", "job_item", "job_event", "schedule", "saved_view"}


@pytest.fixture
def library(tmp_path):
    """A real file: currency reads a monitor connection, and an
    in-memory database has no second connection to be current against."""
    path = tmp_path / "gallery.db"
    build.build(path)
    conn = connect.connect(str(path))
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")
    for i in range(2, 8):
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (i, uuid.uuid4().bytes, f"f{i}"))
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1,?,0,0)",
            (i, f"f{i}.png", 1_700_000_000 + i),
        )
    conn.commit()
    yield path, conn
    connect.close(conn)


# --- coverage: nothing may drop out quietly ---------------------------------


def _eligible(conn) -> tuple[list[str], set[str]]:
    """(tables that must carry triggers, tables excused) for this build."""
    virtual = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL%'")
    }
    shadow = tuple(f"{one}_" for one in virtual)
    excused = set(virtual) | CHANGES_NO_ANSWER | {"answer_generation"}
    must, skipped = [], set()
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        if name in excused or name.startswith(shadow):
            skipped.add(name)
        else:
            must.append(name)
    return sorted(must), skipped


def test_every_table_that_can_change_an_answer_moves_the_counter(library):
    """The gate. A table added later without its triggers stops
    invalidating answers, silently, and this is what says so."""
    _path, conn = library
    must, _ = _eligible(conn)
    assert len(must) > 40, f"only {len(must)} tables considered; the sweep is not seeing the schema"
    have = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    missing = [
        f"answer_moved_{name}_{short}"
        for name in must
        for short in ("ins", "upd", "del")
        if f"answer_moved_{name}_{short}" not in have
    ]
    assert missing == [], f"tables that can change an answer and would not invalidate it: {missing}"


def test_the_ledger_is_the_only_thing_excused(library):
    """Excusing a table is how a stale answer gets served, so the list
    of excuses is asserted rather than trusted."""
    _path, conn = library
    _, skipped = _eligible(conn)
    virtual_and_shadow = {one for one in skipped if "fts" in one}
    assert skipped - virtual_and_shadow == CHANGES_NO_ANSWER | {"answer_generation"}, skipped - virtual_and_shadow


def test_a_virtual_table_could_not_carry_one_anyway(library):
    """The FTS tables are absent for a reason SQLite enforces, not a
    reason somebody chose -- and their rows only move when `file` or
    `folder` do, which are covered."""
    _path, conn = library
    import sqlite3

    with pytest.raises(sqlite3.OperationalError, match="virtual table"):
        conn.execute("CREATE TRIGGER probe_fts AFTER INSERT ON name_fts BEGIN SELECT 1; END")


# --- behaviour: what moves it and what does not -----------------------------


def _generation(conn) -> int:
    return conn.execute("SELECT value FROM answer_generation").fetchone()[0]


def test_a_ledger_write_leaves_the_answer_alone(library):
    """The whole point. This is the job that runs for hours."""
    _path, conn = library
    was = _generation(conn)
    conn.execute("INSERT INTO job(kind, state, created_at) VALUES('hash','queued',0)")
    job = conn.execute("SELECT id FROM job").fetchone()[0]
    conn.execute("INSERT INTO job_item(job_id, item_id, state) VALUES(?, 2, 'pending')", (job,))
    conn.execute("UPDATE job SET state = 'running' WHERE id = ?", (job,))
    conn.execute(
        "INSERT INTO job_event(job_id, at, type, severity) VALUES(?, 0, 'job.submitted', 'info')",
        (job,),
    )
    conn.commit()
    assert _generation(conn) == was, "a ledger write discarded every cached answer"


def test_a_file_write_does_not(library):
    """The control. Without it the test above passes for a counter
    nothing ever moves."""
    _path, conn = library
    was = _generation(conn)
    conn.execute("UPDATE file SET mtime = mtime + 1 WHERE id = 2")
    conn.commit()
    assert _generation(conn) > was


#: A spread of what answers are actually built FROM -- an authored
#: judgement, a collection, metadata, a person -- each written with its
#: real shape so the row is a row and not a probe.
ELSEWHERE = {
    "favorite": (
        "INSERT INTO user(id, username, password_hash, role, created_at) VALUES(1,'w','x','ADMIN',0)",
        "INSERT INTO favorite(file_id, user_id, created_at) VALUES(2,1,0)",
    ),
    "rating": (
        "INSERT INTO user(id, username, password_hash, role, created_at) VALUES(1,'w','x','ADMIN',0)",
        "INSERT INTO rating(file_id, user_id, rating, created_at) VALUES(2,1,4,0)",
    ),
    "collection": (
        "INSERT INTO entity(id,uuid,kind,slug) VALUES(90, randomblob(16), 'collection','c')",
        "INSERT INTO collection(id,parent_id,name,kind,created_at,updated_at) VALUES(90,NULL,'C','album',0,0)",
    ),
    "file_param": ("INSERT INTO file_param(file_id, source, key, value_text) VALUES(2,'generation','k','v')",),
    "person": (
        "INSERT INTO entity(id,uuid,kind,slug) VALUES(91, randomblob(16), 'person','p')",
        "INSERT INTO person(id, name, created_at) VALUES(91,'P',0)",
    ),
}


@pytest.mark.parametrize("table", sorted(ELSEWHERE))
def test_a_write_anywhere_else_does_not_either(library, table):
    _path, conn = library
    was = _generation(conn)
    for statement in ELSEWHERE[table]:
        conn.execute(statement)
    conn.commit()
    assert _generation(conn) > was, f"a write to {table} did not invalidate anything"


# --- the projection actually survives ---------------------------------------


def test_a_cached_answer_survives_a_job_committing(library):
    """End to end, through the ResultSet: the projection is REUSED
    across a ledger commit and rebuilt across a file commit."""
    path, conn = library
    asked = resultset.parse()
    first = resultset.describe(conn, "", asked, NOW)

    writer = connect.connect(str(path))
    try:
        writer.execute("INSERT INTO job(kind, state, created_at) VALUES('hash','queued',0)")
        writer.commit()
        after_ledger = resultset.describe(conn, "", asked, NOW)
        assert after_ledger["currency"] == first["currency"], "a ledger commit changed the library state"

        writer.execute("UPDATE file SET mtime = mtime + 100 WHERE id = 2")
        writer.commit()
        after_file = resultset.describe(conn, "", asked, NOW)
        assert after_file["currency"] != first["currency"], "a file commit did NOT change the library state"
    finally:
        connect.close(writer)


def test_a_restored_snapshot_cannot_resurrect_a_cached_answer(library, tmp_path):
    """The counter lives in the FILE, so restoring one rewinds it.

    `PRAGMA data_version` could not do this: it counts one connection's
    own observations, so another connection's writes only ever push it
    up -- and replacing the whole file pushes it up too. A counter in
    the database is comparable across connections, which is why it is
    here, and rewindable, which is why this test is.

    A rewound counter is worse than a stale one. The projection cache is
    process-lifetime and keyed on this string, so after a restore an old
    key matches again and an answer built from rows that no longer exist
    is served as current -- with no read that could notice.

    Found by a suite, not by reasoning: a module-scoped harness that
    restores its template between tests started reporting one test's
    collection counts under another's question.
    """
    path, conn = library
    asked = resultset.parse()
    before = resultset.describe(conn, "", asked, NOW)

    snapshot = tmp_path / "snapshot.db"
    source = connect.connect(str(path), read_only=True)
    try:
        kept = connect.connect(str(snapshot))
        try:
            source.backup(kept)
        finally:
            connect.close(kept)
    finally:
        connect.close(source)

    writer = connect.connect(str(path))
    try:
        writer.execute("UPDATE file SET mtime = mtime + 500 WHERE id = 2")
        writer.commit()
        moved = resultset.describe(conn, "", asked, NOW)
        assert moved["currency"] != before["currency"]
    finally:
        connect.close(writer)

    # the snapshot back over the live file: every generation the writes
    # above minted is gone, and the counter is back where it started
    held = connect.connect(str(snapshot), read_only=True)
    try:
        live = connect.connect(str(path))
        try:
            held.backup(live)
        finally:
            connect.close(live)
    finally:
        connect.close(held)

    after = resultset.describe(conn, "", asked, NOW)
    assert after["currency"] not in (before["currency"], moved["currency"]), (
        "a restored snapshot handed back a currency this process had already cached answers under"
    )


def test_a_migrated_database_gets_the_same_coverage(tmp_path):
    """The hole this nearly shipped with.

    Every check above builds from schema.sql. A database that arrives by
    MIGRATION is the other half, and it is the half that will fail
    first: someone adds a table in a future step, writes its DDL, and
    does not write its triggers. A fresh build would still pass -- the
    schema and the step are different files -- while every existing
    library quietly stopped invalidating on that table.

    So the coverage rule is run against a database that got here the
    long way: built, stripped back to v31, and migrated forward through
    the real step.
    """
    from db import migrate
    from tests import schemas

    path = tmp_path / "gallery.db"
    schemas.seed(path, 31)  # the schema that shipped as v31

    # `32 in`, not `== [32]`. What this test is about is that step 32 --
    # the one that writes the triggers -- really ran over a database that
    # got here the long way. Spelling the whole list pins something else
    # entirely: how many steps exist above 31, which is a number every
    # future migration changes, and which broke this test the first time
    # one did.
    ran = migrate.migrate(path)
    assert 32 in ran, ran

    conn = connect.connect(str(path), read_only=True)
    try:
        must, _ = _eligible(conn)
        have = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        missing = [
            f"answer_moved_{name}_{short}"
            for name in must
            for short in ("ins", "upd", "del")
            if f"answer_moved_{name}_{short}" not in have
        ]
        assert missing == [], f"the migration left these tables unable to invalidate an answer: {missing}"
    finally:
        connect.close(conn)
    assert build.drift(path) == [], "the migrated database differs from a fresh build"
