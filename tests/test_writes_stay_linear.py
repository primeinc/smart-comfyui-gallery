"""Writing to this schema must not get slower as the library grows.

Everything else was measured on 5,777 files, where a per-row cost that scales
with the table is invisible. At 100k it is the difference between a scan that
takes seconds and one that never finishes: two triggers here were quadratic,
and the symptom was a build that sat at one core for an hour.

Both had the same shape -- a lookup with no index behind it, run once per row:

* `param_fts` was a standalone table carrying file_id/key/source as UNINDEXED
  columns, and each insert deleted its predecessor by matching on them, so
  every write scanned the whole index. It is external content keyed on
  file_param's rowid now, and the delete is a B-tree lookup.
* `name_fts` carried `entity_id UNINDEXED` and every rename and delete matched
  on it. Its rowid is the entity id now.
* `param_key` recomputed `occurrences` and `value_kind` with aggregate scans
  over every row sharing the key, on every insert. Both are arithmetic now,
  which is only correct because nothing writes the table with INSERT OR
  REPLACE -- see the source gate in test_schema_contract.py.

These measure a ratio rather than an absolute, so they say the same thing on
a fast machine and a slow one, and they fail loudly rather than getting
gradually slower the way the real thing did.
"""

import io
import pathlib
import sqlite3
import time

import pytest

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"

#: Per-row cost may rise by this much across a fourfold size increase. A
#: genuinely quadratic path showed 3.5x here; linear paths measured 1.0-1.1x.
#: The gap is wide enough that timing noise cannot reach it.
TOLERANCE = 2.0

SMALL, LARGE = 2_000, 8_000


@pytest.fixture(scope="module")
def ddl():
    return io.open(SCHEMA, "r", encoding="utf-8", newline="").read()


def a_library(ddl, n):
    """`n` files in one folder, ready to be written to."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'/l','library',0)")
    conn.execute(
        "INSERT INTO entity(id,uuid,kind,slug)"
        " VALUES(1,x'00000000000000000000000000000001','folder','f')"
    )
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'f',0)")
    conn.executemany(
        "INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,?,?)",
        [(i, i.to_bytes(16, "big"), "file", f"f{i}") for i in range(2, n + 2)],
    )
    conn.executemany(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
        " VALUES(?,1,?,'image',1,0,0,0)",
        [(i, f"IMG_{i:06d}.jpg") for i in range(2, n + 2)],
    )
    return conn


def per_row(work, ddl, n, sample=500):
    """Microseconds per row for `work`, measured on a library of `n` files."""
    conn = a_library(ddl, n)
    try:
        started = time.perf_counter()
        rows = work(conn, n, sample)
        return (time.perf_counter() - started) / rows * 1e6
    finally:
        conn.close()


def growth(work, ddl, **kwargs):
    small = per_row(work, ddl, SMALL, **kwargs)
    large = per_row(work, ddl, LARGE, **kwargs)
    return large / small, small, large


def _insert_params(conn, n, sample):
    conn.executemany(
        "INSERT INTO file_param(file_id,source,key,value_text,value_num) VALUES(?,?,?,?,?)",
        [(i, "exif", "Flash", f"value {i}", float(i)) for i in range(2, n + 2)],
    )
    return n


def _rename_files(conn, n, sample):
    conn.executemany(
        "UPDATE file SET name=? WHERE id=?",
        [(f"RENAMED_{i:06d}.jpg", i) for i in range(2, sample + 2)],
    )
    return sample


def _delete_files(conn, n, sample):
    conn.executemany("DELETE FROM file WHERE id=?", [(i,) for i in range(2, sample + 2)])
    return sample


def _upsert_params(conn, n, sample):
    conn.executemany(
        "INSERT INTO file_param(file_id,source,key,value_text) VALUES(?,'exif','Lens',?)"
        " ON CONFLICT(file_id,source,key) DO UPDATE SET value_text = excluded.value_text",
        [(i, "first") for i in range(2, n + 2)],
    )
    conn.executemany(
        "INSERT INTO file_param(file_id,source,key,value_text) VALUES(?,'exif','Lens',?)"
        " ON CONFLICT(file_id,source,key) DO UPDATE SET value_text = excluded.value_text",
        [(i, "second") for i in range(2, n + 2)],
    )
    return n * 2


@pytest.mark.parametrize(
    "label, work",
    [
        ("writing a parsed field", _insert_params),
        ("re-parsing a field", _upsert_params),
        ("renaming a file", _rename_files),
        ("deleting a file", _delete_files),
    ],
)
def test_the_cost_of_a_write_does_not_grow_with_the_library(ddl, label, work):
    ratio, small, large = growth(work, ddl)
    assert ratio < TOLERANCE, (
        f"{label} costs {ratio:.1f}x more per row at {LARGE:,} files than at "
        f"{SMALL:,} ({small:.0f} -> {large:.0f} us/row). Something scales with "
        f"the table -- look for a lookup with no index behind it."
    )


def test_the_registry_stays_exact_at_size(ddl):
    """The counter is arithmetic now, so it is only right if it is right every
    time. A drift of one is invisible; a drift per row is the old bug."""
    conn = a_library(ddl, LARGE)
    try:
        conn.executemany(
            "INSERT INTO file_param(file_id,source,key,value_text) VALUES(?,'exif','Flash','x')",
            [(i,) for i in range(2, LARGE + 2)],
        )
        assert conn.execute(
            "SELECT occurrences FROM param_key WHERE key='Flash'"
        ).fetchone()[0] == LARGE
        conn.executemany(
            "DELETE FROM file_param WHERE file_id=?", [(i,) for i in range(2, 502)]
        )
        assert conn.execute(
            "SELECT occurrences FROM param_key WHERE key='Flash'"
        ).fetchone()[0] == LARGE - 500
    finally:
        conn.close()


def test_the_search_index_stays_consistent_at_size(ddl):
    """integrity-check with a non-zero rank is what actually compares the
    index against its content; count(*) on an external-content table reads
    through to the content and can never disagree."""
    conn = a_library(ddl, LARGE)
    try:
        conn.executemany(
            "INSERT INTO file_param(file_id,source,key,value_text) VALUES(?,'exif','Lens',?)",
            [(i, f"lens {i}") for i in range(2, LARGE + 2)],
        )
        conn.executemany(
            "UPDATE file_param SET value_text=? WHERE file_id=? AND source='exif' AND key='Lens'",
            [(f"changed {i}", i) for i in range(2, 502)],
        )
        conn.executemany(
            "DELETE FROM file_param WHERE file_id=?", [(i,) for i in range(502, 1002)]
        )
        conn.execute("INSERT INTO param_fts(param_fts, rank) VALUES('integrity-check', 1)")
        conn.execute("INSERT INTO name_fts(name_fts, rank) VALUES('integrity-check', 1)")
        assert conn.execute(
            "SELECT count(*) FROM param_fts WHERE param_fts MATCH '\"changed\"'"
        ).fetchone()[0] == 500
    finally:
        conn.close()
