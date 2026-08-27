"""A 128-bit file identifier survives, exactly.

Python 3.12 changed what Windows reports:

    On Windows, st_ino may now be up to 128 bits, depending on the file
    system. Previously it would not be above 64 bits, and larger file
    identifiers would be arbitrarily packed.
        -- cpython Doc/library/os.rst, versionchanged:: 3.12

SQLite's INTEGER is a signed 64-bit quantity (sqlite/sqlite
src/sqliteInt.h:1031 LARGEST_INT64), and a bound parameter above it
raises rather than truncating. So on ReFS, a Dev Drive, or some network
shares, the first directory the walk reached ended the whole scan:

    OverflowError: Python int too large to convert to SQLite INTEGER
      scan.py in ensure_folder
        "SELECT id, parent_id, name FROM folder WHERE root_id = ? AND inode = ?"

The column was the mistake, not the value. Nothing does arithmetic on a
filesystem identifier -- it is compared for equality and nothing else --
so it is stored as what it is: opaque text, exact.

The trap that makes this worth a test file of its own is that SQLite
would have accepted a lossy answer without complaint. Affinity is a
conversion rule, not a constraint: an INTEGER-affinity column silently
turns '340282366920938463463374607431768211455' into the REAL
3.402823669209385e+38. That is the wrong identity, reached with no
error, inside the code whose entire job is to avoid assigning one.
Masking or folding the value to make it fit would have been the same
mistake with more arithmetic.
"""

from __future__ import annotations

import pathlib

import pytest
from PIL import Image

from db import connect, scan
from tests.staging import NOW, fresh_schema

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"

#: The largest value SQLite's INTEGER holds.
LARGEST = 0x7FFFFFFFFFFFFFFF

#: What a filesystem can hand over, spanning the boundary that broke.
IDENTIFIERS = [
    1,
    4096,
    LARGEST - 1,
    LARGEST,
    LARGEST + 1,
    2**64 - 1,
    2**64,
    2**100 + 7,
    2**127,
    2**128 - 1,
    # a real-shaped ReFS FileId128: a volume-ish counter over a file record
    (0x0000000000010000 << 64) | 0x00000000000A1B2C,
]


# --- the representation -----------------------------------------------------


@pytest.mark.parametrize("value", IDENTIFIERS)
def test_an_identifier_is_kept_exactly(value):
    """No fold, no mask, no modulo. The digits that went in come out."""
    assert scan.fs_id(value) == str(value)


def test_a_filesystem_with_no_such_concept_reports_nothing():
    """Zero is absent, not an identifier -- one shared value would
    collapse every folder in the library onto a single row."""
    assert scan.fs_id(0) is None


def test_an_integer_column_would_have_taken_the_wrong_answer_quietly():
    """The reason the column type changed rather than the value.

    Without this the choice of TEXT reads as taste. SQLite's affinity
    converts on store, so the oversized identifier lands in an INTEGER
    column as a float and never comes back.
    """
    conn = connect.memory()
    try:
        conn.execute("CREATE TABLE as_integer(v INTEGER)")
        conn.execute("CREATE TABLE as_text(v TEXT)")
        huge = str(2**128 - 1)
        conn.execute("INSERT INTO as_integer VALUES(?)", (huge,))
        conn.execute("INSERT INTO as_text VALUES(?)", (huge,))

        lost, kind = conn.execute("SELECT v, typeof(v) FROM as_integer").fetchone()
        assert kind == "real", kind
        assert lost != huge, "an INTEGER column would have kept it after all"

        kept, kind = conn.execute("SELECT v, typeof(v) FROM as_text").fetchone()
        assert kind == "text"
        assert kept == huge

        # and binding the raw int is what raised in the first place
        with pytest.raises(OverflowError):
            conn.execute("INSERT INTO as_integer VALUES(?)", (2**128 - 1,))
    finally:
        conn.close()


@pytest.mark.parametrize("value", IDENTIFIERS)
def test_the_column_stores_and_matches_it(value):
    """Equality is the only operation this value has to support."""
    conn = fresh_schema()
    try:
        conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'Z:/x','library',0)")
        held = scan.fs_id(value)
        folder = scan.mint(conn, "folder", "x")
        conn.execute(
            "INSERT INTO folder(id, root_id, parent_id, name, depth, fs_id) VALUES(?, 1, NULL, 'x', 0, ?)",
            (folder, held),
        )
        found = conn.execute("SELECT id, fs_id FROM folder WHERE root_id = 1 AND fs_id = ?", (held,)).fetchone()
        assert found is not None, "the identifier did not match itself"
        assert found[1] == str(value), "stored something other than what it was given"
        # and a DIFFERENT identifier does not match it
        assert conn.execute("SELECT id FROM folder WHERE fs_id = ?", (str(value + 1),)).fetchone() is None
    finally:
        conn.close()


# --- the crash, through the scanner -----------------------------------------


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "pics"
    (root / "inner").mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(root / "top.png")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(root / "inner" / "deep.png")
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    conn.commit()
    yield conn, root
    conn.close()


class _Huge:
    """`os.stat` as a 128-bit filesystem reports it.

    Wraps the real result so every other field is genuine -- size and
    mtime still drive the change detection this walk depends on.
    """

    def __init__(self, real, ino: int):
        self._real = real
        self.st_ino = ino

    def __getattr__(self, name):
        return getattr(self._real, name)


def _reporting_128_bit_ids(monkeypatch) -> dict[str, int]:
    """Every stat in the walk answers with a distinct oversized id."""
    import os as os_module

    real = os_module.stat
    seen: dict[str, int] = {}

    def huge(path, *args, **kwargs):
        held = real(path, *args, **kwargs)
        seen.setdefault(str(path), (2**127) + len(seen) + 1)
        return _Huge(held, seen[str(path)])

    monkeypatch.setattr(scan.os, "stat", huge)
    return seen


def test_a_walk_over_a_128_bit_filesystem_completes(library, monkeypatch):
    """The reported crash, driven through the scanner that raised it."""
    given = _reporting_128_bit_ids(monkeypatch)
    conn, root = library

    told = scan.scan(conn, 1, root, NOW)

    assert told.added == 2, told
    kept = [row[0] for row in conn.execute("SELECT fs_id FROM folder WHERE fs_id IS NOT NULL")]
    assert kept, "no folder recorded an identifier at all"
    # every stored identifier is the EXACT one the filesystem reported
    assert set(kept) <= {str(one) for one in given.values()}
    assert any(int(one) > LARGEST for one in kept), "this walk never exercised an oversized id"


def test_the_files_keep_theirs_too(library, monkeypatch):
    """`observe_tree` reads `st_ino` for files as well, and that one
    reaches an INSERT rather than a SELECT."""
    given = _reporting_128_bit_ids(monkeypatch)
    conn, root = library

    scan.scan(conn, 1, root, NOW)

    kept = [row[0] for row in conn.execute("SELECT fs_id FROM file WHERE fs_id IS NOT NULL")]
    assert len(kept) == 2, kept
    assert set(kept) <= {str(one) for one in given.values()}
    assert all(int(one) > LARGEST for one in kept)


def test_a_second_walk_over_the_same_ids_changes_nothing(library, monkeypatch):
    """Identity across scans is the only thing this value is for: if the
    second walk disagreed, every folder would look new every time."""
    _reporting_128_bit_ids(monkeypatch)
    conn, root = library

    scan.scan(conn, 1, root, NOW)
    was = sorted(conn.execute("SELECT id, name, fs_id FROM folder").fetchall())

    again = scan.scan(conn, 1, root, NOW + 60)

    assert again.added == 0, again
    assert sorted(conn.execute("SELECT id, name, fs_id FROM folder").fetchall()) == was


def test_a_renamed_directory_is_still_the_same_one(library, monkeypatch):
    """What the identifier is FOR. A directory has no bytes to prove
    continuity with, so a rename that minted a new folder would orphan
    the old entity and rot its URL -- and that reasoning has to keep
    working at 128 bits."""
    given = _reporting_128_bit_ids(monkeypatch)
    conn, root = library
    scan.scan(conn, 1, root, NOW)
    before = conn.execute("SELECT id FROM folder WHERE name = 'inner'").fetchone()[0]

    # the same directory under a new name: its identifier follows it
    held = given[str(root / "inner")]
    given[str(root / "renamed")] = held
    (root / "inner").rename(root / "renamed")

    scan.scan(conn, 1, root, NOW + 120)

    after = conn.execute("SELECT id, fs_id FROM folder WHERE name = 'renamed' AND missing_since IS NULL").fetchone()
    assert after is not None, "the renamed directory was not found"
    assert after[0] == before, "a rename minted a new folder instead of following the identifier"
    assert after[1] == str(held)
