"""A database of an older version, seeded from the schema that shipped it.

A test proving a migration works on a vN database has to HAVE a vN
database, and for thirty-five versions this repository made one by
taking today's build and inverting the steps back down. That is one
inversion per step per fixture, it goes stale the moment a step lands,
and -- the part that matters -- it is not a vN database. It is today's
database wearing a lower number.

What that hid, found in this module's first run: an authentic v1 or v2
database reaches today WITHOUT `derived_face_space`, its two agreement
triggers, or `derived_file_hash_space`, and still carrying
`derived_file_hash_phash`. Those objects entered `schema.sql` during the
window it was stamped v3 and no step was written for them -- `@step(2)`
says "purely additive for real this time" and creates one table.
`test_the_shipped_steps_take_a_v1_database_to_the_current_build`
asserted `drift == []` over that for thirty-five versions and was
believed, because the fixture it asserted it about already had every
object today's schema has. An inverted fixture cannot fail this way: it
starts from the answer.

So a fixture seeds from `tests/schemas/vNN.sql`: the file `git show`
returns for the commit that shipped that version, vendored verbatim.
`just schema vendor NN` adds one, `just schema versions` lists what git
has (v1 through v35 today), and `just schema prove` runs the check at
the bottom of this file.

They are vendored rather than read from git at test time because the
suite must run from a checkout with no git available, and they are never
edited -- a historical record that gets corrected is not one.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pathlib
import shutil
import sqlite3

from db.connect import close, connect

HERE = pathlib.Path(__file__).parent / "schemas"

#: One seeded master per version per process: executing a historical
#: schema onto disk costs hundreds of milliseconds and eight tests do it,
#: while a closed database file is a file and copies in single digits.
_MASTERS: dict[int, pathlib.Path] = {}


@functools.cache
def _master_dir() -> pathlib.Path:
    """The masters' home, kept ACROSS processes under `.pytest_cache`.

    A per-process temporary directory made this cache miss every time:
    the eight tests that seed here ask for eight DIFFERENT versions, one
    each, so within a run the dict is written and never read. Measured
    cold, one per version: 0.104s for v1 and 0.033-0.043s for v3, v4,
    v7, v25, v26, v29 -- 0.33s of a 2.69s module spent executing
    historical schemas that are byte-for-byte the same on every run.
    From the file, the same seven cost 0.007s.

    Safe to keep because the key is the vendored file's own bytes and
    those files are a historical record that is never edited: a schema
    corrected in place gets a new digest and a new master. Under
    `.pytest_cache` beside the corpus snapshots, so `just clean` and
    pytest's own `--cache-clear` both reach it.
    """
    where = pathlib.Path(__file__).resolve().parent.parent / ".pytest_cache" / "schema-masters"
    where.mkdir(parents=True, exist_ok=True)
    return where


#: What `_prove` treats as a version that does not carry to today, rather
#: than as the check itself breaking. Everything a bad schema or a step
#: that cannot read one raises.
REFUSED = (sqlite3.Error, AssertionError, OSError, ValueError, KeyError)


def available() -> list[int]:
    """The versions vendored here, ascending."""
    return sorted(int(one.stem[1:]) for one in HERE.glob("v*.sql"))


def sql(version: int) -> str:
    """The schema that shipped as `version`."""
    path = HERE / f"v{version:02d}.sql"
    if not path.exists():
        raise FileNotFoundError(
            f"no vendored schema for v{version}. `just schema vendor {version}` "
            f"writes it from git history; vendored now: {available()}"
        )
    return path.read_text(encoding="utf-8")


def seed(path, version: int) -> None:
    """Write a database at `version` to `path`, and prove it is one.

    The stamp is checked rather than assumed: a vendored file that does
    not stamp the version its name claims would hand every fixture built
    from it a quiet lie, which is the failure this whole module exists
    to stop. Checked when the master is built, which is per CONTENT --
    the cached file's name carries the digest of the SQL it was built
    from, so an edited schema is a different master and is checked
    again.
    """
    master = _MASTERS.get(version)
    if master is None:
        source = sql(version)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        master = _master_dir() / f"v{version:02d}-{digest}.db"
        if not master.exists():
            # Built beside the name it will take and moved onto it, so a
            # run that dies mid-executescript leaves no half-written file
            # for the next one to copy as a database. `os.replace` is
            # atomic within a directory, and two processes racing on the
            # same version write identical bytes either way.
            building = master.with_name(f"{master.stem}.{os.getpid()}.building")
            # Through db.connect, the way `build.build` writes a fresh one: a raw
            # sqlite3.connect leaves the schema's foreign keys inert, and a
            # fixture whose keys were never on is not the database it claims.
            conn = connect(building)
            try:
                conn.executescript(source)
                conn.commit()
                stamped = conn.execute("PRAGMA user_version").fetchone()[0]
                if stamped != version:
                    raise AssertionError(f"tests/schemas/v{version:02d}.sql stamps user_version={stamped}")
            finally:
                close(conn)
            os.replace(building, master)
        _MASTERS[version] = master
    shutil.copy(master, path)


def _prove() -> int:
    """`just schema prove`: every vendored schema builds and migrates.

    Not a pytest test on purpose. It is the check you run when VENDORING
    one, before any fixture depends on it -- and the v1 row is expected
    to fail today, which a test file would have to either xfail or hide.
    """
    import tempfile

    from db import build, migrate
    from db.connect import USER_VERSION

    bad = 0
    for version in available():
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "gallery.db"
            try:
                seed(path, version)
                ran = migrate.migrate(path, take_snapshot=False)
                drift = build.drift(path)
            except REFUSED as exc:
                print(f"v{version:<3} FAILED  {type(exc).__name__}: {exc}")
                bad += 1
                continue
            landed = ran[-1] if ran else version
            note = "ok" if landed == USER_VERSION and not drift else f"landed v{landed}, drift {drift}"
            if note != "ok":
                bad += 1
            print(f"v{version:<3} {note}")
    return bad


if __name__ == "__main__":
    raise SystemExit(_prove())
