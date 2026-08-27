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
import pathlib
import shutil
import sqlite3
import tempfile

from db.connect import close, connect

HERE = pathlib.Path(__file__).parent / "schemas"

#: One seeded master per version per process: executing a historical
#: schema onto disk costs hundreds of milliseconds and eight tests do it,
#: while a closed database file is a file and copies in single digits.
_MASTERS: dict[int, pathlib.Path] = {}


@functools.cache
def _master_dir() -> tempfile.TemporaryDirectory:
    """The masters' home for this process; the finalizer removes it."""
    return tempfile.TemporaryDirectory(prefix="sg-schema-masters-")


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
    to stop.
    """
    master = _MASTERS.get(version)
    if master is None:
        master = pathlib.Path(_master_dir().name) / f"v{version:02d}.db"
        # Through db.connect, the way `build.build` writes a fresh one: a raw
        # sqlite3.connect leaves the schema's foreign keys inert, and a
        # fixture whose keys were never on is not the database it claims.
        conn = connect(master)
        try:
            conn.executescript(sql(version))
            conn.commit()
            stamped = conn.execute("PRAGMA user_version").fetchone()[0]
            if stamped != version:
                raise AssertionError(f"tests/schemas/v{version:02d}.sql stamps user_version={stamped}")
        finally:
            close(conn)
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
