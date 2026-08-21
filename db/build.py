"""Build the gallery database from schema.sql, reproducibly.

Without this there was no process that regenerated the built file, so it drifted
from the DDL and nothing noticed: the database on disk still declared
`content_hash` long after the schema had split it into `content_sha256` and
`quoted_hash`. The whole contract suite loads schema.sql into `:memory:`, so it
stayed green against a file nobody had rebuilt.

    python -m db.build              # build db/gallery.db
    python -m db.build --check      # verify it matches schema.sql, build nothing
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

from db.connect import APPLICATION_ID, USER_VERSION, connect, schema_sql

DEFAULT = pathlib.Path(__file__).resolve().parent / "gallery.db"


def _squeezed(sql: str) -> str:
    """Whitespace-normalised OUTSIDE string literals only.

    A literal's spacing is content -- a RAISE message with a doubled
    space is a different message -- and a comparator that folds it would
    call a migrated trigger equal to a fresh one they no longer match.
    Splitting on the quote keeps SQLite's '' escapes intact: they become
    empty even-indexed segments the rejoin restores verbatim.
    """
    parts = sql.split("'")
    return "'".join(" ".join(part.split()) if index % 2 == 0 else part for index, part in enumerate(parts))


def objects(conn: sqlite3.Connection) -> dict[str, str]:
    """Every schema object, keyed by name, comparably normalised."""
    return {
        name: _squeezed(sql or "")
        for name, sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
    }


def reference() -> sqlite3.Connection:
    """A throwaway database built from the DDL, to compare against."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql())
    return conn


def stamps(conn: sqlite3.Connection) -> tuple[int, int]:
    return (
        conn.execute("PRAGMA application_id").fetchone()[0],
        conn.execute("PRAGMA user_version").fetchone()[0],
    )


def drift(path: pathlib.Path) -> list[str]:
    """What differs between the built file and a fresh build."""
    if not path.exists():
        return [f"{path.name} does not exist"]
    live = connect(path, read_only=True)
    fresh_conn = reference()
    built, fresh = objects(live), objects(fresh_conn)
    out = [f"only in the built file: {n}" for n in sorted(set(built) - set(fresh))]
    out += [f"missing from the built file: {n}" for n in sorted(set(fresh) - set(built))]
    out += [f"differs: {n}" for n in sorted(set(built) & set(fresh)) if built[n] != fresh[n]]
    # The pragmas too, which this check existed to catch and could not see:
    # it read `name, sql FROM sqlite_master` only, so a file stamped with the
    # wrong version -- the exact thing the stamps were added for -- was
    # reported as in sync.
    if stamps(live) != stamps(fresh_conn):
        out.append(f"stamped {stamps(live)}, the DDL stamps {stamps(fresh_conn)}")
    return out


def build(path: pathlib.Path, *, force: bool = False) -> None:
    """Create the database. Refuses to overwrite one holding rows.

    The plan requires initialisation to refuse a destructive cutover without an
    explicit operator action; this is where that refusal lives.
    """
    if path.exists():
        existing = connect(path, read_only=True)
        try:
            rows = existing.execute("SELECT count(*) FROM file").fetchone()[0]
        except sqlite3.Error:
            rows = 0
        existing.close()
        if rows and not force:
            raise SystemExit(
                f"{path} already holds {rows} files. Rebuilding destroys every rating, "
                f"comment, album and name in it. Pass --force if that is what you want."
            )
        for suffix in ("", "-wal", "-shm"):
            pathlib.Path(str(path) + suffix).unlink(missing_ok=True)

    conn = connect(path)
    conn.executescript(schema_sql())
    # The DDL stamps both itself. Re-stamping here is what let schema.sql say
    # v1 while connect.py said v3 for two versions without anything failing:
    # every database this function produced was correct, and every database
    # built from the DDL any other way was two versions behind and unopenable.
    stamped = conn.execute("PRAGMA user_version").fetchone()[0]
    app = conn.execute("PRAGMA application_id").fetchone()[0]
    if stamped != USER_VERSION or app != APPLICATION_ID:
        raise SystemExit(
            f"schema.sql stamps v{stamped}/{app:#x} but this build is "
            f"v{USER_VERSION}/{APPLICATION_ID:#x}. Fix the PRAGMAs at the end "
            f"of the DDL; do not stamp them from here."
        )
    conn.commit()
    conn.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=pathlib.Path, default=DEFAULT)
    ap.add_argument("--check", action="store_true", help="report drift, build nothing")
    ap.add_argument("--force", action="store_true", help="rebuild even if it holds rows")
    args = ap.parse_args(argv)

    if args.check:
        problems = drift(args.path)
        for line in problems:
            print(line)
        print("in sync with schema.sql" if not problems else f"{len(problems)} differences")
        return 1 if problems else 0

    build(args.path, force=args.force)
    problems = drift(args.path)
    if problems:
        print("built, but does not match the DDL:", *problems, sep="\n  ")
        return 1
    conn = connect(args.path, read_only=True)
    counts = {
        kind: conn.execute("SELECT count(*) FROM sqlite_master WHERE type=?", (kind,)).fetchone()[0]
        for kind in ("table", "index", "trigger")
    }
    print(f"built {args.path}: {counts}, schema v{USER_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
