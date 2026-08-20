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


def objects(conn: sqlite3.Connection) -> dict[str, str]:
    """Every schema object, keyed by name, whitespace-normalised."""
    return {
        name: " ".join((sql or "").split())
        for name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }


def reference() -> sqlite3.Connection:
    """A throwaway database built from the DDL, to compare against."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql())
    return conn


def drift(path: pathlib.Path) -> list[str]:
    """Names that differ between the built file and a fresh build."""
    if not path.exists():
        return [f"{path.name} does not exist"]
    built = objects(connect(path, read_only=True))
    fresh = objects(reference())
    out = [f"only in the built file: {n}" for n in sorted(set(built) - set(fresh))]
    out += [f"missing from the built file: {n}" for n in sorted(set(fresh) - set(built))]
    out += [f"differs: {n}" for n in sorted(set(built) & set(fresh)) if built[n] != fresh[n]]
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
    conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    conn.execute(f"PRAGMA user_version = {USER_VERSION}")
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
        kind: conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type=?", (kind,)
        ).fetchone()[0]
        for kind in ("table", "index", "trigger")
    }
    print(f"built {args.path}: {counts}, schema v{USER_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
