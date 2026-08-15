"""CLI for AI DAM derived-state maintenance.

All AI tables are derived, rebuildable state; this proves it operationally:

    python -m smartgallery_ai rebuild --db PATH [--keep-feedback/--drop-feedback]
    python -m smartgallery_ai status  --db PATH

`rebuild` drops the derived tables (human feedback preserved by default)
and recreates the empty schema; the background worker (or `POST
/galleryout/api/aidam/index/<id>`) repopulates everything from source media
and provisioned models.
"""

import argparse
import sqlite3
import sys

from smartgallery_ai import schema


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_rebuild(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        schema.drop_derived_state(conn, keep_feedback=not args.drop_feedback)
        schema.init_schema(conn)
    finally:
        conn.close()
    kept = "kept" if not args.drop_feedback else "DROPPED"
    print(f"Derived AI state dropped and schema recreated (feedback {kept}). "
          f"The worker will repopulate from source media.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    try:
        for table in schema.DERIVED_TABLES:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                n = "missing"
            print(f"{table:24s} {n}")
    finally:
        conn.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="smartgallery_ai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rebuild = sub.add_parser("rebuild", help="drop + recreate derived AI state")
    p_rebuild.add_argument("--db", required=True, help="SmartGallery sqlite file")
    p_rebuild.add_argument("--drop-feedback", action="store_true",
                           help="also drop human feedback (NOT recomputable)")
    p_rebuild.set_defaults(fn=cmd_rebuild)

    p_status = sub.add_parser("status", help="row counts per derived table")
    p_status.add_argument("--db", required=True)
    p_status.set_defaults(fn=cmd_status)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
