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

from smartgallery_ai import provision as P
from smartgallery_ai import resolve_models_dir, schema
from smartgallery_ai.provision import GROUPS  # stdlib-only import; registry drives the help


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the gallery SQLite database with name-addressable rows."""
    return schema.connect(db_path)


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Drop derived AI tables (feedback kept unless --drop-feedback) and recreate
    the empty schema, leaving repopulation to the background worker."""
    conn = _connect(args.db)
    try:
        schema.drop_derived_state(conn, keep_feedback=not args.drop_feedback)
        schema.init_schema(conn)
    finally:
        conn.close()
    kept = "kept" if not args.drop_feedback else "DROPPED"
    print(
        f"Derived AI state dropped and schema recreated (feedback {kept}). "
        f"The worker will repopulate from source media."
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print one row count per derived table; tables absent from the DB print as 'missing'."""
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


def cmd_provision(args: argparse.Namespace) -> int:
    """Show the provisioning plan and, unless --list, download the missing
    weights for the requested groups. The request path never downloads;
    besides this command, only the worker's async auto-provisioning
    (AI_DAM_AUTO_PROVISION, default on) fetches weights."""

    models_dir = resolve_models_dir(explicit=args.models_dir)
    try:
        print(P.format_plan(models_dir, args.groups))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if args.list:
        return 0
    if not args.yes:
        answer = input("\nDownload the MISSING artifacts above? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1
    try:
        result = P.provision(models_dir, args.groups, force=args.force)
    except P.ProvisionError as exc:
        print(f"error: {exc}")
        return 1
    print(f"\ndone: {len(result['downloaded'])} downloaded, {len(result['skipped'])} already present")
    if result["downloaded"]:
        print("Restart the gallery (or wait for the worker's backend retry window) to activate the new backends.")
    return 0


def main(argv=None) -> int:
    """Parse arguments and dispatch to the chosen subcommand; returns its exit code."""
    parser = argparse.ArgumentParser(prog="smartgallery_ai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rebuild = sub.add_parser("rebuild", help="drop + recreate derived AI state")
    p_rebuild.add_argument("--db", required=True, help="SmartGallery sqlite file")
    p_rebuild.add_argument("--drop-feedback", action="store_true", help="also drop human feedback (NOT recomputable)")
    p_rebuild.set_defaults(fn=cmd_rebuild)

    p_status = sub.add_parser("status", help="row counts per derived table")
    p_status.add_argument("--db", required=True)
    p_status.set_defaults(fn=cmd_status)

    p_prov = sub.add_parser(
        "provision",
        help="download model weights into the models dir (the worker also "
        "auto-provisions on start unless AI_DAM_AUTO_PROVISION=false)",
    )
    p_prov.add_argument("groups", nargs="*", default=["all"], help=", ".join(g.name for g in GROUPS) + ", or all")
    # default="" so resolve_models_dir sees "not given" and can fall through
    # to AI_DAM_MODELS_DIR. A literal ".AImodels" here overrode the variable
    # the app itself reads, so `provision` could download into ./.AImodels
    # while the gallery loaded from somewhere else and reported the backend
    # missing -- with provision.py's own out-of-space message telling you to
    # set the variable this ignored.
    p_prov.add_argument(
        "--models-dir",
        default="",
        help="target directory the backends load from (default: $AI_DAM_MODELS_DIR, else ./.AImodels)",
    )
    p_prov.add_argument("--list", action="store_true", help="show the plan only; download nothing")
    p_prov.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_prov.add_argument("--force", action="store_true", help="re-download artifacts that already exist")
    p_prov.set_defaults(fn=cmd_provision)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
