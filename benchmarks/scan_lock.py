"""How long a scan holds SQLite's one write lane.

There is exactly one write lane per database file. Whatever holds it,
every other writer waits -- and `busy_timeout` is 5000 ms
(db/connect.py), after which they do not wait, they fail.

A scan used to hold that lane for its whole duration, hashing included.
`sha256_of` reads every changed file off the disk, so the lane was held
for as long as it took to read the library. On a first scan of a new
root -- where every directory is created, opening the transaction on the
first one -- nothing else in the application could write until the last
byte was hashed. The background worker could not even claim a job; it
got `database is locked` every few seconds and reported it as a crash.

The walk writes nothing now (db/scan.py `survey`), so the lane is taken
only for the rows (`record` + `apply_scan`). This measures both halves
separately against real roots, because the ratio between them IS the
change, and the absolute write figure is what decides whether the lane
is still held past `busy_timeout` on a large library.

Run through `just bench scan-lock`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: What db/connect.py gives every connection. A lane held past this is
#: not slow, it is a failed statement somewhere else.
BUSY_TIMEOUT_MS = 5000


def measure(root: str, *, keep: pathlib.Path | None = None) -> dict:
    """One first scan of one root, timed in its two halves.

    A FIRST scan on purpose: it is the case that hurts, because every
    folder is minted and every file must be hashed. A rescan reuses
    recorded hashes and is cheap in exactly the half this measures.
    """
    from db import build, connect, scan

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="scan-lock-"))
    path = tmp / "gallery.db"
    try:
        build.build(path)
        conn = connect.connect(str(path))
        try:
            conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (root,))
            conn.commit()
            now = time.time()

            started = time.perf_counter()
            held = scan.survey(conn, 1, root, now)
            surveyed = time.perf_counter()
            observed, hashed = scan.record(conn, 1, held, now)
            result = scan.apply_scan(conn, observed, now, hashed=hashed, roots={1})
            conn.commit()
            ended = time.perf_counter()
        finally:
            connect.close(conn)

        survey_ms = (surveyed - started) * 1000
        write_ms = (ended - surveyed) * 1000
        return {
            "root": root,
            "files": len(observed),
            "folders": len(held.dirs),
            "hashed": hashed,
            "added": result.added,
            "survey_ms": round(survey_ms, 1),
            "locked_ms": round(write_ms, 1),
            "total_ms": round(survey_ms + write_ms, 1),
            # what the lane cost before the split: the whole thing
            "locked_ms_before": round(survey_ms + write_ms, 1),
            "share_removed": round(1 - write_ms / (survey_ms + write_ms), 4),
            "over_busy_timeout_before": survey_ms + write_ms > BUSY_TIMEOUT_MS,
            "over_busy_timeout_now": write_ms > BUSY_TIMEOUT_MS,
        }
    finally:
        if keep is None:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="How long a scan holds the write lane")
    parser.add_argument(
        "roots",
        nargs="*",
        default=[
            str(one) for one in sorted(pathlib.Path("C:/ComfyUI/output/sample-datasets").iterdir()) if one.is_dir()
        ]
        if pathlib.Path("C:/ComfyUI/output/sample-datasets").is_dir()
        else [],
        help="directories to scan; defaults to every sample dataset",
    )
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "scan_lock.json"))
    asked = parser.parse_args()
    if not asked.roots:
        raise SystemExit("name at least one root to scan")

    print(f"{'files':>7} {'hashed':>7} {'survey ms':>10} {'LOCKED ms':>10} {'was locked':>11} {'removed':>8}  root")
    rows = []
    for root in asked.roots:
        if not pathlib.Path(root).is_dir():
            print(f"{'-':>7} {'-':>7} {'-':>10} {'-':>10} {'-':>11} {'-':>8}  {root} (not a directory)")
            continue
        row = measure(root)
        rows.append(row)
        flag = "  OVER BUSY TIMEOUT" if row["over_busy_timeout_now"] else ""
        print(
            f"{row['files']:>7} {row['hashed']:>7} {row['survey_ms']:>10.1f} {row['locked_ms']:>10.1f} "
            f"{row['locked_ms_before']:>11.1f} {row['share_removed']:>7.1%}  {pathlib.Path(root).name}{flag}"
        )

    if rows:
        total_before = sum(one["locked_ms_before"] for one in rows)
        total_now = sum(one["locked_ms"] for one in rows)
        files = sum(one["files"] for one in rows)
        print()
        print(f"over {files} files: the lane was held {total_before:.0f} ms, and is now held {total_now:.0f} ms")
        print(f"per 1000 files, locked: {1000 * total_now / files:.0f} ms  (was {1000 * total_before / files:.0f} ms)")
        crosses_now = int(files * BUSY_TIMEOUT_MS / total_now)
        crosses_before = int(files * BUSY_TIMEOUT_MS / total_before)
        print(f"busy_timeout is {BUSY_TIMEOUT_MS} ms, so a library crosses it at roughly")
        print(f"    {crosses_now:,} files now, and {crosses_before:,} files before")

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as sheet:
        json.dump({"busy_timeout_ms": BUSY_TIMEOUT_MS, "rows": rows}, sheet, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
