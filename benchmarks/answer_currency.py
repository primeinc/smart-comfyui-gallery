"""What a page costs while anything else is writing.

The ResultSet materialises the whole ordered answer once and pages it by
slicing, and a projection is valid for one (question, data currency)
pair. Currency is `PRAGMA data_version`, which bumps when ANY other
connection commits ANYTHING -- that is what the pragma measures.

It was `PRAGMA data_version`, which bumps when ANY connection commits
ANYTHING -- so a ledger row landing in `job_event` discarded a
projection over `file` ordering, which cannot have changed. Jobs commit
per item (db/runner.py `committed`) and a precache over a large library
runs for hours, so the invalidation was continuous for hours.

It is now `answer_generation` (db/schema.sql), moved by every table
except `job`, `job_item` and `job_event`.

This measures the same page three ways: with nothing writing, with a
second connection committing a LEDGER row between every request (the
job that runs for hours), and with one committing a row an answer is
actually built from. The first two should agree and the third should
not -- an answer that can have changed must still be rebuilt, which is
the contract, and only the invalidation's granularity was ever wrong.

Rows are synthesised rather than scanned: this is about the ORDERED
ANSWER's cost, which depends on how many rows there are and nothing
else. Run through `just bench answer-currency`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics
import sys
import tempfile
import time
import uuid

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Library sizes to sweep. The point is the SHAPE of the curve: the
#: rebuild is the whole answer, so it grows while the slice does not.
SIZES = (1_000, 10_000, 40_000, 80_000)

#: Requests per measurement. Small on purpose -- the effect is a factor
#: of a hundred, not a few percent, so this is not a microbenchmark.
ROUNDS = 20


def _library(path: pathlib.Path, count: int) -> None:
    """`count` file rows and nothing else. No bytes, no folders beyond
    one: the answer's cost is the row count."""
    from db import build, connect, migrate

    build.build(path)
    conn = connect.connect(str(path))
    try:
        conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/synthetic','library',0)")
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','synthetic')", (uuid.uuid4().bytes,))
        conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'synthetic',0)")
        conn.executemany(
            "INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)",
            [(i, uuid.uuid4().bytes, f"f{i:07d}") for i in range(2, count + 2)],
        )
        conn.executemany(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1000,?,?,0,0)",
            [(i, f"f{i:07d}.png", 1_700_000_000 + i, f"{i:064x}") for i in range(2, count + 2)],
        )
        conn.commit()
        migrate.analyze(conn)
        conn.commit()
    finally:
        connect.close(conn)


def measure(count: int, rounds: int = ROUNDS) -> dict:
    from db import connect, resultset

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="currency-"))
    path = tmp / "gallery.db"
    try:
        _library(path, count)
        conn = connect.connect(str(path))
        writer = connect.connect(str(path))
        try:
            asked = resultset.parse()
            resultset.page(conn, "", asked, 1, time.time())  # build it once

            quiet = []
            for _ in range(rounds):
                started = time.perf_counter()
                resultset.page(conn, "", asked, 5, time.time())
                quiet.append((time.perf_counter() - started) * 1000)

            # A LEDGER commit: what the job running for hours does.
            # Nothing here can change an answer, so nothing should be
            # discarded.
            writer.execute("INSERT INTO job(kind, state, created_at) VALUES('hash','queued',0)")
            writer.commit()
            job = writer.execute("SELECT id FROM job").fetchone()[0]
            ledger = []
            for i in range(rounds):
                writer.execute(
                    "INSERT INTO job_event(job_id, at, type, severity) VALUES(?, ?, 'item.done', 'info')",
                    (job, i),
                )
                writer.commit()
                started = time.perf_counter()
                resultset.page(conn, "", asked, 5, time.time())
                ledger.append((time.perf_counter() - started) * 1000)

            # A commit that CAN change the answer. This must still cost
            # the rebuild, or the fix has broken the contract instead of
            # sharpening it.
            real = []
            for _ in range(rounds):
                writer.execute("UPDATE file SET mtime = mtime + 1 WHERE id = 2")
                writer.commit()
                started = time.perf_counter()
                resultset.page(conn, "", asked, 5, time.time())
                real.append((time.perf_counter() - started) * 1000)
        finally:
            connect.close(writer)
            connect.close(conn)

        still = statistics.median(quiet)
        during_job = statistics.median(ledger)
        after_change = statistics.median(real)
        return {
            "files": count,
            "quiet_ms": round(still, 3),
            "ledger_commit_ms": round(during_job, 3),
            "answer_commit_ms": round(after_change, 2),
            "ledger_factor": round(during_job / still, 1) if still else None,
            "answer_factor": round(after_change / still, 1) if still else None,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="What a page costs while anything else is writing")
    parser.add_argument("--sizes", default=",".join(str(one) for one in SIZES))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "answer_currency.json"))
    asked = parser.parse_args()

    print(f"{'files':>8} {'at rest':>9} {'ledger':>9} {'answer':>9} {'ledger x':>9} {'answer x':>9}")
    rows = []
    for count in [int(one) for one in asked.sizes.split(",") if one.strip()]:
        row = measure(count, asked.rounds)
        rows.append(row)
        print(
            f"{row['files']:>8} {row['quiet_ms']:>9.3f} {row['ledger_commit_ms']:>9.3f} "
            f"{row['answer_commit_ms']:>9.2f} {row['ledger_factor']:>8.1f}x {row['answer_factor']:>8.1f}x"
        )

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as sheet:
        json.dump({"rounds": asked.rounds, "rows": rows}, sheet, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
