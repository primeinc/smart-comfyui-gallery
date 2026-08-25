"""Write a corpus, scan it, and say what the application made of it.

A corpus is a directory until something reads it. This is the step that
tells the difference: it writes a small one into a temporary tree,
serves the application over it, runs the jobs a person would run, and
prints what came back -- kinds, recipes, cameras, duplicate groups, and
how much of the timeline was empty enough to collapse.

Run by `just corpus prove`. Not a test: a test asserts a number somebody
chose, and the point here is to LOOK at the numbers before choosing any.
"""

from __future__ import annotations

import pathlib
import tempfile
import time

from tests import corpus


def _drain(client) -> None:
    from db import connect, runner

    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "corpus-report", time.time() + 86_400) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def main() -> int:
    from litestar.testing import TestClient

    from db import connect
    from sg_web.app import build_app

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sg-corpus-"))
    root = tmp / "library"
    told = corpus.spread(root, scale="small")
    print(f"wrote {told['files']} files + {told['duplicates']} duplicates under {root}\n")

    with TestClient(app=build_app(str(tmp / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        swept = client.post(f"/roots/{made['id']}/scan").json()
        _drain(client)
        print(f"scanned: {swept['added']} added, {swept['hashed']} hashed")
        for job in ("/jobs/ingest", "/jobs/context", "/jobs/phash", "/jobs/dupes"):
            client.post(job)
            _drain(client)

        conn = connect.connect(client.app.state.db_path, read_only=True)
        try:
            print("\nwhat it read")
            for row in conn.execute("SELECT kind, count(*) FROM file GROUP BY kind ORDER BY 2 DESC"):
                print(f"  {row[0]:16} {row[1]:5}")
            print(f"  {'with a recipe':16} {conn.execute('SELECT count(*) FROM generation').fetchone()[0]:5}")
            print(f"  {'with a capture':16} {conn.execute('SELECT count(*) FROM capture').fetchone()[0]:5}")
            print("\n  producers")
            for row in conn.execute("SELECT kind, count(DISTINCT name) FROM artifact GROUP BY kind ORDER BY 1"):
                print(f"    {row[0]:14} {row[1]:5}")
            print(f"\n  duplicate groups {conn.execute('SELECT count(*) FROM derived_dupe_group').fetchone()[0]}")
        finally:
            connect.close(conn)

        surface = client.get("/timeline", headers={"accept": "application/json"}).json()
        coverage = surface["coverage"]
        print("\nthe timeline")
        print(f"  dated          {coverage['interpreted']} of {coverage['present']}")
        print(f"  contested      {coverage['contested']}  (the muddled era, and nothing else)")
        print(f"  opening window {surface['start_spelled']} -> {surface['end_spelled']}")
        print(f"  collapsed      {[one['lasted'] for one in surface['skipped']]}")
        overview = surface["overview"] or {}
        print(f"  whole library  {[one['lasted'] for one in overview.get('skipped', [])]}")
        print(f"\n  the corpus stays at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
