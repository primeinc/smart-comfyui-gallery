"""A folder and an album say when their pictures are from.

The folders and albums index pages carry the span of human moments their
present pictures were interpreted to -- the earliest and the latest,
descendants and members included -- the same way the people index
says when a face was seen. An index page nothing has interpreted says
nothing rather than a made-up date.
"""

from __future__ import annotations

import os
import pathlib

from litestar.testing import TestClient
from PIL import Image

from db import collections, connect, naming, runner
from sg_web.app import build_app

NOW = 1_700_000_000.0
JUNE_10 = 1_686_355_200.0  # 2023-06-10 00:00 as a wall clock
FIRST = JUNE_10 + 14 * 3600 + 23 * 60 + 1
LAST = JUNE_10 + 2 * 86400 + 9 * 3600
AS_MACHINE = {"accept": "application/json"}


def _plain(path: pathlib.Path, at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), (30, 60, 90)).save(path)
    os.utime(path, (at, at))


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def test_the_context_sweep_is_for_what_is_missing(tmp_path, monkeypatch):
    """Interpreted files are not items again; a staled file is; a policy
    bump puts every file back; `everything` does too."""
    from db import context

    root = tmp_path / "lib"
    _plain(root / "a.png", NOW)
    _plain(root / "b.png", NOW)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        client.post("/jobs/ingest")
        first = client.post("/jobs/context")
        assert first.status_code == 201, first.text
        _drain(client)
        assert client.post("/jobs/context").status_code == 204, "every file is interpreted"

        conn = connect.connect(client.app.state.db_path)
        try:
            b = conn.execute("SELECT id FROM file WHERE name = 'b.png'").fetchone()[0]
            context.stale(conn, b)
            conn.commit()
            staled = runner.submit_context(conn, NOW)
            assert [r[0] for r in conn.execute("SELECT item_id FROM job_item WHERE job_id = ?", (staled,))] == [b]
            conn.commit()
        finally:
            connect.close(conn)
        _drain(client)

        monkeypatch.setattr(context, "POLICY_VERSION", context.POLICY_VERSION + 1)
        bumped = client.post("/jobs/context")
        assert bumped.status_code == 201, "an older policy's rows are not current"
        assert bumped.json()["total"] == 2
        monkeypatch.undo()
        assert client.post("/jobs/context", params={"everything": "true"}).json()["total"] == 2


def test_folders_and_albums_carry_the_span_of_their_pictures(tmp_path):
    root = tmp_path / "lib"
    _plain(root / "shots" / "Screenshot 2023-06-10 at 14.23.01.png", NOW)
    _plain(root / "shots" / "deeper" / "Screenshot 2023-06-12 at 09.00.00.png", NOW)
    _plain(root / "empty" / "nothing.txt".replace(".txt", ".png"), NOW)
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        before = client.get("/folders", headers=AS_MACHINE).json()
        shots = next(f for f in before[0]["folders"] if f["name"] == "lib")
        assert (shots["first_seen"], shots["last_seen"]) == (None, None), "not interpreted yet: no date is made up"

        client.post("/jobs/ingest")
        client.post("/jobs/context")
        _drain(client)

        shelves = client.get("/folders", headers=AS_MACHINE).json()
        shots = next(f for f in shelves[0]["folders"] if f["name"] == "lib")
        assert (shots["first_seen"], shots["last_seen"]) == (FIRST, NOW), (
            "descendants included: the named screenshots bound the start, the claimless file's mtime the end"
        )
        page = client.get("/folders", headers={"accept": "text/html"}).text
        assert f'data-epoch="{FIRST}"' in page
        assert "data-seen" in page
        assert "/static/build/timeline.js" in page, "the page spells its epochs"

        made = client.post("/albums", json={"name": "Trip", "kind": "album"})
        assert made.status_code < 300, made.text
        conn = connect.connect(client.app.state.db_path)
        try:
            found = naming.resolve(conn, "collection", "trip")
            assert found is not None
            newest = conn.execute(
                "SELECT id FROM file WHERE name = 'Screenshot 2023-06-12 at 09.00.00.png'"
            ).fetchone()[0]
            collections.set_membership(conn, found[0], newest, True, NOW)
            conn.commit()
        finally:
            connect.close(conn)
        albums = client.get("/albums", headers=AS_MACHINE).json()
        trip = next(a for a in albums if a["slug"] == "trip")
        assert (trip["first_seen"], trip["last_seen"]) == (LAST, LAST)
        tree = client.get("/albums", headers={"accept": "text/html"}).text
        assert f'data-epoch="{LAST}"' in tree
        assert "data-seen" in tree
