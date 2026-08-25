"""Delivery: a thumbnail is bytes at an address, not a query.

The derivative cache was already content-addressed and immutable on disk
-- `<sha[:2]>/<sha>.webp`, keyed on `content_sha256`, safe to delete
because nothing in it cannot be recomputed. That is the shape PhotoPrism
stores thumbnails in and the shape Immich serves them from.

The DELIVERY was not. Every 64-pixel cell went back through the semantic
application: open a SQLite connection, resolve a slug, check whether that
slug is retired, read the file's kind and content hash, build the cache
path, stat it, read the whole file into memory, and hand it back with no
cache headers -- so the browser asked again on the next page view.

Three claims here, and none of them is visible in a picture:

  * the asset route touches NO DATABASE. Asserted by counting the
    connections the application opens, which is the number the whole
    change exists to move.
  * the route resolves thumbnail identity ONCE, so the URL a surface
    emits is already the asset's.
  * a second view can cost nothing, because the URL names the bytes.

And one that is easy to lose: a library whose thumbs job has not run
must still show pictures. The route renders on a miss, exactly as the
slug route always did.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from PIL import Image

from db import connect, ingest, resultset, scan
from vision import thumbs

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0

_HEX64 = re.compile(r"^/thumbs/([0-9a-f]{2})/([0-9a-f]{64})(\.preview)?\.webp$")


# --- what a surface is told to point at -------------------------------------


def test_the_asset_url_mirrors_the_cache_on_disk(tmp_path):
    """One function decides, because a grid cell, a filmstrip frame, a
    rail preview and a tray thumbnail must reach the same conclusion --
    or some of them go on paying for a round trip nobody can see."""
    sha = "b" * 64
    url = thumbs.asset_url(sha, "some-slug")
    found = _HEX64.match(url)
    assert found is not None, url
    assert found.group(1) == sha[:2], "the shard is the path on disk, so the route is a path join"

    # and it IS the path on disk, not merely similar to it
    on_disk = thumbs.path_for(tmp_path, sha, "thumb")
    assert url.endswith(f"{on_disk.parent.name}/{on_disk.name}")
    assert thumbs.asset_url(sha, "s", "preview").endswith(thumbs.path_for(tmp_path, sha, "preview").name)


def test_a_file_nobody_has_hashed_yet_still_has_somewhere_to_point(tmp_path):
    """Ingest has not reached it, so there is no content address. The
    slug route can still respond, at the cost this exists to avoid, which
    is the right trade for a file nobody has finished reading."""
    assert thumbs.asset_url(None, "not-hashed") == "/thumb/not-hashed"
    assert thumbs.asset_url("", "not-hashed") == "/thumb/not-hashed"


def test_it_refuses_a_variant_that_is_not_one(tmp_path):
    with pytest.raises(ValueError, match="not a variant"):
        thumbs.asset_url("c" * 64, "s", "enormous")


# --- the route resolves it once ---------------------------------------------


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "pics"
    root.mkdir()
    for i in range(4):
        Image.new("RGB", (600, 400), (20 + i * 40, 90, 140)).save(root / f"p{i}.png")
    conn = connect.memory()
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, NOW)
    conn.commit()
    yield conn
    conn.close()


def test_the_answer_carries_the_content_hash(library):
    """Resolved ONCE, in the read the ResultSet already does. Without it
    every cell has to ask the database which bytes it is."""
    shape = resultset.page(library, "", resultset.parse(), 1, NOW)
    assert shape["items"], "the library has members"
    for item in shape["items"]:
        assert item["sha"], f"{item['name']} was ingested and has no content hash"
        assert _HEX64.match(thumbs.asset_url(item["sha"], item["slug"])) is not None


# --- served: the claims a unit test cannot see ------------------------------


@pytest.fixture
def served(tmp_path):
    """The real application over a real library, with the derivatives
    rendered -- which is the steady state a person browses in."""
    import time as clock

    from litestar.testing import TestClient

    from sg_web.app import build_app

    root = tmp_path / "pics"
    root.mkdir()
    for i in range(6):
        Image.new("RGB", (900, 600), (30 + i * 30, 100, 150)).save(root / f"s{i}.png")

    app = build_app(str(tmp_path / "run"))
    with TestClient(app=app) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        client.post("/jobs/ingest")
        _drained(client, clock)
        yield client, root


def _drained(client, clock, timeout: float = 300.0) -> None:
    deadline = clock.time() + timeout
    while [job for job in client.get("/jobs").json() if job["state"] in ("queued", "running")]:
        assert clock.time() < deadline, "jobs never drained"
        clock.sleep(0.05)


def _connections_during(work):
    """How many SQLite connections the application opens while doing this.

    Counted at `db.connect.connect`, which every route reaches through,
    so the number is what the application really did rather than what
    this test believes about it.
    """
    seen = {"opened": 0}
    real = connect.connect

    def counting(*args, **kwargs):
        seen["opened"] += 1
        return real(*args, **kwargs)

    connect.connect = counting
    try:
        work()
    finally:
        connect.connect = real
    return seen["opened"]


def test_the_grid_points_at_content_addressed_assets(served):
    client, _ = served
    page = client.get("/g").text
    sources = re.findall(r'<img src="([^"]+)"', page)
    cells = [one for one in sources if "/thumbs/" in one or "/thumb/" in one]
    assert cells, "the grid drew pictures"
    for src in cells:
        assert _HEX64.match(src) is not None, f"{src} is not a content-addressed asset"


def test_serving_one_touches_no_database(served):
    """The number the whole change exists to move. Sixty cells were
    sixty connections; this asserts the per-asset cost is zero."""
    client, _ = served
    src = next(one for one in re.findall(r'<img src="([^"]+)"', client.get("/g").text) if "/thumbs/" in one)
    # warm it, so this measures DELIVERY rather than the render-on-miss
    assert client.get(src).status_code == 200

    opened = _connections_during(lambda: client.get(src))
    assert opened == 0, f"serving one derivative opened {opened} connections"

    # the control, on the same client, in the same test: the slug route
    # is the thing being replaced, and it does open one
    was = _connections_during(lambda: client.get("/thumb/s0"))
    assert was >= 1, "the slug route opens a connection; if it does not, this test proves nothing"


def test_a_whole_page_of_thumbnails_costs_nothing_extra(served):
    client, _ = served
    sources = [one for one in re.findall(r'<img src="([^"]+)"', client.get("/g").text) if "/thumbs/" in one]
    assert len(sources) >= 6
    for src in sources:
        client.get(src)  # warm
    opened = _connections_during(lambda: [client.get(src) for src in sources])
    assert opened == 0, f"{len(sources)} thumbnails opened {opened} connections"


def test_the_bytes_are_cacheable_for_ever_because_the_url_names_them(served):
    """`<sha>.webp` cannot come to mean different pixels, so a browser
    that has it never needs to ask again."""
    client, _ = served
    src = next(one for one in re.findall(r'<img src="([^"]+)"', client.get("/g").text) if "/thumbs/" in one)
    answered = client.get(src)
    control = answered.headers.get("cache-control", "")
    assert "immutable" in control, control
    assert "max-age=31536000" in control, control
    assert answered.headers.get("content-type") == "image/webp"


def test_a_miss_renders_rather_than_showing_a_broken_picture(served):
    """A library whose thumbs job has not run must still show pictures.
    404ing here would trade a slow grid for a broken one."""
    client, home_root = served
    src = next(one for one in re.findall(r'<img src="([^"]+)"', client.get("/g").text) if "/thumbs/" in one)
    assert client.get(src).status_code == 200

    # delete the derivative underneath it, exactly as clearing the cache does
    found = _HEX64.match(src)
    assert found is not None
    cache = home_root.parent / "run" / "thumbs" / found.group(1)
    target = cache / src.rsplit("/", 1)[1]
    assert target.is_file(), target
    target.unlink()

    again = client.get(src)
    assert again.status_code == 200, "a miss renders from any file carrying those bytes"
    assert target.is_file(), "and the render is kept, so the next request is free again"


def test_a_name_that_is_not_a_derivative_is_refused(served):
    """The name is matched, never trusted: nothing that is not a cache
    entry can be named, and `..` cannot appear at all."""
    client, _ = served
    for bad in (
        "/thumbs/ab/not-a-hash.webp",
        "/thumbs/ab/" + "a" * 64 + ".png",
        "/thumbs/zz/" + "a" * 64 + ".webp",
        "/thumbs/ab/" + "A" * 64 + ".webp",
    ):
        assert client.get(bad).status_code == 404, bad
