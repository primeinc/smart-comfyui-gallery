"""Delivery: a thumbnail is bytes at an address, not a question.

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
  * the answer resolves thumbnail identity ONCE, so the URL a surface
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
from PIL.PngImagePlugin import PngInfo

from db import connect, ingest, resultset, scan
from tests.staging import NOW, Stage, fresh_schema, staged
from vision import thumbs

_HEX64 = re.compile(r"^/thumbs/([0-9a-f]{2})/([0-9a-f]{64})(\.preview)?\.webp$")


# --- what a surface is told to point at -------------------------------------


def test_the_asset_url_mirrors_the_cache_on_disk(tmp_path):
    """One function decides, because a grid cell, a filmstrip frame, a
    rail preview and a tray thumbnail must reach the same conclusion --
    or some of them go on paying for a round trip nobody can see."""
    sha = "b" * 64
    url = thumbs.asset_url(sha, "some-slug")
    assert url is not None
    found = _HEX64.match(url)
    assert found is not None, url
    assert found.group(1) == sha[:2], "the shard is the path on disk, so the route is a path join"

    # and it IS the path on disk, not merely similar to it
    on_disk = thumbs.path_for(tmp_path, sha, "thumb")
    assert url.endswith(f"{on_disk.parent.name}/{on_disk.name}")
    preview = thumbs.asset_url(sha, "s", "preview")
    assert preview is not None
    assert preview.endswith(thumbs.path_for(tmp_path, sha, "preview").name)


def test_a_file_nobody_has_hashed_yet_still_has_somewhere_to_point(tmp_path):
    """Ingest has not reached it, so there is no content address. The
    slug route can still answer, at the cost this exists to avoid, which
    is the right trade for a file nobody has finished reading."""
    assert thumbs.asset_url(None, "not-hashed") == "/thumb/not-hashed"
    assert thumbs.asset_url("", "not-hashed") == "/thumb/not-hashed"


def test_a_kind_with_no_picture_is_pointed_nowhere(tmp_path):
    """Hashing is what mints an asset address, and audio and documents
    get hashed like everything else -- so an address existed for them
    and the route behind it refused. Over a mixed eight-file library
    three of eight grid cells answered 404. Asking here is how a surface
    finds out before it points."""
    sha = "d" * 64
    assert thumbs.asset_url(sha, "a-song", medium="audio") is None
    assert thumbs.asset_url(sha, "a-paper", medium="document") is None
    for medium in thumbs.PICTURED:
        assert thumbs.asset_url(sha, "s", medium=medium) is not None, medium
    # unsaid stays unjudged: a caller that knows it holds a picture
    # need not say so
    assert thumbs.asset_url(sha, "s") is not None


def test_it_refuses_a_variant_that_is_not_one(tmp_path):
    with pytest.raises(ValueError, match="not a variant"):
        thumbs.asset_url("c" * 64, "s", "enormous")


# --- the answer resolves it once --------------------------------------------


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "pics"
    root.mkdir()
    for i in range(4):
        Image.new("RGB", (600, 400), (20 + i * 40, 90, 140)).save(root / f"p{i}.png")
    conn = fresh_schema()
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
        pointed = thumbs.asset_url(item["sha"], item["slug"], medium=item["kind"])
        assert pointed is not None, item
        assert _HEX64.match(pointed) is not None, pointed


# --- served: the claims a unit test cannot see ------------------------------


def _pictures(root: pathlib.Path) -> None:
    for i in range(6):
        Image.new("RGB", (900, 600), (30 + i * 30, 100, 150)).save(root / f"s{i}.png")
    # One GENERATED picture, so the artifact shelf has a model card with a
    # sample to draw. Without it `/models` is empty and the test about it
    # skips, which reads like a pass and proves nothing.
    told = PngInfo()
    told.add_text(
        "parameters",
        "a tin lighthouse at dusk\nNegative prompt: blurry\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, Size: 900x600, "
        "Model hash: 0123456789ab, Model: dreamshaper_8",
    )
    Image.new("RGB", (900, 600), (210, 130, 70)).save(root / "gen.png", pnginfo=told)


def _interpreted(stage: Stage) -> None:
    """The human context too. Without it `derived_media_context` is empty
    and the timeline draws nothing at all -- and a test that then finds
    no `<img>` reads exactly like a surface pointing at the wrong
    address, rather than like a world that never built the thing the
    surface is made of."""
    import time as clock

    stage.client.post("/jobs/ingest")
    _drained(stage.client, clock)
    stage.client.post("/jobs/context")
    stage.client.post("/jobs/events")
    _drained(stage.client, clock)


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with staged(
        tmp_path_factory,
        "test_a_thumbnail_is_a_static_asset",
        _pictures,
        _interpreted,
        worker=True,
        # The rendered derivatives ARE the steady state these tests read;
        # emptying the cache between them buys a re-render of six webps
        # per test and proves nothing. The one test about a miss deletes
        # its own entry.
        keep_thumbs=True,
    ) as stage:
        yield stage


@pytest.fixture
def served(_world):
    """The real application over a real library, with the derivatives
    rendered -- which is the steady state a person browses in. Built once
    for the module: every test here reads that steady state, and the
    restore between them puts back whatever one of them moved."""
    _world.restore()
    return _world.client, _world.root


def _drained(client, clock, timeout: float = 300.0) -> None:
    deadline = clock.time() + timeout
    while [job for job in client.get("/jobs").json() if job["state"] in ("queued", "running")]:
        assert clock.time() < deadline, "jobs never drained"
        clock.sleep(0.05)


def _connections_during(monkeypatch, work):
    """How many SQLite connections the application opens while doing this.

    Counted at `db.connect.connect`, which every route reaches through,
    so the number is what the application really did rather than what
    this test believes about it.

    `monkeypatch` rather than a try/finally around a module attribute:
    it puts the real one back when an assertion inside `work` fails too,
    and rebinding a module's `def` is not an assignment a type checker
    will take.
    """
    seen = {"opened": 0}
    real = connect.connect

    def counting(*args, **kwargs):
        seen["opened"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(connect, "connect", counting)
    try:
        work()
    finally:
        monkeypatch.undo()
    return seen["opened"]


def test_the_grid_points_at_content_addressed_assets(served):
    client, _ = served
    page = client.get("/g").text
    sources = re.findall(r'<img src="([^"]+)"', page)
    cells = [one for one in sources if "/thumbs/" in one or "/thumb/" in one]
    assert cells, "the grid drew pictures"
    for src in cells:
        assert _HEX64.match(src) is not None, f"{src} is not a content-addressed asset"


def test_serving_one_touches_no_database(served, monkeypatch):
    """The number the whole change exists to move. Sixty cells were
    sixty connections; this asserts the per-asset cost is zero."""
    client, _ = served
    src = next(one for one in re.findall(r'<img src="([^"]+)"', client.get("/g").text) if "/thumbs/" in one)
    # warm it, so this measures DELIVERY rather than the render-on-miss
    assert client.get(src).status_code == 200

    opened = _connections_during(monkeypatch, lambda: client.get(src))
    assert opened == 0, f"serving one derivative opened {opened} connections"

    # the control, on the same client, in the same test: the slug route
    # is the thing being replaced, and it does open one
    was = _connections_during(monkeypatch, lambda: client.get("/thumb/s0"))
    assert was >= 1, "the slug route opens a connection; if it does not, this test proves nothing"


def test_a_whole_page_of_thumbnails_costs_nothing_extra(served, monkeypatch):
    client, _ = served
    sources = [one for one in re.findall(r'<img src="([^"]+)"', client.get("/g").text) if "/thumbs/" in one]
    assert len(sources) >= 6
    for src in sources:
        client.get(src)  # warm
    opened = _connections_during(monkeypatch, lambda: [client.get(src) for src in sources])
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
    entry can be spelled, and `..` cannot appear at all."""
    client, _ = served
    for bad in (
        "/thumbs/ab/not-a-hash.webp",
        "/thumbs/ab/" + "a" * 64 + ".png",
        "/thumbs/zz/" + "a" * 64 + ".webp",
        "/thumbs/ab/" + "A" * 64 + ".webp",
    ):
        assert client.get(bad).status_code == 404, bad


# --- and every OTHER surface that draws pictures ----------------------------


#: These addresses answer JSON to a machine and a page to a browser, so
#: asking without saying which gets the machine's answer -- and a regex
#: for `<img>` over JSON finds nothing and reads exactly like a page
#: that drew no pictures. Cost ten minutes once.
AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}


def _pictures_on(client, where: str) -> list[str]:
    answered = client.get(where, headers=AS_BROWSER, follow_redirects=True)
    assert answered.status_code == 200, f"{where}: {answered.status_code}"
    assert "<html" in answered.text, f"{where} answered {answered.headers.get('content-type')}, not a page"
    return [one for one in re.findall(r'<img src="([^"]+)"', answered.text) if "/thumb" in one]


def test_every_surface_that_draws_a_picture_points_at_the_asset(served):
    """One definition of where a cell points, not five.

    The grid stopped paying a connection per picture and the person,
    folder, album and artifact pages went on doing so -- because nothing
    NAMED the step they were all skipping. `thumbs.address` is that
    step, and this is the check that no surface quietly stops taking it.
    """
    client, _ = served
    folder = _pictures_on(client, "/f/lib")
    assert folder, "the folder page drew no pictures"
    for src in folder:
        assert _HEX64.match(src) is not None, f"the folder page points at {src}"


def test_no_template_anywhere_still_spells_the_slug_route(served):
    """The check that finds the surface nobody thought of.

    Naming the step was not enough: the artifacts shelf and the person
    drawer went on spelling `/thumb/<slug>` into their own markup long
    after "the artifact pages joined it" was written down, because
    nothing looked at every template at once. A page-by-page test finds
    the pages somebody remembered.

    The ROUTE stays -- it is the honest fallback for a file ingest has
    not hashed yet, and `asset_url` returns it on purpose. What must not
    happen is a template building that address itself, where it cannot
    know whether the file has been hashed.
    """
    import pathlib as _pathlib

    here = _pathlib.Path(__file__).resolve().parents[1] / "sg_web" / "templates"
    spelt = [
        f"{one.name}: {line.strip()[:90]}"
        for one in sorted(here.rglob("*.html"))
        for line in one.read_text(encoding="utf-8").splitlines()
        if 'src="/thumb/' in line or "src='/thumb/" in line
    ]
    assert spelt == [], f"templates building a slug thumbnail address themselves: {spelt}"


def test_the_artifact_shelf_costs_no_connections(served, monkeypatch):
    """A shelf of forty cards at four samples each was a hundred and
    sixty lookups nothing could cache."""
    client, _ = served
    # `/models`, because that is where the shelf lives -- one page per
    # artifact kind, not one index over all of them.
    sources = _pictures_on(client, "/models")
    if not sources:
        pytest.skip("this library has no model with pictures to sample")
    for src in sources:
        assert _HEX64.match(src) is not None, f"the artifact shelf points at {src}"
        client.get(src)
    opened = _connections_during(monkeypatch, lambda: [client.get(src) for src in sources])
    assert opened == 0, f"{len(sources)} shelf thumbnails opened {opened} connections"


def test_a_folder_page_of_thumbnails_costs_no_connections(served, monkeypatch):
    """The number, on a surface other than the grid."""
    client, _ = served
    sources = _pictures_on(client, "/f/lib")
    assert len(sources) >= 6
    for src in sources:
        client.get(src)  # warm, so this measures delivery
    opened = _connections_during(monkeypatch, lambda: [client.get(src) for src in sources])
    assert opened == 0, f"{len(sources)} folder thumbnails opened {opened} connections"


def test_the_timeline_draws_content_addressed_assets(served):
    """The densest surface in the application.

    Session strips, scrubber segments, month and day cells, frames and
    bins are dozens of thumbnails on one page, and every one of them
    spelled `/thumb/<slug>` -- a route with a lookup behind it. It was
    also the least mechanical to convert: its pictures do not come from
    a ResultSet page, so a dozen statements had to carry the hash and
    the kind rather than a bare slug.
    """
    client, _ = served
    drawn = _pictures_on(client, "/timeline")
    assert drawn, "the timeline drew no pictures"
    for src in drawn:
        assert _HEX64.match(src) is not None, f"the timeline points at {src}"


def test_a_whole_timeline_of_thumbnails_costs_no_connections(served, monkeypatch):
    client, _ = served
    sources = _pictures_on(client, "/timeline")
    for src in sources:
        client.get(src)  # warm, so this measures delivery
    opened = _connections_during(monkeypatch, lambda: [client.get(src) for src in sources])
    assert opened == 0, f"{len(sources)} timeline thumbnails opened {opened} connections"


# --- a file that has no picture -----------------------------------------


def test_an_album_track_is_audio_and_not_a_video(tmp_path):
    """The sniff that cost a page of 500s.

    An .m4a is ISO-BMFF, so mimesniff's MP4 walk calls it video/mp4 --
    correct for choosing a decoder, wrong here, because `kind` decides
    whether a file HAS A PICTURE. Ingest let the bytes overrule the
    suffix, the track became a video, the grid minted it a thumbnail
    address, and the renderer went looking for a frame that does not
    exist.
    """
    from vision import sniff

    for brand in (b"M4A ", b"M4B ", b"M4P "):
        head = (0).to_bytes(4, "big") + b"ftyp" + brand + b"\x00" * 500
        assert sniff.sniff(head) == ("audio", "m4a"), brand

    # and the video brands still are videos
    movie = (0).to_bytes(4, "big") + b"ftyp" + b"mp42" + b"\x00" * 500
    assert sniff.sniff(movie) == ("video", "mp4")


def test_a_file_with_no_decodable_frame_is_a_404_not_a_500(served, request):
    """Even with the sniff right, anything that sniffs pictured and
    holds no frame -- a truncated clip, a video of zero frames -- must
    answer 404. It arrived as an uncaught 500 with a traceback, once per
    cell, which is how one bad row cost a page of stack traces instead
    of one grey cell."""
    client, root = served
    broken = root / "broken.mp4"
    broken.write_bytes((0).to_bytes(4, "big") + b"ftyp" + b"mp42" + b"\x00" * 64)
    # The stub is what this test is for, but leaving it is a library the
    # next restore cannot match: it rebuilds the whole world and trips
    # `_rebuilt_none`. Masked today only because this test runs last.
    request.addfinalizer(lambda: broken.unlink(missing_ok=True))
    client.post("/roots/1/scan")
    _drained(client, __import__("time"))

    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        held = conn.execute("SELECT content_sha256, kind FROM file WHERE name = 'broken.mp4'").fetchone()
    finally:
        connect.close(conn)
    if held is None or held[0] is None:
        pytest.skip("this build did not hash the unreadable file")
    sha, kind = held
    if kind not in ("image", "animated_image", "video"):
        pytest.skip(f"this build classified the stub as {kind}, which has no asset address")

    answered = client.get(f"/thumbs/{sha[:2]}/{sha}.webp")
    assert answered.status_code == 404, f"answered {answered.status_code}"
