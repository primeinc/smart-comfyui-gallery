"""The serving layer: originals, ranges, thumbnails, avatars -- over HTTP.

Every claim is a request against the running application. Content types
come from the bytes, ranges obey RFC 9110, thumbnails render once and
serve from cache, an oriented phone photo arrives upright, and a person's
avatar is the face their cluster actually points at.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, derived, ingest, library, naming, scan
from sg_web.app import build_app
from tests.staging import Stage, staged
from vision import decode


def _library(tmp_path, write_media):
    """Real media on disk, scanned and ingested into a run's home."""
    root = tmp_path / "lib"
    root.mkdir()
    write_media(root)
    burrow = tmp_path / "run"
    burrow.mkdir()
    conn = connect.connect(burrow / "gallery.db")
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, 0.0)
    conn.commit()
    return conn, burrow, root


def _rgb(image: Image.Image, xy: tuple[int, int]) -> tuple[int, int, int]:
    """getpixel, narrowed: these fixtures are RGB by construction."""
    pixel = image.convert("RGB").getpixel(xy)
    assert isinstance(pixel, tuple)
    r, g, b = pixel
    return r, g, b


def _slug_of(conn, name: str) -> str:
    return conn.execute("SELECT e.slug FROM entity e JOIN file f ON f.id = e.id WHERE f.name = ?", (name,)).fetchone()[
        0
    ]


def _media(root: pathlib.Path) -> None:
    Image.new("RGB", (900, 400), (200, 30, 30)).save(root / "wide.png")
    turned = Image.new("RGB", (600, 400), (30, 30, 200))
    tag = Image.Exif()
    tag[274] = 6  # stored on its side; upright is 400x600
    turned.save(root / "turned.jpg", exif=tag)
    (root / "voice.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    import av

    with av.open(str(root / "clip.mp4"), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), (0, 0, 255), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _ingested(stage: Stage) -> None:
    conn = stage.conn()
    try:
        for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
            ingest.one(conn, file_id, stage.root / name, 0.0)
        conn.commit()
        stage.held["slugs"] = {name: _slug_of(conn, name) for (name,) in conn.execute("SELECT name FROM file")}
    finally:
        connect.close(conn)


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "bytes", _media, _ingested, worker=True) as stage:
        yield stage


@pytest.fixture
def served(_stage):
    _stage.restore()
    return _stage.client, _stage.held["slugs"], _stage.root


def test_originals_are_served_with_the_type_their_bytes_say(served):
    client, slugs, root = served
    answer = client.get(f"/media/{slugs['wide.png']}")
    assert answer.status_code == 200
    assert answer.headers["content-type"].startswith("image/png")
    assert answer.content == (root / "wide.png").read_bytes()
    assert answer.headers["accept-ranges"] == "bytes"

    clip = client.get(f"/media/{slugs['clip.mp4']}")
    assert clip.headers["content-type"].startswith("video/mp4")
    assert client.get("/media/no-such-thing").status_code == 404


def test_a_range_is_honoured_the_way_a_video_element_seeks(served):
    client, slugs, root = served
    whole = (root / "clip.mp4").read_bytes()
    slug = slugs["clip.mp4"]

    part = client.get(f"/media/{slug}", headers={"range": "bytes=0-3"})
    assert part.status_code == 206
    assert part.content == whole[:4]
    assert part.headers["content-range"] == f"bytes 0-3/{len(whole)}"
    assert part.headers["content-length"] == "4"

    tail = client.get(f"/media/{slug}", headers={"range": "bytes=-5"})
    assert tail.status_code == 206
    assert tail.content == whole[-5:]

    open_ended = client.get(f"/media/{slug}", headers={"range": f"bytes={len(whole) - 2}-"})
    assert open_ended.status_code == 206
    assert open_ended.content == whole[-2:]

    beyond = client.get(f"/media/{slug}", headers={"range": f"bytes={len(whole) * 2}-"})
    assert beyond.status_code == 416
    assert beyond.headers["content-range"] == f"bytes */{len(whole)}"

    garbled = client.get(f"/media/{slug}", headers={"range": "bytes=tuesday"})
    assert garbled.status_code == 200
    assert garbled.content == whole


def test_a_hostile_range_header_is_an_opinion_never_an_error():
    """RFC 9110 defines DIGIT as %x30-39 and nothing else. '²' is one
    latin-1 octet a real socket delivers and int() rejects; an
    Arabic-Indic digit (U+0661) is a numeral int() happily parses and
    the grammar forbids. An unparseable Range on an unauthenticated
    route must come back as "no opinion" -- never as an exception a
    handler did not catch."""
    from sg_web import media

    for hostile in (
        "bytes=-²",
        "bytes=²-",
        "bytes=\u0661-\u0662",
        "bytes=\u0661-",
        "bytes=0-²",
        "bytes=--",
        "bytes=1-2-3",
        "bytes=+1-2",
        "bytes=1_0-",
        "bytes=",
        "bytes=-",
        "bytes= - ",
        "octets=0-1",
        "bytes=\x00-",
    ):
        assert media.parse_range(hostile, 100) is None, hostile


def test_a_raw_latin1_range_octet_is_answered_not_crashed(served):
    """The wire case the test client cannot send: httpx refuses non-ASCII
    header values, but a socket delivers any octet and the server decodes
    it latin-1. Driven at the ASGI layer, `Range: bytes=-\xb2` must get
    the whole file with 200, not a 500."""
    import asyncio

    client, slugs, root = served

    async def drive():
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": f"/media/{slugs['clip.mp4']}",
            "raw_path": f"/media/{slugs['clip.mp4']}".encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver"), (b"range", b"bytes=-\xb2")],
            "client": ("127.0.0.1", 4444),
            "server": ("testserver", 80),
        }
        told = []
        handed = {"count": 0}
        finished = asyncio.Event()

        async def receive():
            # the request body once; then hang up only AFTER the last body
            # event -- a streaming response listens for the disconnect
            # concurrently, and an instant one cancels the stream mid-body
            handed["count"] += 1
            if handed["count"] == 1:
                return {"type": "http.request", "body": b"", "more_body": False}
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            told.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finished.set()

        await client.app(scope, receive, send)
        return told

    told = asyncio.run(drive())
    start = next(message for message in told if message["type"] == "http.response.start")
    assert start["status"] == 200
    body = b"".join(message.get("body", b"") for message in told if message["type"] == "http.response.body")
    assert body == (root / "clip.mp4").read_bytes()


def test_head_answers_what_get_would_say_without_the_body(served):
    """RFC 9110: a resource that answers GET answers HEAD with the same
    headers -- it is how a player learns length and seekability before
    asking for a single byte."""
    client, slugs, root = served
    slug = slugs["clip.mp4"]
    told = client.head(f"/media/{slug}")
    assert told.status_code == 200
    assert told.content == b""
    assert told.headers["content-length"] == str((root / "clip.mp4").stat().st_size)
    assert told.headers["accept-ranges"] == "bytes"
    assert told.headers["content-type"].startswith("video/mp4")
    assert client.head("/media/no-such-thing").status_code == 404


def test_a_thumbnail_renders_once_and_serves_from_cache(served):
    client, slugs, root = served
    slug = slugs["wide.png"]
    first = client.get(f"/thumb/{slug}")
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("image/webp")

    small = decode.open_bytes(first.content)
    assert small.size == (512, 228)

    # New bytes on disk, same recorded hash: a re-request must come from
    # the cache, not a re-decode of whatever sits at the path now.
    Image.new("RGB", (900, 400), (30, 200, 30)).save(root / "wide.png")
    again = decode.open_bytes(client.get(f"/thumb/{slug}").content)
    r, g, _ = _rgb(again, (256, 114))
    assert r > g, "the cache was silently re-rendered"


def test_a_sideways_phone_photo_is_thumbnailed_upright(served):
    client, slugs, _ = served

    answer = client.get(f"/thumb/{slugs['turned.jpg']}")
    small = decode.open_bytes(answer.content)
    assert small.height > small.width, "the EXIF turn was dropped on the way to the grid"


def test_a_video_thumbnail_is_a_poster_frame_at_the_clips_own_size(served):
    """This clip is 320x180, under both derivative edges, so both are the
    poster frame as decoded.

    They used to be enlarged to 512 and 1440. Nothing asked for that --
    the grid is `object-fit: cover` and the lightbox `object-fit:
    contain` -- and encoding the invented pixels was the most expensive
    phase in the pipeline for small sources.
    """
    client, slugs, _ = served

    slug = slugs["clip.mp4"]
    thumb = decode.open_bytes(client.get(f"/thumb/{slug}").content)
    assert thumb.size == (320, 180)
    centre = _rgb(thumb, (160, 90))
    assert centre[2] > centre[0], "the poster frame is not the clip's pixels"
    preview = decode.open_bytes(client.get(f"/preview/{slug}").content)
    assert preview.size == (320, 180)


def test_what_has_no_picture_says_so(served):
    client, slugs, _ = served
    assert client.get(f"/thumb/{slugs['voice.wav']}").status_code == 404


def test_an_avatar_is_the_face_the_cluster_points_at(tmp_path):
    def write(root):
        canvas = Image.new("RGB", (800, 600), (0, 0, 255))
        canvas.paste(Image.new("RGB", (200, 150), (255, 0, 0)), (200, 150))
        canvas.save(root / "ana_1.png")
        canvas.save(root / "ana_2.png")

    conn, burrow, _ = _library(tmp_path, write)
    rng = np.random.default_rng(9)
    vector = rng.standard_normal(32).astype(np.float32)
    sha_by_id = dict(conn.execute("SELECT id, content_sha256 FROM file"))
    for file_id, sha in sha_by_id.items():
        derived.record_faces(
            conn,
            file_id,
            "test/embedder",
            "1",
            sha,
            0.0,
            [
                {
                    "region": derived.region(conn, 0.25, 0.25, 0.25, 0.25),
                    "det_score": 0.9,
                    "embedding": vector.tobytes(),
                }
            ],
        )
    made = derived.cluster(conn, "test/embedder", "1", 0.0, threshold=0.5)
    assert len(made) == 1
    run_id = derived.run_for(conn, "test/embedder", "1", "chinese-whispers", 0.5, 0.0)
    derived.make_primary(conn, run_id)
    person_id = naming.claim(conn, "person", "Ana")
    conn.execute("INSERT INTO person(id,name,created_at) VALUES(?, 'Ana', 0)", (person_id,))
    conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE id = ?", (person_id, made[0]))
    conn.commit()
    conn.close()

    with TestClient(app=build_app(str(burrow))) as client:
        answer = client.get("/avatar/ana")
        assert answer.status_code == 200
        avatar = decode.open_bytes(answer.content)
        assert avatar.size == (256, 256)
        centre = _rgb(avatar, (128, 128))
        assert centre[0] > centre[2], "the avatar is not the asserted face"
        assert client.get("/avatar/nobody").status_code == 404
