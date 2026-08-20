"""The serving layer: originals, ranges, thumbnails, avatars -- over HTTP.

Every claim is a request against the running application. Content types
come from the bytes, ranges obey RFC 9110, thumbnails render once and
serve from cache, an oriented phone photo arrives upright, and a person's
avatar is the face their cluster actually points at.
"""

from __future__ import annotations

import numpy as np
import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, derived, ingest, library, naming, scan
from sg_web.app import build_app


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


def _slug_of(conn, name: str) -> str:
    return conn.execute(
        "SELECT e.slug FROM entity e JOIN file f ON f.id = e.id WHERE f.name = ?", (name,)
    ).fetchone()[0]


@pytest.fixture
def served(tmp_path):
    def write(root):
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

    conn, burrow, root = _library(tmp_path, write)
    slugs = {name: _slug_of(conn, name) for (name,) in conn.execute("SELECT name FROM file")}
    conn.close()
    with TestClient(app=build_app(str(burrow))) as client:
        yield client, slugs, root


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


def test_a_thumbnail_renders_once_and_serves_from_cache(served):
    client, slugs, root = served
    slug = slugs["wide.png"]
    first = client.get(f"/thumb/{slug}")
    assert first.status_code == 200
    assert first.headers["content-type"].startswith("image/webp")
    import io

    small = Image.open(io.BytesIO(first.content))
    assert small.size == (512, 228)

    # New bytes on disk, same recorded hash: a re-request must come from
    # the cache, not a re-decode of whatever sits at the path now.
    Image.new("RGB", (900, 400), (30, 200, 30)).save(root / "wide.png")
    again = Image.open(io.BytesIO(client.get(f"/thumb/{slug}").content)).convert("RGB")
    r, g, b = again.getpixel((256, 114))
    assert r > g, "the cache was silently re-rendered"


def test_a_sideways_phone_photo_is_thumbnailed_upright(served):
    client, slugs, _ = served
    import io

    answer = client.get(f"/thumb/{slugs['turned.jpg']}")
    small = Image.open(io.BytesIO(answer.content))
    assert small.height > small.width, "the EXIF turn was dropped on the way to the grid"


def test_a_video_thumbnail_is_a_frame_and_a_preview_is_bigger(served):
    client, slugs, _ = served
    import io

    slug = slugs["clip.mp4"]
    thumb = Image.open(io.BytesIO(client.get(f"/thumb/{slug}").content)).convert("RGB")
    assert thumb.size == (512, 288)
    centre = thumb.getpixel((256, 144))
    assert centre[2] > centre[0], "the poster frame is not the clip's pixels"
    preview = Image.open(io.BytesIO(client.get(f"/preview/{slug}").content))
    assert preview.size == (1440, 810)


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

    import io

    with TestClient(app=build_app(str(burrow))) as client:
        answer = client.get("/avatar/ana")
        assert answer.status_code == 200
        avatar = Image.open(io.BytesIO(answer.content)).convert("RGB")
        assert avatar.size == (256, 256)
        centre = avatar.getpixel((128, 128))
        assert centre[0] > centre[2], "the avatar is not the asserted face"
        assert client.get("/avatar/nobody").status_code == 404
