"""The derived-image cache serves what layouts need, without waste.

Three variants -- grid thumb, lightbox preview, face avatar -- rendered
once per content hash, written by the jobs that already decoded the
pixels, and chosen so a video is represented by the people in it. The
sampling that backs the video case refuses to conclude "no faces" from a
cadence alone: it bisects, bounded, and keeps its negative evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from db import connect, detect, library, sample, scan
from vision import thumbs
from vision.faces import FaceDetection, StubFaceBackend

SHA = "ab" * 32


def _rgb(image: Image.Image, xy: tuple[int, int]) -> tuple[int, int, int]:
    """getpixel, narrowed: these fixtures are RGB by construction."""
    pixel = image.convert("RGB").getpixel(xy)
    assert isinstance(pixel, tuple)
    r, g, b = pixel
    return r, g, b


def test_thumb_and_preview_are_contained_to_their_edges(tmp_path):
    big = Image.new("RGB", (2000, 1000), (10, 200, 30))
    thumbs.put_all(tmp_path, SHA, big)
    assert Image.open(thumbs.path_for(tmp_path, SHA)).size == (512, 256)
    assert Image.open(thumbs.path_for(tmp_path, SHA, "preview")).size == (1440, 720)


def test_a_tiny_source_is_enlarged_to_grid_size(tmp_path):
    speck = Image.new("RGB", (64, 32), (10, 200, 30))
    thumbs.put(tmp_path, SHA, speck)
    assert Image.open(thumbs.path_for(tmp_path, SHA)).size == (512, 256)


def test_an_unknown_variant_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a variant"):
        thumbs.path_for(tmp_path, SHA, "poster")


def test_a_cache_hit_never_rerenders(tmp_path):
    thumbs.put(tmp_path, SHA, Image.new("RGB", (100, 100), (255, 0, 0)))
    thumbs.put(tmp_path, SHA, Image.new("RGB", (100, 100), (0, 0, 255)))
    kept = Image.open(thumbs.path_for(tmp_path, SHA))
    r, _, b = _rgb(kept, (128, 128))
    assert r > b, "the second render overwrote a cache that was already warm"


def test_an_avatar_is_a_square_crop_centred_on_the_face(tmp_path):
    canvas = Image.new("RGB", (800, 600), (0, 0, 255))
    canvas.paste(Image.new("RGB", (200, 150), (255, 0, 0)), (200, 150))
    thumbs.put_avatar(tmp_path, 7, canvas, (0.25, 0.25, 0.25, 0.25))
    avatar = Image.open(thumbs.avatar_path(tmp_path, 7))
    assert avatar.size == (thumbs.AVATAR, thumbs.AVATAR)
    centre = _rgb(avatar, (128, 128))
    corner = _rgb(avatar, (6, 6))
    assert centre[0] > centre[2], "the face is not in the middle of its own avatar"
    assert corner[2] > corner[0], "the crop kept no context around the face"


def _library_with(tmp_path, write_media):
    """One real file on disk, scanned into a real database."""
    root = tmp_path / "lib"
    root.mkdir()
    media_path = write_media(root)
    conn = connect.connect(tmp_path / "gallery.db")
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    file_id, sha = conn.execute("SELECT id, content_sha256 FROM file").fetchone()
    return conn, file_id, sha, media_path


def _always_one_face(img):
    return [
        FaceDetection(
            bbox=(0.3, 0.3, 0.4, 0.4),
            landmarks=[],
            det_score=0.9,
            embedding=np.ones(16, dtype=np.float32),
        )
    ]


def _face_when_green(img):
    """A stub whose 'person' appears exactly on green frames."""
    pixels = np.asarray(img.convert("RGB"), dtype=np.int64)
    if pixels[..., 1].mean() > pixels[..., 2].mean():
        return _always_one_face(img)
    return []


def _clip(path, colors, *, rate=5):
    """A real H.264 clip, one color per frame, through the same libraries
    the reader uses."""
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=rate)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for color in colors:
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), color, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_a_scanned_but_not_yet_ingested_photo_is_still_upright(tmp_path):
    """Between scan and first ingest there is no capture row, and unknown
    is not upright: the oriented door must read the file's own EXIF then,
    not assume 1. Found by a real browser showing a sideways thumbnail."""
    from db import oriented

    def write(root):
        target = root / "phone.jpg"
        tag = Image.Exif()
        tag[274] = 6  # stored on its side; upright is 400x600
        Image.new("RGB", (600, 400), (30, 30, 200)).save(target, exif=tag)
        return target

    conn, file_id, _, media_path = _library_with(tmp_path, write)
    picture = oriented.for_model(conn, file_id, media_path)
    assert picture.size == (400, 600), "the scan-to-ingest window served the sensor's frame"
    conn.close()


def test_detection_caches_every_variant_as_a_byproduct(tmp_path):
    def write(root):
        target = root / "portrait.png"
        Image.new("RGB", (900, 400), (200, 30, 30)).save(target)
        return target

    conn, file_id, sha, media_path = _library_with(tmp_path, write)
    cache = tmp_path / "thumbs"
    detect.harvest(conn, StubFaceBackend(_always_one_face), file_id, media_path, 0.0, thumbs_dir=str(cache))
    assert Image.open(thumbs.path_for(cache, sha)).size == (512, 228)
    assert Image.open(thumbs.path_for(cache, sha, "preview")).size == (1440, 640)
    conn.close()


def test_a_video_is_represented_by_the_frame_with_its_people(tmp_path):
    """First half establishing shot, second half a person: the cadence sees
    both, and the thumbnail must be the person, not the opening frame."""

    def write(root):
        target = root / "clip.mp4"
        _clip(target, [(0, 0, 255)] * 15 + [(0, 255, 0)] * 15)  # 3s blue, 3s green at 5fps
        return target

    conn, file_id, sha, media_path = _library_with(tmp_path, write)
    cache = tmp_path / "thumbs"
    told = detect.harvest_video(
        conn, StubFaceBackend(_face_when_green), file_id, media_path, 0.0, thumbs_dir=str(cache)
    )
    assert told["faces"] > 0
    poster = Image.open(thumbs.path_for(cache, sha))
    centre = _rgb(poster, (poster.width // 2, poster.height // 2))
    assert centre[1] > centre[2], "the poster frame shows the set, not the person"
    conn.close()


def test_a_face_free_cadence_is_refined_until_the_face_is_found(tmp_path):
    """The person is on screen for one second the 2s cadence never lands
    on. Refinement bisects the gaps and finds them; the moments it looked
    at stay recorded, cadence and bisect alike."""

    def write(root):
        target = root / "clip.mp4"
        # 6s at 5fps; green only inside 2.8s..3.8s -- between cadence points.
        colors = [(0, 255, 0) if 14 <= n <= 19 else (0, 0, 255) for n in range(30)]
        _clip(target, colors)
        return target

    conn, file_id, sha, media_path = _library_with(tmp_path, write)
    cache = tmp_path / "thumbs"
    told = detect.harvest_video(
        conn, StubFaceBackend(_face_when_green), file_id, media_path, 0.0, thumbs_dir=str(cache)
    )
    assert told["faces"] > 0, "the cadence alone was allowed to conclude absence"
    policies = {policy for _, _, policy in sample.taken(conn, file_id)}
    assert "bisect" in policies, "the extra moments are not recorded as refinement"
    poster = Image.open(thumbs.path_for(cache, sha))
    centre = _rgb(poster, (poster.width // 2, poster.height // 2))
    assert centre[1] > centre[2]
    conn.close()


def test_refinement_is_bounded_and_converges(tmp_path):
    """A genuinely face-free video stops being probed: the bisect rows stay
    within the budget, and running the job again adds nothing."""

    def write(root):
        target = root / "clip.mp4"
        _clip(target, [(0, 0, 255)] * 30)
        return target

    conn, file_id, _, media_path = _library_with(tmp_path, write)
    backend = StubFaceBackend(_face_when_green)
    told = detect.harvest_video(conn, backend, file_id, media_path, 0.0)
    assert told["faces"] == 0
    first = sample.taken(conn, file_id)
    assert sum(1 for _, _, policy in first if policy == "bisect") <= detect.REFINE_MOST

    again = detect.harvest_video(conn, backend, file_id, media_path, 0.0)
    assert again["faces"] == 0
    assert len(sample.taken(conn, file_id)) == len(first), "a re-run kept deepening a settled answer"
    conn.close()
