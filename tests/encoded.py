"""One real file per container the application claims and nothing writes.

`db/scan.py KIND_BY_SUFFIX` declares 99 suffixes. Real cameras and real
generators cover the ones people actually own; the rest are containers no
source in this corpus emits -- `.jfif`, `.m2ts`, `.weba`, `.mpo` -- and a
suffix the readers claim and never meet is a claim nothing tests.

These are ENCODED, and labelled encoded. Pillow and FFmpeg write them, which
makes them authoritative for the CONTAINER and for nothing else: a `.3gp`
here proves the reader opens a real 3GP, not that a phone writes one this
way. Where a real specimen exists it wins; these fill the rest.

EVERY PARAMETER BELOW WAS MEASURED, not chosen from memory, and the
measurement mattered:

- `PIL.features.check("tiff")` returns False. Pillow writes TIFF anyway --
  that call asks whether an OPTIONAL feature was compiled in, not whether
  the codec exists, and believing it would have marked twelve formats
  impossible.
- `mpeg1video` refuses rate 10 and accepts 25.
- `mjpeg` refuses `yuv420p` and accepts `yuvj420p`.
- `theora` and `libvorbis` are not in this FFmpeg build; the Ogg containers
  take VP8 and Opus instead.
- `wmav2` refuses a stream with no `bit_rate` set.

Six of these read as "this format is impossible" until the parameter was
corrected. A capability probe that stops at the first error under-reports
what the tree can do.
"""

from __future__ import annotations

import fractions
import hashlib
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
IMAGES = CORPUS / "encoded"
LOCKFILE = REPO / "tests" / "encoded.lock.json"

#: What one encoder attempt may fail with: `ValueError` for an unusable parameter,
#: a `LookupError` subclass for an absent codec, `OSError` for a bad muxer, and
#: `TypeError` for the stream-kind assertion. Anything else propagates as a defect.
ENCODER_FAILURES = (OSError, ValueError, RuntimeError, LookupError, TypeError)

#: Every frame written here, named once. These were typed at each call site --
#: `96, 64` in four places and `(64, 96, 3)` transposed in a fifth -- and a
#: corpus that disagrees about its own frame size is one nobody can re-derive.
WIDTH, HEIGHT = 96, 64

#: Frames per encoded clip. Enough that a container holds more than one
#: keyframe and a seek has somewhere to land; not so many that 21 clips
#: cost anything.
FRAMES = 6

#: Frames per animated still. Three is the smallest number that is
#: unambiguously an animation rather than a still with a duration.
ANIMATION_FRAMES = 3

#: `mpeg1video` refuses a rate it does not recognise, and 10 is one of
#: them. 25 is PAL and every codec here accepts it.
FRAME_RATE = 25

#: VP9 takes any rate, and a slower one makes a smaller file for the
#: same number of frames. Named rather than left as a bare 10 beside
#: twenty appearances of 25, which reads as an oversight.
VP9_FRAME_RATE = 10

#: CD sample rate, which every codec here takes. Opus is specified at
#: 48 kHz and resamples anything else, so it is given its own.
SAMPLE_RATE = 44_100
OPUS_SAMPLE_RATE = 48_000

#: `wmav2` refuses a stream with no bit rate at all. 128 kbps is the rate
#: the format was ubiquitous at.
WMA_BIT_RATE = 128_000

#: One second of tone. The point is a decodable stream, not a duration.
TONE_HZ = 440.0
TONE_AMPLITUDE = 16_000

#: `(suffix, Pillow format)`. Aliases share a format on purpose: what is
#: under test is the suffix table's routing, and the bytes behind `.jfif`
#: really are a JPEG.
STILLS: tuple[tuple[str, str], ...] = (
    (".tiff", "TIFF"),
    (".dib", "BMP"),
    (".jfif", "JPEG"),
    (".jif", "JPEG"),
    (".jpe", "JPEG"),
    (".j2k", "JPEG2000"),
    (".jpf", "JPEG2000"),
    (".jpx", "JPEG2000"),
    (".avif", "AVIF"),
    (".heif", "HEIF"),
    (".heics", "HEIF"),
    (".heifs", "HEIF"),
    (".hif", "HEIF"),
    # A REAL one, wanted alongside the truncated specimen rather than instead of
    # it. The corpus's only `.jxl` was an ExifTool specimen of 22 bytes, so the
    # suffix sat at PARTIAL proving refusal and not that a JPEG XL can be opened.
    (".jxl", "JXL"),
)

#: `(suffix, container, codec, rate, pixel format)`.
CLIPS: tuple[tuple[str, str, str, int, str], ...] = (
    (".webm", "webm", "libvpx-vp9", VP9_FRAME_RATE, "yuv420p"),
    (".mpeg", "mpeg", "mpeg1video", FRAME_RATE, "yuv420p"),
    (".mpg", "mpeg", "mpeg1video", FRAME_RATE, "yuv420p"),
    (".mpe", "mpeg", "mpeg1video", FRAME_RATE, "yuv420p"),
    (".m2v", "mpeg2video", "mpeg2video", FRAME_RATE, "yuv420p"),
    (".ts", "mpegts", "mpeg2video", FRAME_RATE, "yuv420p"),
    (".m2t", "mpegts", "mpeg2video", FRAME_RATE, "yuv420p"),
    (".m2ts", "mpegts", "mpeg2video", FRAME_RATE, "yuv420p"),
    (".vob", "mpeg", "mpeg2video", FRAME_RATE, "yuv420p"),
    (".3gp", "3gp", "mpeg4", FRAME_RATE, "yuv420p"),
    (".3gpp", "3gp", "mpeg4", FRAME_RATE, "yuv420p"),
    (".m4v", "mp4", "mpeg4", FRAME_RATE, "yuv420p"),
    (".asf", "asf", "msmpeg4v3", FRAME_RATE, "yuv420p"),
    (".f4v", "flv", "flv", FRAME_RATE, "yuv420p"),
    (".mjpeg", "mjpeg", "mjpeg", FRAME_RATE, "yuvj420p"),
    (".mjpg", "mjpeg", "mjpeg", FRAME_RATE, "yuvj420p"),
    (".ogv", "ogv", "libvpx", FRAME_RATE, "yuv420p"),
    (".ogm", "ogg", "libvpx", FRAME_RATE, "yuv420p"),
    (".qt", "mov", "mpeg4", FRAME_RATE, "yuv420p"),
    (".divx", "avi", "mpeg4", FRAME_RATE, "yuv420p"),
    (".rmvb", "rm", "rv10", FRAME_RATE, "yuv420p"),
    # REAL ones, for the same reason as `.jxl` above. The corpus's only `.mkv`,
    # `.rm` and `.wmv` were ExifTool specimens truncated to their metadata, so
    # all three sat at PARTIAL and no Matroska, RealMedia or ASF ever opened.
    (".mkv", "matroska", "mpeg4", FRAME_RATE, "yuv420p"),
    (".rm", "rm", "rv10", FRAME_RATE, "yuv420p"),
    (".wmv", "asf", "msmpeg4v3", FRAME_RATE, "yuv420p"),
)

#: `(suffix, container, codec, sample rate, bit rate or None)`.
SOUNDS: tuple[tuple[str, str, str, int, int | None], ...] = (
    (".aiff", "aiff", "pcm_s16be", SAMPLE_RATE, None),
    (".au", "au", "pcm_s16be", SAMPLE_RATE, None),
    (".caf", "caf", "pcm_s16le", SAMPLE_RATE, None),
    (".mka", "matroska", "libmp3lame", SAMPLE_RATE, None),
    (".mp2", "mp2", "mp2", SAMPLE_RATE, None),
    (".oga", "ogg", "libopus", OPUS_SAMPLE_RATE, None),
    (".weba", "webm", "libopus", OPUS_SAMPLE_RATE, None),
    (".wma", "asf", "wmav2", SAMPLE_RATE, WMA_BIT_RATE),
)


def _still(path: pathlib.Path, fmt: str) -> None:
    from PIL import Image

    from vision import decode

    decode.ensure_decoders()
    # Not flat colour: a gradient survives a lossy codec as something a
    # perceptual hash can still tell from its neighbours. A solid block would
    # make every encoded file a duplicate of every other.
    picture = Image.new("RGB", (WIDTH, HEIGHT))
    picture.putdata([(x * 2 % 256, y * 4 % 256, (x + y) % 256) for y in range(HEIGHT) for x in range(WIDTH)])
    picture.save(path, format=fmt)


def _animated(path: pathlib.Path) -> None:
    """An APNG: a PNG with more than one frame, which is its own kind."""
    from PIL import Image

    frames = [Image.new("RGB", (WIDTH, HEIGHT), (30 + i * 60, 90, 160)) for i in range(ANIMATION_FRAMES)]
    frames[0].save(path, format="PNG", save_all=True, append_images=frames[1:], duration=120)


def _multi(path: pathlib.Path) -> None:
    """An MPO: two JPEGs in one file, as a stereo camera writes."""
    from PIL import Image

    left = Image.new("RGB", (WIDTH, HEIGHT), (200, 80, 40))
    right = Image.new("RGB", (WIDTH, HEIGHT), (40, 80, 200))
    left.save(path, format="MPO", save_all=True, append_images=[right])


def _clip(path: pathlib.Path, container: str, codec: str, rate: int, pix: str) -> None:
    import av
    import numpy as np

    held = av.open(str(path), "w", format=container)
    try:
        stream = held.add_stream(codec, rate=fractions.Fraction(rate, 1))
        # `add_stream` is typed as returning any stream kind, so the checker
        # cannot know a video codec yields a VideoStream. Asserted rather than
        # assumed, so a codec name that routes elsewhere says so here.
        if not isinstance(stream, av.VideoStream):
            raise TypeError(f"{codec} did not open a video stream, got {type(stream).__name__}")
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, pix
        for i in range(FRAMES):
            shade = np.full((HEIGHT, WIDTH, 3), 20 + i * 35, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(shade, format="rgb24")):
                held.mux(packet)
        for packet in stream.encode():
            held.mux(packet)
    finally:
        held.close()


def _sound(path: pathlib.Path, container: str, codec: str, rate: int, bitrate: int | None) -> None:
    import av
    import numpy as np

    held = av.open(str(path), "w", format=container)
    try:
        stream = held.add_stream(codec, rate=rate)
        if not isinstance(stream, av.AudioStream):
            raise TypeError(f"{codec} did not open an audio stream, got {type(stream).__name__}")
        if bitrate is not None:
            stream.bit_rate = bitrate
        moment = np.arange(rate) / rate
        tone = (np.sin(2 * np.pi * TONE_HZ * moment) * TONE_AMPLITUDE).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="s16p", layout="stereo")
        frame.rate = rate
        for packet in stream.encode(frame):
            held.mux(packet)
        for packet in stream.encode():
            held.mux(packet)
    finally:
        held.close()


def write(into: pathlib.Path | None = None) -> dict:
    """Write one file per listed suffix. Every failure is recorded."""
    where = into or IMAGES
    where.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    trouble: list[dict] = []

    def attempt(path: pathlib.Path, kind: str, how: str, run) -> None:
        try:
            run()
            raw = path.read_bytes()
            rows.append(
                {
                    "path": f"{where.name}/{path.name}",
                    "suffix": path.suffix.lower(),
                    "kind": kind,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "written_by": how,
                }
            )
        except ENCODER_FAILURES as why:
            trouble.append({"path": path.name, "why": f"{type(why).__name__}: {why}"[:160]})

    for suffix, fmt in STILLS:
        target = where / f"encoded{suffix}"
        attempt(target, "image", f"Pillow {fmt}", lambda t=target, f=fmt: _still(t, f))

    apng = where / "encoded.apng"
    attempt(apng, "animated_image", "Pillow PNG, 3 frames", lambda: _animated(apng))
    mpo = where / "encoded.mpo"
    attempt(mpo, "image", "Pillow MPO, 2 images", lambda: _multi(mpo))

    for suffix, container, codec, rate, pix in CLIPS:
        target = where / f"encoded{suffix}"
        attempt(
            target,
            "video",
            f"FFmpeg {container}/{codec}",
            lambda t=target, c=container, k=codec, r=rate, p=pix: _clip(t, c, k, r, p),
        )

    for suffix, container, codec, rate, bitrate in SOUNDS:
        target = where / f"encoded{suffix}"
        attempt(
            target,
            "audio",
            f"FFmpeg {container}/{codec}",
            lambda t=target, c=container, k=codec, r=rate, b=bitrate: _sound(t, c, k, r, b),
        )

    held = {
        "what": "One file per container the application claims and no real source in this corpus writes.",
        "written_by": "Pillow and FFmpeg, through PyAV",
        "not_evidence_of": "how a camera or a phone writes this container",
        "files": rows,
        "trouble": trouble,
    }
    # newline="" or Windows writes CRLF into a file the repo stores as LF,
    # dirtying a tracked lockfile with zero content delta and reddening the
    # commit gate for whoever is holding a candidate.
    with LOCKFILE.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(held, indent=2) + "\n")
    return held


if __name__ == "__main__":
    import collections

    got = write()
    by = collections.Counter(one["kind"] for one in got["files"])
    print(f"  {len(got['files'])} files into {IMAGES}  {dict(by)}")
    print(f"  {sum(one['bytes'] for one in got['files']) / 1e6:.1f} MB")
    for one in got["trouble"]:
        print(f"  TROUBLE {one['path']}: {one['why']}")
