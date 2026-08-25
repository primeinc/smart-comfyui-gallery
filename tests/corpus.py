"""A library shaped like a real one, written to disk.

Twenty `write_library` functions in this suite each hand-roll their own
media, `_plain` is copy-pasted into three of them, and not one of them
spans more than a fortnight. So every surface that has to survive a
LIBRARY -- the timeline above all -- was only ever exercised against
eight files in one afternoon, and the shapes that actually break it went
untested until somebody looked at their own pictures and said so.

What a real library does that a fixture never did:

**It spans decades, unevenly.** A scan from 2004, nothing at all until
2009, a camera era, a phone era, and a wall of generated images this
year. The gaps are the point -- three years, six months, three weeks --
because a surface that draws elapsed time spends itself on them.

**It mixes what it knows.** Some files carry EXIF to the subsecond and a
body serial; some carry a folder name and nothing else; some carry an
A1111 infotext, a ComfyUI graph or a SwarmUI manifest; some carry
nothing at all and are dated by the filesystem, which is to say barely
dated. A corpus of one dialect proves one adapter.

**It disagrees with itself.** A file whose EXIF says 2019 and whose name
says 2021 is not a corner case, it is most of anybody's downloads
folder. So are exact duplicates in two places, and near-duplicates one
seed apart.

Deterministic: seeded, no clock, no `random` without one. The same
arguments write the same bytes, so a fixture built from this is a
fixture that can be compared against itself.

    python -m tests.corpus <dir> [--scale wide] [--real <dir>]

`just corpus` is the recipe; `spread()` is the import.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import random
import shutil
import sys

from PIL import ExifTags, Image
from PIL.PngImagePlugin import PngInfo
from PIL.TiffImagePlugin import IFDRational

from vision import decode

UTC = datetime.UTC


def _at(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> float:
    return datetime.datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()


@dataclasses.dataclass(frozen=True)
class Era:
    """One stretch of a life with one way of recording it.

    `holds` is how many files a `wide` corpus puts here; `small` takes a
    fraction. The eras deliberately do not touch: what is between them
    is the gap, and the gaps are what most of this exists to produce.
    """

    name: str
    start: tuple[int, int, int]
    end: tuple[int, int, int]
    holds: int
    #: how the file says when it happened
    dating: str  # "exif" | "folder" | "none" | "conflicting"
    #: what made it
    maker: str  # "scanner" | "compact" | "dslr" | "phone" | "a1111" | "comfy" | "swarm"
    #: Which kinds appear, and how often -- REPEATS are the weight. A
    #: library is overwhelmingly stills, and an even cycle would make a
    #: third of it video, which is both a lie and slow to write.
    kinds: tuple[str, ...] = ("image",)


#: The shape of a life, and the holes in it. Chosen so every zoom has
#: something to collapse: two decades between the first two eras, three
#: years inside the camera one, and days inside a burst.
ERAS: tuple[Era, ...] = (
    Era("scanned", (2002, 6, 1), (2004, 9, 30), 6, "folder", "scanner"),
    # -- a five-year hole: the years nobody photographed anything --
    Era("compact", (2009, 4, 1), (2011, 10, 31), 40, "exif", "compact"),
    # -- a three-year hole --
    Era("dslr", (2014, 5, 1), (2018, 9, 30), 120, "exif", "dslr", ("image",) * 11 + ("video",)),
    # -- a fourteen-month hole --
    Era("phone", (2021, 1, 1), (2023, 11, 30), 260, "exif", "phone", ("image",) * 12 + ("video", "animated_image")),
    Era("undated", (2023, 12, 1), (2023, 12, 31), 14, "none", "phone", ("image", "image", "document")),
    Era("muddled", (2024, 2, 1), (2024, 6, 30), 24, "conflicting", "phone"),
    Era("a1111", (2025, 3, 1), (2025, 9, 30), 150, "none", "a1111"),
    Era("comfy", (2026, 1, 5), (2026, 8, 20), 220, "none", "comfy", ("image",) * 15 + ("video",)),
    Era("swarm", (2026, 6, 1), (2026, 8, 24), 60, "none", "swarm"),
)

CHECKPOINTS = ("dreamshaper_8", "juggernautXL_v9", "sd_xl_base_1.0", "flux1-dev", "realvisxlV40")
LORAS = ("filmGrain", "detailTweaker", "add_detail", "epiNoiseoffset", "polyhedron_skin")
SAMPLERS = ("Euler a", "DPM++ 2M Karras", "DPM++ SDE", "UniPC", "Heun")
SUBJECTS = (
    "a brass diving helmet at dusk",
    "a lighthouse in fog",
    "an orange tabby asleep on a radiator",
    "a rain-slick street under sodium lamps",
    "a greenhouse full of ferns",
    "a paper boat on still water",
)
CAMERAS = (
    ("Canon", "Canon EOS 5D Mark III", "EF24-105mm f/4L IS USM", "182029002226"),
    ("NIKON CORPORATION", "NIKON D750", "24.0-120.0 mm f/4.0", "6041234"),
    ("Apple", "iPhone 13 Pro", "iPhone 13 Pro back camera 5.7mm f/1.5", None),
    ("FUJIFILM", "X-T4", "XF16-55mmF2.8 R LM WR", "1AB23456"),
)
PLACES = (
    ("Reykjavik", 64.1466, -21.9426),
    ("Kyoto", 35.0116, 135.7681),
    ("Iowa City", 41.6611, -91.5302),
    ("Lisbon", 38.7223, -9.1393),
)


def _paint(size: tuple[int, int], seed: int) -> Image.Image:
    """A picture whose bytes depend on its seed, so two files are alike
    only when they are meant to be. Flat colour would make every file a
    duplicate of every other and drown the duplicate surfaces."""
    rng = random.Random(seed)  # noqa: S311 -- shape, not secrecy: the seed IS the contract
    base = Image.new("RGB", size, (rng.randrange(30, 90), rng.randrange(40, 120), rng.randrange(60, 160)))
    pixels = base.load()
    assert pixels is not None
    for _ in range(60):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        shade = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for dx in range(min(6, size[0] - x)):
            for dy in range(min(6, size[1] - y)):
                pixels[x + dx, y + dy] = shade
    return base


def _stamp(path: pathlib.Path, when: float) -> None:
    os.utime(path, (when, when))


def _exif_for(when: float, camera: tuple, *, place: tuple | None, seed: int) -> Image.Exif:
    made, model, lens, serial = camera
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = made
    exif[ExifTags.Base.Model] = model
    photo = exif.get_ifd(ExifTags.IFD.Exif)
    # LOCAL wall clock, with no zone, which is what EXIF stores and what
    # a camera writes. Spelling it in UTC on a machine that is not UTC
    # makes the file disagree with its own mtime by the offset -- and
    # every photograph then arrives CONTESTED, which buries the ones that
    # are contested on purpose. Measured: 146 of 146, before this line.
    local = datetime.datetime.fromtimestamp(when, UTC).astimezone()
    photo[ExifTags.Base.DateTimeOriginal] = local.strftime("%Y:%m:%d %H:%M:%S")
    photo[ExifTags.Base.SubsecTimeOriginal] = f"{seed % 100:02d}"
    photo[ExifTags.Base.LensModel] = lens
    photo[ExifTags.Base.ISOSpeedRatings] = (100, 200, 400, 800, 1600, 3200)[seed % 6]
    photo[ExifTags.Base.FNumber] = (1.8, 2.8, 4.0, 5.6, 8.0)[seed % 5]
    photo[ExifTags.Base.ExposureTime] = (1 / 30, 1 / 60, 1 / 125, 1 / 500)[seed % 4]
    photo[ExifTags.Base.FocalLength] = (24.0, 35.0, 50.0, 85.0, 105.0)[seed % 5]
    if serial:
        photo[ExifTags.Base.BodySerialNumber] = serial
    if place:
        _, lat, lon = place
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps[ExifTags.GPS.GPSLatitudeRef] = "N" if lat >= 0 else "S"
        gps[ExifTags.GPS.GPSLatitude] = _degrees(abs(lat))
        gps[ExifTags.GPS.GPSLongitudeRef] = "E" if lon >= 0 else "W"
        gps[ExifTags.GPS.GPSLongitude] = _degrees(abs(lon))
    return exif


def _degrees(value: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    """Degrees, minutes and seconds as EXIF stores them -- RATIONALS.

    A bare tuple of integer pairs looks like the same thing and is not:
    Pillow calls `abs()` on each element and dies on a tuple. The one
    other place in this tree that writes GPS spells it this way
    (tests/test_camera_metadata_lands_in_the_schema.py GPS_LONDON).
    """
    d = int(value)
    m = int((value - d) * 60)
    s = int((((value - d) * 60) - m) * 60 * 100)
    return (IFDRational(d, 1), IFDRational(m, 1), IFDRational(s, 100))


def _infotext(seed: int, *, rng: random.Random) -> str:
    """An A1111 / Forge infotext block, in the grammar the reader knows
    (metaparse/adapters.py `parse_infotext`)."""
    subject = SUBJECTS[seed % len(SUBJECTS)]
    lora = LORAS[seed % len(LORAS)]
    return (
        f"{subject} <lora:{lora}:{rng.choice((0.3, 0.45, 0.6, 0.8))}>\n"
        "Negative prompt: blurry, watermark, text\n"
        f"Steps: {rng.choice((20, 24, 28, 32))}, Sampler: {rng.choice(SAMPLERS)}, "
        f"CFG scale: {rng.choice((4.5, 6.0, 7.0, 8.5))}, Seed: {seed}, "
        f"Size: {rng.choice(('832x1216', '1024x1024', '1216x832'))}, "
        f"Model: {CHECKPOINTS[seed % len(CHECKPOINTS)]}, "
        f"Model hash: {seed:08x}, Version: v1.10.1"
    )


def _comfy_graph(seed: int, *, rng: random.Random) -> dict:
    """A ComfyUI API-format graph -- the `prompt` chunk, which is the one
    a node graph is actually recovered from."""
    subject = SUBJECTS[seed % len(SUBJECTS)]
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": rng.choice((20, 25, 30)),
                "cfg": rng.choice((5.0, 7.0, 8.0)),
                "sampler_name": rng.choice(("euler", "dpmpp_2m", "uni_pc")),
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINTS[seed % len(CHECKPOINTS)]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": subject, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, watermark", "clip": ["4", 1]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "SG", "images": ["8", 0]}},
    }


def _swarm(seed: int, *, rng: random.Random) -> dict:
    """SwarmUI's `sui_image_params` manifest, which names its LoRAs as
    records rather than as prompt substrings."""
    return {
        "sui_image_params": {
            "prompt": SUBJECTS[seed % len(SUBJECTS)],
            "negativeprompt": "blurry",
            "model": CHECKPOINTS[seed % len(CHECKPOINTS)],
            "seed": seed,
            "steps": rng.choice((20, 30)),
            "cfgscale": rng.choice((5.0, 7.5)),
            "loras": [LORAS[seed % len(LORAS)]],
            "loraweights": ["0.55"],
        }
    }


def _write(path: pathlib.Path, era: Era, when: float, seed: int, rng: random.Random, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (320, 240) if kind != "document" else (200, 260)

    if kind == "video":
        # A container the prober can open, written by PyAV rather than
        # faked: a file that only LOOKS like a video is a file the video
        # arm never really sees.
        _video(path, when, seed)
        _stamp(path, when)
        return
    if kind == "animated_image":
        frames = [_paint((160, 120), seed + n) for n in range(4)]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
        _stamp(path, when)
        return
    if kind == "document":
        _paint(size, seed).save(path, "PDF", resolution=72)
        _stamp(path, when)
        return

    image = _paint(size, seed)
    if era.maker in ("a1111", "comfy", "swarm"):
        info = PngInfo()
        if era.maker == "a1111":
            info.add_text("parameters", _infotext(seed, rng=rng))
        elif era.maker == "comfy":
            info.add_text("prompt", json.dumps(_comfy_graph(seed, rng=rng)))
            info.add_text("workflow", json.dumps({"nodes": [], "links": [], "version": 0.4}))
        else:
            info.add_text("parameters", json.dumps(_swarm(seed, rng=rng)))
        image.save(path, pnginfo=info)
    elif era.dating in ("exif", "conflicting"):
        camera = CAMERAS[seed % len(CAMERAS)]
        place = PLACES[seed % len(PLACES)] if seed % 3 == 0 else None
        # A muddled file's EXIF deliberately disagrees with the name and
        # the mtime around it -- which is most of anybody's downloads.
        said = when - 400 * 86_400 if era.dating == "conflicting" else when
        image.save(path, exif=_exif_for(said, camera, place=place, seed=seed))
    else:
        image.save(path)
    _stamp(path, when)


def _video(path: pathlib.Path, when: float, seed: int) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height = 160, 120
        stream.pix_fmt = "yuv420p"
        for n in range(8):
            frame = av.VideoFrame.from_image(_paint((160, 120), seed + n))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _moments(era: Era, holds: int, rng: random.Random) -> list[float]:
    """When an era's files happened: in BURSTS, not evenly.

    A life is days you took forty pictures and months you took none, and
    an even spread would hide every grouping surface this library has --
    sessions, events, day sheets and the collapse the timeline now does.
    """
    lo, hi = _at(*era.start), _at(*era.end)
    out: list[float] = []
    while len(out) < holds:
        day = lo + rng.random() * max(1.0, hi - lo)
        burst = rng.choice((1, 1, 2, 3, 8, 20))
        start = day - (day % 86_400) + rng.randrange(7, 20) * 3_600
        out.extend(start + n * rng.randrange(20, 400) for n in range(min(burst, holds - len(out))))
    return sorted(out)


def _folder_for(era: Era, when: float) -> str:
    d = datetime.datetime.fromtimestamp(when, UTC)
    if era.dating == "folder":
        return f"scans/{d:%Y-%m-%d}"
    if era.maker in ("a1111", "comfy", "swarm"):
        return f"generated/{era.maker}/{d:%Y-%m}"
    if era.name == "undated":
        return "downloads"
    if era.dating == "conflicting":
        # Where they belong, and where the docstring already said they
        # come from: a file whose EXIF and whose name disagree is most of
        # anybody's downloads folder. It also keeps them out of
        # `photos/`, so "these agree with themselves" is a claim about a
        # directory rather than about a naming convention.
        return f"downloads/{d:%Y-%m}"
    return f"photos/{d:%Y}/{d:%Y-%m-%d}"


def _name_for(era: Era, when: float, seed: int, kind: str) -> str:
    d = datetime.datetime.fromtimestamp(when, UTC)
    suffix = {"image": ".png", "video": ".mp4", "animated_image": ".gif", "document": ".pdf"}[kind]
    if era.maker in ("a1111", "comfy", "swarm"):
        return f"{era.maker}_{seed:08d}{suffix}"
    if era.dating == "conflicting":
        # the NAME claims a date the EXIF contradicts
        return f"IMG_{d:%Y%m%d}_{d:%H%M%S}_{seed % 1000:03d}{suffix}"
    if era.dating == "none":
        return f"download_{seed:06d}{suffix}"
    return f"{'DSC' if era.maker == 'dslr' else 'IMG'}_{seed % 10000:04d}{suffix}"


SCALES = {"small": 0.08, "medium": 0.35, "wide": 1.0}


def spread(
    root: pathlib.Path,
    *,
    seed: int = 20260825,
    scale: str = "wide",
    real: pathlib.Path | None = None,
    eras: tuple[Era, ...] = ERAS,
) -> dict:
    """Write the corpus. Returns what it wrote, by era and by kind.

    Deterministic in `seed`: the same arguments write the same bytes, so
    a fixture built from this can be compared against itself.
    """
    if scale not in SCALES:
        raise ValueError(f"scale is one of {', '.join(SCALES)}, not {scale!r}")
    share = SCALES[scale]
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)  # noqa: S311 -- shape, not secrecy: the seed IS the contract
    told: dict = {"eras": {}, "kinds": {}, "files": 0, "root": str(root)}
    n = 0

    for era in eras:
        holds = max(1, round(era.holds * share))
        for when in _moments(era, holds, rng):
            n += 1
            kind = era.kinds[n % len(era.kinds)]
            path = root / _folder_for(era, when) / _name_for(era, when, n, kind)
            _write(path, era, when, seed + n, rng, kind)
            told["eras"][era.name] = told["eras"].get(era.name, 0) + 1
            told["kinds"][kind] = told["kinds"].get(kind, 0) + 1
            told["files"] += 1

    told["duplicates"] = _duplicates(root)
    told["real"] = _real(root, real, rng) if real else 0
    return told


def _duplicates(root: pathlib.Path) -> int:
    """The same photograph filed twice, and one re-encode of it.

    Not decoration: an exact copy and a near copy are different things
    to the duplicate surfaces -- one can become a single stored payload
    and the other cannot -- and a corpus with neither proves neither.
    """
    originals = sorted(root.glob("photos/**/*.png"))[:6]
    made = 0
    for one in originals:
        twin = root / "backup" / one.parent.name / one.name
        twin.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(one, twin)
        made += 1
        # and a re-encode: alike to a perceptual hash, different bytes
        alike = root / "backup" / one.parent.name / f"{one.stem}_web.jpg"
        with decode.open_still(one) as image:
            image.convert("RGB").save(alike, quality=72)
        os.utime(alike, (one.stat().st_mtime, one.stat().st_mtime))
        made += 1
    return made


def _real(root: pathlib.Path, source: pathlib.Path, rng: random.Random) -> int:
    """Mix real media in, re-dated across the range.

    The synthetic files are 320x240 flat-ish paint: they exercise every
    seam and lie about every cost. One real 24-megapixel raw tells the
    thumbnail, decode and embed paths something no amount of generated
    ones will.
    """
    if not source.is_dir():
        raise ValueError(f"no such directory to mix in: {source}")
    into = root / "real"
    into.mkdir(parents=True, exist_ok=True)
    made = 0
    for one in sorted(p for p in source.rglob("*") if p.is_file())[:60]:
        target = into / one.name
        shutil.copy2(one, target)
        # spread them over the whole range rather than leaving them in a
        # clump at whatever date they came with
        _stamp(target, _at(rng.randrange(2005, 2026), rng.randrange(1, 13), rng.randrange(1, 28)))
        made += 1
    return made


def main(argv: list[str] | None = None) -> int:
    parsed = argparse.ArgumentParser(prog="tests.corpus", description=__doc__)
    parsed.add_argument("root", type=pathlib.Path)
    parsed.add_argument("--scale", default="wide", choices=sorted(SCALES))
    parsed.add_argument("--seed", type=int, default=20260825)
    parsed.add_argument("--real", type=pathlib.Path, default=None)
    parsed.add_argument("--force", action="store_true", help="write into a directory that already holds files")
    args = parsed.parse_args(argv)

    if args.root.exists() and any(args.root.iterdir()) and not args.force:
        print(f"{args.root} is not empty; pass --force to write into it anyway", file=sys.stderr)
        return 2
    told = spread(args.root, seed=args.seed, scale=args.scale, real=args.real)
    print(f"{told['files']} files under {told['root']}")
    for name, held in told["eras"].items():
        print(f"  {name:10} {held:5}")
    print(f"  {'kinds':10} {told['kinds']}")
    print(f"  {'duplicates':10} {told['duplicates']}")
    if told["real"]:
        print(f"  {'real':10} {told['real']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
