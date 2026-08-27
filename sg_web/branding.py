"""The application's own face, drawn from one mark: PWA icons, the iOS
splash set, and the install-sheet screenshots.

`python -m sg_web.branding icons` writes every raster the manifest and
base.html reference into `static/pwa/` -- deterministic Pillow drawing,
no source images. `python -m sg_web.branding screenshots` boots the real
application over a generated sample library and photographs /g at a
phone and a desktop viewport, because a hand-made screenshot drifts from
the product immediately and the manifest's `sizes` must state the PNG's
real dimensions (w3c/manifest-app-info: a UA may use the aspect ratio
to decide whether to show one).

The mark is the favicon's drawing (static/favicon.svg): four tiles on
the gallery's ground, one lit in the accent. gallery.css owns the
palette; these constants restate it because CSS custom properties do
not reach Pillow -- change one, regenerate with `just pwa-assets`.

The maskable icon is its own drawing, not a re-export: every pixel a
platform mask may keep must sit inside a centred circle of radius 40%
of the icon's edge (w3c/manifest index.html:2226-2231), so its tile
block is shrunk until the block's half-diagonal fits that circle and
the ground bleeds square to the edge.
"""

from __future__ import annotations

import pathlib

GROUND = (0x14, 0x15, 0x1A, 255)
INK = (0xE8, 0xE6, 0xDF, 255)
ACCENT = (0xD8, 0xA2, 0x4A, 255)

STATIC = pathlib.Path(__file__).resolve().parent / "static"
PWA = STATIC / "pwa"

#: The iOS launch-screen matrix, one row per CSS media triple in
#: base.html: (device-width, device-height, device-pixel-ratio). The
#: PNG for a row is width*ratio x height*ratio pixels. A near-miss
#: triple is ignored in silence, so base.html's links and this table
#: are held together by tests/test_the_app_installs_like_an_app.py.
SPLASH: tuple[tuple[int, int, int], ...] = (
    (440, 956, 3),
    (430, 932, 3),
    (420, 912, 3),
    (402, 874, 3),
    (393, 852, 3),
    (390, 844, 3),
    (375, 812, 3),
    (414, 896, 3),
    (414, 896, 2),
    (375, 667, 2),
    (1024, 1366, 2),
)


def splash_name(width: int, height: int, ratio: int) -> str:
    return f"splash-{width}x{height}@{ratio}x.png"


#: Screenshot captures: manifest `form_factor` -> viewport. Both ship
#: because a UA shows only its own form factor's screenshots
#: (w3c/manifest-app-info index.html:471-474) -- a narrow-only set
#: leaves every desktop install sheet blank, and vice versa.
SHOTS: dict[str, tuple[int, int]] = {"narrow": (750, 1334), "wide": (1920, 1080)}


def _mark(size: int, *, tile_span: float, radius_frac: float | None, background: tuple):
    """The four-tile mark on its ground, supersampled then resized.

    `tile_span` is the tile block's half-extent on the 32-unit grid:
    the favicon's drawing uses 12 (tiles from 4 to 28); the maskable
    icon 9, because 9*sqrt(2) = 12.73 units of half-diagonal is what
    fits inside the 40% safe-zone radius (12.8 units). `radius_frac`
    rounds the ground's corners; None bleeds it square to the edge.
    """
    from PIL import Image, ImageDraw

    over = 8
    px = size * over
    unit = px / 32
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if radius_frac is None:
        draw.rectangle((0, 0, px, px), fill=background)
    else:
        draw.rounded_rectangle((0, 0, px, px), radius=int(px * radius_frac), fill=background)
    gap = 1.0
    tile = tile_span - gap / 2
    corners = {
        (16 - tile_span, 16 - tile_span): INK,
        (16 + gap / 2, 16 - tile_span): ACCENT,
        (16 - tile_span, 16 + gap / 2): (*INK[:3], 140),
        (16 + gap / 2, 16 + gap / 2): (*INK[:3], 204),
    }
    for (x, y), fill in corners.items():
        draw.rounded_rectangle(
            (x * unit, y * unit, (x + tile) * unit, (y + tile) * unit), radius=int(2 * unit), fill=fill
        )
    return img.resize((size, size), Image.LANCZOS)


def icons() -> list[pathlib.Path]:
    """Every raster the manifest and base.html name, into static/pwa."""
    from PIL import Image

    PWA.mkdir(exist_ok=True)
    written: list[pathlib.Path] = []

    def keep(name: str, image) -> None:
        target = PWA / name
        image.save(target)
        written.append(target)

    # Transparent rounded corners for the ordinary icons; a launcher
    # that wants its own shape uses the maskable one instead.
    keep("icon-192.png", _mark(192, tile_span=12, radius_frac=6 / 32, background=GROUND))
    keep("icon-512.png", _mark(512, tile_span=12, radius_frac=6 / 32, background=GROUND))
    # Full bleed, content inside the safe zone.
    keep("icon-maskable-512.png", _mark(512, tile_span=9, radius_frac=None, background=GROUND))
    # iOS composes its own corner mask and renders transparency black.
    keep("apple-touch-icon-180.png", _mark(180, tile_span=12, radius_frac=None, background=GROUND))

    for width, height, ratio in SPLASH:
        w, h = width * ratio, height * ratio
        screen = Image.new("RGBA", (w, h), GROUND)
        edge = w // 4
        mark = _mark(edge, tile_span=12, radius_frac=6 / 32, background=GROUND)
        screen.alpha_composite(mark, ((w - edge) // 2, (h - edge) // 2))
        keep(splash_name(width, height, ratio), screen.convert("RGB"))
    return written


def _sample_library(root: pathlib.Path, count: int = 24) -> None:
    """A believable grid: vertical two-tone gradients, staggered hours."""
    import os

    from PIL import Image, ImageDraw

    palette = [GROUND[:3], (0x2C, 0x2F, 0x38), (0xD8, 0xA2, 0x4A), (0x9A, 0x97, 0x8D), (0x6B, 0x8C, 0xAE)]
    stamped = 1_700_000_000
    for i in range(count):
        top = palette[i % len(palette)]
        bottom = palette[(i + 2) % len(palette)]
        img = Image.new("RGB", (640, 640))
        draw = ImageDraw.Draw(img)
        for y in range(640):
            mixed = tuple(int(a + (b - a) * y / 639) for a, b in zip(top, bottom, strict=True))
            draw.line((0, y, 640, y), fill=mixed)
        path = root / f"sample_{i:02}.png"
        img.save(path)
        os.utime(path, (stamped + i * 3600, stamped + i * 3600))


def screenshots() -> list[pathlib.Path]:
    """Photograph /g at both form factors, over a generated library."""
    import os
    import tempfile

    import httpx
    from litestar.testing.client.subprocess_client import run_app
    from playwright.sync_api import sync_playwright

    PWA.mkdir(exist_ok=True)
    written: list[pathlib.Path] = []
    repo = pathlib.Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="sg-press-") as tmp:
        home = pathlib.Path(tmp)
        root = home / "lib"
        root.mkdir()
        _sample_library(root)
        before = os.environ.get("SG_TEST_HOME")
        os.environ["SG_TEST_HOME"] = str(home / "run")
        # run_app launches the `litestar` console script, which lives
        # beside the interpreter and off PATH when this module is run
        # by interpreter path (tests/conftest.py does the same).
        held_path = os.environ.get("PATH") or ""
        os.environ["PATH"] = str(repo / ".venv" / "Scripts") + os.pathsep + held_path
        try:
            with run_app(workdir=repo, app="tests.live_app:create_app", capture_output=False) as url:
                with httpx.Client(base_url=url, timeout=30) as api:
                    api.post("/roots", json={"path": str(root)})
                    api.post("/roots/1/scan")
                with sync_playwright() as pw:
                    browser = pw.chromium.launch()
                    for factor, (w, h) in SHOTS.items():
                        page = browser.new_page(viewport={"width": w, "height": h})
                        page.goto(f"{url}/g")
                        page.wait_for_selector(".cell img", timeout=30_000)
                        target = PWA / f"screenshot-{factor}.png"
                        page.screenshot(path=str(target))
                        written.append(target)
                        page.close()
                    browser.close()
        finally:
            os.environ["PATH"] = held_path
            if before is None:
                os.environ.pop("SG_TEST_HOME", None)
            else:
                os.environ["SG_TEST_HOME"] = before
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="sg_web.branding", description="Draw the app's PWA rasters.")
    parser.add_argument("what", choices=("icons", "screenshots", "all"))
    asked = parser.parse_args()
    made: list[pathlib.Path] = []
    if asked.what in ("icons", "all"):
        made += icons()
    if asked.what in ("screenshots", "all"):
        made += screenshots()
    for path in made:
        print(path.relative_to(STATIC.parent.parent))


if __name__ == "__main__":
    main()
