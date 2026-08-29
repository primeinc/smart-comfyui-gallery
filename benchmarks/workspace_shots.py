"""The query workspace, photographed at three widths.

Every claim about the filter surface, the presentations, the compare
tray and the viewer has a browser test behind it, and every one of those
tests asserts BEHAVIOUR: the control is clickable, the count is right,
the URL survives a reload. None of them can see that a panel overlaps a
heading, that a column runs off the screen, that a section is a wall of
empty rows, or that the whole thing looks like an aircraft maintenance
console.

So this boots the real application over a real mixed library --
generated stills of three recipes, photographs with real camera tags, a
generated clip and a plain one -- and photographs the surfaces at a wide
desktop, a laptop and a phone. The images are the evidence; the contact
sheet is so a person can see them together.

Nothing is staged: every picture is a Chromium screenshot of a page the
running server rendered, and the library is built through the
application's own scan and ingest routes.

Run through `just bench workspace`.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EVIDENCE = REPO / "benchmarks" / "results" / "workspace"

#: The widths a person actually has. Not a sweep: three real machines.
WIDTHS: tuple[tuple[str, int, int], ...] = (
    ("wide", 1920, 1080),
    ("laptop", 1440, 900),
    ("phone", 390, 844),
)

A1111 = (
    "{prompt} <lora:{lora}:{weight}>\n"
    "Negative prompt: blurry, low quality, watermark\n"
    "Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, "
    "Size: 832x1216, Model: {checkpoint}"
)

#: Enough of each recipe that a count can be wrong in a visible way, and
#: enough variety that the drawer's lists are worth looking at.
MADE = [
    *[("dreamshaper_8", "filmGrain", "0.35", "Euler a", 28, 7.0, "a brass diving helmet at dusk")] * 9,
    *[("juggernautXL", "detailTweaker", "0.80", "DPM++ 2M Karras", 20, 4.5, "a paper boat over a waterfall")] * 6,
    *[("fluxDev", "cinematicLight", "1.00", "Euler", 12, 3.5, "a lighthouse in a storm at dawn")] * 4,
]

CAMERAS = (
    ("Canon", "Canon EOS R5", "RF24-70mm F2.8 L IS USM", 100, 2.8, 35.0),
    ("Canon", "Canon EOS R5", "RF24-70mm F2.8 L IS USM", 400, 4.0, 70.0),
    ("FUJIFILM", "X-T4", "XF16-55mmF2.8 R LM WR", 800, 5.6, 23.0),
    ("FUJIFILM", "X-T4", "XF16-55mmF2.8 R LM WR", 3200, 8.0, 55.0),
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _answers(web) -> bool:
    import httpx

    try:
        return web.get("/health").text == "ok"
    except httpx.TransportError:
        return False


def _wait_healthy(web, server) -> None:
    """Bounded readiness gate. A child that already exited is reported as
    the crash it is, never as forty quiet seconds of 'never answered'."""
    deadline = time.time() + 40
    while not _answers(web):
        if server.poll() is not None:
            raise RuntimeError(f"the server exited with {server.returncode} before answering")
        if time.time() > deadline:
            raise RuntimeError("the server never answered /health")
        time.sleep(0.2)


def _library(root: pathlib.Path) -> None:
    """A mixed library, built on disk the way one arrives."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    root.mkdir(parents=True, exist_ok=True)
    for i, (checkpoint, lora, weight, sampler, steps, cfg, prompt) in enumerate(MADE):
        info = PngInfo()
        info.add_text(
            "parameters",
            A1111.format(
                prompt=prompt,
                lora=lora,
                weight=weight,
                steps=steps,
                sampler=sampler,
                cfg=cfg,
                seed=1000 + i,
                checkpoint=checkpoint,
            ),
        )
        # portrait, like most generations, so the grid has a real shape
        shade = (40 + (i * 11) % 180, 60 + (i * 7) % 150, 120 + (i * 13) % 120)
        Image.new("RGB", (416, 608), shade).save(root / f"made_{i:02d}.png", pnginfo=info)

    # Photographs, with the tags a photograph has -- so the Camera
    # section of the drawer and the camera columns of the table have
    # something real in them rather than being empty by construction.
    for i, (make, model, lens, iso, fnum, focal) in enumerate(CAMERAS):
        shot = Image.new("RGB", (720, 480), (30 + i * 30, 110, 70))
        tags = Image.Exif()
        tags[0x010F] = make
        tags[0x0110] = model
        exif = tags.get_ifd(0x8769)
        exif[0x8827] = iso  # ISOSpeedRatings
        exif[0x829D] = fnum  # FNumber
        exif[0x920A] = focal  # FocalLength
        exif[0xA434] = lens  # LensModel
        exif[0x9003] = f"2024:07:{14 + i:02d} 1{i}:20:00"  # DateTimeOriginal
        shot.save(root / f"shot_{i:02d}.jpg", exif=tags)

    import av
    import numpy as np

    graph = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"seed": 424242, "steps": 30, "cfg": 6.0, "sampler_name": "euler", "scheduler": "simple"},
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "wan2_1_t2v.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a paper boat going over a waterfall"}},
    }
    for name, tags in (("generated_clip.mp4", {"prompt": json.dumps(graph)}), ("handheld.mp4", None)):
        # `movflags=use_metadata_tags` or the muxer drops custom tags --
        # which is exactly what ComfyUI passes, and why.
        with av.open(str(root / name), "w", options={"movflags": "use_metadata_tags"}) as container:
            if tags:
                for key, value in tags.items():
                    container.metadata[key] = value
            stream = container.add_stream("h264", rate=8)
            stream.width, stream.height = 480, 270
            stream.pix_fmt = "yuv420p"
            for i in range(16):
                frame = av.VideoFrame.from_ndarray(
                    np.full((270, 480, 3), (200 - i * 8, 40, 60 + i * 6), dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def _drained(web, timeout: float = 300.0) -> None:
    deadline = time.time() + timeout
    while True:
        running = [job["id"] for job in web.get("/jobs").json() if job["state"] in ("queued", "running")]
        if not running:
            return
        if time.time() > deadline:
            raise RuntimeError(f"jobs still running: {running}")
        time.sleep(0.2)


#: Scripts run in the page before the shutter, so a capture of an open
#: drawer is a capture of an OPEN drawer rather than one mid-animation.
SETTLE = {
    "open-filters": (
        "() => {"
        " const open = document.querySelector('[data-filters-open]');"
        " if (open && open.getAttribute('aria-expanded') !== 'true') open.click();"
        " for (const key of ['generation.checkpoint', 'kind']) {"
        "   const one = document.querySelector('[data-filter=\"' + key + '\"]');"
        "   if (one && !one.open) one.querySelector('summary').click(); }"
        "}"
    ),
    "open-inspector": (
        "() => {"
        " const root = document.querySelector('[data-viewer]');"
        " const toggle = document.querySelector('[data-inspector-toggle]');"
        " if (root && root.dataset.inspector !== 'open' && toggle) toggle.click();"
        " const panel = document.querySelector('[data-panel=\"creation\"]');"
        " if (panel && !panel.open) panel.querySelector('summary').click();"
        "}"
    ),
    # The tray is workspace state, so filling it is writing that state and
    # reloading -- which is also a real proof that it survives a reload.
    "keep": (
        "() => {"
        " const cells = [...document.querySelectorAll('a.cell[data-slug]')].slice(0, 3);"
        " const compare = cells.map(c => ({slug: c.dataset.slug, name: c.querySelector('img').alt,"
        "   kind: c.dataset.kind || ''}));"
        " const held = JSON.parse(localStorage.getItem('sg.workspace.v1') || '{}');"
        " localStorage.setItem('sg.workspace.v1', JSON.stringify({...held, compare, tray: 'open'}));"
        "}"
    ),
}


def _surfaces(slug: str) -> tuple[dict, ...]:
    return (
        {"name": "gallery", "url": "/g", "caption": "The gallery, nothing asked."},
        {
            "name": "filters-open",
            "url": "/g",
            "caption": "The filter drawer, with Generation and Media disclosed and their values counted.",
            "settle": "open-filters",
        },
        {
            "name": "filtered",
            "url": "/g?f=has.generation%3Aeq%3A1&f=generation.checkpoint%3Aeq%3A{checkpoint}",
            "caption": "Two clauses held: the chips, the count, the badge.",
        },
        {
            "name": "analyze-generated",
            "url": "/g?view=analyze&f=has.generation%3Aeq%3A1",
            "caption": "Analyze over generated media: prompts, LoRAs, and the recipe broken down.",
        },
        {
            "name": "analyze-photos",
            "url": "/g?view=analyze&f=has.capture%3Aeq%3A1",
            "caption": "Analyze over photographs -- the SAME surface, different dimensions.",
        },
        {"name": "analyze-everything", "url": "/g?view=analyze", "caption": "Analyze over the whole mixed library."},
        {"name": "table", "url": "/g?view=table", "caption": "The table over mixed media."},
        {
            "name": "table-generated",
            "url": "/g?view=table&f=has.generation%3Aeq%3A1",
            "caption": "The table over generated media only: the recipe columns appear.",
        },
        {
            "name": "viewer",
            "url": f"/i/{slug}",
            "caption": "The viewer with the recipe panel open.",
            "settle": "open-inspector",
        },
        {"name": "compare-tray", "url": "/g", "caption": "Three kept in the compare tray.", "settle": "keep"},
        {
            "name": "comparing",
            "url": "/g",
            "caption": "The comparison itself, above everything.",
            "settle": "keep",
            "then": "compare",
        },
    )


def _capture(scratch: pathlib.Path, staging: pathlib.Path) -> list[dict]:
    import httpx
    from playwright.sync_api import sync_playwright

    taken: list[dict] = []
    root = scratch / "library"
    _library(root)

    port = _free_port()
    home = scratch / "run"
    with (staging / "server.log").open("wb") as server_log:
        server = subprocess.Popen(
            [sys.executable, "-m", "sg_web", "--home", str(home), "--port", str(port)],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(base_url=base, timeout=30.0) as web, sync_playwright() as play:
                _wait_healthy(web, server)
                made = web.post("/roots", json={"path": str(root)}).json()
                swept = web.post(f"/roots/{made['id']}/scan").json()
                print(f"scanned {swept['added']} files")
                web.post("/jobs/ingest")
                _drained(web)

                listed = web.get("/g/peek", params={"page": 1, "count": 9}).json()["items"]
                slug = next(one["slug"] for one in listed if one["name"].startswith("made_"))
                offered = web.get("/g/options", params={"key": "generation.checkpoint"}).json()["options"]
                busiest = offered[0]["value"] if offered else ""

                browser = play.chromium.launch()
                for label, width, height in WIDTHS:
                    context = browser.new_context(viewport={"width": width, "height": height}, base_url=base)
                    page = context.new_page()
                    for surface in _surfaces(slug):
                        url = surface["url"].replace("{checkpoint}", str(busiest))
                        page.goto(url, wait_until="networkidle")
                        # Workspace state persists across navigation ON PURPOSE, so
                        # one capture's open drawer follows it into the next.
                        # Each surface is photographed from the default arrangement.
                        page.evaluate("() => localStorage.removeItem('sg.workspace.v1')")
                        page.reload(wait_until="networkidle")
                        settle = surface.get("settle")
                        if settle:
                            page.evaluate(SETTLE[settle])
                            if settle == "keep":
                                page.reload(wait_until="networkidle")
                            page.wait_for_timeout(300)
                        if surface.get("then") == "compare":
                            page.click("[data-compare-open]")
                            page.wait_for_selector("[data-compare-view]", timeout=10_000)
                            page.wait_for_timeout(600)
                        page.wait_for_timeout(250)
                        name = f"{surface['name']}--{label}"
                        (staging / f"{name}.png").write_bytes(page.screenshot())
                        taken.append(
                            {
                                "name": name,
                                "surface": surface["name"],
                                "width": label,
                                "px": f"{width}x{height}",
                                "file": f"{name}.png",
                                "caption": surface["caption"],
                                "url": url,
                            }
                        )
                        print(f"  shot {name}")
                    context.close()
                browser.close()
        finally:
            server.terminate()
            with contextlib.suppress(Exception):
                server.wait(timeout=15)
    return taken


def _sheet(taken: list[dict]) -> str:
    by_surface: dict[str, list[dict]] = {}
    for one in taken:
        by_surface.setdefault(one["surface"], []).append(one)
    blocks = []
    for surface, shots in by_surface.items():
        columns = "".join(
            f"<figure><figcaption>{html.escape(one['width'])} &middot; {html.escape(one['px'])}</figcaption>"
            f'<img src="data:image/png;base64,'
            f'{base64.b64encode((EVIDENCE / one["file"]).read_bytes()).decode()}"></figure>'
            for one in shots
        )
        blocks.append(
            f"<section><h2>{html.escape(surface)}</h2>"
            f"<p>{html.escape(shots[0]['caption'])}</p>"
            f"<code>{html.escape(shots[0]['url'])}</code>"
            f'<div class="row">{columns}</div></section>'
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>the query workspace, at three widths</title>"
        "<style>body{background:#14151a;color:#e8e6df;font:14px/1.5 system-ui;margin:0;padding:24px}"
        "h1{font-size:20px}h2{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:#d8a24a}"
        "code{color:#9a978d;font-size:12px}"
        ".row{display:flex;gap:14px;align-items:flex-start;overflow-x:auto;padding:10px 0}"
        "figure{margin:0;flex:none}figcaption{color:#9a978d;font-size:11px;padding-bottom:4px}"
        "img{border:1px solid #2c2f38;max-height:560px}"
        "section{border-top:1px solid #2c2f38;padding:14px 0}</style>"
        "<h1>The query workspace, photographed</h1>"
        "<p>Every image is a Chromium screenshot of a page the running application rendered, over a"
        " library built through its own scan and ingest routes.</p>" + "".join(blocks)
    )


def main() -> int:
    argparse.ArgumentParser(description="Photograph the query workspace at three widths.").parse_args()
    staging = pathlib.Path(tempfile.mkdtemp())
    scratch = pathlib.Path(tempfile.mkdtemp())
    taken = _capture(scratch, staging)
    if not taken:
        print("nothing was captured", file=sys.stderr)
        return 1
    if EVIDENCE.exists():
        shutil.rmtree(EVIDENCE)
    EVIDENCE.mkdir(parents=True)
    for one in staging.iterdir():
        shutil.copy2(one, EVIDENCE / one.name)
    # An explicit newline, so the tracked manifest is committed with the
    # line endings this repository requires (sglint SG804) rather than
    # whatever the platform's default writer produces.
    with (EVIDENCE / "manifest.json").open("w", encoding="utf-8", newline="\n") as sheet:
        json.dump(taken, sheet, indent=2)
    (EVIDENCE / "contact-sheet.html").write_text(_sheet(taken), encoding="utf-8")
    print(f"\n{len(taken)} captures -> {EVIDENCE}")
    print(f"contact sheet: {EVIDENCE / 'contact-sheet.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
