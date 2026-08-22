"""A real browser against the real server: the assumptions, on screen.

TestClient proves the routes; this proves the CLIENT SIDE of every
serving claim, in Chromium via Playwright (microsoft/playwright-python@
eab2bca README.md, sync API): the WebP variants decode to the pixels we
rendered, the EXIF turn survives all the way to the screen, a <video>
element seeks through OUR range implementation issuing real 206s, and
the job feed streams into a browser WebSocket. Pixels are read from
element screenshots, not canvas readback -- a cross-origin video taints
a canvas and getImageData throws, while a screenshot sees what the
compositor drew.
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.spawns


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _centre(shot_bytes: bytes) -> tuple[int, int, int]:
    import io

    image = Image.open(io.BytesIO(shot_bytes)).convert("RGB")
    pixel = image.getpixel((image.width // 2, image.height // 2))
    assert isinstance(pixel, tuple)
    r, g, b = pixel
    return r, g, b


@pytest.fixture(scope="module")
def gallery(tmp_path_factory):
    """The real server over a real library, plus one Chromium."""
    import httpx
    from playwright.sync_api import sync_playwright

    tmp = tmp_path_factory.mktemp("browser")
    root = tmp / "lib"
    root.mkdir()
    Image.new("RGB", (900, 400), (200, 30, 30)).save(root / "wide.png")
    turned = Image.new("RGB", (600, 400), (30, 30, 200))
    tag = Image.Exif()
    tag[274] = 6  # stored on its side; upright is 400x600
    turned.save(root / "turned.jpg", exif=tag)

    import av

    with av.open(str(root / "clip.mp4"), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for n in range(30):  # 3s blue, then 3s green
            color = (0, 0, 255) if n < 15 else (0, 255, 0)
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), color, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    port = _free_port()
    # The child's stdout is its access log, one line per request; a pipe
    # nobody drains blocks the server at the OS buffer. A file has no
    # such ceiling, and holds the log for a post-mortem.
    server_log = (tmp / "server.log").open("wb")
    base = f"http://127.0.0.1:{port}"
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "sg_web", "--home", str(tmp / "run"), "--port", str(port)],
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        with httpx.Client(base_url=base, timeout=5.0) as web:
            deadline = time.time() + 30
            while True:
                try:
                    if web.get("/health").text == "ok":
                        break
                except httpx.TransportError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.2)
            made = web.post("/roots", json={"path": str(root)}).json()
            assert web.post(f"/roots/{made['id']}/scan").json()["added"] == 3

        with sync_playwright() as p:
            browser = p.chromium.launch()
            yield browser, base
            browser.close()
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        server_log.close()


def test_the_thumbnails_are_pixels_a_browser_shows(gallery):
    """The grid's <img> tags decode our WebP variants to the colors we
    stored, at the contained sizes, with the EXIF turn applied."""
    browser, base = gallery
    page = browser.new_page()
    try:
        page.set_content(
            f'<img id="wide" src="{base}/thumb/wide">'
            f'<img id="turned" src="{base}/thumb/turned">'
            f'<img id="big" src="{base}/preview/wide">'
        )
        page.wait_for_function(
            "() => ['wide','turned','big'].every(i => document.getElementById(i).complete"
            " && document.getElementById(i).naturalWidth > 0)"
        )
        sizes = page.evaluate(
            "() => Object.fromEntries(['wide','turned','big'].map(i => {"
            " const e = document.getElementById(i);"
            " return [i, [e.naturalWidth, e.naturalHeight]]; }))"
        )
        assert sizes["wide"] == [512, 228]
        assert sizes["big"] == [1440, 640]
        assert sizes["turned"][1] > sizes["turned"][0], "the EXIF turn never reached the screen"

        r, g, b = _centre(page.locator("#wide").screenshot())
        assert r > g, "the thumbnail on screen is not the picture's color"
        assert r > b, "the thumbnail on screen is not the picture's color"
    finally:
        page.close()


def test_a_video_element_seeks_through_our_range_implementation(gallery):
    """Chromium's media stack fetches with Range; a seek to the green half
    must land on green pixels, and at least one of those fetches must have
    been answered 206 -- the whole reason the range code exists."""
    browser, base = gallery
    page = browser.new_page()
    served = []
    page.on("response", lambda response: served.append((response.url, response.status)))
    try:
        page.set_content(f'<video id="v" src="{base}/media/clip" muted preload="auto" width="320"></video>')
        page.wait_for_function("() => document.getElementById('v').readyState >= 2")

        page.evaluate(
            "() => new Promise(done => {"
            " const v = document.getElementById('v');"
            " v.addEventListener('seeked', done, { once: true });"
            " v.currentTime = 4.5; })"
        )
        told = page.evaluate(
            "() => { const v = document.getElementById('v');"
            " return { at: v.currentTime, w: v.videoWidth, h: v.videoHeight, secs: v.duration }; }"
        )
        assert told["w"] == 320
        assert told["h"] == 180
        assert 5.5 <= told["secs"] <= 6.5, "the container's clock did not survive to the browser"
        assert abs(told["at"] - 4.5) < 0.05

        r, g, b = _centre(page.locator("#v").screenshot())
        assert g > b, "the seeked frame is not the moment that was asked for"
        assert g > r, "the seeked frame is not the moment that was asked for"

        ranged = [(url, status) for url, status in served if "/media/clip" in url]
        assert any(status == 206 for _, status in ranged), f"Chromium never got a 206 from the media route: {ranged}"
    finally:
        page.close()


def test_job_progress_streams_into_a_browser_websocket(gallery):
    """The realtime contract holds where it will actually be consumed:
    a browser WebSocket receives the snapshot first, then deltas through
    to the terminal state, with no polling anywhere."""
    import httpx

    browser, base = gallery
    page = browser.new_page()
    try:
        ws_base = base.replace("http://", "ws://")
        page.evaluate(
            "(feed) => { window.__got = [];"
            " window.__ws = new WebSocket(feed);"
            " window.__ws.onmessage = (event) => window.__got.push(JSON.parse(event.data)); }",
            f"{ws_base}/ws/jobs",
        )
        page.wait_for_function("() => window.__ws.readyState === 1")

        with httpx.Client(base_url=base, timeout=5.0) as web:
            job_id = web.post("/jobs/verify").json()["id"]

        page.wait_for_function(
            "() => window.__got.some(m => ['done','failed','cancelled'].includes(m.state))",
            timeout=30_000,
        )
        got = page.evaluate("() => window.__got")
        assert got[0]["type"] == "snapshot", "the first message must render the rows, not an event"
        deltas = [m for m in got[1:] if m.get("job") == job_id]
        assert deltas, "no deltas for the submitted job reached the browser"
        assert deltas[-1]["state"] == "done"
        done_counts = [m["done"] for m in deltas]
        assert done_counts == sorted(done_counts), "progress went backwards in the browser"
    finally:
        page.close()


def test_the_activity_surface_shows_a_sweep_started_from_operations(gallery):
    """WI-51's proof on screen: no `new WebSocket()` injected anywhere.
    A person opens /operations, presses a sweep button, and the shell's
    own activity surface -- the htmx ws extension swapping the server's
    fragments -- shows the job appear as queued and run through to done,
    with progress that never goes backwards. The same surface is mounted
    on the gallery, so a reload elsewhere shows the persisted row too."""
    browser, base = gallery
    page = browser.new_page()
    try:
        page.goto(f"{base}/operations")
        page.wait_for_selector("[data-operations]")
        # the feed is connected before anything is pressed
        page.wait_for_function("() => document.querySelector('[data-activity-jobs]') !== null")
        page.click('[data-launch="verify"]')
        page.wait_for_selector("[data-shell-notice] [data-notice]")
        notice = page.text_content("[data-shell-notice] [data-notice]")
        assert notice is not None
        assert "queued #" in notice
        job_id = int(notice.rsplit("#", 1)[1].split(",")[0].strip())
        # a refusal lands in the same notice, as an error, with no reload
        page.evaluate(
            "() => htmx.ajax('POST', '/operations/roots', {target: '#operations-roots', values: {path: ' '}})"
        )
        page.wait_for_selector("[data-shell-notice] [data-error]")
        refused = page.text_content("[data-shell-notice] [data-error]")
        assert refused is not None
        assert "needs a path" in refused, refused

        page.wait_for_selector(f'[data-activity-jobs] [data-job="{job_id}"]', state="attached")
        # the surface is a dropdown that opens inside the viewport, not a
        # list painted off the right edge of the shell
        page.click("[data-activity] summary")
        page.wait_for_selector(f'[data-activity-jobs] [data-job="{job_id}"]', state="visible")
        box = page.locator("[data-activity-jobs]").bounding_box()
        size = page.viewport_size
        assert box is not None
        assert size is not None
        assert box["x"] >= 0, box
        assert box["y"] >= 0, box
        assert box["x"] + box["width"] <= size["width"], (box, size)
        assert box["width"] >= 200, box
        shot = os.environ.get("SG_SCREENSHOT_DIR")
        if shot:
            page.screenshot(path=str(pathlib.Path(shot) / "activity-open.png"), full_page=False)
        page.wait_for_function(
            "(id) => { const li = document.querySelector(`[data-job='${id}']`);"
            " return li && ['done','failed','cancelled'].includes(li.dataset.state); }",
            arg=job_id,
            timeout=30_000,
        )
        assert page.get_attribute(f'[data-job="{job_id}"]', "data-state") == "done"
        count = page.text_content(f'[data-job="{job_id}"] .job-count')
        assert count is not None
        assert count.strip().startswith("3 / 3")
        # a terminal job has no cancel; a live one would
        assert page.locator(f'[data-job="{job_id}"] .job-cancel').count() == 0
    finally:
        page.close()
