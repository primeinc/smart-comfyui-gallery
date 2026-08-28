"""The gallery installs as an app, on every engine that can.

There is no single "install a PWA" API: Chromium prompts through a
non-standard event, Safari installs only by hand gesture, Android reads
the manifest itself. Each clause here pins one seam of that contract --
the manifest and its MIME type, the truthfulness of every raster it
names, the root-scoped service worker, the iOS head lines, and the
shell's affordances -- so a drifted asset or a silently unloadable
manifest fails a test instead of failing an install in silence.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from litestar.testing import TestClient

from sg_web import branding
from sg_web.app import build_app
from vision import decode

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
STATIC = pathlib.Path(__file__).resolve().parent.parent / "sg_web" / "static"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    with TestClient(app=build_app(str(tmp_path_factory.mktemp("pwa") / "run"), worker=False)) as client:
        yield client


def manifest_dict() -> dict:
    return json.loads((STATIC / "manifest.webmanifest").read_text(encoding="utf-8"))


def test_the_manifest_is_served_as_json_with_the_fields_installability_needs(served):
    told = served.get("/manifest.webmanifest")
    assert told.status_code == 200
    assert told.headers["content-type"].startswith("application/manifest+json"), (
        "a non-JSON MIME type makes Chromium ignore the manifest in silence"
    )
    body = told.json()
    assert body["id"] == "/g", "an explicit id keeps install identity off start_url"
    assert body["start_url"] == "/g"
    assert body["scope"] == "/", "every product route must fall inside the installed scope"
    assert body["display"] == "standalone"
    assert body["name"]
    assert body["short_name"]
    assert body["theme_color"] == "#14151a"


def test_every_page_links_the_manifest_exactly_once(served):
    page = served.get("/g", headers=AS_BROWSER).text
    assert page.count('rel="manifest"') == 1, "only one manifest link per page is honored"
    assert '<link rel="manifest" href="/manifest.webmanifest">' in page


def test_every_icon_the_manifest_names_exists_at_its_declared_size(served):
    body = manifest_dict()
    sizes = {tuple(int(n) for n in icon["sizes"].split("x")) for icon in body["icons"]}
    assert (192, 192) in sizes
    assert (512, 512) in sizes, "Chromium's install criteria want 192 and 512 PNGs"
    assert any(icon.get("purpose") == "maskable" for icon in body["icons"]), (
        "without a maskable icon Android letterboxes the mark"
    )
    for icon in body["icons"]:
        answer = served.get(icon["src"])
        assert answer.status_code == 200, icon["src"]
        with decode.open_still(STATIC / "pwa" / icon["src"].rsplit("/", 1)[1]) as held:
            assert f"{held.width}x{held.height}" == icon["sizes"], f"{icon['src']} lies about its pixels"


def test_the_maskable_icon_bleeds_to_its_edges_and_keeps_its_mark_in_the_safe_zone():
    """Any platform mask may discard pixels outside a centred circle of
    radius 40% of the edge (w3c/manifest index.html:2226-2231): the
    ground must reach every corner, and nothing but ground may sit
    outside the circle."""
    with decode.open_still(STATIC / "pwa" / "icon-maskable-512.png") as raw:
        held = raw.convert("RGBA")
    for corner in ((0, 0), (511, 0), (0, 511), (511, 511)):
        # RGBA, so every pixel is the 4-tuple -- `getpixel` also answers a
        # bare band value for a one-band image and None off the canvas.
        pixel = held.getpixel(corner)
        assert isinstance(pixel, tuple), f"{corner} is not an RGBA pixel"
        assert pixel[3] == 255, "a transparent corner letterboxes under a mask"
    center, radius = 256, 0.4 * 512
    ground = held.getpixel((2, 2))
    for x in range(0, 512, 8):
        for y in range(0, 512, 8):
            if ((x - center) ** 2 + (y - center) ** 2) ** 0.5 > radius + 8:
                assert held.getpixel((x, y)) == ground, f"mark pixels at ({x},{y}) sit outside the safe zone"


def test_the_screenshots_carry_both_form_factors_truthfully(served):
    """A UA shows only its own form factor's screenshots, so a
    narrow-only set leaves every desktop install sheet blank -- and a
    `sizes` that disagrees with the PNG can cost the selection."""
    shots = manifest_dict()["screenshots"]
    assert {shot["form_factor"] for shot in shots} == {"narrow", "wide"}
    for shot in shots:
        assert shot["label"], "the label is the screenshot's accessible name"
        assert served.get(shot["src"]).status_code == 200
        with decode.open_still(STATIC / "pwa" / shot["src"].rsplit("/", 1)[1]) as held:
            assert f"{held.width}x{held.height}" == shot["sizes"], f"{shot['src']} lies about its pixels"


def test_the_service_worker_is_root_scoped_real_and_update_safe(served):
    told = served.get("/sw.js")
    assert told.status_code == 200
    assert told.headers["content-type"].startswith("text/javascript")
    assert told.headers["cache-control"] == "no-cache", "a cached worker script stalls every future update"
    body = told.text
    assert 'addEventListener("fetch"' in body, "Chromium's ambient prompt requires a fetch handler"
    assert "skipWaiting()" in body
    assert "SKIP_WAITING" in body, "activation must wait for the page's reload prompt"
    assert re.search(r"addEventListener\(\"install\"", body)
    assert "caches.delete" in body, "stale cache versions must die at activate"
    page = served.get("/g", headers=AS_BROWSER).text
    assert "/static/build/app.js" in page, "nothing registers the worker without the bundle"


def test_offline_navigation_has_a_real_page_to_land_on(served):
    """The worker's whole cache is the offline page; it must exist, be
    the URL sw.js precaches, and own its icon inline -- a request onto
    /static is exactly what just failed."""
    sw = (STATIC / "sw.js").read_text(encoding="utf-8")
    assert '"/static/offline.html"' in sw
    told = served.get("/static/offline.html")
    assert told.status_code == 200
    assert 'rel="icon" href="data:image/svg+xml' in told.text


def test_the_ios_head_lines_match_files_that_exist(served):
    """iOS ignores manifest icons and draws no launch screen it is not
    handed -- and a media triple that misses by one pixel is ignored in
    silence, indistinguishable from shipping nothing. Every link in
    base.html must resolve, at the pixel size its triple implies, and
    the generator's table must be exactly the linked set."""
    page = served.get("/g", headers=AS_BROWSER).text
    assert '<link rel="apple-touch-icon" href="/static/pwa/apple-touch-icon-180.png">' in page
    assert "apple-mobile-web-app-capable" not in page, "the deprecated meta masks a broken manifest"
    linked = re.findall(
        r'apple-touch-startup-image" media="\(device-width: (\d+)px\) and \(device-height: (\d+)px\)'
        r' and \(-webkit-device-pixel-ratio: (\d+)\)[^"]*" href="([^"]+)"',
        page,
    )
    assert {(int(w), int(h), int(r)) for w, h, r, _ in linked} == set(branding.SPLASH), (
        "base.html and branding.SPLASH disagree; regenerate or relink"
    )
    for w, h, ratio, href in linked:
        assert served.get(href).status_code == 200, href
        with decode.open_still(STATIC / "pwa" / href.rsplit("/", 1)[1]) as held:
            assert held.size == (int(w) * int(ratio), int(h) * int(ratio)), f"{href} is not its triple's box"


def test_the_shell_offers_install_without_depending_on_chromiums_event(served):
    """Both affordances exist in the document and both start hidden:
    install.ts reveals the button only when Chromium hands it a prompt,
    and the iOS hint only on iOS outside the installed app. A page that
    only ever renders the button has silently dropped Safari."""
    page = served.get("/g", headers=AS_BROWSER).text
    assert re.search(r"<button[^>]*data-install\b[^>]*hidden", page)
    assert re.search(r"<p[^>]*data-install-ios[^>]*hidden", page)
    assert "Add to Home Screen" in page
