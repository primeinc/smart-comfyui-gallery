"""File-serving routes must not escape their directory.

Three routes take a user-controlled name and hand it to
`send_from_directory`: the zip download, the ComfyUI input passthrough
(whose `<path:filename>` converter accepts slashes), and the storyboard
frame server. Each is a candidate for path traversal, and one of them is
reachable without a session on a non-forced-login install.

The secret is planted ONLY outside every served root, so a 200 carrying it
can only mean the route escaped. A positive control asserts the probe can
see a genuine serve, otherwise a uniform 404 would prove nothing -- an
all-404 result is equally consistent with a broken test.
"""

from __future__ import annotations

import contextlib
import os

import pytest

_SECRET = b"this file is outside every served directory"
_NAME = "traversal_target.txt"

# Slash, encoded slash, double-encoded, backslash, and the "....//" trick
# that defeats naive single-pass stripping.
_PAYLOADS = [
    "../" + _NAME,
    "..%2f" + _NAME,
    "%2e%2e%2f" + _NAME,
    "....//" + _NAME,
    ".." + chr(92) + _NAME,
    "..%5c" + _NAME,
    "%2e%2e%5c" + _NAME,
    "%252e%252e%252f" + _NAME,
    "../../" + _NAME,
]

_ROUTES = [
    "/galleryout/serve_zip/{}",
    "/galleryout/input_file/{}",
    "/galleryout/storyboard_frame/{}/frame.png",
    "/galleryout/storyboard_frame/somehash/{}",
]


@pytest.fixture
def planted_secret(smartgallery_app):
    """A file one level above the input folder, guaranteed to be outside
    every directory any route is allowed to serve from."""
    served = {os.path.abspath(p) for p in (
        smartgallery_app.BASE_INPUT_PATH,
        smartgallery_app.BASE_OUTPUT_PATH,
        smartgallery_app.BASE_SMARTGALLERY_PATH,
        smartgallery_app.ZIP_CACHE_DIR,
        smartgallery_app.THUMBNAIL_CACHE_DIR)}
    for directory in served:
        os.makedirs(directory, exist_ok=True)

    outside = os.path.abspath(
        os.path.join(smartgallery_app.BASE_INPUT_PATH, os.pardir))
    if outside in served:
        pytest.skip("no directory above the input folder is out of range here")

    path = os.path.join(outside, _NAME)
    with open(path, "wb") as fh:
        fh.write(_SECRET)
    # No same-named decoy inside a served root, or a legitimate hit would
    # read as an escape (this is exactly how a first draft of this probe
    # produced a false positive).
    for directory in served:
        decoy = os.path.join(directory, _NAME)
        if os.path.exists(decoy):
            os.remove(decoy)
    yield path
    with contextlib.suppress(OSError):
        os.remove(path)


def test_the_probe_can_see_a_genuine_serve(smartgallery_app):
    """Positive control. Without this, a uniform 404 below would be equally
    consistent with the routes being unreachable or the assertion broken."""
    os.makedirs(smartgallery_app.ZIP_CACHE_DIR, exist_ok=True)
    inside = os.path.join(smartgallery_app.ZIP_CACHE_DIR, "control_probe.zip")
    with open(inside, "wb") as fh:
        fh.write(_SECRET)
    try:
        resp = smartgallery_app.app.test_client().get(
            "/galleryout/serve_zip/control_probe.zip")
        assert resp.status_code == 200
        assert _SECRET in resp.get_data(), "the probe cannot detect a real serve"
    finally:
        with contextlib.suppress(OSError):
            os.remove(inside)


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("payload", _PAYLOADS)
def test_no_route_serves_a_file_outside_its_directory(
        smartgallery_app, planted_secret, route, payload):
    resp = smartgallery_app.app.test_client().get(route.format(payload))
    assert _SECRET not in resp.get_data(), (
        f"{route.format(payload)} escaped its directory (status "
        f"{resp.status_code})")
