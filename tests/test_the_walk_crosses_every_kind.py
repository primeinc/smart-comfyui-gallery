"""The filmstrip over a library that is not all photographs.

Five kinds are peers in the ResultSet, so a walk through an answer
crosses audio and documents the same way it crosses pictures -- and the
raster routes refuse exactly those two (`_variant_bytes`: "a {kind} has
no {variant}"). The strip pointed every neighbour at `/thumb/{slug}`
regardless, so a mixed walk emitted a first-party 404 per audio file and
per document and drew a broken image where a member of the answer was.

Every filmstrip test passed through this, because the corpus they walk
is twenty-three PNGs. A contract about five kinds proved over one kind
is a contract nobody checked.

Two things are asserted here and neither is visible in a screenshot:

  * nothing the browser fetches for this walk fails (`unbroken`), which
    is the assertion the old cells could not have passed;
  * an audio file and a document are still MEMBERS -- their cells are
    drawn, arrowing onto them works, and the strip's ordinals stay
    contiguous. Dropping them would have silenced the 404 by making the
    walk lie about the answer it is walking.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page

from db import facets
from tests.conftest import POLL, Live
from vision import thumbs

pytestmark = pytest.mark.slow

#: One library, every kind the vocabulary names, interleaved so the
#: window around any member contains something that is not a picture.
#: Ordered by mtime below, so the walk is this order.
WRITTEN = (
    "a_first.png",
    "b_notes.pdf",
    "c_second.png",
    "d_voice.wav",
    "e_clip.mp4",
    "f_third.png",
    "g_more.pdf",
    "h_fourth.png",
)

#: What each of those becomes once the scanner has read it. A document is a
#: PDF and nothing else -- `vision/sniff.py` recognises `%PDF-` and
#: `db/scan.py` KIND_BY_SUFFIX maps `.pdf`; a `.txt` is not scanned at all.
KIND_OF = {
    "a_first.png": "image",
    "b_notes.pdf": "document",
    "c_second.png": "image",
    "d_voice.wav": "audio",
    "e_clip.mp4": "video",
    "f_third.png": "image",
    "g_more.pdf": "document",
    "h_fourth.png": "image",
}

#: The kinds with no picture to take, and the cells that must not ask for a
#: raster: the COMPLEMENT of `vision/thumbs.py PICTURED` within the kind
#: vocabulary, computed so a sixth kind cannot land in neither set.
UNPICTURED = set(facets.KINDS) - set(thumbs.PICTURED)


def _pdf(words: str) -> bytes:
    """The smallest PDF that is one: header, four objects, xref, trailer.

    Written by hand rather than pulled from a library because the only
    property this fixture needs is that `vision/sniff.py` sees `%PDF-`
    and the file is not a lie -- a real reader can open it.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(words) + 1, words.encode("ascii")),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for at in offsets:
        out += b"%010d 00000 n \n" % at
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, start)
    return bytes(out)


def write_library(root) -> None:
    import os

    import av

    for i, name in enumerate(WRITTEN):
        path = root / name
        if name.endswith(".png"):
            Image.new("RGB", (60, 40), (20 + i * 25, 90, 150)).save(path)
        elif name.endswith(".pdf"):
            path.write_bytes(_pdf(f"document {i}"))
        elif name.endswith(".wav"):
            # A REAL wav, written by the standard library: a hand-cut RIFF stub
            # sniffs as audio but the prober cannot read it, so the scan job
            # logs a failed item this fixture would be teaching people to ignore.
            import wave

            with wave.open(str(path), "wb") as sound:
                sound.setnchannels(1)
                sound.setsampwidth(2)
                sound.setframerate(8000)
                sound.writeframes(b"\x00\x00" * 800)
        else:
            with av.open(str(path), "w") as container:
                stream = container.add_stream("h264", rate=5)
                stream.width, stream.height = 160, 120
                stream.pix_fmt = "yuv420p"
                for _ in range(5):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((120, 160, 3), (0, 0, 255), dtype=np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        # mtime IS the walk order under sort=oldest
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == len(WRITTEN), swept
    api.post("/jobs/ingest")
    deadline = time.monotonic() + 90
    while [job for job in api.get("/jobs").json() if job["state"] in ("queued", "running")]:
        assert time.monotonic() < deadline, "jobs never drained"
        time.sleep(POLL)


#: Oldest first, so the walk is WRITTEN's order.
WALK = "sort=oldest"


def _surface(live: Live, slug: str) -> dict:
    """One member's own answer, through the address that negotiates.

    `/g` is a page and does not negotiate JSON: asking it for
    `application/json` returns HTML that `.json()` then fails to parse.
    `/i/{slug}` DOES negotiate, and that is the seam the filmstrip's own
    answer comes through.
    """
    told = live.api.get(f"/i/{slug}", params={"sort": "oldest"}, headers={"accept": "application/json"})
    assert told.status_code == 200, told.text
    return told.json()


#: A file's slug is minted from its name (`a_first.png` -> `a-first`), so the
#: walk is addressable by the names this module wrote, with no scraping of the
#: grid for them.
SLUG_OF = {name: name.rsplit(".", 1)[0].replace("_", "-") for name in WRITTEN}


def _grid(live: Live) -> str:
    told = live.api.get("/g", params={"sort": "oldest", "size": 100})
    assert told.status_code == 200, told.status_code
    return told.text


def _cells(live: Live) -> list[tuple[str, str]]:
    """(slug, kind) per grid cell, in answer order."""
    return re.findall(r'data-slug="([^"]+)" data-kind="([^"]+)"', _grid(live))


# --- the answer itself ------------------------------------------------------


def test_the_library_really_holds_every_kind(live: Live):
    """Without this the rest is a test over six PNGs again."""
    found = _cells(live)
    assert [slug for slug, _ in found] == [SLUG_OF[name] for name in WRITTEN], found
    assert dict(found) == {SLUG_OF[name]: kind for name, kind in KIND_OF.items()}


def test_an_unpictured_member_keeps_its_place_in_the_strip(live: Live):
    """The fix must not be "leave them out". A walk that skips its own
    members is walking a different answer from the one it names."""
    strip = _surface(live, SLUG_OF["e_clip.mp4"])["context"]["filmstrip"]
    ordinals = [one["ordinal"] for one in strip["items"]]
    assert ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))), ordinals
    walked = [one["kind"] for one in strip["items"]]
    assert set(walked) & UNPICTURED, f"this window contains no unpictured member: {walked}"


def test_a_kind_with_no_picture_is_offered_no_raster(live: Live):
    """The defect, stated where it lives: the strip's own answer."""
    strip = _surface(live, SLUG_OF["c_second.png"])["context"]["filmstrip"]
    for one in strip["items"]:
        if one["kind"] in UNPICTURED:
            assert one["thumb"] is None, f"{one['slug']} is a {one['kind']} and was given {one['thumb']}"
        else:
            assert one["thumb"], one


def test_the_grid_offers_none_either(live: Live):
    """The same defect on the bigger surface, and the one that was NOT
    in the backlog: hashing mints an asset address for every kind, so
    three of these eight cells pointed at `/thumbs/<sha>.webp` and got
    404 -- as uncaught exceptions, because rendering was where the kind
    finally got consulted."""
    page = _grid(live)
    cells = re.findall(r'data-kind="([^"]+)"(.*?)</a>', page, re.DOTALL)
    assert len(cells) == len(WRITTEN), len(cells)
    for kind, body in cells:
        if kind in UNPICTURED:
            assert "<img" not in body, f"a {kind} cell still asks for a picture"
            assert "cell-kind" in body, f"a {kind} cell drew nothing at all"
        else:
            assert "<img" in body, f"a {kind} cell lost its picture"


def test_the_raster_routes_really_do_refuse_those_kinds(live: Live):
    """The control. Without it, `thumb is None` proves only that this
    test and the view agree with each other."""
    assert live.api.get(f"/thumb/{SLUG_OF['d_voice.wav']}").status_code == 404
    assert live.api.get(f"/thumb/{SLUG_OF['b_notes.pdf']}").status_code == 404
    assert live.api.get(f"/thumb/{SLUG_OF['a_first.png']}").status_code in (200, 301)


# --- in a browser -----------------------------------------------------------


def test_walking_a_mixed_library_breaks_nothing(page: Page, live: Live, unbroken):
    """`unbroken` is the whole assertion: it fails the test on any
    first-party response >= 400. Before the fix this walk produced one
    per audio file and one per document, every time a strip drew."""
    page.goto(f"/i/{SLUG_OF['a_first.png']}?{WALK}")
    page.wait_for_selector("[data-filmstrip-item]")
    for _ in range(len(WRITTEN) - 1):
        page.keyboard.press("ArrowRight")
        page.wait_for_selector("[data-filmstrip-item]")
    assert unbroken == [], unbroken


def test_the_grid_of_a_mixed_library_breaks_nothing(page: Page, live: Live, unbroken):
    page.goto(f"/g?{WALK}&size=100")
    page.wait_for_selector("[data-grid] .cell")
    page.wait_for_function("() => Array.from(document.images).every(i => i.complete)")
    assert unbroken == [], unbroken


def test_the_unpictured_cells_say_their_kind(page: Page, live: Live, unbroken):
    """Drawn, not blank: a member of the walk you can see and arrow onto."""
    page.goto(f"/i/{SLUG_OF['c_second.png']}?{WALK}")
    page.wait_for_selector("[data-filmstrip-kind]")
    labelled = page.eval_on_selector_all(
        "[data-filmstrip-kind]", "els => els.map(e => [e.dataset.filmstripKind, e.textContent.trim()])"
    )
    assert labelled, "no unpictured cell was drawn"
    for kind, words in labelled:
        assert kind in UNPICTURED, kind
        assert words, f"a {kind} cell drew nothing at all"
    pictured = page.eval_on_selector_all("[data-filmstrip-item] img", "els => els.map(e => e.src)")
    assert pictured, "the pictures stopped being drawn"
    assert all("/thumbs/" in src or "/thumb/" in src for src in pictured), pictured
