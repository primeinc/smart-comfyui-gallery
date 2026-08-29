"""Re-reading, bounded to the folder you are looking at.

Improving a parser is a re-parse of the database -- the schema says so
at `param_key` -- and the sniffer that decides a file's KIND has to keep up
with what cameras, phones and generators actually write. So the application
has to be able to re-read.

It always could, over EVERYTHING. That is a price nobody pays to fix one
folder of album tracks, and a correction too expensive to apply is not a
correction.

It is a job, and it has to be: the same per-file work the whole-library
sweep does, resumable, cancellable, and able to survive a restart --
four thousand files inline would hold the request open. But a job is not
a reason to send somebody to another page, so the button reports where
it is.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from tests.conftest import POLL, Live

pytestmark = pytest.mark.slow

INSIDE = 3
OUTSIDE = 2


def write_library(root) -> None:
    music = root / "music" / "album"
    music.mkdir(parents=True)
    for i in range(INSIDE):
        Image.new("RGB", (16, 12), (30 * i, 90, 140)).save(music / f"track{i}.png")
    pictures = root / "pictures"
    pictures.mkdir()
    for i in range(OUTSIDE):
        Image.new("RGB", (16, 12), (10, 40 * i, 60)).save(pictures / f"p{i}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    api.post(f"/roots/{made['id']}/scan")


def _items(live: Live, job_id: int) -> int:
    return live.api.get(f"/jobs/{job_id}").json()["total"]


def test_a_folder_re_read_takes_its_subtree_and_nothing_else(page: Page, live: Live, unbroken):
    """Somebody pointing at `music` means the albums inside it -- a scope
    that stopped at the top level would silently do a fraction of what
    was asked."""
    told = live.api.post("/jobs/ingest?everything=true&folder=music")
    assert told.status_code in (200, 201), told.text
    assert _items(live, told.json()["id"]) == INSIDE, "the subtree, and nothing outside it"

    whole = live.api.post("/jobs/ingest?everything=true")
    assert _items(live, whole.json()["id"]) == INSIDE + OUTSIDE


def test_the_button_is_on_the_folder_it_would_read(page: Page, live: Live, unbroken):
    """Where the problem is visible. The whole-library sweep already
    exists on the operations page; this is the one somebody runs."""
    page.goto("/f/album")
    # Two retrying assertions rather than a wait and two one-shot reads:
    # `expect` waits for the element itself, so a `wait_for_selector` is a
    # round trip, and a read that answers once races the render it reads.
    button = page.locator("[data-folder-reread]")
    expect(button).to_have_attribute("data-folder-reread", "album")
    expect(button).to_contain_text(str(INSIDE))


def test_it_says_where_it_got_to_without_leaving_the_page(page: Page, live: Live, unbroken):
    """A job is not a reason to send somebody somewhere else: a folder of
    three is done in a second, and "watch it in operations" asks them to
    go and look for something that already finished."""
    page.goto("/f/album")
    button = page.locator("[data-folder-reread]")
    expect(button).to_be_visible()
    was = page.url
    button.click()
    # `wait_for_function`, NOT `expect(...).to_have_text`: the evaluation runs
    # on every animation frame, so it sees the words as they are written, where
    # `expect` polls on a timer and answers late.
    page.wait_for_function(
        "() => /read \\d+ again/.test(document.querySelector('[data-folder-reread]').textContent)",
        timeout=30_000,
    )
    assert page.url == was, "it navigated away to report"
    expect(button).to_be_disabled()


def test_a_folder_with_nothing_to_do_says_so_rather_than_looking_broken(page: Page, live: Live, unbroken):
    """204 is an answer. A button that looked like it failed when the
    truth was "already read" is why the state is drawn from the response
    rather than from the click."""
    deadline = time.monotonic() + 60
    while [one for one in live.api.get("/jobs").json() if one["state"] in ("queued", "running")]:
        assert time.monotonic() < deadline
        time.sleep(POLL)
    assert live.api.post("/jobs/ingest").status_code in (200, 201, 204)
    deadline = time.monotonic() + 60
    while [one for one in live.api.get("/jobs").json() if one["state"] in ("queued", "running")]:
        assert time.monotonic() < deadline
        time.sleep(POLL)

    # everything is read now, so the UNBOUNDED freshness sweep has no
    # items -- which is the 204 the button has to render as words
    assert live.api.post("/jobs/ingest").status_code == 204
