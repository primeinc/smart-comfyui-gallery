"""A thumbnail that will not load degrades instead of breaking.

The server already declines to point at a picture that cannot exist --
`thumbs.asset_url` answers None for a medium with no picture to take,
and every grid draws the kind instead. That is the right answer and it
is not enough, because it depends on the server being right about the
kind.

It was wrong once, exactly the way this catches. An `.m4a` is ISO-BMFF,
mimesniff's MP4 walk calls it video/mp4, ingest lets the bytes overrule
the suffix -- so a folder of album tracks became videos, the server
minted thumbnail addresses in good faith, and every one failed. The row
was wrong; the page had no way to notice; the person got a wall of
broken-image icons.

So this is the second line, and it is about the SYMPTOM rather than
about m4a: any thumbnail that fails, for any reason a page cannot know
-- a row that lies about its kind, a derivative deleted from the cache,
a file gone offline mid-scroll -- becomes the same grey label the server
would have drawn.
"""

from __future__ import annotations

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

FILES = 3


def write_library(root) -> None:
    for i in range(FILES):
        Image.new("RGB", (48, 36), (40 * i, 90, 140)).save(root / f"p{i}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    api.post(f"/roots/{made['id']}/scan")


def _cells(page: Page) -> int:
    return page.evaluate("() => document.querySelectorAll('[data-grid] a.cell').length")


def _broken_labels(page: Page) -> list[str]:
    return page.evaluate("() => [...document.querySelectorAll('[data-broken-picture]')].map(s => s.textContent.trim())")


@pytest.mark.expects_broken
def test_a_thumbnail_that_fails_becomes_the_kind_it_is(page: Page, live: Live):
    """The whole claim. The address is minted, the request fails, and
    the cell says what the file is rather than showing a broken icon."""
    # Fail every thumbnail, which is what a library of mis-kinded rows
    # looked like -- the route answers 404 for a file with no decodable
    # frame, and the page has no way to know that in advance.
    page.route("**/thumbs/**", lambda route: route.fulfill(status=404, body=""))
    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell", timeout=10_000)
    page.wait_for_function(
        f"() => document.querySelectorAll('[data-broken-picture]').length === {FILES}", timeout=10_000
    )
    assert _cells(page) == FILES, "the files are still members of the answer"
    assert page.locator("[data-grid] a.cell img").count() == 0, "a broken image was left on the page"


@pytest.mark.expects_broken
def test_it_says_the_kind_the_row_claims(page: Page, live: Live):
    """Read off the cell, which carries `data-kind`. The label is the
    same shape the server draws for a medium with no picture, so the two
    paths are indistinguishable to somebody looking at the page."""
    page.route("**/thumbs/**", lambda route: route.fulfill(status=404, body=""))
    page.goto("/g")
    page.wait_for_selector("[data-broken-picture]", timeout=10_000)
    held = page.evaluate("() => [...document.querySelectorAll('[data-broken-picture]')].map(s => s.dataset.cellKind)")
    assert set(held) == {"image"}, held
    # and it reads as words, never as an empty box
    assert all(one for one in _broken_labels(page)), _broken_labels(page)


def test_a_picture_that_loads_is_left_alone(page: Page, live: Live):
    """The control. Without it this suite passes over a page that
    replaced every thumbnail in the library."""
    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell img", timeout=10_000)
    page.wait_for_function(
        "() => [...document.querySelectorAll('[data-grid] a.cell img')].every(i => i.complete && i.naturalWidth > 0)",
        timeout=15_000,
    )
    assert _broken_labels(page) == []


@pytest.mark.expects_broken
def test_an_image_that_is_not_a_thumbnail_is_not_touched(page: Page, live: Live):
    """A decorative image, an icon, an avatar with its own fallback --
    none of those want a "doc" label dropped where they were. Only
    pictures OF something in the library are handled."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.evaluate("""() => {
        const img = document.createElement('img');
        img.id = 'not-a-thumbnail';
        img.src = '/static/there-is-no-such-file.png';
        document.body.append(img);
    }""")
    page.wait_for_timeout(500)
    assert page.locator("#not-a-thumbnail").count() == 1, "an unrelated image was replaced"


@pytest.mark.expects_broken
def test_it_catches_a_picture_that_arrives_later(page: Page, live: Live):
    """One listener, in the capture phase, because `error` does not
    bubble -- which is also what makes it enough for images that do not
    exist yet: an endless grid, a swapped fragment, a remounted strip."""
    page.route("**/thumbs/**", lambda route: route.fulfill(status=404, body=""))
    page.goto("/g")
    page.wait_for_selector("[data-broken-picture]", timeout=10_000)
    was = len(_broken_labels(page))

    page.evaluate("""() => {
        const cell = document.querySelector('[data-grid] a.cell');
        const img = document.createElement('img');
        img.src = '/thumbs/00/' + '0'.repeat(64) + '.webp';
        cell.append(img);
    }""")
    page.wait_for_function(
        f"() => document.querySelectorAll('[data-broken-picture]').length === {was + 1}", timeout=10_000
    )
