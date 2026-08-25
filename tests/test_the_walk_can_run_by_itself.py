"""A slideshow is the walk on a timer, and nothing more than that.

It takes the same step an arrow takes, to the same address, through the
same `walk` the container gave the viewer. No second ordering, no list of
slugs cached in the browser, no prefetch queue -- which is what keeps it
honest when the answer moves underneath it.

The decision worth pinning is where "playing" lives. The viewer is
REMOUNTED on every step (the overlay replaces its contents, the page
navigates), so a timer held in the module dies with each picture. Playing
is a fact about the walk, not about the mount, so it lives in the
workspace -- and that one choice is also why the slideshow works
identically in both containers.

Two settings, not one, because they answer different questions:

    wrap    what the ARROWS do at either end of the answer
    loop    what the SLIDESHOW does when it reaches the end

Somebody can reasonably want arrows that stop -- so they can tell when
they have seen everything -- over a slideshow that repeats all night.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

FILES = 4


def write_library(root) -> None:
    for i in range(FILES):
        # distinct sizes, so which picture is on the stage is legible from
        # the address alone and no test has to compare pixels
        Image.new("RGB", (8 + i, 8), (60 * i, 40, 200)).save(root / f"p{i}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    api.post(f"/roots/{made['id']}/scan")


def _open_first(page: Page) -> None:
    page.goto("/g?sort=oldest")
    page.wait_for_selector("[data-grid] a.cell", timeout=10_000)
    page.locator("[data-grid] a.cell").first.click()
    page.wait_for_selector("[data-viewer]", timeout=10_000)


def _ordinal(page: Page) -> int:
    return int(page.inner_text(".viewer-ordinal").split("/")[0].strip())


def _arrange(page: Page, **said) -> None:
    """Write the workspace before any script reads it.

    `add_init_script` runs before the page's own scripts on every
    navigation, which is the only place a per-browser arrangement can be
    stated: there is no server setting to post, on purpose -- wrap and
    loop change no membership and belong in no fingerprint.
    """
    held = json.dumps(json.dumps(said))
    page.add_init_script(f"try {{ localStorage.setItem('sg.workspace.v1', {held}); }} catch (e) {{}}")


def test_playing_walks_without_anybody_touching_it(page: Page, live: Live):
    _arrange(page, showEvery=1)
    _open_first(page)
    assert _ordinal(page) == 1
    page.click("[data-show-play]")
    # The whole claim: nothing below this line touches the page.
    page.wait_for_function(
        "() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('2')", timeout=10_000
    )
    page.wait_for_function(
        "() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('3')", timeout=10_000
    )


def test_it_survives_the_remount_that_each_step_is(page: Page, live: Live):
    """The reason `showPlaying` is workspace state and not a variable.

    Every step replaces the viewer, so a slideshow that lived in the
    module would take exactly one step and stop. Two steps is the proof
    that the second mount picked the walk up.
    """
    _arrange(page, showEvery=1)
    _open_first(page)
    page.click("[data-show-play]")
    page.wait_for_function(
        "() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('3')", timeout=10_000
    )
    assert page.evaluate("() => JSON.parse(localStorage.getItem('sg.workspace.v1')).showPlaying") is True
    assert page.get_attribute("[data-slideshow]", "data-playing") == "yes"


def test_without_loop_it_stops_at_the_end_and_stays_there(page: Page, live: Live):
    _arrange(page, showEvery=1, loop=False)
    _open_first(page)
    page.click("[data-show-play]")
    page.wait_for_function(
        f"() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('{FILES}')", timeout=20_000
    )
    # It stopped ITSELF: the control says so, and the state it left is the
    # state the next mount reads.
    page.wait_for_selector('[data-slideshow][data-playing="no"]', timeout=10_000)
    assert page.evaluate("() => JSON.parse(localStorage.getItem('sg.workspace.v1')).showPlaying") is False
    assert _ordinal(page) == FILES, "it walked past the end of the answer"


def test_with_loop_the_end_is_the_start_again(page: Page, live: Live):
    _arrange(page, showEvery=1, loop=True)
    _open_first(page)
    page.click("[data-show-play]")
    page.wait_for_function(
        f"() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('{FILES}')", timeout=20_000
    )
    # Round again, to the SAME answer -- crossing the end is this walk
    # restarting, never a silent slide into a different question.
    page.wait_for_function(
        "() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('1')", timeout=20_000
    )
    assert page.get_attribute("[data-slideshow]", "data-playing") == "yes"


def test_wrap_is_about_the_arrows_and_loop_does_not_turn_it_on(page: Page, live: Live):
    """The two settings are separate, and this is where that is visible:
    a looping slideshow does not give a person wrapping arrows."""
    _arrange(page, loop=True, wrap=False)
    _open_first(page)
    assert page.get_attribute("[data-viewer]", "data-wrap") == "off"
    # The wrapping arrow IS in the document -- the server spelled its
    # address -- and is not a STEP: `data-nav` means "a step you can take
    # right now", so an arrow nobody can reach must not carry it. It is
    # hidden as well, because an invisible link over a tenth of the stage
    # would still take the click.
    around = page.locator('[data-nav-wrap="previous"]')
    assert around.count() == 1
    assert not around.is_visible()
    assert page.locator('[data-nav="previous"]').count() == 0
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(300)
    assert _ordinal(page) == 1, "the arrow wrapped with wrap off"


def test_wrap_on_makes_the_arrows_come_round(page: Page, live: Live):
    _arrange(page, wrap=True)
    _open_first(page)
    assert page.get_attribute("[data-viewer]", "data-wrap") == "on"
    page.keyboard.press("ArrowLeft")
    page.wait_for_function(
        f"() => document.querySelector('.viewer-ordinal').textContent.trim().startsWith('{FILES}')", timeout=10_000
    )


def test_the_setting_changes_the_arrows_on_the_open_picture(page: Page, live: Live):
    """No step and no reload: the checkbox writes the workspace and sets
    the attribute the stylesheet reads, so the arrow appears under the
    pointer that was already there."""
    _open_first(page)
    page.click("[data-show-settings-toggle]")
    assert page.get_attribute("[data-viewer]", "data-wrap") == "off"
    page.check("[data-show-wrap]")
    assert page.get_attribute("[data-viewer]", "data-wrap") == "on"
    # visible AND promoted to a step, in the same turn
    assert page.locator('[data-nav-wrap="previous"]').is_visible()
    assert page.locator('[data-nav="previous"]').count() == 1
    assert page.evaluate("() => JSON.parse(localStorage.getItem('sg.workspace.v1')).wrap") is True


def test_escape_stops_the_slideshow_before_it_means_leave(page: Page, live: Live):
    """The outermost thing the viewer is doing. A person reaching for
    Escape while pictures are moving means stop, not leave."""
    _arrange(page, showEvery=8)
    _open_first(page)
    page.click("[data-show-play]")
    page.wait_for_selector('[data-slideshow][data-playing="yes"]', timeout=10_000)
    here = page.url
    page.keyboard.press("Escape")
    page.wait_for_selector('[data-slideshow][data-playing="no"]', timeout=10_000)
    assert page.url == here, "Escape left the picture as well as stopping the walk"


def test_a_picture_with_no_walk_offers_no_slideshow(page: Page, live: Live):
    """A control that does nothing is worse than no control.

    The query defines the walk, so a picture the question does not
    contain has no neighbourhood at all -- no arrows, no filmstrip, and
    nothing to put on a timer. Nothing here is favorited, so this is that
    picture opened inside an answer it is not a member of.
    """
    page.goto("/g?sort=oldest")
    page.wait_for_selector("[data-grid] a.cell", timeout=10_000)
    href = page.get_attribute("[data-grid] a.cell", "href")
    assert href is not None
    page.goto(f"{href.split('?')[0]}?favorite=1")
    page.wait_for_selector("[data-viewer]", timeout=10_000)
    assert page.locator("[data-slideshow]").count() == 0
    assert page.locator("[data-nav]").count() == 0
