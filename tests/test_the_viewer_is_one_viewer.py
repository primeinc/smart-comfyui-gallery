"""The viewer, witnessed in a browser, in BOTH of its containers.

Everything here needs a real one: cursor anchoring is a claim about where
a pixel lands after a transform, promotion is a claim about which bytes an
`<img>` ended up holding, and "a drag released over the backdrop does not
close the overlay" is a claim about pointer capture. None of it can be
decided from source, AST or types, so none of it belongs at a cheaper
layer.

The pairing is the point. Each behaviour is asserted on the page AND on
the gallery overlay, because "one viewer, two containers" is only true if
nobody can quietly fix one of them.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

#: Wider than the preview's 1440 box, so the original has more to give.
BIG = (2400, 1800)
#: Smaller than that box, so `contain` UPSCALED it and the original has less.
SMALL = (320, 240)


def write_library(root) -> None:
    Image.new("RGB", BIG, (30, 90, 160)).save(root / "a_big.png")
    Image.new("RGB", SMALL, (160, 90, 30)).save(root / "b_small.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == 2
    # the stage's arithmetic is built from file.width/height, which ingest
    # is what records -- a scan alone leaves them NULL
    api.post("/jobs/ingest")
    _drained(api)
    if swept["precache"] is not None:
        _settled(api, swept["precache"])


def _settled(api, job_id, timeout=60.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        state = api.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state}"
        time.sleep(0.05)


def _drained(api, timeout=60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [j["id"] for j in api.get("/jobs").json() if j["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(0.05)


def _address(api, name: str) -> str:
    """The library's own answer, by name. `/g/grid` is a fragment whatever
    the Accept says; peek is the typed listing of the same ordering."""
    listed = api.get("/g/peek", params={"page": 1, "count": 9}).json()["items"]
    for row in listed:
        if row["name"] == name:
            return row["slug"]
    raise AssertionError(f"no picture called {name} among {[r['name'] for r in listed]}")


# --- the two containers, opened the two ways --------------------------------


def _open_page(page: Page, live: Live, name: str) -> None:
    """The direct address: a viewer with nothing underneath it."""
    page.goto(f"/i/{_address(live.api, name)}")
    page.wait_for_selector("[data-viewer] [data-stage] img[data-stage-media]", timeout=15_000)
    _painted(page)


def _open_overlay(page: Page, live: Live, name: str) -> None:
    """The mounted gallery: the same viewer over a grid that stays put."""
    slug = _address(live.api, name)
    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell", timeout=15_000)
    page.click(f'[data-grid] a.cell[href^="/i/{slug}"]')
    page.wait_for_selector("[data-lightbox] [data-viewer] img[data-stage-media]", timeout=15_000)
    _painted(page)


def _painted(page: Page) -> None:
    page.wait_for_function(
        "() => { const i = document.querySelector('img[data-stage-media]');"
        " return i && i.complete && i.naturalWidth > 0; }",
        timeout=15_000,
    )


OPENERS = [("page", _open_page), ("overlay", _open_overlay)]


def _box(page: Page):
    return page.evaluate(
        "() => { const r = document.querySelector('img[data-stage-media]').getBoundingClientRect();"
        " return {x: r.x, y: r.y, w: r.width, h: r.height}; }"
    )


def _zoom(page: Page) -> int:
    return int(page.get_attribute("[data-viewer]", "data-zoom") or "0")


def _placement(page: Page):
    """Where the stage and the inspector actually landed."""
    return page.evaluate(
        "() => ({stage: document.querySelector('[data-stage]').getBoundingClientRect().toJSON(),"
        " panel: document.querySelector('[data-inspector-panel]').getBoundingClientRect().toJSON()})"
    )


def _walk(page: Page) -> dict:
    """Whichever ends of the walk this picture has, read without waiting --
    "there is no next" is an answer, not a slow yes."""
    return page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('[data-nav]')]"
        ".map(a => [a.dataset.nav, a.getAttribute('href')]))"
    )


#: `visibility`, not display: the strip keeps its space so hiding it
#: cannot resize the stage the zoom is measured against.
_STRIP_SEEN = "document.querySelector('[data-filmstrip]').checkVisibility({visibilityProperty: true})"


def _strip_shown(page: Page) -> None:
    page.wait_for_function(f"() => {_STRIP_SEEN}", timeout=5_000)


def _strip_hidden(page: Page) -> None:
    page.wait_for_function(f"() => !{_STRIP_SEEN}", timeout=5_000)


def _actual(page: Page) -> None:
    page.keyboard.press("z")
    page.wait_for_function("() => document.querySelector('[data-stage]').dataset.framing === 'actual'")


# --- what the viewer does ---------------------------------------------------


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_the_filmstrip_shows_the_walk_and_marks_where_you_are(page: Page, live: Live, where, open_it, unbroken):
    """Rendered by the server, in answer order, one of them current.

    Nothing in the browser sorts or pages it: the assertion is that the
    DOM order IS the ordinals the server sent, ascending, with no gaps
    invented between them.
    """
    open_it(page, live, "a_big.png")
    strip = page.evaluate(
        "() => [...document.querySelectorAll('[data-filmstrip-item]')].map(a => ({"
        " ordinal: Number(a.dataset.ordinal), href: a.getAttribute('href'),"
        " current: a.getAttribute('aria-current') === 'true' }))"
    )
    assert strip, f"{where}: a library of two has a neighbourhood"
    assert [one["ordinal"] for one in strip] == sorted(one["ordinal"] for one in strip)
    assert [one["ordinal"] for one in strip] == list(range(strip[0]["ordinal"], strip[0]["ordinal"] + len(strip)))
    assert [one["current"] for one in strip].count(True) == 1, "exactly one item is where you are"
    for one in strip:
        assert one["href"].startswith("/i/"), f"{where}: the strip is addresses, not client state"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_clicking_the_filmstrip_walks_the_way_the_arrows_do(page: Page, live: Live, where, open_it, unbroken):
    """One Walk adapter, so the overlay REPLACES its mount and the page
    navigates -- and fifty strip steps stay one Back out of the gallery,
    exactly as fifty arrow presses do."""
    open_it(page, live, "a_big.png")
    depth = page.evaluate("() => history.length")
    was = page.evaluate("() => location.pathname")
    other = page.evaluate(
        "(here) => [...document.querySelectorAll('[data-filmstrip-item]')]"
        ".find(a => a.getAttribute('aria-current') !== 'true')?.getAttribute('href')",
        was,
    )
    assert other, f"{where}: a library of two offers a neighbour to click"

    page.click("[data-filmstrip-item]:not([aria-current='true'])")
    page.wait_for_function("(before) => location.pathname !== before", arg=was, timeout=15_000)
    _painted(page)

    assert page.evaluate("() => location.pathname") == other.split("?")[0]
    assert page.is_visible("[data-viewer]"), f"{where}: the strip lost the viewer"
    if where == "overlay":
        assert page.evaluate("() => history.length") == depth, (
            "an overlay step REPLACES: fifty of them are still one Back out of the gallery"
        )


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_the_filmstrip_stands_down_while_the_picture_is_being_inspected(
    page: Page, live: Live, where, open_it, unbroken
):
    """Local context is for browsing. Zoomed in, the neighbours are not
    what anybody is looking at; lights-out means the photograph alone."""
    open_it(page, live, "a_big.png")
    assert page.is_visible("[data-filmstrip]"), f"{where}: it is there while browsing"

    _actual(page)
    _strip_hidden(page)
    page.keyboard.press("Escape")  # back to fit
    _strip_shown(page)

    page.keyboard.press("l")
    _strip_hidden(page)
    page.keyboard.press("l")
    _strip_shown(page)


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_dragging_across_the_filmstrip_leaves_the_photograph_alone(page: Page, live: Live, where, open_it, unbroken):
    """The strip is outside the stage and scrolls natively. A finger or a
    pointer dragged along it must not reach the picture's own pan/zoom."""
    open_it(page, live, "a_big.png")
    _actual(page)
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelector('[data-stage]').dataset.framing === 'fit'")
    before = _box(page)
    held = _zoom(page)

    box = page.evaluate("() => document.querySelector('[data-filmstrip-track]').getBoundingClientRect().toJSON()")
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + 10, box["y"] + box["height"] / 2, steps=8)
    page.mouse.up()
    page.mouse.wheel(0, -300)
    page.wait_for_timeout(150)

    assert _zoom(page) == held, f"{where}: a wheel over the strip zoomed the picture"
    assert _box(page) == before, f"{where}: dragging the strip moved the picture"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_no_keystroke_means_two_things_at_once(page: Page, live: Live, where, open_it, unbroken):
    """The defect this exists for: the viewer and the authored strip each
    listened to the document, so F was focus AND favorite, 1 was
    actual-pixels AND one star, 0 was fit AND clear-rating -- and every
    one of them fired both, silently rating a photograph somebody was
    only looking at.

    That no key CAN mean two things is proved elsewhere and more cheaply:
    the registry throws on a second claim, so a colliding build fails to
    mount the viewer at all and every other test here goes red, and sglint
    SG503 refuses a module that listens to the document directly. What is
    left for a browser is the half neither can see -- that the keys which
    DID collide now do their authored job and leave the picture alone.
    """
    open_it(page, live, "a_big.png")

    # F is the authored strip's, and must not touch the picture
    before = _zoom(page)
    framing = page.get_attribute("[data-stage]", "data-framing")
    page.keyboard.press("f")
    page.wait_for_function("() => document.querySelector('[data-fav]').getAttribute('aria-pressed') === 'true'")
    assert _zoom(page) == before, f"{where}: favorite changed the zoom"
    assert page.get_attribute("[data-stage]", "data-framing") == framing, f"{where}: favorite reframed the picture"
    assert page.get_attribute("[data-viewer]", "data-chrome") != "focus", f"{where}: favorite entered focus"

    # and 1 is one star, not actual pixels
    page.keyboard.press("1")
    page.wait_for_function("() => document.querySelector('[data-stars]').dataset.rating === '1'")
    assert page.get_attribute("[data-stage]", "data-framing") == framing, f"{where}: one star reframed the picture"
    page.keyboard.press("0")
    page.wait_for_function("() => document.querySelector('[data-stars]').dataset.rating === '0'")
    assert page.get_attribute("[data-stage]", "data-framing") == framing, f"{where}: clearing a rating reframed"
    page.keyboard.press("f")  # leave the library as it was found
    page.wait_for_function("() => document.querySelector('[data-fav]').getAttribute('aria-pressed') === 'false'")


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_every_inspector_section_opens_by_pointer_and_by_keyboard(page: Page, live: Live, where, open_it, unbroken):
    """A heading with `cursor: pointer` and nothing listening is a control
    that looks like one and is not. The sections are `<details>`, so the
    browser owns the disclosure: focusable, Enter and Space, announced.
    Every section is exercised, not a sample -- the bug was that four of
    six were inert while two worked."""
    open_it(page, live, "a_big.png")
    page.keyboard.press("i")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.inspector === 'open'")

    named = page.evaluate(
        "() => [...document.querySelectorAll('[data-inspector-panel] [data-panel]')].map(d => d.dataset.panel)"
    )
    assert len(named) >= 3, f"{where}: the inspector renders its sections: {named}"

    for panel in named:
        held = page.locator(f'[data-panel="{panel}"]')
        summary = held.locator("summary")
        assert summary.count() == 1, f"{where}: {panel} is a real disclosure, not a heading"
        was = held.evaluate("d => d.open")
        summary.click()
        page.wait_for_function(
            "([p, before]) => document.querySelector(`[data-panel='${p}']`).open !== before",
            arg=[panel, was],
            timeout=5_000,
        )
        # and by keyboard, which is the half a click test cannot see
        summary.press("Enter")
        page.wait_for_function(
            "([p, back]) => document.querySelector(`[data-panel='${p}']`).open === back",
            arg=[panel, was],
            timeout=5_000,
        )


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_a_picture_opens_fitted_in_either_container(page: Page, live: Live, where, open_it, unbroken):
    """Fit is where every picture starts, in both presentations -- and
    fitted means inside its stage, not merely small."""
    open_it(page, live, "a_big.png")
    assert _zoom(page) == 100, where
    assert page.get_attribute("[data-stage]", "data-framing") == "fit"
    stage = page.evaluate("() => document.querySelector('[data-stage]').getBoundingClientRect().toJSON()")
    held = _box(page)
    # the computed bound rides along, because "the picture is too big" and
    # "the rule that bounds it never matched" are different bugs
    bound = page.evaluate(
        "() => { const i = document.querySelector('img[data-stage-media]');"
        " const c = getComputedStyle(i); return {maxW: c.maxWidth, maxH: c.maxHeight}; }"
    )
    assert held["w"] <= stage["width"] + 1, f"{where}: picture {held} in stage {stage}, bounded by {bound}"
    assert held["h"] <= stage["height"] + 1, f"{where}: picture {held} in stage {stage}, bounded by {bound}"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_the_wheel_zooms_around_the_pointer(page: Page, live: Live, where, open_it, unbroken):
    """The pixel under the cursor stays under the cursor.

    Anchoring is the whole difference between a zoom and a jump: zoom on
    a face near a corner and a centre-anchored transform throws it off
    screen. Measured as the pointer's position WITHIN the picture, before
    and after -- which is exactly what "stays under" means.
    """
    open_it(page, live, "a_big.png")
    before = _box(page)
    # well off centre, so a centre-anchored transform would move it
    at_x = before["x"] + before["w"] * 0.25
    at_y = before["y"] + before["h"] * 0.25
    held_x = (at_x - before["x"]) / before["w"]
    held_y = (at_y - before["y"]) / before["h"]

    page.mouse.move(at_x, at_y)
    page.mouse.wheel(0, -400)
    page.wait_for_function("(was) => Number(document.querySelector('[data-viewer]').dataset.zoom) > was", arg=100)

    after = _box(page)
    now_x = (at_x - after["x"]) / after["w"]
    now_y = (at_y - after["y"]) / after["h"]
    assert abs(now_x - held_x) < 0.02, f"{where}: the point under the cursor moved horizontally"
    assert abs(now_y - held_y) < 0.02, f"{where}: the point under the cursor moved vertically"


def _wheel(page: Page, modifier: str | None, amount: float) -> None:
    """A wheel gesture over the middle of the picture."""
    at = _box(page)
    page.mouse.move(at["x"] + at["w"] / 2, at["y"] + at["h"] / 2)
    if modifier:
        page.keyboard.down(modifier)
    page.mouse.wheel(0, amount)
    if modifier:
        page.keyboard.up(modifier)


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_the_chosen_modifier_walks_the_library_on_the_wheel(page: Page, live: Live, where, open_it, unbroken):
    """Alt+wheel steps to the next picture instead of zooming.

    The default, and the reason it is alt: a browser already reads
    shift+wheel as horizontal scroll and ctrl+wheel as its own page zoom,
    so alt is the only one nothing else has taken.
    """
    open_it(page, live, "a_big.png")
    assert page.get_attribute("[data-viewer]", "data-wheel-modifier") == "alt", (
        f"{where}: the run's chosen key reaches the viewer"
    )
    walk = _walk(page)
    assert walk, f"{where}: a library of two offers a step"
    was = page.evaluate("() => location.pathname")
    held = _zoom(page)

    _wheel(page, "Alt", 300 if "next" in walk else -300)
    try:
        page.wait_for_function("(before) => location.pathname !== before", arg=was, timeout=15_000)
    except Exception as never:
        # the zoom is the discriminator: unchanged means the modifier WAS
        # honoured and the step itself failed; changed means the modifier
        # never reached the viewer and the wheel fell through to zooming
        raise AssertionError(
            f"{where}: alt+wheel did not walk from {was}; zoom {held} -> {_zoom(page)}, walk {walk}"
        ) from never
    _painted(page)

    assert _zoom(page) == held, f"{where}: the walk zoomed on its way past"
    assert page.is_visible("[data-viewer]"), f"{where}: walking on the wheel lost the viewer"


@pytest.mark.parametrize("modifier", ["Shift", "Control"])
@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_a_gesture_the_viewer_does_not_claim_reaches_the_browser(
    page: Page, live: Live, where, open_it, modifier, unbroken
):
    """The viewer cancels only what it acts on.

    Ctrl+wheel is how a browser delivers a trackpad PINCH, and shift+wheel
    is its horizontal scroll. Neither may walk the library -- pinching a
    photograph must not skip to the next one -- and neither may be
    swallowed either, which is what an unconditional preventDefault at the
    top of the handler did while the setting's own docstring promised the
    opposite.

    `defaultPrevented`, read after the viewer's listener has had the
    event, is the only honest way to ask "did you leave this to me?".
    """
    open_it(page, live, "a_big.png")
    was = page.evaluate("() => location.pathname")
    held = _zoom(page)
    page.evaluate(
        "() => { window.__wheel = null;"
        " document.addEventListener('wheel', e => { window.__wheel = e.defaultPrevented; }, {passive: true}); }"
    )

    _wheel(page, modifier, -300)
    page.wait_for_function("() => window.__wheel !== null", timeout=5_000)

    assert page.evaluate("() => window.__wheel") is False, (
        f"{where}: {modifier}+wheel was cancelled; the viewer claims only a plain wheel and Alt"
    )
    assert page.evaluate("() => location.pathname") == was, f"{where}: {modifier} walked the library"
    assert _zoom(page) == held, f"{where}: {modifier}+wheel zoomed the picture"


@pytest.mark.parametrize(("where", "open_it", "pace_ms"), [(w, o, p) for w, o in OPENERS for p in (0, 90)])
def test_one_flick_of_the_wheel_is_one_picture(page: Page, live: Live, where, open_it, pace_ms, unbroken):
    """A gesture is a run of events, not an event.

    A hard flick or a trackpad swipe is a stream of dozens decaying over
    hundreds of milliseconds. Run at pace 0 that stream arrives as a
    burst; run at 90ms it spans about a second, which is the case a
    cooldown counted from the STEP cannot survive -- it expires while the
    gesture's own inertia is still arriving and walks a second picture.
    The boundary is silence since the last EVENT, so a gesture keeps
    pushing its own boundary ahead of itself.

    The library holds two, so a second step returns to where it started
    and the pathname is what catches it.
    """
    open_it(page, live, "a_big.png")
    walk = _walk(page)
    assert walk, f"{where}: a library of two offers a step"
    was = page.evaluate("() => location.pathname")

    # positioned ONCE and then flicked: re-reading the picture's box
    # between events would evaluate into a navigation already in flight
    at = _box(page)
    page.mouse.move(at["x"] + at["w"] / 2, at["y"] + at["h"] / 2)
    forward = 300 if "next" in walk else -300
    page.keyboard.down("Alt")
    for _ in range(10):
        page.mouse.wheel(0, forward)
        if pace_ms:
            page.wait_for_timeout(pace_ms)
    page.keyboard.up("Alt")
    page.wait_for_function("(before) => location.pathname !== before", arg=was, timeout=15_000)
    landed = page.evaluate("() => location.pathname")

    page.wait_for_timeout(600)  # long enough for a second step to land
    assert page.evaluate("() => location.pathname") == landed, (
        f"{where}: a {pace_ms}ms-paced flick walked more than one picture ({was} -> {landed} -> onwards)"
    )


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_turning_the_wheel_back_is_a_new_gesture(page: Page, live: Live, where, open_it, unbroken):
    """The other half of the boundary rule, stated so it cannot drift.

    Silence ends a gesture -- and so does reversing, immediately, because
    turning the wheel back is somebody correcting themselves and making
    them wait out the inertia of the flick they are undoing would feel
    broken. Without this the test above would pass for a viewer that
    simply refused every second step.
    """
    open_it(page, live, "a_big.png")
    walk = _walk(page)
    assert walk, f"{where}: a library of two offers a step"
    was = page.evaluate("() => location.pathname")

    at = _box(page)
    page.mouse.move(at["x"] + at["w"] / 2, at["y"] + at["h"] / 2)
    forward = 300 if "next" in walk else -300
    page.keyboard.down("Alt")
    page.mouse.wheel(0, forward)
    page.wait_for_function("(before) => location.pathname !== before", arg=was, timeout=15_000)
    there = page.evaluate("() => location.pathname")

    # straight back, with no pause at all: a reversal is its own gesture
    page.mouse.wheel(0, -forward)
    page.keyboard.up("Alt")
    page.wait_for_function("(from) => location.pathname !== from", arg=there, timeout=15_000)
    assert page.evaluate("() => location.pathname") == was, f"{where}: turning back did not return to {was}"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_actual_pixels_are_the_sources_own(page: Page, live: Live, where, open_it, unbroken):
    """`1` means one source pixel per DEVICE pixel, not per CSS pixel.

    The two differ by devicePixelRatio, and a 1:1 that quietly showed
    half the resolution would make the promotion machinery pointless.
    """
    open_it(page, live, "a_big.png")
    _actual(page)
    held = page.evaluate(
        "() => document.querySelector('img[data-stage-media]').getBoundingClientRect().width"
        " * (window.devicePixelRatio || 1)"
    )
    assert abs(held - BIG[0]) < 2, f"{where}: 1:1 showed {held} device pixels for a {BIG[0]}px source"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_zooming_a_large_source_promotes_to_the_original(page: Page, live: Live, where, open_it, unbroken):
    """The preview is enough until it is not, and then the original
    arrives -- without the transform moving."""
    open_it(page, live, "a_big.png")
    assert page.get_attribute("[data-stage]", "data-quality") == "preview"
    assert "/preview/" in (page.get_attribute("img[data-stage-media]", "src") or "")

    _actual(page)
    page.wait_for_function(
        "() => document.querySelector('[data-stage]').dataset.quality === 'original'", timeout=15_000
    )
    assert "/media/" in (page.get_attribute("img[data-stage-media]", "src") or ""), where


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_zooming_a_small_source_does_not_fetch_a_worse_original(page: Page, live: Live, where, open_it, unbroken):
    """The trap the server's arithmetic exists to close.

    The preview is `ImageOps.contain`ed to a 1440 box, and `contain`
    RESIZES rather than shrinks -- so a 320px picture is served as a
    1440px preview. A viewer promoting on "displayed pixels exceed the
    preview's naturalWidth" would fetch the original and get four times
    FEWER pixels. `promotable` is the server's answer; this is the
    control that proves the browser obeys it rather than measuring.
    """
    open_it(page, live, "b_small.png")
    _actual(page)
    page.wait_for_timeout(400)  # every chance to promote, before denying it did
    assert page.get_attribute("[data-stage]", "data-quality") == "preview", where
    assert "/preview/" in (page.get_attribute("img[data-stage-media]", "src") or ""), (
        f"{where}: promoted a {SMALL[0]}px source to bytes smaller than the preview it already had"
    )


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_a_pan_cannot_lose_the_photograph(page: Page, live: Live, where, open_it, unbroken):
    """Dragged hard enough, the picture used to leave the stage entirely
    and there was no way back but Escape. Pan is bounded by the overhang:
    a zoomed picture can be pushed exactly to its own edge and no
    further, so some of it is always on screen."""
    open_it(page, live, "a_big.png")
    _actual(page)
    stage = page.evaluate("() => document.querySelector('[data-stage]').getBoundingClientRect().toJSON()")

    for corner in ((4, 4), (2000, 2000), (4, 2000), (2000, 4)):
        at = _box(page)
        page.mouse.move(at["x"] + at["w"] / 2, at["y"] + at["h"] / 2)
        page.mouse.down()
        page.mouse.move(corner[0], corner[1], steps=8)
        page.mouse.up()
        held = _box(page)
        overlap = {
            "left edge": held["x"] < stage["right"],
            "right edge": held["x"] + held["w"] > stage["left"],
            "top edge": held["y"] < stage["bottom"],
            "bottom edge": held["y"] + held["h"] > stage["top"],
        }
        gone = [edge for edge, on in overlap.items() if not on]
        assert not gone, f"{where}: dragged to {corner} the picture left the stage past its {gone}: {held} vs {stage}"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_actual_pixels_stay_actual_when_the_stage_changes_size(page: Page, live: Live, where, open_it, unbroken):
    """`actual` is a scale computed from the fitted size, and the fitted
    size moves whenever the stage does. Opening the inspector and resizing
    the window both change it, so a picture still labelled 1:1 had quietly
    stopped being 1:1 -- the one thing that label promises."""
    page.set_viewport_size({"width": 1280, "height": 900})
    open_it(page, live, "a_big.png")
    _actual(page)

    def device_pixels() -> float:
        return page.evaluate(
            "() => document.querySelector('img[data-stage-media]').getBoundingClientRect().width"
            " * (window.devicePixelRatio || 1)"
        )

    assert abs(device_pixels() - BIG[0]) < 2, where

    page.keyboard.press("i")  # the inspector takes a column from the stage
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.inspector === 'open'")
    page.wait_for_timeout(200)
    assert abs(device_pixels() - BIG[0]) < 2, f"{where}: opening the inspector broke 1:1"
    assert page.get_attribute("[data-stage]", "data-framing") == "actual", f"{where}: it stopped calling itself actual"

    page.set_viewport_size({"width": 700, "height": 620})  # and the sheet layout
    page.wait_for_timeout(300)
    assert abs(device_pixels() - BIG[0]) < 2, f"{where}: resizing the window broke 1:1"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_a_drag_released_off_the_picture_pans_and_does_not_dismiss(page: Page, live: Live, where, open_it, unbroken):
    """Pointer capture, proven by releasing exactly where dismissal lives.

    The overlay's backdrop click IS Back (frontend/src/overlay.ts), so a
    pan ending outside the picture would close the viewer if the pointer
    were not captured. Released at the corner of the window on purpose.
    """
    open_it(page, live, "a_big.png")
    _actual(page)
    before = _box(page)

    page.mouse.move(before["x"] + before["w"] / 2, before["y"] + before["h"] / 2)
    page.mouse.down()
    page.mouse.move(6, 6, steps=10)  # up into the chrome, off the picture entirely
    page.mouse.up()

    assert page.is_visible("[data-viewer]"), f"{where}: a pan closed the viewer"
    after = _box(page)
    assert (after["x"], after["y"]) != (before["x"], before["y"]), f"{where}: the drag did not pan"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_escape_unwinds_the_viewer_before_it_means_leave(page: Page, live: Live, where, open_it, unbroken):
    """One ladder, both containers: a zoomed picture fits, an open
    inspector closes, and only then does Escape mean "leave"."""
    open_it(page, live, "a_big.png")
    _actual(page)

    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelector('[data-stage]').dataset.framing === 'fit'")
    assert page.is_visible("[data-viewer]"), f"{where}: the first Escape left instead of unwinding the zoom"

    page.keyboard.press("i")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.inspector === 'open'")
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.inspector === 'closed'")
    assert page.is_visible("[data-viewer]"), f"{where}: Escape on an open inspector left the picture"

    page.keyboard.press("Escape")
    page.wait_for_url(lambda url: "/i/" not in url, timeout=15_000)


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_the_inspector_docks_wide_and_sheets_narrow(page: Page, live: Live, where, open_it, unbroken):
    """Placement is geometry's answer, not a preference.

    Nothing in the browser decides this and nothing is stored: the same
    markup and the same open state land beside the picture when there is
    room, and under it when there is not.
    """
    page.set_viewport_size({"width": 1280, "height": 900})
    open_it(page, live, "a_big.png")
    page.keyboard.press("i")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.inspector === 'open'")

    wide = _placement(page)
    assert wide["panel"]["width"] > 0, f"{where}: an open inspector is on screen: {wide}"
    assert wide["panel"]["left"] >= wide["stage"]["right"] - 1, (
        f"{where}: a wide viewer docks the inspector beside the picture: {wide}"
    )

    page.set_viewport_size({"width": 620, "height": 900})
    page.wait_for_timeout(150)  # one layout pass
    narrow = _placement(page)
    assert narrow["panel"]["width"] > 0, f"{where}: the inspector stays open across the reflow: {narrow}"
    assert narrow["panel"]["top"] >= narrow["stage"]["bottom"] - 1, (
        f"{where}: a narrow viewer sheets the inspector under the picture: {narrow}"
    )


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_lights_out_is_a_decision_a_mouse_move_does_not_undo(page: Page, live: Live, where, open_it, unbroken):
    """Resting is the pointer being still; lights out is a decision. A
    mouse move undoes the first and must not undo the second."""
    open_it(page, live, "a_big.png")
    assert page.get_attribute("[data-viewer]", "data-chrome") == "visible"
    page.keyboard.press("l")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.chrome === 'focus'")
    page.mouse.move(300, 300)
    page.wait_for_timeout(200)
    assert page.get_attribute("[data-viewer]", "data-chrome") == "focus", f"{where}: a mouse move cancelled focus"
    page.keyboard.press("l")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.chrome === 'visible'")


def _lights_on_button(page: Page) -> dict:
    """Whether the way out of lights-out can be SEEN, and whether it is in
    the page's tab order -- `visibility`, which `checkVisibility` only
    reports when asked for it."""
    return page.evaluate(
        "() => { const b = document.querySelector('[data-viewer] .viewer-lights-on');"
        " if (!b) return {there: false};"
        " return {there: true, seen: b.checkVisibility({visibilityProperty: true}),"
        "         opacity: Number(getComputedStyle(b).opacity)}; }"
    )


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_lights_out_leaves_a_visible_way_back(page: Page, live: Live, where, open_it, unbroken):
    """L hides every control the viewer has, including the one that turned
    the lights off -- so without this, the only exits are two keys nothing
    on screen names. The button appears only while the lights ARE off, and
    it is a mouse's answer, not a keyboard's: pressed, not typed."""
    open_it(page, live, "a_big.png")
    lit = _lights_on_button(page)
    assert lit["there"], f"{where}: no way out of lights-out is rendered at all"
    assert not lit["seen"], f"{where}: the way out is on screen while the lights are on: {lit}"

    page.keyboard.press("l")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.chrome === 'focus'")
    # it fades in, so the settled value is the claim -- a button that is
    # technically visible at zero opacity is still nothing anybody can see
    page.wait_for_function(
        "() => Number(getComputedStyle(document.querySelector('[data-viewer] .viewer-lights-on')).opacity) > 0",
        timeout=2_000,
    )
    dark = _lights_on_button(page)
    assert dark["seen"], f"{where}: lights out, and nothing on screen says how to get back: {dark}"

    page.click("[data-viewer] .viewer-lights-on")
    page.wait_for_function("() => document.querySelector('[data-viewer]').dataset.chrome !== 'focus'")
    assert not _lights_on_button(page)["seen"], f"{where}: the way out stayed after the lights came on"


@pytest.mark.parametrize(("where", "open_it"), OPENERS)
def test_the_walk_is_the_servers_and_the_viewer_survives_it(page: Page, live: Live, where, open_it, unbroken):
    """Next means what the ResultSet says it means: the arrows are the
    server's addresses, and the viewer never computes an ordering."""
    open_it(page, live, "a_big.png")
    walk = _walk(page)
    assert walk, f"{where}: a library of two offers a step in one direction"
    for href in walk.values():
        assert href.startswith("/i/"), f"{where}: the walk is addresses, not client state"

    was = page.evaluate("() => location.pathname")
    page.keyboard.press("ArrowRight" if "next" in walk else "ArrowLeft")
    page.wait_for_function("(before) => location.pathname !== before", arg=was, timeout=15_000)
    _painted(page)
    assert page.is_visible("[data-viewer]"), f"{where}: walking lost the viewer"
    assert page.get_attribute("[data-stage]", "data-framing") == "fit", (
        f"{where}: the next picture opens fitted, not under the last one's zoom"
    )
