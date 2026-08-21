"""The gallery in a real browser: WI-35's acceptance walk, on screen.

One Chromium drives the whole promised path: a sorted question, normal
paging, a far rail jump that walks no intermediate page, a rail preview
holding real members of the exact destination page, opening an item,
and browser-back restoring the state the URL owns. DOM stays bounded:
one page of cells at a time, never the library.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time

import pytest
from PIL import Image

pytestmark = pytest.mark.spawns


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    """150 stills behind the real server, and one Chromium."""
    import os

    import httpx
    from playwright.sync_api import sync_playwright

    tmp = tmp_path_factory.mktemp("grid")
    root = tmp / "lib"
    root.mkdir()
    for i in range(150):
        path = root / f"pic_{i:03d}.png"
        Image.new("RGB", (12, 12), (i % 256, 80, 120)).save(path)
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = (tmp / "server.log").open("wb")
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
            assert web.post(f"/roots/{made['id']}/scan").json()["added"] == 150

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


def test_the_promised_walk_holds_on_screen(driven):
    browser, base = driven
    page = browser.new_page()
    fetched: list[str] = []
    page.on("request", lambda request: fetched.append(request.url))
    try:
        # A sorted question, from the URL alone. size=30 makes five pages,
        # so the rail jump below is a real skip over the middle.
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")
        assert page.locator(".cell").count() == 30, "the DOM holds one page, not the library"
        assert page.get_attribute("[data-grid]", "data-pages") == "5"
        first = page.locator(".cell").first.get_attribute("data-slug")
        assert first is not None
        assert "pic-000" in first, "sort=oldest must lead with the oldest"

        # Normal paging: the htmx swap moves the grid AND the address bar.
        page.click('[data-pager] a[rel="next"]')
        page.wait_for_function("() => document.querySelector('[data-grid]').dataset.page === '2'")
        assert "page=2" in page.url
        ordinal = page.locator(".cell").first.get_attribute("data-ordinal")
        assert ordinal == "31"

        # The rail preview: hover near the bottom shows real members of
        # the page a click there would land on.
        rail = page.locator("[data-rail]")
        box = rail.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] - 2)
        page.wait_for_selector("[data-rail-pop]:not([hidden]) [data-rail-pop-grid] img")
        label = page.text_content("[data-rail-pop-label]")
        assert label is not None
        assert "page 5 of 5" in label
        shown = page.eval_on_selector_all("[data-rail-pop-grid] img", "els => els.map(e => new URL(e.src).pathname)")
        import httpx

        with httpx.Client(base_url=base, timeout=5.0) as web:
            told = web.get("/g/peek", params={"sort": "oldest", "size": 30, "page": 5, "count": 9}).json()
        assert shown == [f"/thumb/{item['slug']}" for item in told["items"]], (
            "the preview must be the exact destination page's members"
        )

        # The far jump: one navigation to page 5, no walk through 3 and 4.
        fetched.clear()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] - 2)
        page.wait_for_function("() => document.querySelector('[data-grid]').dataset.page === '5'")
        assert "page=5" in page.url
        walked = [url for url in fetched if re.search(r"page=[34]\b", url)]
        assert walked == [], f"random access walked prior pages: {walked}"
        last = page.locator(".cell").last.get_attribute("data-slug")
        assert last is not None
        assert "pic-149" in last

        # Opening an item is a navigation to its own address.
        opened = page.locator(".cell").first.get_attribute("href")
        page.click(".cell")
        page.wait_for_url(f"**{opened}")

        # And browser-back restores the state the URL was carrying.
        page.go_back()
        page.wait_for_selector("[data-grid]")
        assert "page=5" in page.url
        assert page.get_attribute("[data-grid]", "data-page") == "5"
        assert page.locator(".cell").count() == 30
    finally:
        page.close()


def test_the_search_form_asks_a_valid_question(driven):
    """Typing a phrase while the sort select still says "newest" must not
    submit the contradiction the seam refuses -- the form re-shapes
    itself into the question the phrase implies (regression: /g?q=banana
    &kind=&sort=newest answered 400 straight from the form)."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g")
        page.wait_for_selector("[data-ask]")
        page.fill('[data-ask] [name="q"]', "banana")
        page.click("[data-ask] button")
        page.wait_for_selector("[data-grid]")
        assert "q=banana" in page.url
        assert "sort=newest" not in page.url
        assert "kind=" not in page.url, "empty fields stay out of the URL"
        # No embeddings exist in this library: the honest answer is a
        # degraded/empty result page, never an error body.
        assert page.locator("[data-degraded], .empty").count() >= 1
        # And the form is still usable after coming back.
        page.go_back()
        page.wait_for_selector("[data-ask]")
        assert page.locator('[data-ask] [name="q"]').is_enabled()
    finally:
        page.close()


def test_a_commit_between_render_and_hover_redraws_instead_of_lying(driven):
    """The cross-request generation gap: the grid renders, a job commits,
    the rail is hovered. The preview request carries the displayed
    currency, the server answers 409, and the page redraws itself from
    the URL -- two generations are never on screen as one answer."""
    import httpx

    browser, base = driven
    page = browser.new_page()
    statuses: list[tuple[str, int]] = []
    page.on("response", lambda response: statuses.append((response.url, response.status)))
    try:
        page.goto(f"{base}/g?size=30")
        page.wait_for_selector("[data-grid]")
        before = page.get_attribute("[data-grid]", "data-currency")
        assert before is not None

        # Any commit moves the library's currency -- an album is cheap.
        with httpx.Client(base_url=base, timeout=5.0) as web:
            assert web.post("/albums", json={"name": "mid-hover"}).status_code == 201

        rail = page.locator("[data-rail]")
        box = rail.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

        page.wait_for_function(
            "(before) => { const g = document.querySelector('[data-grid]');"
            " return g && g.dataset.currency !== before; }",
            arg=before,
            timeout=15_000,
        )
        refused = [status for url, status in statuses if "/g/peek" in url and status == 409]
        assert refused, "the stale preview request was never refused; the redraw happened for another reason"
        after = page.get_attribute("[data-grid]", "data-currency")
        assert after != before
    finally:
        page.close()


def test_the_lightbox_is_an_address_not_a_mode(driven):
    """WI-36's walk: open pushes the item URL once over the mounted
    gallery, every arrow REPLACES it, Back leaves the whole viewer in
    one step, Forward re-opens what the URL names, Escape on a pasted
    standalone page goes to the computed results URL, and a library
    commit mid-walk forces a whole redraw instead of mixing
    generations."""
    import httpx

    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")

        # Open: one PUSH, gallery still mounted underneath.
        page.click('.cell[data-ordinal="1"]')
        page.wait_for_selector("[data-lightbox]")
        assert "/i/pic-000?sort=oldest&size=30" in page.url
        assert page.locator("[data-grid]").count() == 1, "the gallery must stay mounted behind the overlay"

        # Arrows: REPLACE, never another push.
        page.keyboard.press("ArrowRight")
        page.wait_for_function("() => window.location.pathname === '/i/pic-001'")
        page.keyboard.press("ArrowRight")
        page.wait_for_function("() => window.location.pathname === '/i/pic-002'")
        assert "sort=oldest" in page.url, "the walk's context rides every replaced URL"

        # Back: ONE step out of the whole viewer, gallery state intact.
        page.go_back()
        page.wait_for_function("() => window.location.pathname === '/g'")
        page.wait_for_function("() => document.querySelector('[data-lightbox-root]').hidden === true")
        assert "sort=oldest" in page.url

        # Forward: the URL names an item, so the screen shows it again.
        page.go_forward()
        page.wait_for_selector("[data-lightbox]")
        assert "/i/pic-002" in page.url

        # Escape behaves as Back from the overlay.
        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/g'")

        # A pasted item URL is a complete page; Escape there goes to the
        # computed results address, never blindly off-site.
        page.goto(f"{base}/i/pic-100?sort=oldest&size=30")
        page.wait_for_selector(".detail")
        assert page.locator('link[rel="canonical"][href="/i/pic-100"]').count() == 1
        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/g'")
        assert "page=4" in page.url, "ordinal 101 at size 30 returns to its own results page"

        # A commit mid-walk moves the LIBRARY generation but not this
        # answer: the next arrow is refused with 409, the media adapter
        # proves the mounted answer unchanged through locate, adopts the
        # fresh currency, and the walk CONTINUES -- the whole redraw is
        # reserved for an answer that really changed (pinned by the
        # unfiling walk below).
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")
        before = page.get_attribute("[data-grid]", "data-currency")
        page.click('.cell[data-ordinal="1"]')
        page.wait_for_selector("[data-lightbox]")
        with httpx.Client(base_url=base, timeout=5.0) as web:
            assert web.post("/albums", json={"name": "mid-walk"}).status_code == 201
        page.keyboard.press("ArrowRight")
        page.wait_for_function("() => window.location.pathname === '/i/pic-001'")
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').hidden") is False, (
            "an unchanged answer must keep the walk mounted; redraws are for changed answers"
        )
        page.wait_for_function(
            "(before) => document.querySelector('[data-lightbox]').dataset.currency !== before", arg=before
        )
    finally:
        page.close()


def test_clicking_the_backdrop_dismisses_like_back(driven):
    """A click on the dimmed backdrop is the same gesture as Escape: one
    step Back to the mounted gallery. A click on the media itself is
    not a dismissal."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")
        page.click('.cell[data-ordinal="1"]')
        page.wait_for_selector("[data-lightbox]")

        # A click ON the content must not dismiss.
        page.click(".lightbox-media img")
        assert page.locator("[data-lightbox]").count() == 1

        # A click on the backdrop (far left edge, outside the content) is Back.
        size = page.viewport_size
        assert size is not None
        page.mouse.click(4, size["height"] // 2)
        page.wait_for_function("() => window.location.pathname === '/g'")
        page.wait_for_function("() => document.querySelector('[data-lightbox-root]').hidden === true")
    finally:
        page.close()


def test_the_overlay_shell_survives_hostile_navigation(driven):
    """The AddressableOverlay's deep obligations, each phase hostile:
    a slow fragment must not overwrite a newer open (reversed network
    completion); a dismissal mid-flight must bury the late response; a
    modified click is the browser's link, not ours; focus moves into
    the overlay and returns to the trigger; and a failed fragment falls
    back to the exact entity URL as a full page."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")

        # Phase 1: A (slow) then B (fast); A completes last and must land
        # nowhere. One history entry, so one Back leaves the viewer.
        def slowly(route):
            if route.request.header_value("hx-request") == "true":
                time.sleep(0.6)
            route.continue_()

        page.route("**/i/pic-000*", slowly)
        page.click('.cell[data-ordinal="1"]')
        page.click('.cell[data-ordinal="2"]')
        page.wait_for_selector("[data-lightbox]")
        page.wait_for_timeout(900)  # let the slow loser arrive and be discarded
        label = page.text_content(".lightbox-label")
        assert label is not None
        assert "pic_001" in label, "the slow first open overwrote the newer view"
        assert "/i/pic-001" in page.url
        page.unroute("**/i/pic-000*")

        # Focus lives in the overlay while it is open.
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').contains(document.activeElement)")

        # Phase 2: dismissal buries an in-flight arrow fetch.
        page.route("**/i/pic-002*", slowly)
        page.keyboard.press("ArrowRight")  # pic-002, slow
        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/g'")
        page.wait_for_timeout(900)
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').hidden") is True, (
            "a response that lost to a dismissal re-opened the overlay"
        )
        page.unroute("**/i/pic-002*")

        # Focus returned to the element that opened the overlay.
        assert page.evaluate("() => document.activeElement === document.querySelector('.cell[data-ordinal=\"2\"]')")

        # Phase 3: a modified click belongs to the browser.
        page.click('.cell[data-ordinal="3"]', modifiers=["Control"])
        page.wait_for_timeout(300)
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').hidden") is True
        assert page.url.endswith("/g?sort=oldest&size=30")

        # Phase 4: a failed fragment falls back to the address as a page.
        def refuse(route):
            if route.request.header_value("hx-request") == "true":
                route.fulfill(status=500, body="no")
            else:
                route.continue_()

        page.route("**/i/pic-003*", refuse)
        page.click('.cell[data-ordinal="4"]')
        page.wait_for_selector(".detail")
        assert "/i/pic-003" in page.url, "the fallback must land on the exact entity URL"
        page.unroute("**/i/pic-003*")
    finally:
        page.close()


def test_a_losing_failure_lands_nowhere(driven):
    """The Module that makes stale asynchronous SUCCESS harmless must
    make stale asynchronous FAILURE harmless too: a fetch that rejects
    after losing to a newer open or a dismissal must not full-navigate
    the browser to its obsolete address. And generation evidence fails
    CLOSED -- a 200 fragment with no data-currency at all, answering a
    view that expects one, is refused whole rather than mounted."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")

        # A dies late at the transport layer; B wins first. B must stay
        # mounted and the browser must stay at B's address.
        def dies_slowly(route):
            if route.request.header_value("hx-request") == "true":
                time.sleep(0.6)
                route.abort()
            else:
                route.continue_()

        page.route("**/i/pic-000*", dies_slowly)
        page.click('.cell[data-ordinal="1"]')
        page.click('.cell[data-ordinal="2"]')
        page.wait_for_selector("[data-lightbox]")
        page.wait_for_timeout(900)  # A's rejection arrives and must land nowhere
        assert "/i/pic-001" in page.url, "a stale transport failure navigated the whole browser"
        label = page.text_content(".lightbox-label")
        assert label is not None
        assert "pic_001" in label
        page.unroute("**/i/pic-000*")

        # An arrow fetch that rejects AFTER a dismissal: the gallery
        # stays, nothing resurrects, nothing navigates.
        page.route("**/i/pic-002*", dies_slowly)
        page.keyboard.press("ArrowRight")  # pic-002, doomed
        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/g'")
        page.wait_for_timeout(900)
        assert page.evaluate("() => window.location.pathname") == "/g", (
            "a dismissed request's failure dragged the browser to its address"
        )
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').hidden") is True
        page.unroute("**/i/pic-002*")

        # A 200 fragment stripped of its generation, against a view that
        # expects one: full-page fallback, the fragment never mounts.
        def strips_currency(route):
            if route.request.header_value("hx-request") == "true":
                route.fulfill(status=200, body='<div data-lightbox><p class="lightbox-label">forged</p></div>')
            else:
                route.continue_()

        page.route("**/i/pic-004*", strips_currency)
        page.click('.cell[data-ordinal="5"]')
        page.wait_for_selector(".detail")
        assert "/i/pic-004" in page.url
        assert page.evaluate("() => document.body.textContent.includes('forged')") is False, (
            "a fragment that could not prove its generation was mounted anyway"
        )
        page.unroute("**/i/pic-004*")
    finally:
        page.close()


def test_the_folder_page_walks_into_the_gallery(driven):
    """The FolderView on screen: a bounded ResultSet preview (one page,
    never the directory), media links that carry the folder context so
    the arrows walk THE FOLDER, and Escape returning to the folder's
    own results in the gallery."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/f/lib")
        page.wait_for_selector(".grid")
        assert page.locator(".cell").count() == 60, "the preview is one ResultSet page, not the directory"
        first = page.locator(".cell").first
        href = first.get_attribute("href")
        assert href is not None
        assert "?folder=lib" in href, "preview media links must carry the folder context"
        first.click()
        page.wait_for_selector(".detail")
        assert "folder=lib" in page.url

        walk = page.locator('.detail-walk a[rel="next"]')
        onward = walk.get_attribute("href")
        assert onward is not None
        assert "folder=lib" in onward, "the arrows must keep walking the folder question"
        walk.click()
        page.wait_for_selector(".detail")
        assert "folder=lib" in page.url

        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/g'")
        assert "folder=lib" in page.url, "Escape returns to the folder's own results, not the library"
    finally:
        page.close()


def test_a_favorite_does_not_break_the_walk(driven):
    """The answer-identity contract on screen: a favorite commits, the
    library generation moves, but the walked answer did not -- so the
    strip adopts the new currency in place and the NEXT ARROW still
    walks the lightbox instead of 409-redrawing over an unchanged
    answer."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")
        before = page.get_attribute("[data-grid]", "data-currency")
        page.click('.cell[data-ordinal="1"]')
        page.wait_for_selector("[data-lightbox] [data-authored]")

        page.click("[data-lightbox] [data-fav]")
        page.wait_for_function(
            "() => document.querySelector('[data-lightbox] [data-fav]').getAttribute('aria-pressed') === 'true'"
        )
        # The commit moved the generation; the strip must have adopted it.
        page.wait_for_function(
            "(before) => document.querySelector('[data-lightbox]').dataset.currency !== before", arg=before
        )

        page.keyboard.press("ArrowRight")
        page.wait_for_function("() => window.location.pathname === '/i/pic-001'")
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').hidden") is False, (
            "the favorite broke the walk: the arrow forced a full redraw over an unchanged answer"
        )
        assert page.locator("[data-grid]").count() == 1, "the gallery must still be mounted underneath"
    finally:
        page.close()


def test_unfiling_the_walked_item_redraws_the_walk(driven):
    """The opposite case: removing the item from the album being walked
    really changes the answer, and the strip's coherence check redraws
    whole instead of leaving arrows that claim a membership the
    database no longer holds."""
    import httpx

    browser, base = driven
    with httpx.Client(base_url=base, timeout=5.0) as web:
        assert web.post("/albums", json={"name": "Walkout"}).json()["slug"] == "walkout"
        for slug in ("pic-010", "pic-011"):
            assert web.post("/t/walkout/add", json={"file": slug}).status_code < 300

    page = browser.new_page()
    try:
        page.goto(f"{base}/g?album=walkout")
        page.wait_for_selector("[data-grid]")
        page.click(".cell")  # newest member first
        page.wait_for_selector("[data-lightbox] [data-authored]")

        page.click("[data-lightbox] [data-album-picker]")
        page.wait_for_selector("[data-lightbox] [data-album-choices]:not([hidden]) input")
        boxes = page.locator("[data-lightbox] [data-album-choices] label", has_text="Walkout")
        boxes.locator("input").uncheck()

        # The answer really changed: the coherence check redraws whole,
        # and the URL owns what renders -- the item's full page.
        page.wait_for_selector(".detail", timeout=15_000)
        assert page.locator("[data-lightbox-root]").count() == 0
        assert "/i/pic-011" in page.url
    finally:
        page.close()


def test_walking_your_own_judgement_curates_itself(driven):
    """The authored facets under the answer-identity contract, on
    screen: raising a rating inside a rating_min walk preserves the
    answer and the walk continues; dropping below the threshold, or
    unfavoriting inside favorite=1, really changes the answer and
    redraws whole."""
    import httpx

    browser, base = driven
    with httpx.Client(base_url=base, timeout=5.0) as web:
        for slug in ("pic-030", "pic-031"):
            assert web.post(f"/i/{slug}/rating", json={"value": 4}).status_code < 300
        for slug in ("pic-040", "pic-041"):
            assert web.post(f"/i/{slug}/favorite", json={"value": True}).status_code < 300

    page = browser.new_page()
    try:
        # rating_min=4 walk: 4 -> 5 stars moves the generation, not the
        # membership -- the strip adopts and the arrow keeps walking.
        page.goto(f"{base}/g?rating_min=4")
        page.wait_for_selector("[data-grid]")
        assert page.locator(".cell").count() == 2
        page.click(".cell")  # pic-031, the newer
        page.wait_for_selector("[data-lightbox] [data-authored]")
        page.keyboard.press("5")
        page.wait_for_function("() => document.querySelector('[data-lightbox] [data-stars]').dataset.rating === '5'")
        page.keyboard.press("ArrowRight")
        page.wait_for_function("() => window.location.pathname === '/i/pic-030'")
        assert page.evaluate("() => document.querySelector('[data-lightbox-root]').hidden") is False, (
            "a rating raise kept the same answer; the walk must continue in the overlay"
        )

        # 4 -> 3 stars drops pic-030 below the threshold: the answer
        # really changed, and the coherence check redraws whole.
        page.keyboard.press("3")
        page.wait_for_selector(".detail", timeout=15_000)
        assert "/i/pic-030" in page.url

        # favorite=1 walk: unfavoriting the open item leaves the walked
        # answer -- same contract, other facet.
        page.goto(f"{base}/g?favorite=1")
        page.wait_for_selector("[data-grid]")
        page.click(".cell")  # pic-041, the newer favorite
        page.wait_for_selector("[data-lightbox] [data-authored]")
        page.click("[data-lightbox] [data-fav]")
        page.wait_for_selector(".detail", timeout=15_000)
        assert page.locator("[data-lightbox-root]").count() == 0
        assert "/i/pic-041" in page.url
    finally:
        page.close()


def test_saving_the_view_walks_to_its_smart_collection(driven):
    """The product payoff on screen: any gallery question becomes a
    smart collection with one button, and the saved address answers
    with the same membership the view showed."""
    import httpx

    browser, base = driven
    with httpx.Client(base_url=base, timeout=5.0) as web:
        assert web.post("/i/pic-050/rating", json={"value": 5}).status_code < 300
        assert web.post("/i/pic-050/favorite", json={"value": True}).status_code < 300

    page = browser.new_page()
    page.on("dialog", lambda dialog: dialog.accept("Five star"))
    try:
        # favorite AND five stars names exactly one file, whatever the
        # other walks on this shared server have judged.
        page.goto(f"{base}/g?rating_min=5&favorite=1")
        page.wait_for_selector("[data-grid]")
        assert page.locator(".cell").count() == 1
        page.click("[data-save-smart]")
        page.wait_for_url("**/t/five-star")
        page.wait_for_selector(".grid .cell")
        assert page.locator(".cell").count() == 1
        assert 'data-slug="pic-050"' in page.content()
    finally:
        page.close()


def test_a_selection_spans_pages_and_curates_as_one(driven):
    """Selection belongs to an answer: it survives page swaps while the
    grid's answer is unchanged, spans pages, and one toolbar action
    curates the whole set in one request. An answer-preserving write
    keeps the selection mounted; a write the question depends on clears
    it and redraws; a different question mounting clears it outright."""
    browser, base = driven
    page = browser.new_page()
    try:
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")
        page.check('.cell-shell:has(.cell[data-ordinal="1"]) [data-pick]')
        page.check('.cell-shell:has(.cell[data-ordinal="2"]) [data-pick]')
        page.wait_for_selector("[data-curate]:not([hidden])")
        assert page.text_content("[data-curate-count]") == "2 selected"

        # Page 2 of the SAME answer: the selection survives the swap.
        page.click('[data-pager] a[rel="next"]')
        page.wait_for_function("() => document.querySelector('[data-grid]').dataset.page === '2'")
        assert page.text_content("[data-curate-count]") == "2 selected"
        page.check('.cell-shell:has(.cell[data-ordinal="31"]) [data-pick]')
        assert page.text_content("[data-curate-count]") == "3 selected"

        # One answer-preserving write: three files, one request, and the
        # selection stays mounted for the next operation.
        before = page.get_attribute("[data-grid]", "data-currency")
        page.click('[data-curate] [data-bulk-favorite="1"]')
        page.wait_for_function(
            "(before) => document.querySelector('[data-grid]').dataset.currency !== before", arg=before
        )
        assert page.text_content("[data-curate-count]") == "3 selected"

        # A different answer mounting clears the selection.
        page.goto(f"{base}/g?favorite=1&size=30")
        page.wait_for_selector("[data-grid]")
        assert page.locator(".cell").count() >= 3
        assert page.evaluate("() => document.querySelector('[data-curate]').hidden") is True

        # And a write THIS question depends on: un-favoriting inside
        # favorite=1 changes the answer, clears, and redraws whole.
        shown = page.locator(".cell").count()
        page.check(".cell-shell [data-pick]")
        page.wait_for_selector("[data-curate]:not([hidden])")
        page.click('[data-curate] [data-bulk-favorite="0"]')
        page.wait_for_function(
            "(shown) => document.querySelectorAll('.cell').length === shown - 1", arg=shown, timeout=15_000
        )
        assert page.evaluate("() => document.querySelector('[data-curate]').hidden") is True
    finally:
        page.close()


def test_the_navigation_indexes_walk_into_the_entities(driven):
    """The two front doors on screen: /folders enters the physical axis
    through a root shelf's folder entities, /albums enters the authored
    axis through the tree -- each click landing on the entity's own
    address, no ids and no paths anywhere on the way."""
    import httpx

    browser, base = driven
    with httpx.Client(base_url=base, timeout=5.0) as web:
        assert web.post("/albums", json={"name": "Shelf"}).json()["slug"] == "shelf"

    page = browser.new_page()
    try:
        page.goto(f"{base}/folders")
        page.wait_for_selector('[data-folder="lib"]')
        page.click('[data-folder="lib"]')
        page.wait_for_selector(".grid")
        assert page.url.endswith("/f/lib")

        page.goto(f"{base}/albums")
        page.wait_for_selector('[data-album="shelf"]')
        page.click('[data-album="shelf"]')
        page.wait_for_selector(".detail")
        assert page.url.endswith("/t/shelf")
    finally:
        page.close()


def test_the_album_page_walks_into_the_gallery(driven):
    """The CollectionView on screen: the authored membership as a
    bounded grid whose media links carry the album context, and Escape
    landing back on the album's results in the gallery."""
    import httpx

    browser, base = driven
    with httpx.Client(base_url=base, timeout=5.0) as web:
        assert web.post("/albums", json={"name": "Walk"}).json()["slug"] == "walk"
        for slug in ("pic-000", "pic-001"):
            assert web.post("/t/walk/add", json={"file": slug}).status_code < 300

    page = browser.new_page()
    try:
        page.goto(f"{base}/t/walk")
        page.wait_for_selector(".grid")
        assert page.locator(".cell").count() == 2
        href = page.locator(".cell").first.get_attribute("href")
        assert href is not None
        assert "?album=walk" in href, "preview media links must carry the album context"
        page.click(".cell")
        page.wait_for_selector(".detail")
        assert "album=walk" in page.url

        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/g'")
        assert "album=walk" in page.url, "Escape returns to the album's own results, not the library"
    finally:
        page.close()
