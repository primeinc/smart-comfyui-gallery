"""When this application asks a person something, it asks in its own
words, on its own page.

`window.prompt`, `window.confirm` and `window.alert` were the last
surfaces here that were not this application. They are worse than ugly:
prompt has ONE unlabelled text field and no way to offer choices, so
"which smart collection?" became a comma-joined list of internal slugs
pasted into a sentence, with an instruction to type one of them back --
the application asking a person to remember its own spelling, which is
the one thing it exists to do for them.

The check that matters is negative and it is the first test here:
Playwright's `dialog` event fires for every native box, and nothing may
fire one. A positive test of the new dialog would pass just as happily
with a `window.prompt` still sitting beside it on some other path.

Built on the native <dialog> (frontend/src/ask.ts), so what these tests
DO NOT have to assert is the interesting part: focus, inertness, the top
layer, Escape and the backdrop are the browser's, not ours.
"""

from __future__ import annotations

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

FILES = 4


def write_library(root) -> None:
    for i in range(FILES):
        Image.new("RGB", (8, 8), (40 * i, 90, 140)).save(root / f"p{i}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    api.post(f"/roots/{made['id']}/scan")


@pytest.fixture
def no_native_box(page: Page) -> list[str]:
    """Record every native box the page opens, and dismiss it.

    Dismissing rather than leaving it is what makes the failure legible:
    an unhandled `dialog` blocks the page, so a regression would time
    out somewhere unrelated instead of failing here with what it said.
    """
    opened: list[str] = []

    def caught(dialog) -> None:
        opened.append(f"{dialog.type}: {dialog.message}")
        dialog.dismiss()

    page.on("dialog", caught)
    return opened


def test_no_surface_opens_a_native_box(page: Page, live: Live, no_native_box: list[str]):
    """The whole point, asserted where it can see every path at once."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.click("[data-save-smart]")
    page.wait_for_selector("dialog.ask-box[open]", timeout=10_000)
    page.keyboard.press("Escape")
    page.click("[data-replace-smart]")
    page.wait_for_selector("dialog.ask-box[open]", timeout=10_000)
    page.keyboard.press("Escape")
    assert no_native_box == [], "a native browser box was opened"


def test_naming_a_view_is_a_labelled_field_and_a_named_button(page: Page, live: Live):
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.click("[data-save-smart]")
    box = page.locator("dialog.ask-box[open]")
    box.wait_for(timeout=10_000)
    assert "name this smart collection" in box.inner_text()
    # The field is focused without the test focusing it: `autofocus` on a
    # modal dialog is the browser's job, and a name box that needs a click
    # first is the same rudeness in a nicer frame.
    page.keyboard.type("Blue four")
    box.locator(".ask-take").click()
    page.wait_for_url("**/t/*", timeout=20_000)
    slug = page.url.rsplit("/t/", 1)[1].split("?", 1)[0]
    told = live.api.get(f"/t/{slug}", headers={"accept": "application/json"}).json()
    assert told["name"] == "Blue four"


def test_escape_saves_nothing(page: Page, live: Live):
    """Dismissal is a real answer, and the browser's own Escape is it."""
    before = len(live.api.get("/albums", headers={"accept": "application/json"}).json())
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.click("[data-save-smart]")
    page.wait_for_selector("dialog.ask-box[open]", timeout=10_000)
    page.keyboard.type("never saved")
    page.keyboard.press("Escape")
    page.wait_for_selector("dialog.ask-box", state="detached", timeout=10_000)
    assert page.url.endswith("/g"), "a dismissed dialog navigated somewhere"
    after = live.api.get("/albums", headers={"accept": "application/json"}).json()
    assert len(after) == before
    assert not any(one["name"] == "never saved" for one in after)


def test_enter_in_the_field_saves(page: Page, live: Live):
    """Implicit submission picks the FIRST submit button in tree order,
    which is why the affirmative is written before the dismissal and
    moved after it by the stylesheet. Written the obvious way round,
    Enter would have cancelled."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.click("[data-save-smart]")
    page.wait_for_selector("dialog.ask-box[open]", timeout=10_000)
    page.keyboard.type("Entered")
    page.keyboard.press("Enter")
    page.wait_for_url("**/t/*", timeout=20_000)
    slug = page.url.rsplit("/t/", 1)[1].split("?", 1)[0]
    assert live.api.get(f"/t/{slug}", headers={"accept": "application/json"}).json()["name"] == "Entered"


def test_replacing_a_rule_offers_the_collections_by_name(page: Page, live: Live):
    """The defect this replaced: a prompt whose question CONTAINED the
    comma-joined slugs, with the first one pre-filled, asking a person to
    type one back. Names are what somebody chose; the slug is how the
    application spells it, and belongs underneath as an aside."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.click("[data-save-smart]")
    page.wait_for_selector("dialog.ask-box[open]", timeout=10_000)
    page.keyboard.type("First question")
    page.keyboard.press("Enter")
    page.wait_for_url("**/t/*", timeout=20_000)
    slug = page.url.rsplit("/t/", 1)[1].split("?", 1)[0]

    page.goto("/g?kind=image")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    page.click("[data-replace-smart]")
    box = page.locator("dialog.ask-box[open]")
    box.wait_for(timeout=10_000)
    choice = box.locator(f'.ask-choice[value="{slug}"]')
    assert choice.count() == 1, box.inner_text()
    assert "First question" in choice.inner_text(), "the collection is offered by its slug, not its name"
    # One click IS the choice: there is no select-then-confirm step, so a
    # choice nobody made cannot be submitted.
    choice.click()
    page.wait_for_url(f"**/t/{slug}*", timeout=20_000)
    told = live.api.get(f"/t/{slug}", headers={"accept": "application/json"}).json()
    assert told["rule"] is not None
    assert "kind" in told["rule"]["nl"], told["rule"]["nl"]


def test_a_dialog_owns_the_keyboard_while_it_is_up(page: Page, live: Live):
    """Escape belongs to whatever is in front.

    The keyboard registry (frontend/src/keys.ts) is one document listener
    every surface claims keys from, and it PREVENTS a claimed key. A
    modal dialog closes on Escape as the default action of a close
    request -- so the prevention landed on the dialog, not beside it.

    Measured without the guard rather than reasoned about: the dialog
    stayed open for the full ten seconds while the viewer underneath
    quietly unwound. A modal that cannot be dismissed by the one key
    every modal answers to, sitting over a page acting on the keystroke
    it swallowed."""
    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell", timeout=10_000)
    page.click("[data-grid] a.cell")
    page.wait_for_selector("[data-lightbox-root]:not([hidden])", timeout=10_000)
    opened = page.url
    page.evaluate("""() => {
        const box = document.createElement('dialog');
        box.className = 'ask-box';
        box.innerHTML = '<form method="dialog"><button autofocus value="">cancel</button></form>';
        document.body.append(box);
        box.showModal();
    }""")
    page.keyboard.press("Escape")
    page.wait_for_selector("dialog.ask-box", state="hidden", timeout=10_000)
    assert page.url == opened, "Escape closed the dialog AND the overlay behind it"
    page.wait_for_selector("[data-lightbox-root]:not([hidden])", timeout=5_000)
