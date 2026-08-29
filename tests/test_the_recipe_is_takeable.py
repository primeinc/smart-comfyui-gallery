"""The generation recipe, witnessed in a browser.

The one question a person has about a generated picture is "how do I make
this again, or make it slightly differently". Answering it means two
things this suite can only check with a real browser:

  * what lands on the CLIPBOARD -- a copy button is a claim about the
    system clipboard, not about the DOM it read
  * that the editable fields are a SCRATCH copy -- a caret, a selection
    and a delete that leave nothing behind, on this picture or the next

The recipe fixture is the A1111 infotext from tests/test_metaparse.py,
which mirrors the format reference cited in metaparse/adapters.py. It has
two LoRAs at different weights on purpose: a LoRA copied without its
weight is the field people report missing from every other gallery's copy
button, and it is the one this panel exists to carry.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from playwright.sync_api import Page

from tests.conftest import POLL, Live

pytestmark = pytest.mark.slow

PROMPT = "a castle on a hill <lora:castleLora:0.8> <lora:filmGrain:0.35>"
NEGATIVE = "blurry, ugly"
INFOTEXT = (
    f"{PROMPT}\n"
    f"Negative prompt: {NEGATIVE}\n"
    "Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, "
    "Seed: 12345, Size: 512x768, Model hash: abc123def, Model: dreamshaper_8, "
    'Denoising strength: 0.4, Clip skip: 2, Lora hashes: "castleLora: deadbeef", '
    "Version: v1.10.1"
)
OTHER = "a lighthouse at dawn"
OTHER_INFOTEXT = f"{OTHER}\nNegative prompt: fog\nSteps: 8, Sampler: Euler, CFG scale: 3, Seed: 777"


def _generated(path, infotext: str) -> None:
    info = PngInfo()
    info.add_text("parameters", infotext)
    Image.new("RGB", (320, 240), (40, 80, 140)).save(path, pnginfo=info)


def write_library(root) -> None:
    _generated(root / "a_castle.png", INFOTEXT)
    _generated(root / "b_lighthouse.png", OTHER_INFOTEXT)


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == 2
    # the recipe is what INGEST reads out of the tEXt chunk; a scan alone
    # records that there is a file
    api.post("/jobs/ingest")
    _drained(api)


def _drained(api, timeout=60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [j["id"] for j in api.get("/jobs").json() if j["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(POLL)


def _address(api, name: str) -> str:
    for row in api.get("/g/peek", params={"page": 1, "count": 9}).json()["items"]:
        if row["name"] == name:
            return row["slug"]
    raise AssertionError(f"no picture called {name}")


def _open_recipe(page: Page, live: Live, name: str = "a_castle.png") -> None:
    """The viewer, with the recipe disclosed and its fields laid out.

    The panel is opened through the UI rather than by setting `open`:
    sizing the scratch fields happens when they are first laid out, and a
    field grown by a test that reached past the disclosure would prove
    nothing about the one a person sees.
    """
    page.goto(f"/i/{_address(live.api, name)}")
    page.wait_for_selector("[data-viewer] [data-stage] img[data-stage-media]", timeout=15_000)
    if page.get_attribute("[data-viewer]", "data-inspector") != "open":
        page.keyboard.press("i")
    page.wait_for_selector("[data-recipe]", state="visible", timeout=5_000)
    panel = page.locator("[data-recipe]")
    if not panel.evaluate("p => p.open"):
        panel.locator("summary").click()
    page.wait_for_selector("[data-recipe-field='prompt'] [data-scratch]", state="visible", timeout=5_000)


def _clipboard(page: Page) -> str:
    return page.evaluate("() => navigator.clipboard.readText()")


#: Parked on the clipboard before a copy, so the value read back after
#: it is known to be that press and not the one before it.
UNCOPIED = "nothing-copied-yet"


def _copy(page: Page, selector: str) -> str:
    """Press a copy button and read what it put on the clipboard.

    The sentinel is what makes a second copy in the same test a real
    wait rather than a no-op reading the previous answer. The button's
    own `data-done` says the same thing a moment later, but it stays lit
    for 1200ms (frontend/src/recipe.ts `copied`), so waiting for it to
    clear before pressing again bought that flash on every copy after
    the first. `copied` has no guard on the flag -- a press during it
    writes like any other -- and the clipboard is the thing being
    claimed about anyway.
    """
    page.evaluate("(idle) => navigator.clipboard.writeText(idle)", UNCOPIED)
    page.click(selector)
    page.wait_for_function(
        "(idle) => navigator.clipboard.readText().then(text => text !== idle)",
        arg=UNCOPIED,
        timeout=5_000,
    )
    return _clipboard(page)


@pytest.fixture
def clipboard(page: Page, live: Live):
    """The system clipboard, readable. Without this the page can write and
    nothing can check what it wrote."""
    page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=live.url)
    return page


def test_the_recipe_leads_with_the_prompt_and_keeps_the_weights(page: Page, live: Live, unbroken):
    """What is on screen: the prompt at full height, and every LoRA with
    the number that makes it reproduce."""
    _open_recipe(page, live)
    assert page.input_value("[data-recipe-field='prompt'] [data-scratch]") == PROMPT
    assert page.input_value("[data-recipe-field='negative'] [data-scratch]") == NEGATIVE

    # sized to its content rather than left at the one row the markup asks
    # for: a prompt behind a scrollbar is a prompt nobody reads
    lines = page.evaluate(
        "() => { const f = document.querySelector(\"[data-recipe-field='prompt'] [data-scratch]\");"
        " return {height: f.getBoundingClientRect().height, scroll: f.scrollHeight}; }"
    )
    assert lines["height"] >= lines["scroll"] - 1, f"the prompt is scrollable at {lines}"

    weighted = page.evaluate(
        "() => [...document.querySelectorAll('.recipe-lora')].map(row => ({"
        " name: row.querySelector('span:not(.recipe-label)').textContent.trim(),"
        " weight: row.querySelector('.recipe-weight')?.textContent.trim() }))"
    )
    assert {one["name"]: one["weight"] for one in weighted} == {"castleLora": "0.80", "filmGrain": "0.35"}


def test_copy_everything_is_enough_to_make_the_picture_again(clipboard: Page, live: Live, unbroken):
    """The whole recipe, in the shape the tools that read these already
    use. This is the assertion that a copy button which "looks like it
    worked" is not enough."""
    page = clipboard
    _open_recipe(page, live)
    text = _copy(page, "[data-copy-all]")

    assert text.startswith(PROMPT), text
    assert f"Negative prompt: {NEGATIVE}" in text, text
    for pair in ("Seed: 12345", "Steps: 20", "CFG scale: 7", "Sampler: Euler a", "Model: dreamshaper_8"):
        assert pair in text, f"{pair!r} missing from:\n{text}"
    # the LoRAs are already inline in this tool's prompt, so they are NOT
    # listed a second time -- pasting that would apply each one twice
    assert "Loras:" not in text, text

    # the control for that absence, on the same panel: take the inline tags out
    # of the prompt and they come back as a listed pair, with their weights.
    # Without it the assertion above would pass on dead LoRA emission too.
    page.fill("[data-recipe-field='prompt'] [data-scratch]", "a castle on a hill")
    listed = _copy(page, "[data-copy-all]")
    assert "Loras: <lora:castleLora:0.80> <lora:filmGrain:0.35>" in listed, listed


def test_one_field_copies_what_is_on_screen_including_an_edit(clipboard: Page, live: Live, unbroken):
    """A scratch edit is the thing somebody meant to take. The button
    copies the field, not the file."""
    page = clipboard
    _open_recipe(page, live)
    field = page.locator("[data-recipe-field='prompt'] [data-scratch]")
    field.fill("a castle on a hill at night")
    assert _copy(page, "[data-recipe-field='prompt'] [data-copy]") == "a castle on a hill at night"


def test_an_edit_is_scratch_and_survives_nothing(page: Page, live: Live, unbroken):
    """Empty the prompt entirely, walk to the next picture and back: the
    file's own text is there. What a file says it is is a fact, not a
    preference -- only the ARRANGEMENT of the panels is remembered."""
    _open_recipe(page, live)
    page.fill("[data-recipe-field='prompt'] [data-scratch]", "")
    revert = page.locator("[data-recipe-field='prompt'] [data-revert]")
    assert revert.is_visible(), "the way back appears once there is something to go back from"

    revert.click()
    assert page.input_value("[data-recipe-field='prompt'] [data-scratch]") == PROMPT
    assert not revert.is_visible(), "and goes away again"

    page.fill("[data-recipe-field='prompt'] [data-scratch]", "gone")
    _open_recipe(page, live, "b_lighthouse.png")
    assert page.input_value("[data-recipe-field='prompt'] [data-scratch]") == OTHER
    _open_recipe(page, live)
    assert page.input_value("[data-recipe-field='prompt'] [data-scratch]") == PROMPT
