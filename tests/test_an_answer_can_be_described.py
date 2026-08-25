"""Understanding, in a browser: one answer, described rather than shown.

The gallery could produce a result set and then offered exactly one
thing to do with it -- look at thumbnails. "Show me these, and tell me
which prompts and which LoRAs made them" had nowhere to be asked.

Three claims here, and the first is the one the whole design rests on.

**Gallery and Analyze describe the SAME members.** They are two
presentations of one question, and `view` never reaches the
GalleryQuery. If switching could move the total, an analysis would be a
report about a different library than the one on screen -- which is how
one surface says 412 and another says 407.

**Every number is a question.** A count that cannot be clicked back into
the query is a dashboard. Each row is an ordinary link that adds its own
clause, so refining is navigation and works with the middle mouse
button.

**Exact prompts, counted, copyable.** `prompt.text_hash` is a real
identity, so these are counts and not estimates -- and the point of
seeing them is to take one somewhere else.

Cross-media on purpose: the library holds generated stills, a
photograph and a real video, because an analysis proved only on
generated images is an analysis that will surprise somebody.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

CASTLE = "a castle on a hill"
LIGHTHOUSE = "a lighthouse at dawn"

#: Four of one recipe, two of another. With one of each, every wrong
#: count is still 1 and nothing is proved.
MADE = [
    *[("dreamshaper_8", "filmGrain", "0.35", "Euler a", CASTLE)] * 4,
    *[("juggernautXL", "detailTweaker", "0.80", "DPM++ 2M", LIGHTHOUSE)] * 2,
]


def write_library(root) -> None:
    for i, (checkpoint, lora, weight, sampler, prompt) in enumerate(MADE):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"{prompt} <lora:{lora}:{weight}>\n"
            "Negative prompt: blurry\n"
            f"Steps: 28, Sampler: {sampler}, CFG scale: 7, Seed: 4242, Size: 832x1216, "
            f"Model: {checkpoint}",
        )
        Image.new("RGB", (64, 48), (20 + i * 7, 60, 90)).save(root / f"made_{i:02d}.png", pnginfo=info)
    Image.new("RGB", (64, 48), (10, 120, 10)).save(root / "taken.png")

    import av

    with av.open(str(root / "clip.mp4"), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), (0, 0, 255), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


WHOLE = len(MADE) + 2


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == WHOLE
    api.post("/jobs/ingest")
    _drained(api)


def _drained(api, timeout=90.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [j["id"] for j in api.get("/jobs").json() if j["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(0.05)


def _analyze(page: Page, question: str = "") -> None:
    page.goto(f"/g?{question}&view=analyze" if question else "/g?view=analyze")
    page.wait_for_selector("[data-analyze]", timeout=15_000)


def _total(page: Page) -> str:
    return page.inner_text("[data-total-count]").strip()


def _bars(page: Page, key: str) -> dict[str, int]:
    return page.evaluate(
        "key => Object.fromEntries("
        '[...document.querySelectorAll(`[data-breakdown="${key}"] .analyze-bar`)].map(bar => ['
        " bar.querySelector('.analyze-bar-label').textContent.trim(),"
        " Number(bar.querySelector('.analyze-bar-count').textContent)]))",
        key,
    )


def _prompts(page: Page) -> dict[str, int]:
    return page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('.prompt-use')].map(one => ["
        " one.querySelector('[data-prompt-text]').textContent.trim(),"
        " Number(one.querySelector('.prompt-use-count').textContent)]))"
    )


# --- one question, two presentations ----------------------------------------


def test_switching_presentation_does_not_change_the_answer(page: Page, live: Live, unbroken):
    """The claim the whole design rests on. `view` never reaches the
    GalleryQuery, so the membership and the total are untouched."""
    page.goto("/g?f=has.generation%3Aeq%3A1")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    grid_total = _total(page)
    cells = page.evaluate("() => document.querySelectorAll('[data-grid] a.cell').length")
    assert cells == len(MADE)

    page.click('[data-view="analyze"]')
    page.wait_for_selector("[data-analyze]", timeout=15_000)
    assert _total(page) == grid_total, "the analysis describes the answer it is standing on"
    assert page.locator('[data-view="analyze"]').get_attribute("aria-current") == "page"
    # and the chips -- the question itself -- are untouched
    assert page.evaluate("() => document.querySelectorAll('[data-chip-edit]').length") == 1

    page.click('[data-view="gallery"]')
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert _total(page) == grid_total


def test_the_door_to_it_is_visible(page: Page, live: Live, unbroken):
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert page.locator('[data-view="analyze"]').is_visible(), (
        "a presentation reachable only by typing a URL parameter is a presentation for whoever wrote it"
    )


# --- what it says -----------------------------------------------------------


def test_it_says_which_prompts_made_these(page: Page, live: Live, unbroken):
    """Exact prompts, counted. Two files carrying one prompt share one
    `prompt` row, so this is a count and not an estimate."""
    _analyze(page, "f=has.generation%3Aeq%3A1")
    said = _prompts(page)
    assert said == {
        f"{CASTLE} <lora:filmGrain:0.35>": 4,
        f"{LIGHTHOUSE} <lora:detailTweaker:0.80>": 2,
    }, said


def test_it_says_which_loras_and_at_what_strength(page: Page, live: Live, unbroken):
    """A LoRA without its weight does not reproduce the picture, and it
    is the field people report missing from every other gallery."""
    _analyze(page, "f=has.generation%3Aeq%3A1")
    rows = page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('[data-analyze-loras] tbody tr')].map(tr => ["
        " tr.dataset.lora, [...tr.querySelectorAll('td')].slice(1, 3).map(td => td.textContent.trim())]))"
    )
    assert rows == {"filmGrain": ["4", "0.35"], "detailTweaker": ["2", "0.80"]}, rows


def test_it_breaks_the_answer_down_and_the_shares_add_up(page: Page, live: Live, unbroken):
    _analyze(page, "f=has.generation%3Aeq%3A1")
    assert _bars(page, "generation.checkpoint") == {"dreamshaper_8": 4, "juggernautXL": 2}
    assert _bars(page, "generation.sampler") == {"Euler a": 4, "DPM++ 2M": 2}

    shares = page.evaluate(
        "() => [...document.querySelectorAll('[data-breakdown=\"generation.checkpoint\"] .analyze-bar-share')]"
        ".map(s => parseFloat(s.textContent))"
    )
    assert sum(shares) == pytest.approx(100.0, abs=0.2), shares


def test_a_breakdown_says_how_much_of_the_answer_it_covers(page: Page, live: Live, unbroken):
    """ "18 of 684 have a camera" and "18 of 18 are a Canon" are
    different sentences, and a share drawn against the wrong one is a bar
    that lies quietly."""
    _analyze(page)
    covered = page.inner_text('[data-breakdown="generation.checkpoint"] .analyze-of').strip()
    assert covered == f"{len(MADE)} of {WHOLE}", covered


# --- every number is a question ---------------------------------------------


def test_clicking_a_bar_narrows_the_question(page: Page, live: Live, unbroken):
    """A count that cannot be clicked back into the query is a
    dashboard, and a dashboard is where data goes to be looked at
    instead of used."""
    _analyze(page)
    page.click('[data-breakdown="generation.checkpoint"] .analyze-bar a')
    page.wait_for_selector("[data-analyze]", timeout=15_000)

    assert "generation.checkpoint" in page.url, page.url
    assert "view=analyze" in page.url, "refining stays in the presentation it was refined from"
    assert _total(page) == "4 results"
    assert page.evaluate("() => document.querySelectorAll('[data-chip-edit]').length") == 1
    # and the answer it now describes is the narrowed one
    assert _prompts(page) == {f"{CASTLE} <lora:filmGrain:0.35>": 4}


def test_clicking_a_lora_narrows_to_the_files_that_used_it(page: Page, live: Live, unbroken):
    _analyze(page)
    page.click('[data-analyze-loras] tr[data-lora="detailTweaker"] a')
    page.wait_for_selector("[data-analyze]", timeout=15_000)
    assert _total(page) == "2 results"
    assert page.evaluate("() => [...document.querySelectorAll('[data-chip-edit]')].map(c => c.textContent.trim())") == [
        "LoRA detailTweaker"
    ]


def test_a_prompt_can_be_taken_somewhere_else(page: Page, live: Live, unbroken):
    """Seeing the prompt is half of it. The point is to use it."""
    page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=live.url)
    _analyze(page, "f=has.generation%3Aeq%3A1")
    page.click(".prompt-use:first-child [data-copy-prompt]")
    page.wait_for_function(
        "() => document.querySelector('.prompt-use:first-child [data-copy-prompt]').dataset.done === 'true'",
        timeout=5_000,
    )
    assert page.evaluate("() => navigator.clipboard.readText()") == f"{CASTLE} <lora:filmGrain:0.35>"


# --- cross-media ------------------------------------------------------------


def test_an_answer_of_mixed_media_is_described_as_one(page: Page, live: Live, unbroken):
    """Not a generated-image feature. The whole library, including a
    clip and a photograph, is one answer with one description."""
    _analyze(page)
    assert _total(page) == f"{WHOLE} results"
    assert _bars(page, "media.kind") == {"image": WHOLE - 1, "video": 1}
    assert _bars(page, "has.generation") == {"yes": len(MADE), "no": 2}


def test_describing_a_medium_with_nothing_to_say_says_nothing(page: Page, live: Live, unbroken):
    """An honest empty, rather than a page of bars at 100%.

    The clip in THIS library carries no ComfyUI tags, so it has no
    recipe to describe -- and a one-member answer has nothing to break
    down either: every dimension it does carry would read "video, 1,
    100%", which is the question repeated back as an answer.

    Both halves are deliberate. A breakdown whose single value covers
    the whole answer is dropped, so a narrow question does not produce a
    wall of full bars saying what the chips already say.
    """
    _analyze(page, "kind=video")
    assert _total(page) == "1 result"
    assert _prompts(page) == {}
    assert page.locator("[data-analyze-loras]").count() == 0
    assert _bars(page, "media.kind") == {}, "one video, described as 'video, 100%', is the question read back"
    assert page.locator(".analyze-bar").count() == 0

    # the control: the SAME surface over an answer that does hold a
    # spread has plenty to say, so the emptiness above is about this
    # answer and not about the analysis being broken
    _analyze(page)
    assert page.locator(".analyze-bar").count() > 0


# --- the third presentation: the same answer as facts ------------------------


def _table(page: Page, question: str = "") -> None:
    page.goto(f"/g?{question}&view=table" if question else "/g?view=table")
    page.wait_for_selector("[data-table]", timeout=15_000)


def test_the_table_shows_the_same_members_as_the_grid(page: Page, live: Live, unbroken):
    """Three presentations, one membership. `view` never reaches the
    GalleryQuery, so all three have to agree about the total."""
    page.goto("/g?f=has.generation%3Aeq%3A1")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    was = _total(page)

    page.click('[data-view="table"]')
    page.wait_for_selector("[data-table]", timeout=15_000)
    assert _total(page) == was
    rows = page.evaluate("() => document.querySelectorAll('[data-table] tbody tr').length")
    assert rows == len(MADE)

    page.click('[data-view="analyze"]')
    page.wait_for_selector("[data-analyze]", timeout=15_000)
    assert _total(page) == was, "and the third agrees with the other two"


def test_the_table_carries_what_a_grid_cell_deliberately_does_not(page: Page, live: Live, unbroken):
    _table(page, "f=has.generation%3Aeq%3A1")
    first = page.evaluate(
        "() => Object.fromEntries([...document.querySelectorAll('[data-table] thead th')]"
        ".map((th, i) => [th.textContent.trim(),"
        " document.querySelector('[data-table] tbody tr').children[i].textContent.trim()]))"
    )
    assert first["checkpoint"] in {"dreamshaper_8", "juggernautXL"}, first
    assert first["steps"] == "28"
    assert first["seed"] == "4242"
    assert first["pixels"], "the pixels on disk, which a cell never says"


def test_the_answer_decides_which_columns_exist(page: Page, live: Live, unbroken):
    """Asking about photographs must not produce eight empty recipe
    columns, and asking about generated pictures must not produce four
    empty camera ones."""
    _table(page, "f=has.generation%3Aeq%3A1")
    generated = page.evaluate(
        "() => [...document.querySelectorAll('[data-table] thead th')].map(th => th.textContent.trim())"
    )
    assert "sampler" in generated
    assert "camera" not in generated, "nothing in this answer came from a camera"
    assert "length" not in generated, "and nothing in it has a length"

    _table(page, "kind=video")
    clips = page.evaluate(
        "() => [...document.querySelectorAll('[data-table] thead th')].map(th => th.textContent.trim())"
    )
    assert "length" in clips, "a clip has one"


def _named(page: Page) -> list[str]:
    return page.evaluate(
        "() => [...document.querySelectorAll('[data-table] tbody .answer-name a')].map(a => a.textContent.trim())"
    )


def test_a_heading_reorders_the_answer_and_the_url_says_so(page: Page, live: Live, unbroken):
    """A heading is a LINK carrying the reordered question, never a
    click that shuffles the rows already on screen: the sort decides
    which pictures are on page one, so a local shuffle would be a table
    disagreeing with its own pager."""
    _table(page)
    page.click('[data-table] [data-sort-by="name"]')
    page.wait_for_selector('[data-table] [data-sort-by="name"][data-sorted]', timeout=15_000)
    assert "sort=name" in page.url, page.url
    ascending = _named(page)
    assert ascending == sorted(ascending, key=str.casefold), ascending


def test_clicking_the_same_heading_turns_it_round(page: Page, live: Live, unbroken):
    """Two clicks means the other way, and the heading says which way it
    is -- to a reader by the arrow, and to a screen reader by aria-sort
    on the cell, which is the same fact said in the right place."""
    _table(page)
    page.click('[data-table] [data-sort-by="name"]')
    page.wait_for_selector('[data-table] [data-sort-by="name"][data-sorted="asc"]', timeout=15_000)
    assert page.get_attribute('[data-table] th:has([data-sort-by="name"])', "aria-sort") == "ascending"
    up = _named(page)

    page.click('[data-table] [data-sort-by="name"]')
    page.wait_for_selector('[data-table] [data-sort-by="name"][data-sorted="desc"]', timeout=15_000)
    assert page.get_attribute('[data-table] th:has([data-sort-by="name"])', "aria-sort") == "descending"
    assert _named(page) == list(reversed(up))


def test_a_number_column_opens_at_the_big_end(page: Page, live: Live, unbroken):
    """Text ascending, numbers descending, because they mean different
    things: "sort by size" almost always means show me what is huge, and
    opening on the smallest file is a click somebody undoes every time."""
    _table(page)
    page.click('[data-table] [data-sort-by="size"]')
    page.wait_for_selector('[data-table] [data-sort-by="size"][data-sorted]', timeout=15_000)
    assert "sort=size-desc" in page.url, page.url


def test_the_order_survives_a_reload(page: Page, live: Live, unbroken):
    """Because it is part of the question rather than something the page
    is holding, which is what makes a shared link land on the order it
    was shared in."""
    _table(page)
    page.click('[data-table] [data-sort-by="pixels"]')
    page.wait_for_selector('[data-table] [data-sort-by="pixels"][data-sorted]', timeout=15_000)
    was = _named(page)
    page.reload()
    page.wait_for_selector("[data-table]", timeout=15_000)
    assert _named(page) == was
    assert page.get_attribute('[data-table] th:has([data-sort-by="pixels"])', "aria-sort") is not None


def test_a_row_is_a_way_into_the_picture(page: Page, live: Live, unbroken):
    """A table of facts about pictures still has to get you to one."""
    _table(page)
    page.click("[data-table] tbody tr:first-child .answer-name a")
    page.wait_for_selector("[data-viewer]", timeout=15_000)
    assert "/i/" in page.url


# --- and the terms, which are a reading rather than a count -----------------


def test_the_terms_panel_stands_beside_the_prompts_and_says_what_it_is(page: Page, live: Live, unbroken):
    """Two panels, two kinds of claim, next to each other on purpose.

    The prompts above are counts -- `prompt.text_hash` is a real
    identity. These are a reading: commas separate terms, which is a
    convention every prompt box follows and no grammar enforces. Mixing
    them into one panel would let the reading borrow the count's
    certainty, so they are apart and the second says what it assumes.
    """
    _analyze(page)
    page.wait_for_selector("[data-analyze-terms]", timeout=10_000)

    said = page.inner_text("[data-analyze-terms]")
    assert "split on commas" in said, "the panel does not say what it assumes"

    # This library's prompts are prose, so each is ONE term -- which is
    # the reading being wrong, visibly, rather than hidden.
    terms = page.evaluate("() => [...document.querySelectorAll('[data-term]')].map(one => one.dataset.term)")
    assert CASTLE in terms, terms
    assert LIGHTHOUSE in terms, terms


def test_a_term_carries_the_question_it_would_make(page: Page, live: Live, unbroken):
    """Every row of an analysis is a way to ask a narrower question, and
    a term is no different: clicking one searches for it."""
    _analyze(page)
    page.wait_for_selector("[data-analyze-terms]", timeout=10_000)

    page.click(f'[data-term-search="{CASTLE}"]')
    page.wait_for_function("() => new URLSearchParams(location.search).get('q') !== null", timeout=10_000)
    asked = page.evaluate("() => new URLSearchParams(location.search).get('q')")
    assert asked == CASTLE
