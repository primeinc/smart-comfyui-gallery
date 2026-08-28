"""The keyword shelf: the page that fixes what a year of typing did.

A vocabulary is only worth having if one picture-idea gets one word, and
a year of typing produces "beach", "Beaches" and "beech" whether anybody
meant it to. `db/authored.py rename_tag` has folded one word into
another since the tables shipped, correctly and under test -- and until
this page nothing called it. The fixing was possible and unreachable,
which is the same as absent.

Two gestures, and they are not symmetrical.

**Renaming FOLDS.** A collision is the ordinary case rather than an
error: somebody typing "Beaches" over a word that is already "beach" is
saying they were always one word, which is the whole reason to be here.
The row that would collide -- a picture already wearing both -- is the
one this has to get right, and it is tested directly.

**Forgetting DESTROYS.** Retyping a word onto two hundred pictures is
not a recovery, so it follows the doctrine removing a root already sets
rather than a lighter one invented for keywords: the page shows the
count, the request carries the count back, and a count that moved
between the two is refused. That refusal is the most important test
here, because it is the one nobody would notice was missing.

The ordering is commonest-first and that is a decision, not a default.
The question this page answers is "what have I actually been calling
things", and the three-picture typo sitting under the four-hundred-
picture word is exactly the row somebody came to fix.
"""

from __future__ import annotations

import contextlib
import re

import pytest
from litestar.testing import TestClient
from PIL import Image
from playwright.sync_api import Page, expect

from db import connect
from sg_web.app import build_app
from tests.conftest import Live
from tests.staging import staged

pytestmark = pytest.mark.slow

FILES = 4

AS_JSON = {"accept": "application/json"}
AS_BROWSER = {"accept": "text/html"}


# --- the served run the browser tests share ---------------------------------


def write_library(root) -> None:
    for i in range(FILES):
        Image.new("RGB", (64, 48), (20 + i * 30, 90, 140)).save(root / f"p{i:02d}.png")


def prepare(api, root) -> list[str]:
    """The scan, and the slugs it minted -- read off the page rather than
    guessed from the filenames, because minting a slug is a rule this
    test does not own.

    One baseline keyword and nothing else. The browser tests below share
    one served run and one database, so each of them writes its OWN
    words: a test that inherited another's rows would pass in file order
    and fail the day anything reorders them.
    """
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == FILES
    grid = api.get("/g?sort=oldest", headers=AS_BROWSER).text
    # No closing quote in the pattern: a cell's href is
    # `/i/{slug}?{qs}` (sg_web/templates/_grid.html:33), so anchoring on
    # one matched nothing -- and an exception raised in `prepare` HANGS
    # the run rather than failing it, which is how a silent zero here
    # costs twenty minutes instead of a red line.
    slugs = re.findall(r'href="/i/([^"?]+)', grid)
    assert len(slugs) >= FILES, f"the grid named {len(slugs)} pictures, not {FILES}"
    # so the shelf is never empty, and no test may touch this one
    assert api.post(f"/i/{slugs[0]}/tags", json={"name": "Baseline"}).status_code in (200, 201)
    return slugs


def _wearing(live: Live, word: str, *at: int) -> None:
    """Put one word on these pictures, through the running application."""
    slugs = live.prepared
    assert isinstance(slugs, list)
    for one in at:
        told = live.api.post(f"/i/{slugs[one]}/tags", json={"name": word})
        assert told.status_code in (200, 201), told.text


# --- the vocabulary, without a browser --------------------------------------


def _small_library(root) -> None:
    for i in range(4):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")


def _typed(stage) -> None:
    """The year of typing this page exists to fix, once."""
    client = stage.client
    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        slugs = [
            slug for (slug,) in conn.execute("SELECT e.slug FROM file f JOIN entity e ON e.id = f.id ORDER BY f.name")
        ]
    finally:
        connect.close(conn)
    # p0: Beaches. p1: BOTH -- the row a fold would collide on.
    # p2: beach. p3: Sunset.
    for at, word in ((0, "Beaches"), (1, "beach"), (1, "Beaches"), (2, "beach"), (3, "Sunset")):
        assert client.post(f"/i/{slugs[at]}/tags", json={"name": word}).status_code in (200, 201)


@pytest.fixture(scope="module")
def _shelf_stage(tmp_path_factory):
    with staged(tmp_path_factory, "keyword_vocabulary", _small_library, _typed) as stage:
        yield stage


@pytest.fixture
def shelf(_shelf_stage):
    """One world with the typing already done, restored between tests.

    Every test here renames or forgets a word, so none can inherit
    another's shelf -- but building four pictures, an application, a scan
    and five taggings per test spent a quarter of a second to answer a
    question that costs a hundredth. The snapshot isolates identically.
    """
    _shelf_stage.restore()
    return _shelf_stage.client


def _listed(client) -> list[dict]:
    told = client.get("/keywords", headers=AS_JSON)
    assert told.status_code == 200, told.text
    return told.json()


def test_the_shelf_is_every_word_with_how_many_wear_it(shelf):
    """Commonest first, because the question is "what have I been
    calling things" and the answer to that is a shape."""
    told = _listed(shelf)
    assert [(one["tag"], one["pictures"]) for one in told] == [("beach", 2), ("beaches", 2), ("sunset", 1)]
    assert [one["label"] for one in told] == ["beach", "Beaches", "Sunset"]
    # and each row carries the question it asks, ready for a link
    assert told[0]["qs"] == "f=tag%3Aeq%3Abeach"


def test_renaming_folds_onto_the_word_already_there(shelf):
    """The gesture this page exists for, and the row that would break
    it: p1 wears BOTH words, so the fold has a collision to survive."""
    told = shelf.post("/keywords/rename", json={"name": "Beaches", "to": "beach"})
    assert told.status_code in (200, 201), told.text
    assert [(one["tag"], one["pictures"]) for one in told.json()] == [("beach", 3), ("sunset", 1)]
    # the answer IS the shelf, so the page never computes the result
    assert told.json() == _listed(shelf)


def test_renaming_to_a_free_word_keeps_every_picture(shelf):
    told = shelf.post("/keywords/rename", json={"name": "Sunset", "to": "Golden Hour"})
    assert told.status_code in (200, 201), told.text
    renamed = next(one for one in told.json() if one["pictures"] == 1)
    assert (renamed["tag"], renamed["label"]) == ("golden hour", "Golden Hour")


def test_forgetting_takes_the_word_off_every_picture(shelf):
    told = shelf.post("/keywords/forget", json={"name": "beach", "pictures": 2})
    assert told.status_code in (200, 201), told.text
    assert [one["tag"] for one in told.json()] == ["beaches", "sunset"]


def test_a_count_that_moved_since_you_looked_is_refused(shelf):
    """The most important test here and the one nobody would notice was
    missing. Forgetting is the only gesture on this page that destroys
    authored work, so it proves what it is acting on -- the same
    doctrine removing a root sets, not a lighter one for keywords."""
    refused = shelf.post("/keywords/forget", json={"name": "beach", "pictures": 99})
    assert refused.status_code == 400, refused.text
    # BOTH numbers, because "it changed" without saying to what is not
    # something a person can act on
    assert "99" in refused.text
    assert "2 picture" in refused.text
    assert _listed(shelf), "the refusal took the keyword anyway"
    assert any(one["tag"] == "beach" for one in _listed(shelf))


def test_a_word_that_is_not_there_is_refused_by_name(shelf):
    for body in ({"name": "never-typed", "to": "x"},):
        refused = shelf.post("/keywords/rename", json=body)
        assert refused.status_code == 400, refused.text
        assert "never-typed" in refused.text
    gone = shelf.post("/keywords/forget", json={"name": "never-typed", "pictures": 0})
    assert gone.status_code == 400, gone.text


def test_renaming_to_nothing_is_refused(shelf):
    """An emptied box is what somebody does before typing, and it must
    not become a keyword nobody can see or delete."""
    for empty in ("", "   "):
        refused = shelf.post("/keywords/rename", json={"name": "Sunset", "to": empty})
        assert refused.status_code == 400, (repr(empty), refused.text)
        assert "word" in refused.text


def test_a_library_with_no_keywords_says_so_rather_than_failing(tmp_path):
    """Nothing typed yet is the commonest state and a real one."""
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        assert client.get("/keywords", headers=AS_JSON).json() == []
        page = client.get("/keywords", headers=AS_BROWSER)
        assert page.status_code == 200
        assert "data-keywords-none" in page.text


# --- the page a person meets ------------------------------------------------


@pytest.fixture
def page(_shared_context):
    """A page of this module's OWN, overriding the shared one.

    `test_agreeing_takes_it_off_every_picture` fails on the module's
    shared page and passes on this one: the confirmation it agrees to is
    never accepted, so the row it waits to see go stays, and the module
    goes red after five seconds against a test that is not the broken
    one. The test before it is the one that installs a dialog handler
    answering NO.

    What carries over is not the handler itself -- taking it off between
    tests was tried in `conftest` and changed nothing, and Playwright's
    own bookkeeping says it should have worked
    (`_connection.py:245`, `pyee/base.py:282`). So the mechanism is page
    state this does not name yet, and the honest fix is a page that
    cannot carry anything rather than a guess about what it carried.

    ~40 ms per test, on four tests.
    """
    page = _shared_context.new_page()
    yield page
    with contextlib.suppress(Exception):
        page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
    page.close()


def test_the_shelf_is_reachable_without_knowing_it_exists(live: Live, page: Page):
    """A surface reachable only by typing its URL is a surface for
    whoever wrote it. It sits in the shell beside albums, because both
    are a person's own organisation of their own library."""
    page.goto("/g")
    page.click('nav.shell a[href="/keywords"]')
    page.wait_for_url("**/keywords")
    expect(page.locator("[data-keywords]")).to_be_visible()
    expect(page.locator('nav.shell a[href="/keywords"]')).to_have_attribute("aria-current", "page")


def test_renaming_redraws_the_list_from_the_answer(live: Live, page: Page):
    """A fold changes two rows -- one absorbs the other's pictures and
    one vanishes -- so the page redraws from the authoritative answer
    rather than patching the row it clicked."""
    _wearing(live, "Cormorant", 1)
    _wearing(live, "Gannet", 2, 3)
    page.goto("/keywords")
    box = page.locator('[data-keyword="cormorant"] [data-keyword-rename-input]')
    expect(box).to_be_visible()
    box.fill("Gannet")
    box.press("Enter")
    # one row gone, the other holding all three pictures -- and no reload
    expect(page.locator('[data-keyword="cormorant"]')).to_have_count(0)
    expect(page.locator('[data-keyword="gannet"] .keyword-count')).to_have_text("3 pictures")


def test_forgetting_asks_first_and_names_the_number(live: Live, page: Page):
    """ "forget fulmar" and "take fulmar off 3 pictures" are different
    things to agree to, and only one of them is a question somebody can
    answer."""
    _wearing(live, "Fulmar", 0, 1, 2)
    page.goto("/keywords")
    asked: list[str] = []

    def refuse(dialog):
        asked.append(dialog.message)
        dialog.dismiss()

    page.on("dialog", refuse)
    page.locator('[data-keyword="fulmar"] [data-forget]').click()
    assert asked, "nothing was asked before destroying authored work"
    assert "3 pictures" in asked[0], asked[0]
    expect(page.locator('[data-keyword="fulmar"]')).to_have_count(1)


def test_agreeing_takes_it_off_every_picture(live: Live, page: Page):
    _wearing(live, "Petrel", 0, 1)
    page.goto("/keywords")
    expect(page.locator('[data-keyword="petrel"]')).to_have_count(1)
    page.on("dialog", lambda dialog: dialog.accept())
    page.locator('[data-keyword="petrel"] [data-forget]').click()
    expect(page.locator('[data-keyword="petrel"]')).to_have_count(0)
    # gone from the library rather than only from this page
    page.reload()
    expect(page.locator('[data-keyword="petrel"]')).to_have_count(0)
    # and the word nobody touched is still standing
    expect(page.locator('[data-keyword="baseline"]')).to_have_count(1)
