"""A wrong face is corrected on the page where it is seen.

Saying "that is not her" has been possible since the claim existed, and
reachable from one place: the inspector of a single picture, one picture
at a time. That is not where anybody notices the mistake. They notice it
looking at a wall of somebody's photographs -- the person's own page --
and that page could only send them somewhere else to say it.

So the correction lives over every thumbnail there, and this drives it
the way a person does: click the thing, and read what the page says
afterwards.

What it says afterwards is the part worth pinning. Denying takes the
name off the picture NOW (db/derived.py `withdraw_attribution`), so the
cell cannot keep showing the thumbnail under this person. Undo withdraws
the CLAIM and only the claim: the next clustering run is free to decide
it again, but no run has said so since, so the picture does not come
back by itself. The cell says exactly that, because a browser that put
the picture back would be inventing derived state nothing produced.
"""

from __future__ import annotations

import pathlib
import time as clock

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from tests.conftest import Live

pytestmark = pytest.mark.slow

MODEL = ("face-test", "1")
FILES = 3


def write_library(root: pathlib.Path) -> None:
    for i in range(FILES):
        Image.new("RGB", (48, 36), (40 * i, 90, 140)).save(root / f"p{i}.png")


def prepare(api, root, where: pathlib.Path) -> dict:
    """Scan through the routes, then name one person in every picture.

    The naming is written straight to the database rather than run
    through a detector: what this file is about is the correction, and a
    real face model would make it a test about face models.

    Takes the home, which is why it is spelled with three parameters
    (tests/conftest.py `_prepared`): the run's database is not reachable
    through the routes, and `SG_TEST_HOME` is not a way to find it -- the
    harness boots the next module's server while this one's tests run,
    and re-points that variable while it does.
    """
    from db import authored, connect, derived, naming
    from sg_web import home

    made = api.post("/roots", json={"path": str(root)}).json()
    api.post(f"/roots/{made['id']}/scan")

    conn = connect.connect(home.db_path(where))
    try:
        who = authored.person(conn, "Hannah", clock.time())
        run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, 0.5, clock.time())
        pictures = []
        for file_id, sha in conn.execute("SELECT id, content_sha256 FROM file ORDER BY id").fetchall():
            derived.record_faces(
                conn,
                file_id,
                MODEL[0],
                MODEL[1],
                sha,
                clock.time(),
                [
                    {
                        "region": derived.region(conn, 0.1, 0.1, 0.3, 0.3),
                        "embedding": np.ones(4, np.float32).tobytes(),
                    }
                ],
            )
            derived.attribute(conn, file_id, who, run_id, MODEL[0], MODEL[1], face_count=1)
            named = naming.entity_slug(conn, file_id)
            assert named is not None
            pictures.append(named[1])
        # The page shows the PRIMARY run's people and nothing else, so a
        # run nobody chose renders an empty page -- which reads exactly
        # like a missing button.
        derived.make_primary(conn, run_id)
        conn.commit()
        person = naming.entity_slug(conn, who)
        assert person is not None
    finally:
        connect.close(conn)
    return {"person": person[1], "pictures": pictures}


def _shells(page: Page) -> int:
    return page.evaluate("() => document.querySelectorAll('[data-person-picture]').length")


def test_every_picture_carries_the_correction(page: Page, live: Live):
    """The control, and the claim: it is on all of them, not on the one
    somebody already opened."""
    held = live.prepared
    assert isinstance(held, dict)
    page.goto(f"/p/{held['person']}")
    page.wait_for_selector("[data-person-pictures]", timeout=10_000)
    assert _shells(page) == FILES
    expect(page.locator("[data-person-not-here]")).to_have_count(FILES)


def test_saying_it_takes_that_picture_off_the_person(page: Page, live: Live):
    """Clicked, not posted. The button has to be reachable, enabled, and
    wired -- three things a route test cannot see."""
    held = live.prepared
    assert isinstance(held, dict)
    page.goto(f"/p/{held['person']}")
    page.wait_for_selector("[data-person-not-here]", timeout=10_000)
    was = _shells(page)

    page.locator("[data-person-not-here]").first.click()
    page.wait_for_selector("[data-person-denied]", timeout=10_000)
    assert _shells(page) == was - 1, "the picture is still standing under this person"
    assert "not them" in page.locator("[data-person-denied]").inner_text()

    # and it holds across a reload, because it is a record and not a
    # thing the browser drew
    page.reload()
    page.wait_for_selector("[data-person-pictures]", timeout=10_000)
    assert _shells(page) == was - 1


def test_undo_says_what_it_actually_undoes(page: Page, live: Live):
    """The honest half. Withdrawing deletes the record that the
    attribution was wrong; it does not put the name back, because no
    clustering run has said so since. The cell says so rather than
    quietly restoring a thumbnail."""
    held = live.prepared
    assert isinstance(held, dict)
    page.goto(f"/p/{held['person']}")
    # No wait before the click: `click` waits for the button itself.
    page.locator("[data-person-not-here]").first.click()
    page.wait_for_selector("[data-person-denied]", timeout=10_000)
    page.locator("[data-person-denied] button").first.click()
    page.wait_for_selector("[data-person-withdrawn]", timeout=10_000)

    said = page.locator("[data-person-withdrawn]").inner_text()
    assert "withdrawn" in said, said
    assert "clustering" in said, "it does not say when the picture could come back"
    expect(page.locator("[data-person-withdrawn] a")).to_have_count(1)
