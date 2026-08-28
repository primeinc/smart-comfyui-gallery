"""Scanning a root lists files. It does not make a gallery.

A scan finds names and sizes and queues thumbnails, and that is all: no
metadata is read, nothing is fingerprinted, no face is found, no caption
is written. So the person who has just registered their library is
looking at pictures with no search, no people and no dates, having done
what the page told them to do.

The button that fixes it exists -- `catch_up`, first of the launchers,
"bring the library up to date, in order" -- and it is in a different
section further down the page. That is the same defect this surface was
built to remove, at a smaller scale: an order only the application
knows, which a person is charged with carrying.

So the roots panel says what the scan left undone, names the sweeps and
their counts, and offers the chain in place. Sweep names rather than a
total, because there is no honest total: a file missing both a reading
and a caption is one file and two sweeps.
"""

from __future__ import annotations

import pathlib

import pytest
from PIL import Image

from tests.staging import hosting

pytestmark = pytest.mark.slow

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}


@pytest.fixture(scope="module")
def _host(tmp_path_factory):
    with hosting(tmp_path_factory, "a_scan_says_what_it_left_undone") as stage:
        yield stage


@pytest.fixture
def client(_host):
    """One application, restored between tests.

    Every test here needs its own library at its own SIZE -- three files,
    two, none -- which is why each was booting a whole application to get
    one. An empty home plus a restore gives each of them the same thing:
    no roots, numbering from 1, and a library they write themselves.
    """
    _host.restore()
    return _host.client


def _root(tmp_path: pathlib.Path, files: int) -> pathlib.Path:
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(files):
        Image.new("RGB", (12, 12), (20 * i, 80, 130)).save(root / f"p{i}.png")
    return root


def test_a_scan_names_the_work_it_left_and_offers_it_in_place(client, tmp_path):
    """The whole entry. What the scan did not do, beside the thing that
    did not do it, with one press that does it in the right order."""
    root = _root(tmp_path, 3)
    made = client.post("/operations/roots", data={"path": str(root), "kind": "library"})
    assert made.status_code == 200, made.text
    root_id = client.get("/roots").json()[0]["id"]

    scanned = client.post(f"/operations/roots/{root_id}/scan")
    assert scanned.status_code == 200, scanned.text
    assert "3 added" in scanned.text

    assert "data-root-behind" in scanned.text, "the scan said nothing about what it left undone"
    assert "data-root-catch-up" in scanned.text, "the chain is not offered where the scan happened"
    assert 'hx-post="/operations/jobs/catch_up"' in scanned.text
    # and it names sweeps with their counts, not a total
    assert 'data-behind="ingest">ingest 3<' in scanned.text, scanned.text


def test_the_offer_points_at_a_route_that_queues_the_chain(client, tmp_path):
    """A button is only worth rendering if pressing it works. The panel
    posts where the launcher lives, and that queues more than one step --
    which is the difference between this and any single sweep."""
    root = _root(tmp_path, 2)
    client.post("/operations/roots", data={"path": str(root), "kind": "library"})
    root_id = client.get("/roots").json()[0]["id"]
    client.post(f"/operations/roots/{root_id}/scan")

    pressed = client.post("/operations/jobs/catch_up")
    assert pressed.status_code == 200, pressed.text
    assert "queued #" in pressed.text, pressed.text
    assert pressed.text.count("#") > 1, "catch-up queued a single step; it is meant to be the chain"


def test_a_library_with_nothing_outstanding_is_not_nagged(client, tmp_path):
    """The discriminating half. A panel that always says "still to do"
    is a decoration, so an empty root -- nothing present, nothing
    outstanding -- must say nothing at all."""
    root = _root(tmp_path, 0)
    client.post("/operations/roots", data={"path": str(root), "kind": "library"})
    root_id = client.get("/roots").json()[0]["id"]

    scanned = client.post(f"/operations/roots/{root_id}/scan")
    assert scanned.status_code == 200, scanned.text
    assert "0 added" in scanned.text
    assert "data-root-behind" not in scanned.text, "it offered work on a library with no files"


def test_the_console_itself_carries_the_same_answer(client, tmp_path):
    """Same fact, both renders. The panel is included on the cold page as
    well as returned from the form, and StrictUndefined means a context
    the view forgot is a 500 rather than a silently empty section."""
    root = _root(tmp_path, 3)
    client.post("/operations/roots", data={"path": str(root), "kind": "library"})
    root_id = client.get("/roots").json()[0]["id"]
    client.post(f"/operations/roots/{root_id}/scan")

    page = client.get("/operations", headers=AS_BROWSER)
    assert page.status_code == 200, page.text
    assert "data-root-behind" in page.text
    assert 'data-behind="ingest">ingest 3<' in page.text


def test_registering_a_root_renders_the_panel_at_all(client, tmp_path):
    """The other render site. `add_root` returns the same partial, so a
    context it does not pass is a crash on the most ordinary press on
    the page."""
    root = _root(tmp_path, 3)
    made = client.post("/operations/roots", data={"path": str(root), "kind": "library"})
    assert made.status_code == 200, made.text
    assert str(root) in made.text
    # nothing is scanned yet, so there is no present file and no work
    assert "data-root-behind" not in made.text
