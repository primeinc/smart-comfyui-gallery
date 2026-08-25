"""The gallery grid over HTTP: presentation adapters, one ResultSet.

The shell renders whole from nothing but the URL, the fragment swaps
the same result set, the rail's peek previews the exact destination page,
and locate agrees with where the grid actually shows a file. None of
these routes owns a second opinion -- change the query and every
surface moves together, which is what WI-35's acceptance pins.
"""

from __future__ import annotations

import re

import pytest
from litestar.testing import TestClient
from PIL import Image

from sg_web.app import build_app


@pytest.fixture(scope="module")
def grid_client(tmp_path_factory):
    """150 real files behind the running application -- past the old
    60/120-row dead end by construction."""
    tmp = tmp_path_factory.mktemp("gallery")
    root = tmp / "lib"
    (root / "bay").mkdir(parents=True)
    (root / "cove").mkdir()
    import os

    for i in range(150):
        folder = "bay" if i < 90 else "cove"
        path = root / folder / f"pic_{i:03d}.png"
        Image.new("RGB", (12, 12), (i % 256, 80, 120)).save(path)
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))

    with TestClient(app=build_app(str(tmp / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        told = client.post(f"/roots/{made['id']}/scan").json()
        assert told["added"] == 150
        yield client


def _slugs(html: str) -> list[str]:
    return re.findall(r'data-slug="([^"]+)"', html)


def test_the_shell_renders_whole_from_the_url(grid_client):
    answer = grid_client.get("/g")
    assert answer.status_code == 200
    html = answer.text
    assert 'data-pages="3"' in html
    assert 'data-total="150"' in html
    assert len(_slugs(html)) == 60, "the first page carries a full page of cells, not a hard cap"
    assert "page 1 of 3" in html
    # Newest first: the last-written file leads.
    assert "pic-149" in _slugs(html)[0]


def test_page_three_exists_beyond_the_old_dead_end(grid_client):
    answer = grid_client.get("/g", params={"page": 3})
    assert answer.status_code == 200
    assert len(_slugs(answer.text)) == 30
    assert "page 3 of 3" in answer.text


def test_the_fragment_is_the_same_answer_as_the_shell(grid_client):
    shell = grid_client.get("/g", params={"page": 2}).text
    fragment = grid_client.get("/g/grid", params={"page": 2}).text
    assert "<html" not in fragment, "a fragment is swapped into a page, never a page in a page"
    assert _slugs(fragment) == _slugs(shell)


def test_the_peek_previews_the_exact_destination_page(grid_client):
    told = grid_client.get("/g/peek", params={"page": 2, "count": 9}).json()
    page_two = _slugs(grid_client.get("/g/grid", params={"page": 2}).text)
    assert [item["slug"] for item in told["items"]] == page_two[:9]
    assert told["first_ordinal"] == 61
    assert told["last_ordinal"] == 120
    assert told["total"] == 150


def test_locate_agrees_with_the_grid(grid_client):
    page_two = _slugs(grid_client.get("/g/grid", params={"page": 2}).text)
    found = grid_client.get(f"/g/locate/{page_two[5]}").json()
    assert found["in_answer"] is True
    assert found["page"] == 2
    assert found["ordinal"] == 66
    assert found["previous"] == page_two[4]
    assert found["next"] == page_two[6]


def test_a_scoped_question_moves_every_surface_together(grid_client):
    shell = grid_client.get("/g", params={"folder": "cove", "size": 25})
    assert 'data-total="60"' in shell.text
    assert 'data-pages="3"' in shell.text
    told = grid_client.get("/g/peek", params={"folder": "cove", "size": 25, "page": 3}).json()
    assert told["total"] == 60
    assert told["first_ordinal"] == 51
    outside = _slugs(grid_client.get("/g/grid").text)[0]  # newest overall lives in cove
    inside = grid_client.get(f"/g/locate/{outside}", params={"folder": "bay"}).json()
    assert inside == {"in_answer": False}


def test_the_url_is_canonical_state_and_the_pager_pushes_it(grid_client):
    html = grid_client.get("/g", params={"folder": "bay", "size": 30}).text
    assert 'hx-push-url="/g?folder=bay&amp;size=30&amp;page=2"' in html
    assert 'hx-get="/g/grid?folder=bay&amp;size=30&amp;page=2"' in html


def test_a_malformed_question_is_refused_not_emptied(grid_client):
    assert grid_client.get("/g", params={"sort": "best"}).status_code == 400
    assert grid_client.get("/g", params={"sort": "similarity"}).status_code == 400
    assert grid_client.get("/g", params={"size": 0}).status_code == 400
    assert grid_client.get("/g", params={"folder": "nowhere"}).status_code == 404


def test_the_static_assets_serve(grid_client):
    for asset in ("gallery.css", "build/gallery.js", "htmx-2.0.7.min.js"):
        answer = grid_client.get(f"/static/{asset}")
        assert answer.status_code == 200, asset
        assert len(answer.content) > 500, asset


def test_the_peek_refuses_to_cross_currencies(grid_client):
    """A preview must belong to the same result-set generation as the
    grid it floats beside: a stale expectation is a 409, never a
    preview from the new ordering shown against the old geometry."""
    import re as regex

    held = regex.search(r'data-currency="([^"]+)"', grid_client.get("/g").text)
    assert held is not None
    current = held.group(1)
    fresh = grid_client.get("/g/peek", params={"page": 1, "expect": current})
    assert fresh.status_code == 200
    assert fresh.json()["currency"] == current
    stale = grid_client.get("/g/peek", params={"page": 1, "expect": "v0-long-gone"})
    assert stale.status_code == 409
    assert grid_client.get("/g/peek", params={"page": 1}).status_code == 200, (
        "a caller with no expectation still gets the current answer"
    )
