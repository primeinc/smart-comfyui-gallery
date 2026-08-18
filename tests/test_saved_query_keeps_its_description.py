"""A saved query's description was accepted and thrown away.

/queries/list reads a description back out of a leading `-- Description:`
comment in the saved SQL and returns it for the picker. /queries/save took
a `description` from the caller, bound it to a local, and never wrote it
anywhere -- so the comment `list` looks for was never produced by anything
and every saved query listed with a blank description.

Nothing in templates/ calls either route, so the fault survived: it is an
API-only pair and no page was visibly wrong. Ruff's F841 on the unused
local is what surfaced it.

The description is flattened to one line before it goes in. A newline
would close the comment and put the remainder of the text into a file that
is later executed as SQL.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def client(smartgallery_app, monkeypatch, tmp_path):
    """A gallery folder of this test's own: these routes write real files."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(tmp_path))
    return smartgallery_app.app.test_client()


def _saved(smartgallery_app, name):
    path = os.path.join(smartgallery_app.BASE_SMARTGALLERY_PATH,
                        ".omniquery", "saved_queries", name)
    return open(path, encoding="utf-8").read()


def test_a_description_survives_the_round_trip(smartgallery_app, client):
    """The bug: save accepted it, list always answered with ''."""
    # Arrange / Act
    saved = client.post("/galleryout/api/omniquery/queries/save",
                        json={"name": "recent",
                              "description": "everything from this week",
                              "sql": "SELECT 1"}).get_json()
    listed = client.get("/galleryout/api/omniquery/queries/list").get_json()

    # Assert
    assert saved["status"] == "success", saved
    entry = next(e for e in listed["queries"] if e["name"] == "recent.txt")
    assert entry["description"] == "everything from this week", listed


def test_a_query_saved_without_one_is_left_alone(smartgallery_app, client):
    """No description means no comment: the SQL is stored as it arrived."""
    # Arrange / Act
    client.post("/galleryout/api/omniquery/queries/save",
                json={"name": "bare", "sql": "SELECT 1"})

    # Assert
    assert _saved(smartgallery_app, "bare.txt") == "SELECT 1"


def test_a_description_cannot_carry_sql_of_its_own(smartgallery_app, client):
    """A newline would end the comment and leave the rest executable."""
    # Arrange / Act
    client.post("/galleryout/api/omniquery/queries/save",
                json={"name": "injected",
                      "description": "harmless\nDROP TABLE files; --",
                      "sql": "SELECT 1"})
    body = _saved(smartgallery_app, "injected.txt")

    # Assert
    first, rest = body.split("\n", 1)
    assert first == "-- Description: harmless DROP TABLE files; --", body
    assert rest == "SELECT 1", body


def test_the_description_it_reads_back_is_the_one_it_wrote(smartgallery_app, client):
    """Flattening happens on the way in, so the picker shows one line."""
    # Arrange / Act
    client.post("/galleryout/api/omniquery/queries/save",
                json={"name": "wrapped",
                      "description": "line one\nline two",
                      "sql": "SELECT 1"})
    listed = client.get("/galleryout/api/omniquery/queries/list").get_json()

    # Assert
    entry = next(e for e in listed["queries"] if e["name"] == "wrapped.txt")
    assert entry["description"] == "line one line two", listed
