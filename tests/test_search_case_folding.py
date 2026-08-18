"""Searching by name has to fold case in every script, not just English.

SQLite's LIKE is case-insensitive for ASCII and nothing else, so `CAFE`
matched `cafe` while `CAFÉ` did not match `café`, and `РИСУНОК` never
matched `рисунок`. Search worked in English and quietly failed for
everyone else -- and it fails by returning nothing, which reads as an
empty library rather than a broken comparison.

`ulower` is registered on every connection and applied to both sides of
the name comparison, so folding follows Python's rules, which know about
every script.

The tests match on file id rather than on the name: the page carries its
file data as JSON with non-ASCII escaped to \\uXXXX, so looking for the
name itself finds nothing whatever the search did. That mistake made an
earlier probe of this report "no matches" for every term, including the
ASCII control.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "fold_"

_FILES = {
    "chinese": f"{_PREFIX}测试.png",
    "french": f"{_PREFIX}café.png",
    "russian": f"{_PREFIX}рисунок.png",
    "greek": f"{_PREFIX}Ελλάδα.png",
    "german": f"{_PREFIX}straße.png",
}


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    for name in _FILES.values():
        Image.new("RGB", (16, 16), (30, 30, 90)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {r[0]: r[1] for r in conn.execute(f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()}
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    for name in _FILES.values():
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _search(smartgallery_app, ids, term, path="/galleryout/view/_root_"):
    client = smartgallery_app.app.test_client()
    html = client.get(f"{path}?search={term}").get_data(as_text=True)
    return sorted(name for name, fid in ids.items() if fid and fid in html)


def test_the_fixture_is_searchable_at_all(smartgallery_app, library):
    """Control: an ASCII term common to every file finds all of them, so a
    later empty result means the folding failed and not the fixture."""
    assert _search(smartgallery_app, library, _PREFIX) == sorted(_FILES.values())


@pytest.mark.parametrize(
    ("term", "expected_key"),
    [
        ("测试", "chinese"),
        ("café", "french"),
        ("CAFÉ", "french"),  # the regression
        ("Café", "french"),
        ("рисунок", "russian"),
        ("РИСУНОК", "russian"),  # the regression
        ("Ελλάδα", "greek"),
        ("ΕΛΛΆΔΑ", "greek"),  # the regression
        ("straße", "german"),
        ("STRASSE", "german"),  # folds to the same thing in Python
    ],
)
def test_a_search_finds_its_file_whatever_the_case(smartgallery_app, library, term, expected_key):
    found = _search(smartgallery_app, library, term)

    assert _FILES[expected_key] in found, f"searching {term!r} did not find {_FILES[expected_key]!r}; found {found}"


def test_ascii_search_still_folds(smartgallery_app, library):
    """The behaviour that always worked has to keep working."""
    assert _search(smartgallery_app, library, _PREFIX.upper()) == sorted(_FILES.values())


def test_a_search_that_matches_nothing_still_matches_nothing(smartgallery_app, library):
    """Folding must not turn every search into a match."""
    assert _search(smartgallery_app, library, "zzz_no_such_file") == []


def _filtered(smartgallery_app, library, query):
    client = smartgallery_app.app.test_client()
    html = client.get(f"/galleryout/view/_root_?{query}").get_data(as_text=True)
    return sorted(name for name, fid in library.items() if fid and fid in html)


@pytest.fixture
def annotated(smartgallery_app, library):
    """The Greek file carries a comment and a prompt, both non-English."""
    conn = smartgallery_app.get_db_connection()
    try:
        file_id = library[_FILES["greek"]]
        conn.execute(
            "INSERT INTO file_comments (file_id, client_uuid, author_name, "
            "comment_text, target_audience, created_at) "
            "VALUES (?, 'admin', 'Me', 'Καλημέρα κόσμε', 'public', 1.0)",
            (file_id,),
        )
        conn.execute("UPDATE files SET workflow_prompt = ? WHERE id = ?", ("портрет девушки", file_id))
        conn.commit()
    finally:
        conn.close()
    return library


@pytest.mark.parametrize("term", ["κόσμε", "ΚΌΣΜΕ", "Κόσμε"])
def test_comment_search_folds_case_too(smartgallery_app, annotated, term):
    """Comments are prose, so they are the likeliest thing in the gallery
    not to be written in English."""
    found = _filtered(smartgallery_app, annotated, f"comment_search={term}")

    assert found == [_FILES["greek"]], f"searching comments for {term!r} found {found}"


@pytest.mark.parametrize("term", ["портрет", "ПОРТРЕТ"])
def test_prompt_search_folds_case_too(smartgallery_app, annotated, term):
    found = _filtered(smartgallery_app, annotated, f"workflow_prompt={term}")

    assert found == [_FILES["greek"]], f"searching prompts for {term!r} found {found}"


def test_comment_search_still_excludes_what_it_should(smartgallery_app, annotated):
    """The negation form has to keep working: folding both sides must not
    turn a NOT LIKE into a match-everything."""
    found = _filtered(smartgallery_app, annotated, "comment_search=ΚΌΣΜΕ")
    assert _FILES["french"] not in found, "a file with no such comment came back from a comment search"
