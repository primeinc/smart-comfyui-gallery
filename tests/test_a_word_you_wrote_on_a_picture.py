"""The oldest idea in digital asset management, arriving forty-one
schema versions late.

A keyword. Not a rating, not an album, not a face -- a word somebody
types on a picture because it is the word they will look for. This
library could slice itself forty-one ways and none of them was that.

Three decisions are what these tests are actually about.

**It is authored, so it survives a rebuild.** The obvious home was
`derived_annotation` with `kind='tag'`, which already exists and already
holds words about pictures. It would have been wrong: that namespace
requires a `model_id` and a `source_sha256`, and `derived.drop_all`
deletes all of it. A word a person typed would have vanished at the next
re-annotate with nothing reporting it. That is the last test here and
the reason for the table.

**Case is not identity.** "New York" and "new york" are one keyword or
the library is answering two questions with one word. Folded in Python
rather than by COLLATE NOCASE, which folds ASCII only.

**A keyword with no pictures is not a keyword.** It is the typo somebody
just corrected, and left standing it haunts the filter menu forever.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import authored, connect, derived
from sg_web.app import build_app

pytestmark = pytest.mark.slow


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        slugs = [
            slug for (slug,) in conn.execute("SELECT e.slug FROM file f JOIN entity e ON e.id = f.id ORDER BY f.name")
        ]
        yield client, conn, slugs
        connect.close(conn)


def _tag(client, slug: str, name: str, value: bool = True):
    told = client.post(f"/i/{slug}/tags", json={"name": name, "value": value})
    assert told.status_code in (200, 201), told.text
    return told.json()


def _id_of(conn, slug: str) -> int:
    return int(conn.execute("SELECT id FROM entity WHERE kind = 'file' AND slug = ?", (slug,)).fetchone()[0])


def test_a_word_lands_on_a_picture_and_comes_back_with_it(library):
    """The whole feature in one gesture: type a word, and the picture
    now carries it."""
    client, _conn, slugs = library
    answered = _tag(client, slugs[0], "Sunset")
    assert [one["label"] for one in answered["authored"]["tags"]] == ["Sunset"]
    assert [one["tag"] for one in answered["authored"]["tags"]] == ["sunset"]

    # and the surface reads it back, not just the write's own answer
    surface = client.get(f"/i/{slugs[0]}").json()
    assert [one["label"] for one in surface["authored"]["tags"]] == ["Sunset"]


def test_case_and_spacing_do_not_split_one_keyword(library):
    """New York and new  york are the same word. A library that let them
    apart would answer two questions with one keyword and be wrong both
    times."""
    client, conn, slugs = library
    _tag(client, slugs[0], "New York")
    _tag(client, slugs[1], "new  york")
    assert [(one, count) for one, _label, count in authored.vocabulary(conn)] == [("new york", 2)]


def test_the_spelling_is_the_one_that_named_it(library):
    """Whoever wrote it down first named it. A later posting of the same
    word must not silently restyle it on every page that shows it."""
    client, conn, slugs = library
    _tag(client, slugs[0], "New York")
    _tag(client, slugs[1], "NEW YORK")
    assert [label for _one, label, _count in authored.vocabulary(conn)] == ["New York"]


def test_the_same_word_twice_is_one_keyword_not_a_toggle(library):
    """Desired state, like every authored route beside it: a retry after
    a network hiccup lands where the person already put it."""
    client, _conn, slugs = library
    _tag(client, slugs[0], "beach")
    answered = _tag(client, slugs[0], "beach")
    assert len(answered["authored"]["tags"]) == 1


def test_taking_the_last_picture_off_a_keyword_takes_the_keyword(library):
    """A word with nothing under it is the typo somebody just corrected.
    Left standing it haunts the filter menu, where the cost of being
    wrong is a list nobody trusts."""
    client, conn, slugs = library
    _tag(client, slugs[0], "typpo")
    assert authored.vocabulary(conn)
    _tag(client, slugs[0], "typpo", value=False)
    assert authored.vocabulary(conn) == []
    assert conn.execute("SELECT count(*) FROM tag").fetchone()[0] == 0


def test_taking_one_picture_off_leaves_the_keyword_standing(library):
    """The other half of that rule, and the case that would break it."""
    client, conn, slugs = library
    _tag(client, slugs[0], "beach")
    _tag(client, slugs[1], "beach")
    _tag(client, slugs[0], "beach", value=False)
    assert [(one, count) for one, _label, count in authored.vocabulary(conn)] == [("beach", 1)]


def test_a_keyword_is_shared_and_not_one_persons_opinion(library):
    """The deliberate difference from `rating` and `favorite` beside it.
    Those are what one person thinks and two people may disagree; a
    keyword is a fact about the picture that everybody reads."""
    client, conn, slugs = library
    _tag(client, slugs[0], "beach")
    other = authored.add_user(conn, "someone-else", "x", "USER", 0.0)
    conn.commit()
    held = authored.media_state(conn, _id_of(conn, slugs[0]), other)
    assert [one["label"] for one in held.tags] == ["beach"]


def test_a_word_with_no_word_in_it_is_refused_and_says_so(library):
    """Blank and whitespace-only are what an empty box posts. A refusal
    beats a keyword nobody can see or delete."""
    client, _conn, slugs = library
    for empty in ("", "   ", chr(9) + chr(10)):
        refused = client.post(f"/i/{slugs[0]}/tags", json={"name": empty, "value": True})
        assert refused.status_code == 400, (repr(empty), refused.status_code, refused.text)
        assert "word" in refused.text


def test_a_pasted_paragraph_is_not_a_keyword(library):
    """The cap is on the normalised form, so it counts characters
    somebody meant rather than the spaces between them."""
    client, _conn, slugs = library
    refused = client.post(f"/i/{slugs[0]}/tags", json={"name": "x" * 101, "value": True})
    assert refused.status_code == 400, refused.text
    assert "100" in refused.text


def test_renaming_folds_into_the_word_that_is_already_there(library):
    """The reason a keyword is not an entity: renaming is an UPDATE
    rather than a retired address somebody still holds a bookmark to.

    A fold is the ordinary case -- somebody typed "beach" and "Beaches"
    over a year and is now saying they were always one word -- so a
    collision merges rather than refusing.
    """
    client, conn, slugs = library
    _tag(client, slugs[0], "Beaches")
    _tag(client, slugs[1], "beach")
    # slugs[1] wears BOTH, which is the row that would collide
    _tag(client, slugs[1], "Beaches")
    conn.commit()

    authored.rename_tag(conn, "Beaches", "beach", 0.0)
    conn.commit()
    assert [(one, count) for one, _label, count in authored.vocabulary(conn)] == [("beach", 2)]
    assert [one["label"] for one in authored.tags_of(conn, _id_of(conn, slugs[1]))] == ["beach"]


def test_renaming_to_a_free_word_keeps_every_picture(library):
    client, conn, slugs = library
    _tag(client, slugs[0], "beech")
    conn.commit()
    authored.rename_tag(conn, "beech", "Beach", 0.0)
    conn.commit()
    assert authored.vocabulary(conn) == [("beach", "Beach", 1)]


def test_a_keyword_outlives_the_derived_layer(library):
    """The decision the table exists for.

    `derived_annotation` already holds words about pictures and already
    has a `kind` column, and putting a human keyword there would have
    cost one migration instead of two tables. `drop_all` is why not: it
    deletes that whole namespace, so a word somebody typed would have
    gone at the next re-annotate with nothing reporting it.
    """
    client, conn, slugs = library
    _tag(client, slugs[0], "Sunset")
    conn.commit()
    derived.drop_all(conn)
    conn.commit()
    assert [one["label"] for one in authored.tags_of(conn, _id_of(conn, slugs[0]))] == ["Sunset"]


def test_a_deleted_picture_does_not_leave_a_dangling_word(library):
    """`file_tag` cascades, which is the ordinary half. The keyword
    itself is left: another picture may still wear it, and this is not
    the gesture that means "I am done with this word"."""
    client, conn, slugs = library
    _tag(client, slugs[0], "beach")
    _tag(client, slugs[1], "beach")
    conn.commit()
    conn.execute("DELETE FROM file WHERE id = ?", (_id_of(conn, slugs[0]),))
    conn.commit()
    assert [(one, count) for one, _label, count in authored.vocabulary(conn)] == [("beach", 1)]
