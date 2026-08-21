"""A place is an entity minted by explicit authoring or enrichment.

The refusal contract is the hostile part: a writer that validates AFTER
creating rows leaves half an entity behind when the caller catches the
refusal and commits its own work -- the exact strand the collection
lifecycle already paid for. So `places.place` proves its refusals are
atomic against the realistic caller, the one that does not roll back.
"""

from __future__ import annotations

import pytest

from db import connect, places
from db.build import build


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "gallery.db"
    build(path)
    conn = connect.connect(path)
    yield conn
    connect.close(conn)


def test_a_place_is_a_full_entity_citizen(db):
    """Minting writes the entity AND the place, addressable by slug."""
    hawaii = places.place(db, "Hawaii", "region", now=1000.0)
    beach = places.place(db, "Waimanalo Beach", "locality", now=1000.0, parent_id=hawaii)
    db.commit()
    kind, slug = db.execute("SELECT kind, slug FROM entity WHERE id = ?", (hawaii,)).fetchone()
    assert (kind, slug) == ("place", "hawaii")
    parent = db.execute("SELECT parent_id FROM place WHERE id = ?", (beach,)).fetchone()[0]
    assert parent == hawaii


def test_a_refused_parent_leaves_no_orphan_entity(db):
    """Invalid parent -> refusal -> the caller catches it and commits its
    own transaction anyway. Zero entity rows may survive that commit: the
    parent check runs BEFORE the mint, so the refusal is atomic without
    relying on the caller to roll back."""
    with pytest.raises(ValueError, match="not a place"):
        places.place(db, "Nowhere Beach", "locality", now=1000.0, parent_id=99_999)
    db.commit()
    assert db.execute("SELECT count(*) FROM entity WHERE kind = 'place'").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM place").fetchone()[0] == 0


def test_the_vocabulary_refusals_are_names_not_rows(db):
    """A bad kind or an empty name refuses the same way -- before any
    row exists to strand."""
    with pytest.raises(ValueError, match="place kind"):
        places.place(db, "Hawaii", "archipelago", now=1000.0)
    with pytest.raises(ValueError, match="non-empty"):
        places.place(db, "   ", "region", now=1000.0)
    db.commit()
    assert db.execute("SELECT count(*) FROM entity WHERE kind = 'place'").fetchone()[0] == 0
