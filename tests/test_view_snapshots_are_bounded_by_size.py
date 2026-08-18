"""Browsing a large library must not be able to exhaust memory.

Every full gallery render stores its computed file list server-side so
that /load_more and /api/current_view_ids can page through it. The store
kept the last thirty-two of those. Thirty-two of what was never bounded:
a snapshot holds one row per file in the view, and a view can be the whole
library.

Measured on a seeded library, rendering the recursive root view:

    entries in the snapshot : 5000
    keys per entry          : 26
    per file                : 2278 bytes

    at capacity (32 snapshots), a library of:
       10000 files ->    21.7 MB per snapshot,    695.1 MB held
       50000 files ->   108.6 MB per snapshot,   3475.4 MB held
      200000 files ->   434.4 MB per snapshot,  13901.8 MB held

Half of each row is workflow_prompt and workflow_files, and those were
modest stand-ins; real prompts run longer. Fifty thousand renders is an
ordinary few months of generating, and browsing thirty-two views is an
afternoon.

One snapshot of a large library is what paging through it costs, and the
page rendering it needs that snapshot to exist -- so the newest is never
evicted, however large. What is bounded is the multiplier. There is now a
ceiling on rows held across all snapshots as well as on their number, and
whichever bites first wins.
"""

from __future__ import annotations

import smartgallery


def _rows(n, prefix="f"):
    return [{"id": f"{prefix}{i}"} for i in range(n)]


def test_a_small_library_still_keeps_every_snapshot():
    """Over-reach guard, and the case almost everybody is in. Nothing
    about ordinary browsing may change: thirty-two views of a few
    thousand files each are nowhere near the ceiling."""
    store = smartgallery.ViewSnapshotStore(capacity=32)

    tokens = [store.put("", _rows(1000, f"v{i}_")) for i in range(32)]

    kept = [token for token in tokens if store.get(token, "") is not None]
    assert len(kept) == 32, (
        f"only {len(kept)} of 32 small snapshots survived; ordinary browsing has started losing views it used to keep"
    )


def test_the_count_ceiling_still_applies():
    """The bound that was already there has to stay."""
    store = smartgallery.ViewSnapshotStore(capacity=4)

    tokens = [store.put("", _rows(10, f"v{i}_")) for i in range(6)]

    assert store.get(tokens[0], "") is None
    assert store.get(tokens[1], "") is None
    assert store.get(tokens[-1], "") is not None


def test_rows_held_stay_under_the_ceiling():
    """The bug: thirty-two large views were all kept."""
    store = smartgallery.ViewSnapshotStore(capacity=32, max_rows=10_000)

    for i in range(32):
        store.put("", _rows(4_000, f"v{i}_"))

    held = sum(len(files) for _owner, files in store._snapshots.values())
    assert held <= 10_000, (
        f"{held} rows held across snapshots against a ceiling of 10000; at "
        f"the measured 2278 bytes a row that is {held * 2278 / 1048576:.0f} MB"
    )


def test_the_newest_view_is_kept_even_when_it_alone_is_too_big():
    """The one that must not regress into a fix. A library larger than the
    ceiling has to remain pageable: the render hands out this token and
    immediately asks for the next page with it."""
    store = smartgallery.ViewSnapshotStore(capacity=32, max_rows=10_000)
    store.put("", _rows(5_000, "old_"))

    token = store.put("", _rows(250_000, "huge_"))

    snapshot = store.get(token, "")
    assert snapshot is not None, (
        "a view bigger than the ceiling was evicted the moment it was "
        "stored, so a library that size cannot be paged at all"
    )
    assert len(snapshot) == 250_000


def test_paging_still_works_after_a_large_view_arrives(smartgallery_app, monkeypatch):
    """End to end, through the route that consumes these: the token from a
    large view must still serve its pages."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    store = smartgallery_app.ViewSnapshotStore(capacity=32, max_rows=1_000)
    monkeypatch.setattr(smartgallery_app, "VIEW_SNAPSHOTS", store)

    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    token = store.put("1", [{"id": f"big{i}"} for i in range(5_000)])
    page = client.get(f"/galleryout/load_more?view_token={token}&offset=0")

    assert page.status_code == 200, page.status_code
    assert page.get_json() is not None


def test_an_evicted_token_is_reported_as_expired():
    """Eviction is not silent corruption: the caller gets nothing back and
    the page re-renders, which is the contract that already existed for
    the count ceiling."""
    store = smartgallery.ViewSnapshotStore(capacity=32, max_rows=1_000)
    first = store.put("", _rows(900, "a_"))
    store.put("", _rows(900, "b_"))

    assert store.get(first, "") is None


def test_the_shipped_store_bounds_its_rows():
    """The store the gallery actually uses, not one built for a test."""
    store = smartgallery.VIEW_SNAPSHOTS

    assert getattr(store, "_max_rows", None), (
        "VIEW_SNAPSHOTS has no row ceiling, so its memory is whatever the "
        "library size multiplied by its snapshot count happens to be"
    )
    assert store._max_rows <= 200_000, store._max_rows


def test_owners_are_still_kept_apart():
    """Unchanged behaviour that the eviction work sits on top of."""
    store = smartgallery.ViewSnapshotStore(capacity=8)
    token = store.put("7", _rows(3))

    assert store.get(token, "7") is not None
    assert store.get(token, "9") is None
