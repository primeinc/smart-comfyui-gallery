"""A collection is one authored entity with a whole life.

Identity (entity + slug history), definition (name, kind, color,
description, parent), membership definition (filed rows or a typed
rule) and lifecycle (active or archived) are separable facts with one
owner each: db/collections.py owns the definition and the lifecycle,
db/collection_rules.py owns what a rule means, the ResultSet owns the
media answer. Every definition write is desired state under an
optimistic revision claim; membership never bumps the revision; archive
retires discoverability and nothing else; and no user-facing hard
delete exists, because an address that can someday resolve to a
different entity is the lie the whole addressing doctrine forbids.

A smart collection is a typed rule the ResultSet evaluates.

The rule owns MEMBERSHIP -- durable, uuid-referenced, actor-pinned,
never executable; the ResultSet still owns the ordered answer, so the
outer question's facets intersect the rule's members instead of
replacing them. Every failure state stays loud: unevaluated, broken
and unavailable are never presented as an empty collection.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3

import pytest
from PIL import Image

from db import authored, collection_rules, collections, connect, naming, resultset
from tests.staging import Stage, staged

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}


def _library(root: pathlib.Path) -> None:
    stamped = 1_700_000_000
    for i in range(6):
        path = root / f"pic_{i}.png"
        Image.new("RGB", (12, 12), (40 + i * 30, 120, 90)).save(path)
        os.utime(path, (stamped + i * 60, stamped + i * 60))


def _prepare(stage: Stage) -> None:
    client = stage.client
    for slug, stars in (("pic-1", 4), ("pic-3", 5), ("pic-4", 2)):
        client.post(f"/i/{slug}/rating", json={"value": stars})
    for slug in ("pic-1", "pic-2"):
        client.post(f"/i/{slug}/favorite", json={"value": True})


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_collection_has_a_lifecycle", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def curated(_stage):
    _stage.restore()
    return _stage.client


@pytest.fixture
def saved(curated):
    """The same world, under the name the smart-collection tests use."""
    return curated


def _view(client, slug: str) -> dict:
    told = client.get(f"/t/{slug}", headers=AS_MACHINE)
    assert told.status_code == 200, told.text
    return told.json()


def _raw(client) -> sqlite3.Connection:
    return connect.connect(client.app.state.db_path)


# --- creation --------------------------------------------------------------


def test_a_collection_is_born_whole_and_a_refused_smart_leaves_nothing(curated):
    """One lifecycle Module makes all three kinds; the answer is the
    authoritative CollectionView at revision 1; a smart collection whose
    rule refuses leaves NO collection behind."""
    made = curated.post(
        "/albums",
        json={"name": "  Portfolio ", "kind": "flag", "color": "#7C3AED", "description": "  keepers  "},
    )
    assert made.status_code == 201
    body = made.json()
    assert (body["name"], body["kind"], body["slug"]) == ("Portfolio", "flag", "portfolio")
    assert body["color"] == "#7c3aed", "one stored spelling, lowercased"
    assert body["description"] == "keepers", "whitespace is not authored state"
    assert (body["definition_rev"], body["archived"]) == (1, False)

    child = curated.post("/albums", json={"name": "Homepage", "parent": "portfolio"})
    assert child.status_code == 201
    assert child.json()["parent"] == "portfolio"

    assert curated.post("/albums", json={"name": "   "}).status_code == 400
    assert curated.post("/albums", json={"name": "Q", "kind": "smart"}).status_code == 400
    assert curated.post("/albums", json={"name": "Q", "color": "purple"}).status_code == 400

    doomed = curated.post("/albums/smart", json={"name": "Doomed", "q": "sunset"})  # semantic without take
    assert doomed.status_code == 400
    conn = _raw(curated)
    try:
        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Doomed'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entity WHERE slug = 'doomed'").fetchone()[0] == 0
    finally:
        connect.close(conn)


# --- the definition patch --------------------------------------------------


def test_rename_is_one_operation_with_a_permanent_forwarding_address(curated):
    curated.post("/albums", json={"name": "Keepers"})
    told = curated.patch("/t/keepers", json={"name": "Best Keepers", "expected_rev": 1})
    assert told.status_code == 200, told.text
    body = told.json()
    assert (body["slug"], body["name"], body["definition_rev"]) == ("best-keepers", "Best Keepers", 2)
    moved = curated.get("/t/keepers", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/t/best-keepers")
    assert _view(curated, "best-keepers")["name"] == "Best Keepers"


def test_description_and_color_are_desired_facts_with_null_to_clear(curated):
    curated.post("/albums", json={"name": "Moody"})
    told = curated.patch("/t/moody", json={"description": "blue hour", "color": "#123ABC", "expected_rev": 1}).json()
    assert (told["description"], told["color"], told["definition_rev"]) == ("blue hour", "#123abc", 2)

    # Absent means UNCHANGED: a rename touches nothing it did not name.
    told = curated.patch("/t/moody", json={"name": "Moodier", "expected_rev": 2}).json()
    assert (told["description"], told["color"]) == ("blue hour", "#123abc")

    # Null means CLEARED, and the clear retried is still cleared.
    told = curated.patch("/t/moodier", json={"description": None, "color": None, "expected_rev": 3}).json()
    assert (told["description"], told["color"], told["definition_rev"]) == (None, None, 4)

    assert curated.patch("/t/moodier", json={"color": "papayawhip", "expected_rev": 4}).status_code == 400
    assert curated.patch("/t/moodier", json={"expected_rev": 4}).status_code == 400, "an empty patch says nothing"
    strange = curated.patch("/t/moodier", json={"colour": "#000000", "expected_rev": 4})
    assert strange.status_code == 400, "a misspelled fact is refused, never silently ignored"
    assert _view(curated, "moodier")["definition_rev"] == 4, "refusals moved nothing"


def test_reparenting_is_desired_state_with_friendly_refusals(curated):
    for name in ("Trips", "Florida", "Michigan"):
        curated.post("/albums", json={"name": name})
    assert curated.patch("/t/florida", json={"parent": "trips", "expected_rev": 1}).json()["parent"] == "trips"
    assert curated.patch("/t/michigan", json={"parent": "florida", "expected_rev": 1}).json()["parent"] == "florida"

    assert curated.patch("/t/trips", json={"parent": "trips", "expected_rev": 1}).status_code == 400
    descended = curated.patch("/t/trips", json={"parent": "michigan", "expected_rev": 1})
    assert descended.status_code == 400, "a collection cannot move under its own descendant"
    assert curated.patch("/t/trips", json={"parent": "never-was", "expected_rev": 1}).status_code == 400
    assert _view(curated, "trips")["parent"] is None, "every refusal moved nothing"
    assert _view(curated, "trips")["definition_rev"] == 1

    told = curated.patch("/t/michigan", json={"parent": None, "expected_rev": 2}).json()
    assert told["parent"] is None, "null is the top of the hierarchy"

    # The database trigger stays the backstop against raw writes.
    conn = _raw(curated)
    try:
        trips, florida = (
            conn.execute("SELECT id FROM collection WHERE name = ?", (name,)).fetchone()[0]
            for name in ("Trips", "Florida")
        )
        with pytest.raises(sqlite3.IntegrityError, match="cycle"):
            conn.execute("UPDATE collection SET parent_id = ? WHERE id = ?", (florida, trips))
    finally:
        connect.close(conn)


def test_the_parents_offer_excludes_self_and_descendants(curated):
    for name in ("Trips", "Florida", "Michigan", "Elsewhere", "Retired"):
        curated.post("/albums", json={"name": name})
    curated.patch("/t/florida", json={"parent": "trips", "expected_rev": 1})
    curated.patch("/t/michigan", json={"parent": "florida", "expected_rev": 1})
    curated.patch("/t/retired", json={"archived": True, "expected_rev": 1})

    told = curated.patch("/t/trips", json={"description": "x", "expected_rev": 1}).json()
    offered = {row["slug"] for row in told["parents"]}
    assert "elsewhere" in offered
    assert offered & {"trips", "florida", "michigan"} == set(), "self and subtree are never offered"
    assert "retired" not in offered, "an archived collection is not an offered organizer"


def test_a_parent_cannot_be_deleted_out_from_under_its_children(curated):
    """ON DELETE RESTRICT: authored children have independent addresses,
    and destroying a hierarchy is an explicit act -- there is no route
    for it, and even raw SQL cannot cascade through the children."""
    curated.post("/albums", json={"name": "Trips"})
    curated.post("/albums", json={"name": "Michigan", "parent": "trips"})
    curated.post("/albums", json={"name": "Loner"})
    conn = _raw(curated)
    try:
        trips, loner = (
            conn.execute("SELECT id FROM collection WHERE name = ?", (name,)).fetchone()[0]
            for name in ("Trips", "Loner")
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM collection WHERE id = ?", (trips,))
        conn.rollback()
        conn.execute("DELETE FROM collection WHERE id = ?", (loner,))  # childless: the RESTRICT is not a blanket
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Michigan'").fetchone()[0] == 1
    finally:
        connect.close(conn)


def test_no_user_facing_hard_delete_exists(curated):
    curated.post("/albums", json={"name": "Keepers"})
    assert curated.delete("/t/keepers").status_code == 405


# --- optimistic concurrency ------------------------------------------------


def test_a_stale_definition_edit_refuses_with_zero_mutation(curated):
    curated.post("/albums", json={"name": "Draft"})
    first = curated.patch("/t/draft", json={"description": "mine", "expected_rev": 1})
    assert first.status_code == 200
    stale = curated.patch("/t/draft", json={"description": "no, mine", "expected_rev": 1})
    assert stale.status_code == 409
    body = _view(curated, "draft")
    assert (body["description"], body["definition_rev"]) == ("mine", 2), "the stale editor overwrote nothing"

    # The concurrency interface is expected_rev in the body, nowhere
    # else: a header token arrives unbound to the collection in the URL,
    # so NO header may authorize a write -- not even one spelling this
    # collection's own current revision.
    for header in ('W/"draft-r2"', 'W/"some-other-collection-r2"'):
        veiled = curated.patch("/t/draft", json={"description": "x"}, headers={"if-match": header})
        assert veiled.status_code == 400, "a header token must never authorize a definition write"
    assert _view(curated, "draft")["description"] == "mine"
    assert "etag" not in curated.get("/t/draft", headers=AS_MACHINE).headers, (
        "no validator is emitted that could pretend to mean the definition"
    )

    assert curated.patch("/t/draft", json={"description": "x"}).status_code == 400, "no named revision, no write"
    assert curated.patch("/t/draft", json={"description": "x", "expected_rev": True}).status_code == 400


def test_membership_never_creates_a_definition_conflict(curated):
    """Filing pictures is not editing the definition: an open editor's
    revision stays valid through any amount of curation."""
    curated.post("/albums", json={"name": "Keepers"})
    read_rev = _view(curated, "keepers")["definition_rev"]
    curated.post("/t/keepers/add", json={"file": "pic-0"})
    curated.post("/t/keepers/add", json={"file": "pic-1"})
    curated.post("/i/pic-2/collections/keepers", json={"value": True})
    assert _view(curated, "keepers")["definition_rev"] == read_rev
    saved = curated.patch("/t/keepers", json={"description": "still valid", "expected_rev": read_rev})
    assert saved.status_code == 200, "membership manufactured a false definition conflict"
    assert saved.json()["count"] == 3


# --- the rule and the kind transitions -------------------------------------


def test_rule_replacement_is_whole_desired_state(curated):
    made = curated.post("/albums/smart", json={"name": "Starred", "rating_min": 4}).json()
    assert [row["slug"] for row in made["gallery"]["items"]] == ["pic-3", "pic-1"]

    told = curated.put("/t/starred/rule", json={"favorite": "1", "expected_rev": 1})
    assert told.status_code == 200, told.text
    body = told.json()
    assert body["definition_rev"] == 2, "a new meaning is a new definition revision"
    assert [row["slug"] for row in body["gallery"]["items"]] == ["pic-2", "pic-1"], "the WHOLE rule was replaced"

    assert curated.put("/t/starred/rule", json={"rating_min": 2, "expected_rev": 1}).status_code == 409
    assert [row["slug"] for row in _view(curated, "starred")["gallery"]["items"]] == ["pic-2", "pic-1"]

    curated.post("/albums", json={"name": "Listed"})
    assert curated.put("/t/listed/rule", json={"favorite": "1", "expected_rev": 1}).status_code == 400


def test_album_and_flag_exchange_freely_with_members_kept(curated):
    curated.post("/albums", json={"name": "Keepers"})
    for slug in ("pic-0", "pic-5"):
        curated.post("/t/keepers/add", json={"file": slug})
    told = curated.post("/t/keepers/convert", json={"kind": "flag", "expected_rev": 1})
    assert told.status_code == 201, told.text
    body = told.json()
    assert (body["kind"], body["count"], body["definition_rev"]) == ("flag", 2, 2)
    assert {row["slug"] for row in body["gallery"]["items"]} == {"pic-0", "pic-5"}


def test_becoming_smart_requires_emptiness_and_a_rule_in_one_act(curated):
    curated.post("/albums", json={"name": "Full"})
    curated.post("/t/full/add", json={"file": "pic-0"})
    refused = curated.post("/t/full/convert", json={"kind": "smart", "rating_min": 4, "expected_rev": 1})
    assert refused.status_code == 400
    assert "filed member" in refused.json()["detail"]

    curated.post("/albums", json={"name": "Empty"})
    half = curated.post("/t/empty/convert", json={"kind": "smart", "q": "sunset", "expected_rev": 1})
    assert half.status_code == 400, "a rule that refuses converts nothing"
    body = _view(curated, "empty")
    assert (body["kind"], body["definition_rev"]) == ("album", 1), "no half-made smart object exists"

    told = curated.post("/t/empty/convert", json={"kind": "smart", "rating_min": 4, "expected_rev": 1}).json()
    assert (told["kind"], told["state"], told["definition_rev"]) == ("smart", "evaluated", 2)
    assert [row["slug"] for row in told["gallery"]["items"]] == ["pic-3", "pic-1"]


def test_leaving_smart_requires_the_rules_discard_said_out_loud(curated):
    curated.post("/albums/smart", json={"name": "Starred", "rating_min": 4})
    kept = curated.post("/t/starred/convert", json={"kind": "album", "expected_rev": 1})
    assert kept.status_code == 400
    assert "discard" in kept.json()["detail"]
    assert _view(curated, "starred")["kind"] == "smart"

    told = curated.post("/t/starred/convert", json={"kind": "album", "discard_rule": True, "expected_rev": 1}).json()
    assert (told["kind"], told["definition_rev"], told["count"]) == ("album", 2, 0)
    conn = _raw(curated)
    try:
        assert conn.execute("SELECT count(*) FROM collection_rule").fetchone()[0] == 0, "the rule was discarded"
    finally:
        connect.close(conn)
    filed = curated.post("/t/starred/add", json={"file": "pic-0"})
    assert filed.status_code == 201, "a listed collection takes filings again"


# --- archive ---------------------------------------------------------------


def test_archive_is_a_lifecycle_not_a_deletion(curated):
    curated.post("/albums", json={"name": "Old Project"})
    curated.post("/albums", json={"name": "Still Going", "parent": "old-project"})
    for slug in ("pic-0", "pic-1"):
        curated.post("/t/old-project/add", json={"file": slug})

    told = curated.patch("/t/old-project", json={"archived": True, "expected_rev": 1}).json()
    assert (told["archived"], told["definition_rev"]) == (True, 2)

    # Identity, members, children and address all stand.
    body = _view(curated, "old-project")
    assert (body["archived"], body["count"]) == (True, 2)
    assert [child["slug"] for child in body["collections"]] == ["still-going"]

    # Discoverability is what changed: off the shelf, out of the picker,
    # while the ACTIVE child surfaces at the top instead of vanishing.
    shelf = {row["slug"] for row in curated.get("/albums", headers=AS_MACHINE).json()}
    assert "old-project" not in shelf
    assert "still-going" in shelf
    page = curated.get("/albums", headers=AS_BROWSER).text
    assert 'data-album="old-project"' not in page
    assert 'data-album="still-going"' in page, "an active child does not disappear with its organizer"
    choices = {row["slug"] for row in curated.get("/i/pic-3/collection-choices").json()}
    assert "old-project" not in choices
    assert "still-going" in choices
    retired = curated.get("/albums", params={"state": "archived"}, headers=AS_MACHINE).json()
    assert [row["slug"] for row in retired] == ["old-project"]

    # Desired state: archiving what is archived keeps the original fact.
    conn = _raw(curated)
    try:
        first = conn.execute("SELECT archived_at FROM collection WHERE name = 'Old Project'").fetchone()[0]
    finally:
        connect.close(conn)
    curated.patch("/t/old-project", json={"archived": True, "expected_rev": 2})
    conn = _raw(curated)
    try:
        assert conn.execute("SELECT archived_at FROM collection WHERE name = 'Old Project'").fetchone()[0] == first
    finally:
        connect.close(conn)

    restored = curated.patch("/t/old-project", json={"archived": False, "expected_rev": 3}).json()
    assert (restored["archived"], restored["slug"]) == (False, "old-project"), "the SAME entity and address return"
    assert "old-project" in {row["slug"] for row in curated.get("/albums", headers=AS_MACHINE).json()}
    assert _view(curated, "old-project")["count"] == 2


def test_an_archived_parent_survives_an_unrelated_edit(curated):
    """The select-fallback trap, closed at the seam: the offer can spell
    the state that already holds, so editing a child beneath an archived
    parent never silently reparents it -- and an archived collection is
    still not a NEW destination."""
    curated.post("/albums", json={"name": "Old Project"})
    curated.post("/albums", json={"name": "Still Going", "parent": "old-project"})
    curated.post("/albums", json={"name": "Retired Too"})
    curated.patch("/t/old-project", json={"archived": True, "expected_rev": 1})
    curated.patch("/t/retired-too", json={"archived": True, "expected_rev": 1})

    told = curated.patch("/t/still-going", json={"description": "still going", "expected_rev": 1}).json()
    assert told["parent"] == "old-project", "an unrelated edit must not move the child"
    offered = {row["slug"]: row["archived"] for row in told["parents"]}
    assert offered.get("old-project") is True, "the archived CURRENT parent is offered, marked"
    assert "retired-too" not in offered, "other archived collections are not destinations"

    kept = curated.patch("/t/still-going", json={"parent": "old-project", "expected_rev": 2})
    assert kept.status_code == 200, "saying the current state out loud is always legal"
    moved = curated.patch("/t/still-going", json={"parent": "retired-too", "expected_rev": 3})
    assert moved.status_code == 400
    assert "restore" in moved.json()["detail"]
    assert _view(curated, "still-going")["parent"] == "old-project"


def test_creation_obeys_the_same_parent_doctrine_as_moving(curated):
    """One definition of "legal parent" however the child comes to
    exist: an archived collection takes no NEW children from a move OR
    from a creation, through HTTP and through the Module with the
    refusal caught and the transaction committed."""
    curated.post("/albums", json={"name": "Old Project"})
    assert curated.post("/albums", json={"name": "Early", "parent": "old-project"}).status_code == 201
    assert (
        curated.post("/albums/smart", json={"name": "Early Smart", "rating_min": 4, "parent": "old-project"})
    ).status_code == 201
    curated.patch("/t/old-project", json={"archived": True, "expected_rev": 1})

    listed = curated.post("/albums", json={"name": "Too Late", "parent": "old-project"})
    assert listed.status_code == 400
    assert "restore" in listed.json()["detail"]
    smart = curated.post("/albums/smart", json={"name": "Too Late Smart", "rating_min": 4, "parent": "old-project"})
    assert smart.status_code == 400
    assert "restore" in smart.json()["detail"]

    conn = _raw(curated)
    try:
        parent = conn.execute("SELECT id FROM collection WHERE name = 'Old Project'").fetchone()[0]
        actor = curated.app.state.actor_id
        with pytest.raises(ValueError, match="restore"):
            collections.create_listed(conn, "Too Late", 5.0, parent_id=parent, actor_id=actor)
        conn.commit()
        rule = collection_rules.from_gallery_query(conn, resultset.parse(rating_min=4), actor_id=actor, take=None)
        with pytest.raises(ValueError, match="restore"):
            collections.create_smart(conn, "Too Late Smart", rule, None, 6.0, parent_id=parent, actor_id=actor)
        conn.commit()
        for name in ("Too Late", "Too Late Smart"):
            assert conn.execute("SELECT count(*) FROM collection WHERE name = ?", (name,)).fetchone()[0] == 0
        for slug in ("too-late", "too-late-smart"):
            assert conn.execute("SELECT count(*) FROM entity WHERE slug = ?", (slug,)).fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_rule r JOIN collection c ON c.id = r.collection_id"
                " WHERE c.name LIKE 'Too Late%'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connect.close(conn)


def test_a_refused_transition_leaves_the_callers_transaction_untouched(curated):
    """The Module's invariant, tested the hostile way: a direct caller
    catches the refusal and COMMITS anyway -- and nothing partial
    persists, because every domain check precedes the first mutation and
    the revision claim leads every multi-step transition."""
    curated.post("/albums/smart", json={"name": "Starred", "rating_min": 4})
    actor = curated.app.state.actor_id
    bad = collection_rules.CollectionRule(
        version=2,
        folder_uuid=None,
        person_uuid=None,
        artifact_uuid=None,
        kind="platypus",
        favorite=None,
        rating_min=None,
        text=None,
        sort=None,
        take=None,
        actor_id=None,
    )
    conn = _raw(curated)
    try:
        starred = conn.execute("SELECT id FROM collection WHERE name = 'Starred'").fetchone()[0]

        def held():
            return conn.execute(
                "SELECT c.kind, c.definition_rev,"
                " (SELECT count(*) FROM collection_rule r WHERE r.collection_id = c.id)"
                " FROM collection c WHERE c.id = ?",
                (starred,),
            ).fetchone()

        # A stale smart->listed must refuse BEFORE the rule is deleted.
        with pytest.raises(collections.CollectionChanged):
            collections.convert_to_listed(conn, starred, "album", actor, 99, 5.0, discard_rule=True)
        conn.commit()
        assert held() == ("smart", 1, 1), "a caught stale refusal, committed, deleted the authored rule"

        # An invalid rule must refuse before any revision or rule write.
        before = conn.execute("SELECT rule_json FROM collection_rule WHERE collection_id = ?", (starred,)).fetchone()
        with pytest.raises(ValueError, match="kind"):
            collections.replace_rule(conn, starred, bad, None, actor, 1, 6.0)
        conn.commit()
        assert held() == ("smart", 1, 1)
        assert (
            conn.execute("SELECT rule_json FROM collection_rule WHERE collection_id = ?", (starred,)).fetchone()
            == before
        )

        # An invalid rule at creation leaves neither collection nor entity.
        with pytest.raises(ValueError, match="kind"):
            collections.create_smart(conn, "Ghost", bad, None, 7.0, actor_id=actor)
        conn.commit()
        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Ghost'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entity WHERE slug = 'ghost'").fetchone()[0] == 0

        # An invalid rule at conversion leaves the listed definition whole.
        plain = collections.collection(conn, "Plain", 8.0)
        conn.commit()
        with pytest.raises(ValueError, match="kind"):
            collections.convert_to_smart(conn, plain, bad, None, actor, 1, 9.0)
        conn.commit()
        row = conn.execute("SELECT kind, definition_rev FROM collection WHERE id = ?", (plain,)).fetchone()
        assert row == ("album", 1)
        assert conn.execute("SELECT count(*) FROM collection_rule WHERE collection_id = ?", (plain,)).fetchone()[0] == 0
    finally:
        connect.close(conn)


# --- one implementation, pinned --------------------------------------------


def test_the_view_is_authoritative_after_every_write(curated):
    """The write's answer and a fresh GET agree on every definition
    fact -- the browser renders what the server read back, never what
    it hoped its click did."""
    curated.post("/albums", json={"name": "Keepers"})
    written = curated.patch(
        "/t/keepers", json={"name": "Kept", "color": "#001122", "description": "d", "expected_rev": 1}
    ).json()
    read = _view(curated, "kept")
    for fact in ("slug", "name", "kind", "color", "description", "parent", "archived", "definition_rev"):
        assert written[fact] == read[fact], f"the write invented its own {fact}"


# --- a smart collection is a saved question --------------------------------


def _slugs(client, **params) -> list[str]:
    import re

    return re.findall(r'data-slug="([^"]+)"', client.get("/g", params=params).text)


def test_a_saved_view_is_the_same_membership_the_resultset_answers(saved):
    made = saved.post("/albums/smart", json={"name": "Best stills", "rating_min": 4, "kind": "image"})
    assert made.status_code < 300
    slug = made.json()["slug"]
    assert _slugs(saved, album=slug) == _slugs(saved, rating_min=4, kind="image") == ["pic-3", "pic-1"]

    # /t/{smart} is the SAME projection /g?album= reads.
    told = saved.get(f"/t/{slug}", headers=AS_MACHINE).json()
    assert told["state"] == "evaluated"
    assert [row["slug"] for row in told["gallery"]["items"]] == ["pic-3", "pic-1"]
    assert told["count"] == 2

    # Membership is DERIVED: a later rating change moves it with no
    # collection_file row anywhere near a smart collection.
    saved.post("/i/pic-4/rating", json={"value": 5})
    assert _slugs(saved, album=slug) == ["pic-4", "pic-3", "pic-1"]
    conn = connect.connect(saved.app.state.db_path)
    filed = conn.execute(
        "SELECT count(*) FROM collection_file cf JOIN collection c ON c.id = cf.collection_id WHERE c.kind = 'smart'"
    ).fetchone()[0]
    connect.close(conn)
    assert filed == 0

    # The outer question's facets INTERSECT the rule's members.
    assert _slugs(saved, album=slug, favorite="1") == ["pic-1"]
    assert _slugs(saved, album=slug, kind="video") == []


def test_the_rule_survives_renames_because_it_holds_the_entity(saved):
    made = saved.post("/albums/smart", json={"name": "In the library", "folder": "lib"})
    slug = made.json()["slug"]
    before = _slugs(saved, album=slug)
    assert len(before) == 6

    conn = connect.connect(saved.app.state.db_path)
    found = naming.resolve(conn, "folder", "lib")
    assert found is not None
    naming.rename(conn, found[0], "library prime", 5.0)
    conn.commit()
    connect.close(conn)

    assert _slugs(saved, album=slug) == before, "a renamed folder must not break or empty the rule"


def test_the_actor_is_pinned_at_creation_never_the_viewer(saved):
    made = saved.post("/albums/smart", json={"name": "My favorites", "favorite": "1"})
    slug = made.json()["slug"]
    mine = saved.app.state.actor_id

    conn = connect.connect(saved.app.state.db_path)
    guest = authored.add_user(conn, "guest", "!", "USER", 0.0)
    pic5 = conn.execute("SELECT id FROM file WHERE name = 'pic_5.png'").fetchone()[0]
    authored.set_favorite(conn, pic5, guest, True, 0.0)
    conn.commit()
    asked = resultset.parse(album=slug)
    for viewer in (mine, guest, None):
        told = resultset.describe(conn, "", asked, 0.0, actor_id=viewer)
        assert told["total"] == 2, "the rule answers the CREATOR's favorites for every viewer"
    # The viewer's own authored facet composes on top of the pinned rule.
    both = resultset.describe(conn, "", resultset.parse(album=slug, favorite="1"), 0.0, actor_id=guest)
    assert both["total"] == 0, "guest favorited none of the creator's favorites"
    connect.close(conn)


def test_a_semantic_rule_needs_take_and_take_bounds_the_set(saved, monkeypatch):
    from db import retrieval

    refused = saved.post("/albums/smart", json={"name": "Sunsets", "q": "sunset"})
    assert refused.status_code == 400
    assert "take" in refused.json()["detail"]

    conn = connect.connect(saved.app.state.db_path)
    ranked = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id")]
    connect.close(conn)

    def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        held = [i for i in ranked if allowed is None or i in allowed]
        return {
            "results": [{"file_id": i, "score": 1.0, "sources": {}} for i in held],
            "participants": ["fake"],
            "contributors": ["fake"],
            "missing": {},
        }

    monkeypatch.setattr(retrieval, "query", fused)
    made = saved.post("/albums/smart", json={"name": "Sunsets", "q": "sunset", "take": 2})
    slug = made.json()["slug"]
    members = _slugs(saved, album=slug)
    assert len(members) == 2, "take must cut the ranked answer down to a set"

    # And a semantic rule nothing can answer is UNAVAILABLE, never empty.
    def refuses(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        raise LookupError("no space can answer")

    monkeypatch.setattr(retrieval, "query", refuses)
    saved.post("/i/pic-0/favorite", json={"value": True})  # move the currency past the cached answer
    body = saved.get(f"/t/{slug}", headers=AS_MACHINE).json()
    assert body["state"] == "unavailable"
    assert body["gallery"] is None
    assert saved.get("/g", params={"album": slug}).status_code == 400


def test_the_failure_states_are_loud_and_distinct(saved):
    conn = connect.connect(saved.app.state.db_path)
    prose = collections.collection(conn, "Old prose", 1.0, kind="smart")
    collection_rules.keep_prose(conn, prose, sql="SELECT 1", now=1.0)
    conn.commit()
    connect.close(conn)
    told = saved.get("/t/old-prose", headers=AS_MACHINE).json()
    assert told["state"] == "unevaluated"
    assert told["rule"] == {"sql": "SELECT 1", "nl": None}, "the preserved prose is shown, never run"
    assert told["gallery"] is None
    assert saved.get("/g", params={"album": "old-prose"}).status_code == 400

    # BROKEN: the referenced entity is gone -- said by name, never empty.
    made = saved.post("/albums/smart", json={"name": "Doomed", "folder": "lib"})
    slug = made.json()["slug"]
    conn = connect.connect(saved.app.state.db_path)
    found = naming.resolve(conn, "folder", "lib")
    assert found is not None
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM file WHERE folder_id = ?", (found[0],))
    conn.execute("DELETE FROM folder WHERE id = ?", (found[0],))
    conn.commit()
    connect.close(conn)
    body = saved.get(f"/t/{slug}", headers=AS_MACHINE).json()
    assert body["state"] == "broken"
    assert "no longer exists" in body["reason"]
    assert body["gallery"] is None
    assert saved.get("/g", params={"album": slug}).status_code == 400


def test_the_albums_index_never_evaluates_a_rule(saved, monkeypatch):
    saved.post("/albums/smart", json={"name": "Lazy", "rating_min": 3})
    asked: list[str] = []
    for name in ("page", "describe", "peek", "locate"):
        real = getattr(resultset, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            asked.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(resultset, name, counted)
    assert saved.get("/albums", headers=AS_BROWSER).status_code == 200
    assert saved.get("/albums", headers=AS_MACHINE).status_code == 200
    assert asked == [], "the album shelf launched a rule evaluation nobody asked for"


def test_an_edited_rule_is_a_new_question(saved):
    made = saved.post("/albums/smart", json={"name": "Moving", "rating_min": 5})
    slug = made.json()["slug"]
    assert _slugs(saved, album=slug) == ["pic-3"]

    conn = connect.connect(saved.app.state.db_path)
    found = naming.resolve(conn, "collection", slug)
    assert found is not None
    fresh = collection_rules.from_gallery_query(
        conn, resultset.parse(rating_min=2), actor_id=saved.app.state.actor_id, take=None
    )
    collection_rules.save(conn, found[0], fresh, source_text="rating_min=2", now=9.0)
    conn.commit()
    connect.close(conn)

    assert _slugs(saved, album=slug) == ["pic-4", "pic-3", "pic-1"], (
        "the edited rule must answer immediately through normal currency"
    )


def _rotten(where=None, select=None, v: int | float = 1) -> str:
    import json

    base = {
        "v": v,
        "where": {"folder": None, "person": None, "kind": None, "favorite": None, "rating_min": None},
        "select": {"sort": None, "text": None, "take": None},
    }
    base["where"].update(where or {})
    base["select"].update(select or {})
    return json.dumps(base)


def test_a_corrupt_stored_rule_is_broken_never_empty(saved):
    """json_valid is syntax; the load gate is MEANING: any semantically
    rotten stored rule -- wrong version, disagreeing versions, vocabulary
    from another planet, a truncated uuid, an authored facet with no
    pinned actor -- is BROKEN by name, never an evaluated empty
    collection."""
    made = saved.post("/albums/smart", json={"name": "Fragile", "kind": "image"})
    slug = made.json()["slug"]
    conn = connect.connect(saved.app.state.db_path)
    fragile = conn.execute("SELECT id FROM collection WHERE name = 'Fragile'").fetchone()[0]

    cases = [
        (47, _rotten(v=47), None),  # a version this build does not understand
        (1, _rotten(v=2), None),  # the column and the stored form disagree
        (1, _rotten(where={"kind": "platypus"}), None),
        (1, _rotten(where={"favorite": "banana"}), None),
        (1, _rotten(where={"rating_min": 700}), None),
        (1, _rotten(select={"take": -5, "sort": "newest"}), None),
        (1, _rotten(select={"sort": "Tuesday"}), None),  # a sort with nothing to cut
        (1, _rotten(where={"folder": "zz"}), None),  # not hex at all
        (1, _rotten(where={"folder": "aabb"}), None),  # hex, but not 16 bytes
        (1, _rotten(where={"favorite": True}), None),  # authored facet, no pinned actor
        # The JSON type system's truthiness corners: falsy is not null,
        # and bool is not an integer however Python coerces them.
        (1, _rotten(where={"folder": ""}), None),
        (1, _rotten(where={"folder": False}), None),
        (1, _rotten(where={"rating_min": True}), 1),
        (1, _rotten(select={"take": True, "sort": "newest"}), None),
        (1, _rotten(v=True), None),
        (1, _rotten(v=1.0), None),
    ]
    for version, payload, actor in cases:
        conn.execute(
            "UPDATE collection_rule SET rule_version = ?, rule_json = ?, actor_id = ? WHERE collection_id = ?",
            (version, payload, actor, fragile),
        )
        conn.commit()
        body = saved.get(f"/t/{slug}", headers=AS_MACHINE).json()
        assert body["state"] == "broken", f"{payload} was not refused"
        assert body["gallery"] is None
        assert saved.get("/g", params={"album": slug}).status_code == 400
    connect.close(conn)


def test_the_persistence_interface_validates_what_it_is_handed(saved):
    """save() owns its invariant: a semantically rotten CollectionRule
    built by hand -- not through from_gallery_query -- is refused at the
    persistence seam, never written for load() to trip over later."""
    conn = connect.connect(saved.app.state.db_path)
    smart = collections.collection(conn, "Handmade", 1.0, kind="smart")
    bad = collection_rules.CollectionRule(
        version=2,
        folder_uuid=None,
        person_uuid=None,
        artifact_uuid=None,
        kind="platypus",
        favorite=None,
        rating_min=None,
        text=None,
        sort=None,
        take=None,
        actor_id=None,
    )
    with pytest.raises(ValueError, match="kind"):
        collection_rules.save(conn, smart, bad, source_text=None, now=1.0)
    assert conn.execute("SELECT count(*) FROM collection_rule WHERE collection_id = ?", (smart,)).fetchone()[0] == 0
    connect.close(conn)


def test_exactly_one_membership_definition_per_collection(saved):
    """The v8 guards, both lanes: a rule belongs only to a smart
    collection -- refused politely by the module and structurally by the
    schema -- and a rule-carrying smart collection cannot quietly become
    listed. Deleting the rule first is the deliberate transition."""
    import sqlite3 as sqlite_module

    conn = connect.connect(saved.app.state.db_path)
    listed = collections.collection(conn, "Listed", 1.0)
    with pytest.raises(ValueError, match="smart"):
        collection_rules.keep_prose(conn, listed, nl="x", now=1.0)
    rule = collection_rules.from_gallery_query(conn, resultset.parse(kind="image"), actor_id=None, take=None)
    with pytest.raises(ValueError, match="smart"):
        collection_rules.save(conn, listed, rule, source_text=None, now=1.0)
    with pytest.raises(sqlite_module.IntegrityError):
        conn.execute("INSERT INTO collection_rule(collection_id, created_at, updated_at) VALUES(?, 1, 1)", (listed,))
    conn.commit()
    connect.close(conn)

    made = saved.post("/albums/smart", json={"name": "Committed", "kind": "image"})
    assert made.status_code < 300
    conn = connect.connect(saved.app.state.db_path)
    committed = conn.execute("SELECT id FROM collection WHERE name = 'Committed'").fetchone()[0]
    with pytest.raises(sqlite_module.IntegrityError, match="rule-defined"):
        conn.execute("UPDATE collection SET kind = 'album' WHERE id = ?", (committed,))
    conn.execute("DELETE FROM collection_rule WHERE collection_id = ?", (committed,))
    conn.execute("UPDATE collection SET kind = 'album' WHERE id = ?", (committed,))
    conn.commit()
    connect.close(conn)
