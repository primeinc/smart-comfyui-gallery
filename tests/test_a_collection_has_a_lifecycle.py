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

import json
import os
import pathlib
import re
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


def _view(client, slug: str) -> dict:
    told = client.get(f"/t/{slug}", headers=AS_MACHINE)
    assert told.status_code == 200, told.text
    return told.json()


def _raw(client) -> sqlite3.Connection:
    return connect.connect(client.app.state.db_path)


def _made(client, path: str, **body) -> dict:
    """A collection created as Arrange: the write must succeed."""
    told = client.post(path, json=body)
    assert told.status_code == 201, told.text
    return told.json()


def _slugs(client, **params) -> list[str]:
    return re.findall(r'data-slug="([^"]+)"', client.get("/g", params=params).text)


# --- shared Arrange --------------------------------------------------------


@pytest.fixture
def keepers(curated) -> str:
    """A listed collection at revision 1; the value is its slug."""
    return _made(curated, "/albums", name="Keepers")["slug"]


@pytest.fixture
def moody(curated) -> str:
    """A listed collection with a description and a color, at revision 2."""
    _made(curated, "/albums", name="Moody")
    told = curated.patch("/t/moody", json={"description": "blue hour", "color": "#123ABC", "expected_rev": 1})
    assert told.status_code == 200, told.text
    return "moody"


@pytest.fixture
def hierarchy(curated) -> None:
    """Trips > Florida > Michigan, built by reparenting; Florida and
    Michigan stand at revision 2, Trips at 1."""
    for name in ("Trips", "Florida", "Michigan"):
        _made(curated, "/albums", name=name)
    assert curated.patch("/t/florida", json={"parent": "trips", "expected_rev": 1}).status_code == 200
    assert curated.patch("/t/michigan", json={"parent": "florida", "expected_rev": 1}).status_code == 200


@pytest.fixture
def starred(curated) -> str:
    """A smart collection, rating_min=4, at revision 1: pic-3 and pic-1."""
    made = _made(curated, "/albums/smart", name="Starred", rating_min=4)
    assert [row["slug"] for row in made["gallery"]["items"]] == ["pic-3", "pic-1"]
    return made["slug"]


@pytest.fixture
def old_project(curated) -> str:
    """A listed collection holding pic-0 and pic-1, with an active child
    Still Going; not yet archived."""
    _made(curated, "/albums", name="Old Project")
    _made(curated, "/albums", name="Still Going", parent="old-project")
    for slug in ("pic-0", "pic-1"):
        assert curated.post("/t/old-project/add", json={"file": slug}).status_code == 201
    return "old-project"


@pytest.fixture
def archived_project(curated, old_project) -> str:
    """Old Project archived at revision 2."""
    told = curated.patch(f"/t/{old_project}", json={"archived": True, "expected_rev": 1})
    assert told.status_code == 200, told.text
    return old_project


@pytest.fixture
def archived_parent_with_child(curated) -> None:
    """Still Going under the archived Old Project, and Retired Too
    archived beside them."""
    _made(curated, "/albums", name="Old Project")
    _made(curated, "/albums", name="Still Going", parent="old-project")
    _made(curated, "/albums", name="Retired Too")
    assert curated.patch("/t/old-project", json={"archived": True, "expected_rev": 1}).status_code == 200
    assert curated.patch("/t/retired-too", json={"archived": True, "expected_rev": 1}).status_code == 200


@pytest.fixture
def bad_rule() -> collection_rules.CollectionRule:
    """A semantically rotten rule built by hand, not through from_gallery_query."""
    return collection_rules.CollectionRule(
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


# --- creation --------------------------------------------------------------


def test_creation_answers_the_authoritative_view_with_authored_fields_normalized(curated):
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


def test_a_collection_can_be_created_with_a_parent(curated):
    _made(curated, "/albums", name="Portfolio")

    child = curated.post("/albums", json={"name": "Homepage", "parent": "portfolio"})

    assert child.status_code == 201
    assert child.json()["parent"] == "portfolio"


def test_creation_refuses_a_blank_name(curated):
    assert curated.post("/albums", json={"name": "   "}).status_code == 400


def test_a_smart_collection_needs_a_rule_not_just_the_kind(curated):
    assert curated.post("/albums", json={"name": "Q", "kind": "smart"}).status_code == 400


def test_creation_refuses_a_color_that_is_not_hex(curated):
    assert curated.post("/albums", json={"name": "Q", "color": "purple"}).status_code == 400


def test_a_refused_smart_collection_leaves_no_entity_behind(curated):
    doomed = curated.post("/albums/smart", json={"name": "Doomed", "q": "sunset"})  # semantic without take

    assert doomed.status_code == 400
    conn = _raw(curated)
    try:
        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Doomed'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entity WHERE slug = 'doomed'").fetchone()[0] == 0
    finally:
        connect.close(conn)


# --- the definition patch --------------------------------------------------


def test_rename_is_one_operation_with_a_permanent_forwarding_address(curated, keepers):
    told = curated.patch(f"/t/{keepers}", json={"name": "Best Keepers", "expected_rev": 1})

    assert told.status_code == 200, told.text
    body = told.json()
    assert (body["slug"], body["name"], body["definition_rev"]) == ("best-keepers", "Best Keepers", 2)
    moved = curated.get("/t/keepers", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/t/best-keepers")
    assert _view(curated, "best-keepers")["name"] == "Best Keepers"


def test_description_and_color_are_desired_facts(curated, keepers):
    told = curated.patch(f"/t/{keepers}", json={"description": "blue hour", "color": "#123ABC", "expected_rev": 1})

    assert told.status_code == 200, told.text
    body = told.json()
    assert (body["description"], body["color"], body["definition_rev"]) == ("blue hour", "#123abc", 2)


def test_a_patch_leaves_the_facts_it_did_not_name_unchanged(curated, moody):
    told = curated.patch(f"/t/{moody}", json={"name": "Moodier", "expected_rev": 2}).json()

    assert (told["description"], told["color"]) == ("blue hour", "#123abc")


def test_null_clears_a_fact(curated, moody):
    told = curated.patch(f"/t/{moody}", json={"description": None, "color": None, "expected_rev": 2}).json()

    assert (told["description"], told["color"], told["definition_rev"]) == (None, None, 3)


def test_a_patch_with_a_bad_color_is_refused_and_moves_nothing(curated, moody):
    assert curated.patch(f"/t/{moody}", json={"color": "papayawhip", "expected_rev": 2}).status_code == 400
    assert _view(curated, moody)["definition_rev"] == 2


def test_an_empty_patch_is_refused(curated, moody):
    assert curated.patch(f"/t/{moody}", json={"expected_rev": 2}).status_code == 400, "an empty patch says nothing"
    assert _view(curated, moody)["definition_rev"] == 2


def test_a_misspelled_fact_is_refused_never_silently_ignored(curated, moody):
    strange = curated.patch(f"/t/{moody}", json={"colour": "#000000", "expected_rev": 2})

    assert strange.status_code == 400
    assert _view(curated, moody)["definition_rev"] == 2


# --- the parent ------------------------------------------------------------


def test_reparenting_sets_the_parent(curated):
    for name in ("Trips", "Florida"):
        _made(curated, "/albums", name=name)

    told = curated.patch("/t/florida", json={"parent": "trips", "expected_rev": 1})

    assert told.status_code == 200, told.text
    assert told.json()["parent"] == "trips"


def test_a_collection_cannot_be_its_own_parent(curated, hierarchy):
    assert curated.patch("/t/trips", json={"parent": "trips", "expected_rev": 1}).status_code == 400
    assert _view(curated, "trips")["definition_rev"] == 1, "the refusal moved nothing"


def test_a_collection_cannot_move_under_its_own_descendant(curated, hierarchy):
    descended = curated.patch("/t/trips", json={"parent": "michigan", "expected_rev": 1})

    assert descended.status_code == 400
    assert _view(curated, "trips")["parent"] is None, "the refusal moved nothing"


def test_an_unknown_parent_is_refused(curated, hierarchy):
    assert curated.patch("/t/trips", json={"parent": "never-was", "expected_rev": 1}).status_code == 400
    assert _view(curated, "trips")["definition_rev"] == 1, "the refusal moved nothing"


def test_a_null_parent_is_the_top_of_the_hierarchy(curated, hierarchy):
    told = curated.patch("/t/michigan", json={"parent": None, "expected_rev": 2}).json()

    assert told["parent"] is None


def test_the_database_trigger_is_the_backstop_against_a_cycle(curated, hierarchy):
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


def test_the_parents_offer_excludes_self_and_descendants(curated, hierarchy):
    for name in ("Elsewhere", "Retired"):
        _made(curated, "/albums", name=name)
    curated.patch("/t/retired", json={"archived": True, "expected_rev": 1})

    told = curated.patch("/t/trips", json={"description": "x", "expected_rev": 1}).json()

    offered = {row["slug"] for row in told["parents"]}
    assert "elsewhere" in offered
    assert offered & {"trips", "florida", "michigan"} == set(), "self and subtree are never offered"
    assert "retired" not in offered, "an archived collection is not an offered organizer"


def test_a_parent_cannot_be_deleted_out_from_under_its_children(curated):
    """ON DELETE RESTRICT: authored children have independent addresses,
    and destroying a hierarchy is an explicit act -- there is no route
    for it, and even raw SQL cannot cascade through the children. The
    childless Loner is the control that the RESTRICT is not a blanket."""
    _made(curated, "/albums", name="Trips")
    _made(curated, "/albums", name="Michigan", parent="trips")
    _made(curated, "/albums", name="Loner")
    conn = _raw(curated)
    try:
        trips, loner = (
            conn.execute("SELECT id FROM collection WHERE name = ?", (name,)).fetchone()[0]
            for name in ("Trips", "Loner")
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM collection WHERE id = ?", (trips,))
        conn.rollback()
        conn.execute("DELETE FROM collection WHERE id = ?", (loner,))
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Michigan'").fetchone()[0] == 1
    finally:
        connect.close(conn)


def test_no_user_facing_hard_delete_exists(curated, keepers):
    assert curated.delete(f"/t/{keepers}").status_code == 405


# --- optimistic concurrency ------------------------------------------------


@pytest.fixture
def draft(curated) -> str:
    """A listed collection whose description was set once: revision 2."""
    _made(curated, "/albums", name="Draft")
    assert curated.patch("/t/draft", json={"description": "mine", "expected_rev": 1}).status_code == 200
    return "draft"


def test_a_stale_definition_edit_refuses_with_zero_mutation(curated, draft):
    stale = curated.patch(f"/t/{draft}", json={"description": "no, mine", "expected_rev": 1})

    assert stale.status_code == 409
    body = _view(curated, draft)
    assert (body["description"], body["definition_rev"]) == ("mine", 2), "the stale editor overwrote nothing"


@pytest.mark.parametrize("header", ['W/"draft-r2"', 'W/"some-other-collection-r2"'], ids=["own", "other"])
def test_an_if_match_header_never_authorizes_a_write(curated, draft, header):
    """The concurrency interface is expected_rev in the body, nowhere
    else: a header token arrives unbound to the collection in the URL,
    so NO header may authorize a write -- not even one spelling this
    collection's own current revision."""
    veiled = curated.patch(f"/t/{draft}", json={"description": "x"}, headers={"if-match": header})

    assert veiled.status_code == 400
    assert _view(curated, draft)["description"] == "mine"


def test_no_etag_is_emitted_that_could_pretend_to_mean_the_definition(curated, draft):
    assert "etag" not in curated.get(f"/t/{draft}", headers=AS_MACHINE).headers


@pytest.mark.parametrize(
    "body", [{"description": "x"}, {"description": "x", "expected_rev": True}], ids=["absent", "bool"]
)
def test_a_write_without_a_named_revision_is_refused(curated, draft, body):
    assert curated.patch(f"/t/{draft}", json=body).status_code == 400


def test_membership_never_creates_a_definition_conflict(curated, keepers):
    """Filing pictures is not editing the definition: an open editor's
    revision stays valid through any amount of curation."""
    read_rev = _view(curated, keepers)["definition_rev"]
    curated.post(f"/t/{keepers}/add", json={"file": "pic-0"})
    curated.post(f"/t/{keepers}/add", json={"file": "pic-1"})
    curated.post(f"/i/pic-2/collections/{keepers}", json={"value": True})
    assert _view(curated, keepers)["definition_rev"] == read_rev

    saved = curated.patch(f"/t/{keepers}", json={"description": "still valid", "expected_rev": read_rev})

    assert saved.status_code == 200, "membership manufactured a false definition conflict"
    assert saved.json()["count"] == 3


# --- the rule and the kind transitions -------------------------------------


def test_rule_replacement_is_whole_desired_state(curated, starred):
    told = curated.put(f"/t/{starred}/rule", json={"favorite": "1", "expected_rev": 1})

    assert told.status_code == 200, told.text
    body = told.json()
    assert body["definition_rev"] == 2, "a new meaning is a new definition revision"
    assert [row["slug"] for row in body["gallery"]["items"]] == ["pic-2", "pic-1"], "the WHOLE rule was replaced"


def test_a_stale_rule_replacement_is_refused_with_zero_mutation(curated, starred):
    assert curated.put(f"/t/{starred}/rule", json={"favorite": "1", "expected_rev": 1}).status_code == 200

    assert curated.put(f"/t/{starred}/rule", json={"rating_min": 2, "expected_rev": 1}).status_code == 409
    assert [row["slug"] for row in _view(curated, starred)["gallery"]["items"]] == ["pic-2", "pic-1"]


def test_a_listed_collection_takes_no_rule(curated, keepers):
    assert curated.put(f"/t/{keepers}/rule", json={"favorite": "1", "expected_rev": 1}).status_code == 400


def test_album_and_flag_exchange_freely_with_members_kept(curated, keepers):
    for slug in ("pic-0", "pic-5"):
        curated.post(f"/t/{keepers}/add", json={"file": slug})

    told = curated.post(f"/t/{keepers}/convert", json={"kind": "flag", "expected_rev": 1})

    assert told.status_code == 201, told.text
    body = told.json()
    assert (body["kind"], body["count"], body["definition_rev"]) == ("flag", 2, 2)
    assert {row["slug"] for row in body["gallery"]["items"]} == {"pic-0", "pic-5"}


def test_becoming_smart_refuses_a_collection_with_filed_members(curated, keepers):
    curated.post(f"/t/{keepers}/add", json={"file": "pic-0"})

    refused = curated.post(f"/t/{keepers}/convert", json={"kind": "smart", "rating_min": 4, "expected_rev": 1})

    assert refused.status_code == 400
    assert "filed member" in refused.json()["detail"]


def test_a_rule_that_refuses_converts_nothing(curated, keepers):
    half = curated.post(f"/t/{keepers}/convert", json={"kind": "smart", "q": "sunset", "expected_rev": 1})

    assert half.status_code == 400
    body = _view(curated, keepers)
    assert (body["kind"], body["definition_rev"]) == ("album", 1), "no half-made smart object exists"


def test_becoming_smart_takes_the_rule_in_the_same_act(curated, keepers):
    told = curated.post(f"/t/{keepers}/convert", json={"kind": "smart", "rating_min": 4, "expected_rev": 1})

    assert told.status_code == 201, told.text
    body = told.json()
    assert (body["kind"], body["state"], body["definition_rev"]) == ("smart", "evaluated", 2)
    assert [row["slug"] for row in body["gallery"]["items"]] == ["pic-3", "pic-1"]


def test_leaving_smart_without_saying_discard_is_refused(curated, starred):
    kept = curated.post(f"/t/{starred}/convert", json={"kind": "album", "expected_rev": 1})

    assert kept.status_code == 400
    assert "discard" in kept.json()["detail"]
    assert _view(curated, starred)["kind"] == "smart"


def test_leaving_smart_with_discard_drops_the_rule(curated, starred):
    told = curated.post(f"/t/{starred}/convert", json={"kind": "album", "discard_rule": True, "expected_rev": 1})

    assert told.status_code == 201, told.text
    assert (told.json()["kind"], told.json()["definition_rev"], told.json()["count"]) == ("album", 2, 0)
    conn = _raw(curated)
    try:
        assert conn.execute("SELECT count(*) FROM collection_rule").fetchone()[0] == 0, "the rule was discarded"
    finally:
        connect.close(conn)


def test_a_collection_that_left_smart_takes_filings_again(curated, starred):
    assert (
        curated.post(f"/t/{starred}/convert", json={"kind": "album", "discard_rule": True, "expected_rev": 1})
    ).status_code == 201

    filed = curated.post(f"/t/{starred}/add", json={"file": "pic-0"})

    assert filed.status_code == 201


# --- archive ---------------------------------------------------------------


def test_archiving_keeps_identity_members_children_and_address(curated, old_project):
    told = curated.patch(f"/t/{old_project}", json={"archived": True, "expected_rev": 1})

    assert told.status_code == 200, told.text
    assert (told.json()["archived"], told.json()["definition_rev"]) == (True, 2)
    body = _view(curated, old_project)
    assert (body["archived"], body["count"]) == (True, 2)
    assert [child["slug"] for child in body["collections"]] == ["still-going"]


def test_an_archived_collection_is_off_the_shelf_and_out_of_the_picker(curated, archived_project):
    shelf = {row["slug"] for row in curated.get("/albums", headers=AS_MACHINE).json()}
    page = curated.get("/albums", headers=AS_BROWSER).text
    choices = {row["slug"] for row in curated.get("/i/pic-3/collection-choices").json()}
    retired = curated.get("/albums", params={"state": "archived"}, headers=AS_MACHINE).json()

    assert archived_project not in shelf
    assert f'data-album="{archived_project}"' not in page
    assert archived_project not in choices
    assert [row["slug"] for row in retired] == [archived_project]


def test_both_album_lists_carry_the_same_keys(curated, archived_project):
    """`/albums` and its archived shelf are one representation.

    The two lists are built from different statements and the archived one
    computes no spans, so it would be easy for them to answer with different
    key sets and for nobody to notice. They do not: a shelf row carries
    first_seen and last_seen as null, because the contract says a listed
    collection has them, not because that list happened to look them up.
    """
    active = curated.get("/albums", headers=AS_MACHINE).json()
    retired = curated.get("/albums", params={"state": "archived"}, headers=AS_MACHINE).json()
    assert active, "the active list needs a row for this to prove anything"
    assert retired, "and so does the archived shelf"

    listed = frozenset({"name", "slug", "kind", "pictures", "first_seen", "last_seen"})
    for row in (*active, *retired):
        assert set(row) == listed, row

    assert all(row["first_seen"] is None and row["last_seen"] is None for row in retired), (
        "the archived shelf computes no spans, and says so with null rather than by omitting the keys"
    )


def test_an_active_child_surfaces_instead_of_vanishing_with_its_organizer(curated, archived_project):
    shelf = {row["slug"] for row in curated.get("/albums", headers=AS_MACHINE).json()}
    page = curated.get("/albums", headers=AS_BROWSER).text
    choices = {row["slug"] for row in curated.get("/i/pic-3/collection-choices").json()}

    assert "still-going" in shelf
    assert 'data-album="still-going"' in page
    assert "still-going" in choices


def test_archiving_what_is_archived_keeps_the_original_fact(curated, archived_project):
    conn = _raw(curated)
    try:
        first = conn.execute("SELECT archived_at FROM collection WHERE name = 'Old Project'").fetchone()[0]
    finally:
        connect.close(conn)

    assert curated.patch(f"/t/{archived_project}", json={"archived": True, "expected_rev": 2}).status_code == 200

    conn = _raw(curated)
    try:
        assert conn.execute("SELECT archived_at FROM collection WHERE name = 'Old Project'").fetchone()[0] == first
    finally:
        connect.close(conn)


def test_restoring_returns_the_same_entity_and_address(curated, archived_project):
    restored = curated.patch(f"/t/{archived_project}", json={"archived": False, "expected_rev": 2})

    assert restored.status_code == 200, restored.text
    assert (restored.json()["archived"], restored.json()["slug"]) == (False, archived_project)
    assert archived_project in {row["slug"] for row in curated.get("/albums", headers=AS_MACHINE).json()}
    assert _view(curated, archived_project)["count"] == 2


def test_an_archived_parent_survives_an_unrelated_edit(curated, archived_parent_with_child):
    """The select-fallback trap, closed at the seam: the offer can spell
    the state that already holds, so editing a child beneath an archived
    parent never silently reparents it."""
    told = curated.patch("/t/still-going", json={"description": "still going", "expected_rev": 1}).json()

    assert told["parent"] == "old-project", "an unrelated edit must not move the child"
    offered = {row["slug"]: row["archived"] for row in told["parents"]}
    assert offered.get("old-project") is True, "the archived CURRENT parent is offered, marked"
    assert "retired-too" not in offered, "other archived collections are not destinations"


def test_saying_the_current_archived_parent_out_loud_is_legal(curated, archived_parent_with_child):
    kept = curated.patch("/t/still-going", json={"parent": "old-project", "expected_rev": 1})

    assert kept.status_code == 200, "saying the current state out loud is always legal"


def test_an_archived_collection_is_not_a_new_destination_for_a_move(curated, archived_parent_with_child):
    moved = curated.patch("/t/still-going", json={"parent": "retired-too", "expected_rev": 1})

    assert moved.status_code == 400
    assert "restore" in moved.json()["detail"]
    assert _view(curated, "still-going")["parent"] == "old-project"


@pytest.mark.parametrize(
    ("path", "body"),
    [("/albums", {"name": "Too Late"}), ("/albums/smart", {"name": "Too Late Smart", "rating_min": 4})],
    ids=["listed", "smart"],
)
def test_creation_under_an_archived_parent_is_refused(curated, archived_project, path, body):
    """One definition of "legal parent" however the child comes to
    exist: an archived collection takes no NEW children from a creation
    any more than from a move."""
    refused = curated.post(path, json={**body, "parent": archived_project})

    assert refused.status_code == 400
    assert "restore" in refused.json()["detail"]


def test_the_module_refuses_a_listed_child_under_an_archived_parent_even_when_the_caller_commits(
    curated, archived_project
):
    conn = _raw(curated)
    try:
        parent = conn.execute("SELECT id FROM collection WHERE name = 'Old Project'").fetchone()[0]
        actor = curated.app.state.actor_id

        with pytest.raises(ValueError, match="restore"):
            collections.create_listed(conn, "Too Late", 5.0, parent_id=parent, actor_id=actor)
        conn.commit()

        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Too Late'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entity WHERE slug = 'too-late'").fetchone()[0] == 0
    finally:
        connect.close(conn)


def test_the_module_refuses_a_smart_child_under_an_archived_parent_even_when_the_caller_commits(
    curated, archived_project
):
    conn = _raw(curated)
    try:
        parent = conn.execute("SELECT id FROM collection WHERE name = 'Old Project'").fetchone()[0]
        actor = curated.app.state.actor_id
        rule = collection_rules.from_gallery_query(conn, resultset.parse(rating_min=4), actor_id=actor, take=None)

        with pytest.raises(ValueError, match="restore"):
            collections.create_smart(conn, "Too Late Smart", rule, None, 6.0, parent_id=parent, actor_id=actor)
        conn.commit()

        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Too Late Smart'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entity WHERE slug = 'too-late-smart'").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT count(*) FROM collection_rule r JOIN collection c ON c.id = r.collection_id"
                " WHERE c.name = 'Too Late Smart'"
            ).fetchone()[0]
            == 0
        )
    finally:
        connect.close(conn)


# --- a refused transition leaves the caller's transaction untouched --------
#
# The Module's invariant, tested the hostile way: a direct caller catches
# the refusal and COMMITS anyway -- and nothing partial persists, because
# every domain check precedes the first mutation and the revision claim
# leads every multi-step transition.


def _held(conn, collection_id: int) -> tuple:
    return conn.execute(
        "SELECT c.kind, c.definition_rev,"
        " (SELECT count(*) FROM collection_rule r WHERE r.collection_id = c.id)"
        " FROM collection c WHERE c.id = ?",
        (collection_id,),
    ).fetchone()


def test_a_stale_smart_to_listed_refuses_before_the_rule_is_deleted(curated, starred):
    actor = curated.app.state.actor_id
    conn = _raw(curated)
    try:
        smart = conn.execute("SELECT id FROM collection WHERE name = 'Starred'").fetchone()[0]

        with pytest.raises(collections.CollectionChanged):
            collections.convert_to_listed(conn, smart, "album", actor, 99, 5.0, discard_rule=True)
        conn.commit()

        assert _held(conn, smart) == ("smart", 1, 1), "a caught stale refusal, committed, deleted the authored rule"
    finally:
        connect.close(conn)


def test_an_invalid_rule_replacement_refuses_before_any_write(curated, starred, bad_rule):
    actor = curated.app.state.actor_id
    conn = _raw(curated)
    try:
        smart = conn.execute("SELECT id FROM collection WHERE name = 'Starred'").fetchone()[0]
        before = conn.execute("SELECT rule_json FROM collection_rule WHERE collection_id = ?", (smart,)).fetchone()

        with pytest.raises(ValueError, match="kind"):
            collections.replace_rule(conn, smart, bad_rule, None, actor, 1, 6.0)
        conn.commit()

        assert _held(conn, smart) == ("smart", 1, 1)
        assert (
            conn.execute("SELECT rule_json FROM collection_rule WHERE collection_id = ?", (smart,)).fetchone() == before
        )
    finally:
        connect.close(conn)


def test_an_invalid_rule_at_creation_leaves_neither_collection_nor_entity(curated, bad_rule):
    actor = curated.app.state.actor_id
    conn = _raw(curated)
    try:
        with pytest.raises(ValueError, match="kind"):
            collections.create_smart(conn, "Ghost", bad_rule, None, 7.0, actor_id=actor)
        conn.commit()

        assert conn.execute("SELECT count(*) FROM collection WHERE name = 'Ghost'").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM entity WHERE slug = 'ghost'").fetchone()[0] == 0
    finally:
        connect.close(conn)


def test_an_invalid_rule_at_conversion_leaves_the_listed_definition_whole(curated, bad_rule):
    actor = curated.app.state.actor_id
    conn = _raw(curated)
    try:
        plain = collections.collection(conn, "Plain", 8.0)
        conn.commit()

        with pytest.raises(ValueError, match="kind"):
            collections.convert_to_smart(conn, plain, bad_rule, None, actor, 1, 9.0)
        conn.commit()

        assert _held(conn, plain) == ("album", 1, 0)
    finally:
        connect.close(conn)


# --- one implementation, pinned --------------------------------------------


def test_the_view_is_authoritative_after_every_write(curated, keepers):
    """The write's answer and a fresh GET agree on every definition
    fact -- the browser renders what the server read back, never what
    it hoped its click did."""
    written = curated.patch(
        f"/t/{keepers}", json={"name": "Kept", "color": "#001122", "description": "d", "expected_rev": 1}
    ).json()

    read = _view(curated, "kept")
    for fact in ("slug", "name", "kind", "color", "description", "parent", "archived", "definition_rev"):
        assert written[fact] == read[fact], f"the write invented its own {fact}"


# --- a smart collection is a saved question --------------------------------


@pytest.fixture
def best_stills(curated) -> str:
    """A smart collection, rating_min=4 AND kind=image: pic-3, pic-1."""
    return _made(curated, "/albums/smart", name="Best stills", rating_min=4, kind="image")["slug"]


def test_a_saved_view_is_the_same_membership_the_resultset_answers(curated):
    made = curated.post("/albums/smart", json={"name": "Best stills", "rating_min": 4, "kind": "image"})

    assert made.status_code == 201, made.text
    slug = made.json()["slug"]
    assert _slugs(curated, album=slug) == _slugs(curated, rating_min=4, kind="image") == ["pic-3", "pic-1"]
    # /t/{smart} is the SAME projection /g?album= reads.
    told = _view(curated, slug)
    assert told["state"] == "evaluated"
    assert [row["slug"] for row in told["gallery"]["items"]] == ["pic-3", "pic-1"]
    assert told["count"] == 2


def test_smart_membership_is_derived_never_filed(curated, best_stills):
    """A later rating change moves the membership with no collection_file
    row anywhere near a smart collection."""
    curated.post("/i/pic-4/rating", json={"value": 5})

    assert _slugs(curated, album=best_stills) == ["pic-4", "pic-3", "pic-1"]
    conn = _raw(curated)
    try:
        filed = conn.execute(
            "SELECT count(*) FROM collection_file cf JOIN collection c ON c.id = cf.collection_id"
            " WHERE c.kind = 'smart'"
        ).fetchone()[0]
    finally:
        connect.close(conn)
    assert filed == 0


@pytest.mark.parametrize(
    ("facet", "members"), [({"favorite": "1"}, ["pic-1"]), ({"kind": "video"}, [])], ids=["favorite", "kind"]
)
def test_the_outer_questions_facets_intersect_the_rules_members(curated, best_stills, facet, members):
    assert _slugs(curated, album=best_stills, **facet) == members


def test_the_rule_survives_renames_because_it_holds_the_entity(curated):
    """TEMPORAL SCENARIO: the rule names the folder by entity, so renaming
    the folder changes nothing the rule answers."""
    slug = _made(curated, "/albums/smart", name="In the library", folder="lib")["slug"]
    before = _slugs(curated, album=slug)
    assert len(before) == 6

    conn = _raw(curated)
    found = naming.resolve(conn, "folder", "lib")
    assert found is not None
    naming.rename(conn, found[0], "library prime", 5.0)
    conn.commit()
    connect.close(conn)

    assert _slugs(curated, album=slug) == before, "a renamed folder must not break or empty the rule"


def test_the_actor_is_pinned_at_creation_never_the_viewer(curated):
    slug = _made(curated, "/albums/smart", name="My favorites", favorite="1")["slug"]
    mine = curated.app.state.actor_id
    conn = _raw(curated)
    try:
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
    finally:
        connect.close(conn)


def test_a_semantic_rule_without_take_is_refused(curated):
    refused = curated.post("/albums/smart", json={"name": "Sunsets", "q": "sunset"})

    assert refused.status_code == 400
    assert "take" in refused.json()["detail"]


@pytest.fixture
def ranked_retrieval(curated, monkeypatch):
    """Retrieval answers every allowed file in id order, no model in the loop."""
    from db import retrieval

    conn = _raw(curated)
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


def test_take_bounds_a_semantic_rule_to_a_set(curated, ranked_retrieval):
    made = curated.post("/albums/smart", json={"name": "Sunsets", "q": "sunset", "take": 2})

    assert made.status_code == 201, made.text
    assert len(_slugs(curated, album=made.json()["slug"])) == 2, "take must cut the ranked answer down to a set"


def test_a_semantic_rule_nothing_can_answer_is_unavailable_never_empty(curated, ranked_retrieval, monkeypatch):
    from db import retrieval

    slug = _made(curated, "/albums/smart", name="Sunsets", q="sunset", take=2)["slug"]

    def refuses(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        raise LookupError("no space can answer")

    monkeypatch.setattr(retrieval, "query", refuses)
    curated.post("/i/pic-0/favorite", json={"value": True})  # move the currency past the cached answer

    body = _view(curated, slug)

    assert body["state"] == "unavailable"
    assert body["gallery"] is None
    assert curated.get("/g", params={"album": slug}).status_code == 400


def test_preserved_prose_is_unevaluated_and_shown_never_run(curated):
    conn = _raw(curated)
    prose = collections.collection(conn, "Old prose", 1.0, kind="smart")
    collection_rules.keep_prose(conn, prose, sql="SELECT 1", now=1.0)
    conn.commit()
    connect.close(conn)

    told = _view(curated, "old-prose")

    assert told["state"] == "unevaluated"
    assert told["rule"] == {"sql": "SELECT 1", "nl": None}, "the preserved prose is shown, never run"
    assert told["gallery"] is None
    assert curated.get("/g", params={"album": "old-prose"}).status_code == 400


def test_a_rule_whose_entity_is_gone_is_broken_by_name_never_empty(curated):
    slug = _made(curated, "/albums/smart", name="Doomed", folder="lib")["slug"]
    conn = _raw(curated)
    found = naming.resolve(conn, "folder", "lib")
    assert found is not None
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM file WHERE folder_id = ?", (found[0],))
    conn.execute("DELETE FROM folder WHERE id = ?", (found[0],))
    conn.commit()
    connect.close(conn)

    body = _view(curated, slug)

    assert body["state"] == "broken"
    assert "no longer exists" in body["reason"]
    assert body["gallery"] is None
    assert curated.get("/g", params={"album": slug}).status_code == 400


def test_the_albums_index_never_evaluates_a_rule(curated, monkeypatch):
    _made(curated, "/albums/smart", name="Lazy", rating_min=3)
    asked: list[str] = []
    for name in ("page", "describe", "peek", "locate"):
        real = getattr(resultset, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            asked.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(resultset, name, counted)

    assert curated.get("/albums", headers=AS_BROWSER).status_code == 200
    assert curated.get("/albums", headers=AS_MACHINE).status_code == 200

    assert asked == [], "the album shelf launched a rule evaluation nobody asked for"


def test_an_edited_rule_is_a_new_question(curated):
    """TEMPORAL SCENARIO: the edited rule answers immediately through
    normal currency."""
    slug = _made(curated, "/albums/smart", name="Moving", rating_min=5)["slug"]
    assert _slugs(curated, album=slug) == ["pic-3"]

    conn = _raw(curated)
    found = naming.resolve(conn, "collection", slug)
    assert found is not None
    fresh = collection_rules.from_gallery_query(
        conn, resultset.parse(rating_min=2), actor_id=curated.app.state.actor_id, take=None
    )
    collection_rules.save(conn, found[0], fresh, source_text="rating_min=2", now=9.0)
    conn.commit()
    connect.close(conn)

    assert _slugs(curated, album=slug) == ["pic-4", "pic-3", "pic-1"]


def _rotten(where=None, select=None, v: int | float = 1) -> str:
    base = {
        "v": v,
        "where": {"folder": None, "person": None, "kind": None, "favorite": None, "rating_min": None},
        "select": {"sort": None, "text": None, "take": None},
    }
    base["where"].update(where or {})
    base["select"].update(select or {})
    return json.dumps(base)


#: (rule_version, rule_json, actor_id) -- every way a stored rule can be
#: semantically rotten while still being valid JSON.
_ROTTEN_RULES = {
    "unknown-version": (47, _rotten(v=47), None),
    "column-disagrees-with-json": (1, _rotten(v=2), None),
    "kind-from-another-planet": (1, _rotten(where={"kind": "platypus"}), None),
    "favorite-banana": (1, _rotten(where={"favorite": "banana"}), None),
    "rating-700": (1, _rotten(where={"rating_min": 700}), None),
    "negative-take": (1, _rotten(select={"take": -5, "sort": "newest"}), None),
    "sort-with-nothing-to-cut": (1, _rotten(select={"sort": "Tuesday"}), None),
    "folder-not-hex": (1, _rotten(where={"folder": "zz"}), None),
    "folder-short-hex": (1, _rotten(where={"folder": "aabb"}), None),
    "authored-facet-no-actor": (1, _rotten(where={"favorite": True}), None),
    # The JSON type system's truthiness corners: falsy is not null, and
    # bool is not an integer however Python coerces them.
    "folder-empty-string": (1, _rotten(where={"folder": ""}), None),
    "folder-false": (1, _rotten(where={"folder": False}), None),
    "rating-true": (1, _rotten(where={"rating_min": True}), 1),
    "take-true": (1, _rotten(select={"take": True, "sort": "newest"}), None),
    "version-true": (1, _rotten(v=True), None),
    "version-float": (1, _rotten(v=1.0), None),
}


@pytest.mark.parametrize(("version", "payload", "actor"), list(_ROTTEN_RULES.values()), ids=list(_ROTTEN_RULES))
def test_a_corrupt_stored_rule_is_broken_never_empty(curated, version, payload, actor):
    """json_valid is syntax; the load gate is MEANING: a semantically
    rotten stored rule is BROKEN by name, never an evaluated empty
    collection."""
    slug = _made(curated, "/albums/smart", name="Fragile", kind="image")["slug"]
    conn = _raw(curated)
    try:
        fragile = conn.execute("SELECT id FROM collection WHERE name = 'Fragile'").fetchone()[0]
        conn.execute(
            "UPDATE collection_rule SET rule_version = ?, rule_json = ?, actor_id = ? WHERE collection_id = ?",
            (version, payload, actor, fragile),
        )
        conn.commit()
    finally:
        connect.close(conn)

    body = _view(curated, slug)

    assert body["state"] == "broken", f"{payload} was not refused"
    assert body["gallery"] is None
    assert curated.get("/g", params={"album": slug}).status_code == 400


def test_the_persistence_interface_validates_what_it_is_handed(curated, bad_rule):
    """save() owns its invariant: a semantically rotten CollectionRule
    built by hand -- not through from_gallery_query -- is refused at the
    persistence seam, never written for load() to trip over later."""
    conn = _raw(curated)
    try:
        smart = collections.collection(conn, "Handmade", 1.0, kind="smart")

        with pytest.raises(ValueError, match="kind"):
            collection_rules.save(conn, smart, bad_rule, source_text=None, now=1.0)

        assert conn.execute("SELECT count(*) FROM collection_rule WHERE collection_id = ?", (smart,)).fetchone()[0] == 0
    finally:
        connect.close(conn)


# --- exactly one membership definition per collection (the v8 guards) -------


def test_the_module_refuses_prose_on_a_listed_collection(curated):
    conn = _raw(curated)
    try:
        listed = collections.collection(conn, "Listed", 1.0)

        with pytest.raises(ValueError, match="smart"):
            collection_rules.keep_prose(conn, listed, nl="x", now=1.0)
    finally:
        connect.close(conn)


def test_the_module_refuses_a_rule_on_a_listed_collection(curated):
    conn = _raw(curated)
    try:
        listed = collections.collection(conn, "Listed", 1.0)
        rule = collection_rules.from_gallery_query(conn, resultset.parse(kind="image"), actor_id=None, take=None)

        with pytest.raises(ValueError, match="smart"):
            collection_rules.save(conn, listed, rule, source_text=None, now=1.0)
    finally:
        connect.close(conn)


def test_the_schema_refuses_a_rule_row_on_a_listed_collection(curated):
    conn = _raw(curated)
    try:
        listed = collections.collection(conn, "Listed", 1.0)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO collection_rule(collection_id, created_at, updated_at) VALUES(?, 1, 1)", (listed,)
            )
    finally:
        connect.close(conn)


def test_a_rule_carrying_smart_collection_cannot_quietly_become_listed(curated):
    _made(curated, "/albums/smart", name="Committed", kind="image")
    conn = _raw(curated)
    try:
        committed = conn.execute("SELECT id FROM collection WHERE name = 'Committed'").fetchone()[0]

        with pytest.raises(sqlite3.IntegrityError, match="rule-defined"):
            conn.execute("UPDATE collection SET kind = 'album' WHERE id = ?", (committed,))
    finally:
        connect.close(conn)


def test_deleting_the_rule_first_is_the_deliberate_transition(curated):
    _made(curated, "/albums/smart", name="Committed", kind="image")
    conn = _raw(curated)
    try:
        committed = conn.execute("SELECT id FROM collection WHERE name = 'Committed'").fetchone()[0]
        conn.execute("DELETE FROM collection_rule WHERE collection_id = ?", (committed,))

        conn.execute("UPDATE collection SET kind = 'album' WHERE id = ?", (committed,))
        conn.commit()

        assert conn.execute("SELECT kind FROM collection WHERE id = ?", (committed,)).fetchone()[0] == "album"
    finally:
        connect.close(conn)


#: The exact top-level keys `/t/{slug}` serves a machine, per representation.
#: ABSENCE IS PART OF THE CONTRACT: a broken smart collection carries no
#: `timeline` at all, and a listed one carries no `state` -- which is why this
#: pins the whole key set rather than asserting a few members are present.
#: Captured from the shape as served today, so any change to how the view is
#: assembled has to be a deliberate edit here rather than a silent difference.
_LISTED_KEYS = frozenset(
    {
        "slug",
        "name",
        "kind",
        "color",
        "description",
        "parent",
        "archived",
        "definition_rev",
        "updated_at",
        "updated_by",
        "collections",
        "count",
        "first_seen",
        "last_seen",
        "timeline",
        "places",
        "gallery",
        "files",
    }
)
#: A smart collection adds the rule and the rule's condition.
_SMART_EXTRA = frozenset({"rule", "state"})
#: The two conditions that explain themselves add a reason.
_REASONED = frozenset({"reason"})
#: A rule that produced no answer has no gallery-shaped facts at all.
_UNGALLERIED = frozenset({"first_seen", "last_seen", "timeline", "places"})


def test_a_lifecycle_write_answers_with_the_managed_shape(curated):
    """The third representation, captured.

    A write answers with the view assembled for MANAGEMENT, not the one
    `GET /t/{slug}` serves a machine: it carries `parents` -- the moves the
    database would allow -- and omits the legacy `files` list. The browser
    reads `slug` and `definition_rev` out of it, so the difference has never
    surfaced, but it is a difference and it is now written down.
    """
    made = curated.post("/albums", json={"name": "Fresh"})
    assert made.status_code == 201, made.text
    written = set(made.json())

    served = set(_view(curated, made.json()["slug"]))

    assert "parents" in written, "a write answers the manage view"
    assert "files" not in written
    assert "files" in served, "the machine GET answers the legacy adapter shape"
    assert "parents" not in served
    assert written - {"parents"} == served - {"files"}, "otherwise they are the same listed collection"


def test_a_listed_collection_says_which_keys_it_carries(curated):
    """An album and a flag are the same representation: no rule, no state,
    and every gallery-shaped fact present because the membership evaluated."""
    for slug in (
        _made(curated, "/albums", name="Plain")["slug"],
        _made(curated, "/albums", name="Flagged", kind="flag")["slug"],
    ):
        assert set(_view(curated, slug)) == _LISTED_KEYS, slug


def _describes_no_answer(told: dict) -> None:
    """What a state that produced no answer says in the keys it does carry.

    Present-and-null is not the same as absent and not the same as empty:
    `count` and `gallery` are null, while `files` is a LIST -- the legacy
    adapter lists the filed members whether or not a rule ever ran. A model
    that made all three null would keep every key set below passing.
    """
    assert told["count"] is None
    assert told["gallery"] is None
    assert isinstance(told["files"], list)


def test_an_evaluated_smart_collection_adds_its_rule_and_state(curated):
    """A rule that ran is a listed collection plus the rule and its
    condition -- and no reason, because nothing needs explaining."""
    told = _view(curated, _made(curated, "/albums/smart", name="Everything")["slug"])
    assert told["state"] == "evaluated"
    assert set(told) == _LISTED_KEYS | _SMART_EXTRA
    assert isinstance(told["count"], int), "an evaluated rule counted its members"
    assert isinstance(told["gallery"], dict)
    assert set(told["gallery"]) == {"items", "total", "pages", "qs"}
    assert set(told["rule"]) == {"sql", "nl"}


def test_an_unevaluated_smart_collection_carries_no_gallery_facts(curated):
    """Preserved prose was never run, so there is no answer to describe:
    the span, the timeline link and the places are absent, not null."""
    conn = _raw(curated)
    prose = collections.collection(conn, "Prose", 1.0, kind="smart")
    collection_rules.keep_prose(conn, prose, sql="SELECT 1", now=1.0)
    conn.commit()
    connect.close(conn)

    told = _view(curated, "prose")

    assert told["state"] == "unevaluated"
    assert set(told) == (_LISTED_KEYS | _SMART_EXTRA) - _UNGALLERIED
    _describes_no_answer(told)
    assert told["rule"] == {"sql": "SELECT 1", "nl": None}, "the preserved prose is shown, never run"


def test_a_broken_smart_collection_carries_a_reason(curated):
    """A rule naming an entity that is gone explains itself and describes
    no answer."""
    slug = _made(curated, "/albums/smart", name="Gone", folder="lib")["slug"]
    conn = _raw(curated)
    found = naming.resolve(conn, "folder", "lib")
    assert found is not None
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DELETE FROM file WHERE folder_id = ?", (found[0],))
    conn.execute("DELETE FROM folder WHERE id = ?", (found[0],))
    conn.commit()
    connect.close(conn)

    told = _view(curated, slug)

    assert told["state"] == "broken"
    assert set(told) == (_LISTED_KEYS | _SMART_EXTRA | _REASONED) - _UNGALLERIED
    _describes_no_answer(told)
    assert isinstance(told["reason"], str)
    assert told["reason"], "a broken rule explains itself in words, never an empty string"
    assert set(told["rule"]) == {"sql", "nl"}, "the rule that broke is still shown"


def test_an_unavailable_smart_collection_carries_a_reason(curated, ranked_retrieval, monkeypatch):
    """A semantic rule nothing can answer right now says so, and describes
    no answer either."""
    from db import retrieval

    slug = _made(curated, "/albums/smart", name="Sunsets", q="sunset", take=2)["slug"]

    def refuses(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        raise LookupError("no space can answer")

    monkeypatch.setattr(retrieval, "query", refuses)
    # move the currency past the cached answer, so the rule is asked again
    assert curated.post("/i/pic-0/favorite", json={"value": True}).status_code == 201

    told = _view(curated, slug)

    assert told["state"] == "unavailable"
    assert set(told) == (_LISTED_KEYS | _SMART_EXTRA | _REASONED) - _UNGALLERIED
    _describes_no_answer(told)
    assert isinstance(told["reason"], str)
    assert told["reason"], "an unanswerable rule says why, never an empty string"
    assert set(told["rule"]) == {"sql", "nl"}
