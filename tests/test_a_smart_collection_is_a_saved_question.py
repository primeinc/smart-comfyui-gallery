"""A smart collection is a typed rule the ResultSet evaluates.

The rule owns MEMBERSHIP -- durable, uuid-referenced, actor-pinned,
never executable; the ResultSet still owns the ordered answer, so the
outer question's facets intersect the rule's members instead of
replacing them. Every failure state stays loud: unevaluated, broken
and unavailable are never presented as an empty collection.
"""

from __future__ import annotations

import os
import pathlib

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
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(path)
        os.utime(path, (stamped + i * 60, stamped + i * 60))


def _prepare(stage: Stage) -> None:
    client = stage.client
    for slug, stars in (("pic-1", 4), ("pic-3", 5), ("pic-4", 2)):
        client.post(f"/i/{slug}/rating", json={"value": stars})
    for slug in ("pic-1", "pic-2"):
        client.post(f"/i/{slug}/favorite", json={"value": True})


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_smart_collection_is_a_saved_question", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def saved(_stage):
    _stage.restore()
    return _stage.client


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
