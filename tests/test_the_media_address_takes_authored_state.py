"""Favorite, rating and album membership: one vertical through MediaView.

`/i/{slug}` has one authored state per actor, assembled inside the same
snapshot as everything else the address shows; the write routes state
DESIRED FACTS so retries are harmless; every adapter -- browser JSON,
the legacy /t membership routes -- runs the same db/authored.py
implementation. And the ResultSet's answer identity keeps the two
generations honest: a favorite moves the library's currency without
moving any answer, while unfiling the walked item moves the answer.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import authored, collections, connect
from sg_web.app import build_app
from tests import retrieving
from tests.staging import hosting, seeded

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}
AS_OVERLAY = {"hx-request": "true"}


@pytest.fixture(scope="module")
def _bare_stage(tmp_path_factory):
    """One application over an EMPTY home, for the tests that bring their
    own library. Each was building its own -- an interpreter's worth of
    imports and a migration -- to register a root."""
    with hosting(tmp_path_factory, "authored_bare") as stage:
        yield stage


@pytest.fixture
def bare(_bare_stage):
    """That application with nothing in it: restored, so `/roots` numbers
    from 1 again and no test inherits another's library."""
    _bare_stage.restore()
    return _bare_stage.client


def _library(tmp) -> tuple:
    """Pictures on disk and a home ready for `build_app` to open.

    The home arrives with the built database already in it: what the
    three tests that call this prove is what the application does with a
    library, never that it can create its own database, and `build_app`
    finding a database costs less than building one.
    """
    root = tmp / "lib"
    root.mkdir()
    _pictures(root)
    burrow = tmp / "run"
    seeded(burrow)
    return burrow, root


def _pictures(root) -> None:
    stamped = 1_700_000_000
    for i in range(4):
        path = root / f"pic_{i}.png"
        Image.new("RGB", (12, 12), (60 + i * 20, 90, 140)).save(path)
        os.utime(path, (stamped + i * 60, stamped + i * 60))


@pytest.fixture(scope="module")
def kept(tmp_path_factory):
    burrow, root = _library(tmp_path_factory.mktemp("authored"))
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 4
        assert client.post("/albums", json={"name": "Keep"}).json()["slug"] == "keep"
        conn = connect.connect(client.app.state.db_path)
        collections.collection(conn, "Rules", 3.0, kind="smart")
        conn.commit()
        connect.close(conn)
        yield client


def test_the_three_faces_report_one_authored_state(kept):
    assert kept.post("/i/pic-1/favorite", json={"value": True}).json()["authored"]["favorite"] is True
    assert kept.post("/i/pic-1/rating", json={"value": 4}).json()["authored"]["rating"] == 4
    kept.post("/i/pic-1/collections/keep", json={"value": True})
    told = kept.post("/i/pic-1/tags", json={"name": "Harbour"}).json()["authored"]
    assert told == {
        "favorite": True,
        "rating": 4,
        "collections": [{"slug": "keep", "name": "Keep"}],
        "tags": [{"tag": "harbour", "label": "Harbour"}],
    }

    body = kept.get("/i/pic-1", headers=AS_MACHINE).json()
    assert body["authored"] == told
    page = kept.get("/i/pic-1", headers=AS_BROWSER).text
    part = kept.get("/i/pic-1", headers=AS_OVERLAY).text
    for face in (page, part):
        assert re.search(r'data-fav\s+aria-pressed="true"', face), "the favorite must render pressed"
        assert 'data-rating="4"' in face
        assert 'href="/t/keep"' in face, "the strip must show the membership in every presentation"
        assert 'data-tag="harbour"' in face, "the strip must show the keyword in every presentation"


def test_desired_state_retries_are_idempotent(kept):
    for _ in range(2):
        assert kept.post("/i/pic-2/favorite", json={"value": True}).json()["authored"]["favorite"] is True
    for _ in range(2):
        assert kept.post("/i/pic-2/rating", json={"value": 5}).json()["authored"]["rating"] == 5
    for _ in range(2):
        told = kept.post("/i/pic-2/collections/keep", json={"value": True}).json()["authored"]
        assert [held["slug"] for held in told["collections"]] == ["keep"]
    count = kept.get("/t/keep", headers=AS_MACHINE).json()["count"]
    for _ in range(2):
        told = kept.post("/i/pic-2/collections/keep", json={"value": False}).json()["authored"]
        assert told["collections"] == []
    assert kept.get("/t/keep", headers=AS_MACHINE).json()["count"] == count - 1
    assert kept.post("/i/pic-2/rating", json={"value": None}).json()["authored"]["rating"] is None
    assert kept.post("/i/pic-2/rating", json={"value": 9}).status_code == 400
    assert kept.post("/i/pic-2/favorite", json={"value": False}).json()["authored"]["favorite"] is False


def test_membership_reflects_immediately_everywhere(kept):
    kept.post("/i/pic-3/collections/keep", json={"value": True})
    assert "pic-3" in [row["slug"] for row in kept.get("/t/keep", headers=AS_MACHINE).json()["gallery"]["items"]]
    assert 'data-slug="pic-3"' in kept.get("/g", params={"album": "keep"}).text
    kept.post("/i/pic-3/collections/keep", json={"value": False})
    assert "pic-3" not in [row["slug"] for row in kept.get("/t/keep", headers=AS_MACHINE).json()["gallery"]["items"]]


def test_a_smart_collection_refuses_through_every_adapter(kept):
    for value in (True, False):
        refused = kept.post("/i/pic-1/collections/rules", json={"value": value})
        assert refused.status_code == 400
        assert "rule" in refused.json()["detail"]
    choices = kept.get("/i/pic-1/collection-choices").json()
    assert "rules" not in [one["slug"] for one in choices], "a rule-derived kind has no membership to offer"
    assert [one["slug"] for one in choices] == ["keep"]


def test_authored_state_is_per_actor_and_survives_restart(tmp_path):
    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        actor = client.app.state.actor_id
        client.post("/i/pic-0/favorite", json={"value": True})
        client.post("/i/pic-0/rating", json={"value": 3})
        conn = connect.connect(client.app.state.db_path)
        other = authored.add_user(conn, "guest", "!", "USER", 0.0)
        file_id = conn.execute("SELECT id FROM file WHERE name = 'pic_0.png'").fetchone()[0]
        authored.set_rating(conn, file_id, other, 1, 0.0)
        conn.commit()
        # Each signature holds its own facts.
        assert authored.media_state(conn, file_id, actor).rating == 3
        assert authored.media_state(conn, file_id, other).rating == 1
        assert authored.media_state(conn, file_id, other).favorite is False
        connect.close(conn)

    with TestClient(app=build_app(str(burrow), worker=False)) as reopened:
        assert reopened.app.state.actor_id == actor, "the local actor must resolve to the same identity"
        told = reopened.get("/i/pic-0", headers=AS_MACHINE).json()["authored"]
        assert (told["favorite"], told["rating"]) == (True, 3)


def test_a_favorite_moves_the_currency_but_not_the_answer(kept):
    """THE answer-identity contract: data_version bumps on every commit,
    so a favorite invalidates the projection -- but the re-answered
    question orders the same files, and the answer hash says so. A
    membership write against the walked album is the opposite case."""
    # Its own arrangement: the walked file is filed and favorited HERE, so
    # this holds when the affected-test selector runs it without its
    # module-mates.
    kept.post("/i/pic-1/collections/keep", json={"value": True})
    kept.post("/i/pic-1/favorite", json={"value": True})
    before = kept.get("/g", params={"album": "keep"})
    held = dict(re.findall(r'data-(currency|answer)="([^"]*)"', before.text))
    kept.post("/i/pic-1/favorite", json={"value": False})
    located = kept.get("/g/locate/pic-1", params={"album": "keep"}).json()
    assert located["currency"] != held["currency"], "a commit must move the library generation"
    assert located["answer"] == held["answer"], "a favorite must not move the album's ordered answer"

    kept.post("/i/pic-1/collections/keep", json={"value": False})
    gone = kept.get("/g/locate/pic-1", params={"album": "keep"}).json()
    assert gone == {"in_answer": False}, "the unfiled item must leave the walked answer"
    emptied = dict(re.findall(r'data-(currency|answer)="([^"]*)"', kept.get("/g", params={"album": "keep"}).text))
    assert emptied["answer"] != located["answer"], "membership changed the album's answer identity"
    kept.post("/i/pic-1/collections/keep", json={"value": True})
    back = kept.get("/g/locate/pic-1", params={"album": "keep"}).json()
    assert back["answer"] == located["answer"], "the same ordered answer must carry the same identity"
    assert back["currency"] != located["currency"]


def test_authored_judgement_is_a_gallery_question(tmp_path, monkeypatch):
    """WI-43's contract: favorite and rating are composable ResultSet
    facets -- eligibility predicates like person and kind, constraining
    each semantic space BEFORE the fusion, spelled canonically in the
    URL, with the ACTOR in the projection identity and never in the
    URL."""
    from db import resultset, retrieval

    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        client.post("/albums", json={"name": "Keep"})
        for slug in ("pic-0", "pic-2"):
            client.post(f"/i/{slug}/favorite", json={"value": True})
        client.post("/i/pic-2/rating", json={"value": 5})
        client.post("/i/pic-3/rating", json={"value": 2})
        client.post("/i/pic-2/collections/keep", json={"value": True})
        client.post("/i/pic-3/collections/keep", json={"value": True})

        # The grid IS the judgement, newest first -- tri-state: liked,
        # NOT liked, unconstrained.
        liked = client.get("/g", params={"favorite": "1"})
        assert re.findall(r'data-slug="([^"]+)"', liked.text) == ["pic-2", "pic-0"]
        assert 'data-qbase="favorite=1&' in liked.text, "the canonical spelling owns the URL"
        unliked = client.get("/g", params={"favorite": "0"}).text
        assert re.findall(r'data-slug="([^"]+)"', unliked) == ["pic-3", "pic-1"]
        assert 'data-qbase="favorite=0&' in unliked

        # Composition: an intersection with every other predicate.
        both = client.get("/g", params={"favorite": "1", "album": "keep"}).text
        assert re.findall(r'data-slug="([^"]+)"', both) == ["pic-2"]
        starred = client.get("/g", params={"rating_min": 2}).text
        assert re.findall(r'data-slug="([^"]+)"', starred) == ["pic-3", "pic-2"], "rating_min means MINIMUM stars"
        assert re.findall(r'data-slug="([^"]+)"', client.get("/g", params={"rating_min": 4}).text) == ["pic-2"]
        crossed = client.get("/g", params={"favorite": "1", "rating_min": 4}).text
        assert re.findall(r'data-slug="([^"]+)"', crossed) == ["pic-2"]

        # Refusals are loud, never empty pages.
        assert client.get("/g", params={"favorite": "yes"}).status_code == 400
        assert client.get("/g", params={"rating_min": 9}).status_code == 400

        # The walk carries the judgement: arrows walk MY favorites.
        walked = client.get("/i/pic-0", params={"favorite": "1"}, headers=AS_MACHINE).json()
        assert walked["context"]["total"] == 2
        assert walked["context"]["previous"] == "pic-2"
        assert walked["context"]["qs"] == "favorite=1"

        # The actor lives in the projection identity, not the spelling:
        # two actors, one URL, two different questions and answers.
        conn = connect.connect(client.app.state.db_path)
        mine = client.app.state.actor_id
        guest = authored.add_user(conn, "guest", "!", "USER", 0.0)
        pic1 = conn.execute("SELECT id FROM file WHERE name = 'pic_1.png'").fetchone()[0]
        authored.set_favorite(conn, pic1, guest, True, 0.0)
        conn.commit()
        asked = resultset.parse(favorite="1")
        ours = resultset.describe(conn, "", asked, 0.0, actor_id=mine)
        theirs = resultset.describe(conn, "", asked, 0.0, actor_id=guest)
        assert ours["qs"] == theirs["qs"] == "favorite=1", "one spelling"
        assert ours["fingerprint"] != theirs["fingerprint"], "never one cached question"
        assert (ours["total"], theirs["total"]) == (2, 1)
        with pytest.raises(ValueError, match="actor"):
            resultset.describe(conn, "", asked, 0.0)
        # A question with no authored facet is ONE cached projection
        # however many actors ask it.
        plain = resultset.parse(kind="image")
        assert (
            resultset.describe(conn, "", plain, 0.0, actor_id=mine)["fingerprint"]
            == resultset.describe(conn, "", plain, 0.0, actor_id=guest)["fingerprint"]
        )
        # Canonical spelling round-trips through parse.
        import urllib.parse

        asked_again = resultset.parse(favorite="0", rating_min=3)
        spelled = resultset.canonical(asked_again)
        assert spelled == "favorite=0&rating_min=3"
        back = dict(urllib.parse.parse_qsl(spelled))
        assert resultset.parse(favorite=back["favorite"], rating_min=int(back["rating_min"])) == asked_again

        # An authored facet constrains each space BEFORE the fusion.
        favorites = {row[0] for row in conn.execute("SELECT file_id FROM favorite WHERE user_id = ?", (mine,))}
        seen: dict = {}

        def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
            seen.update({"allowed": allowed})
            return retrieving.answered([], participants=[], contributors=[])

        monkeypatch.setattr(retrieval, "query", fused)
        rated = {
            row[0] for row in conn.execute("SELECT file_id FROM rating WHERE user_id = ? AND rating >= 4", (mine,))
        }
        # the TABLE, not the loop variable: a for-target takes its type from
        # the iterable's elements, so a declaration above it does not reach.
        # `favorite` is request-shaped ("1") and `rating_min` a number.
        cases: tuple[tuple[dict[str, Any], set], ...] = (
            ({"favorite": "1"}, favorites),
            ({"rating_min": 4}, rated),
            ({"favorite": "1", "rating_min": 4}, favorites & rated),
        )
        for constrained, expected in cases:
            seen.clear()
            resultset.describe(conn, "", resultset.parse(text="sunset", **constrained), 0.0, actor_id=mine)
            assert seen["allowed"] == expected, (
                f"{constrained} must reach retrieval as exactly its eligible set before RRF"
            )
        connect.close(conn)


def test_authored_eligibility_rides_the_indexes(bare, tmp_path):
    """The plan pin: a time-sorted authored question is the file table's
    own ordered walk plus indexed existence probes against the authored
    primary keys -- no read-time sort, no scan of favorite or rating."""
    from db import resultset

    _burrow, root = _library(tmp_path)
    client = bare
    client.post("/roots", json={"path": str(root)})
    client.post("/roots/1/scan")
    client.post("/i/pic-0/favorite", json={"value": True})
    client.post("/i/pic-0/rating", json={"value": 4})
    actor = client.app.state.actor_id
    conn = connect.connect(client.app.state.db_path)
    walked: list[str] = []
    conn.set_trace_callback(walked.append)
    told = resultset.describe(conn, "", resultset.parse(favorite="1", rating_min=3), 0.0, actor_id=actor)
    conn.set_trace_callback(None)
    assert told["total"] == 1
    membership = [one for one in walked if one.lstrip().startswith("SELECT f.id FROM file f")]
    assert len(membership) == 1, walked
    args = tuple([actor] * membership[0].count("?"))
    plan = " | ".join(row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + membership[0], args))
    # The RIGHT index, by name: "an index was involved" is not the
    # contract -- the global time walk rides file_recent whole.
    assert "TEMP B-TREE" not in plan.upper(), plan
    assert "SCAN f USING INDEX file_recent" in plan, plan
    assert "SEARCH fav USING" in plan, plan
    assert "SEARCH r USING" in plan, plan

    # And a folder-scoped authored question rides the folder's own
    # time index -- never the global walk probing folder_id per file.
    walked.clear()
    conn.set_trace_callback(walked.append)
    resultset.describe(conn, "", resultset.parse(folder="lib", favorite="1"), 0.0, actor_id=actor)
    conn.set_trace_callback(None)
    scoped = [one for one in walked if one.lstrip().startswith("SELECT f.id FROM file f")]
    assert len(scoped) == 1, walked
    args = tuple([actor] * scoped[0].count("?"))
    plan = " | ".join(row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + scoped[0], args))
    connect.close(conn)
    assert "TEMP B-TREE" not in plan.upper(), plan
    assert "USING INDEX file_in_folder_by_time" in plan, plan


def test_a_body_the_contract_does_not_name_is_refused(kept):
    """sg_web/wire.py says a JSON contract names every field that crosses
    and refuses the rest, in BOTH directions. For a request that means a
    misspelled field is a 400 at the seam, not a key silently ignored while
    the route reports success for a write that never carried it.

    The pair matters: the same body without the surprise must still be
    accepted, or this would pass just as well against a route that refused
    everything.

    The coercion half is the same contract from the other side. Under a lax
    plugin the `1` posted below would arrive as True and the route would
    answer 201.
    """
    good = kept.post("/i/pic-3/favorite", json={"value": True})
    assert good.status_code == 201, good.text

    surprised = kept.post("/i/pic-3/favorite", json={"value": True, "supriseFieldNobodyAskedFor": 72})
    assert surprised.status_code == 400, surprised.text

    # And the coercion half, which no linter can see: Litestar decodes a body
    # with model_validate(value, strict=...) and that argument beats the model's
    # own config, so Wire is strict only under PydanticPlugin(validate_strict=True).
    coerced = kept.post("/i/pic-3/favorite", json={"value": 1})
    assert coerced.status_code == 400, coerced.text

    # and the closed vocabulary is closed: a place kind no place can be is
    # refused by the contract rather than by the database three calls later
    refused = kept.post("/i/pic-3/place", json={"name": "Lisbon", "kind": "planet"})
    assert refused.status_code == 400, refused.text
    assert kept.post("/i/pic-3/place", json={"name": "Lisbon", "kind": "city"}).status_code == 201
