"""A person says where a picture happened, and the library holds it.

`place_id` had no producer: GPS alone names no place, and no resolver
ships. A person's word does -- POST /i/{slug}/place finds or mints the
place by name and kind, records the claim as authored desired state
(file_place), and the file's context is re-interpreted at once with
`location_basis = 'authored'`. The claim survives every rebuild and
opens a gallery link through the `place.id` facet.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, runner
from sg_web.app import build_app

AS_MACHINE = {"accept": "application/json"}


def _drain(client) -> None:
    import time

    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", time.time()) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _answer_of(page: str) -> str:
    import re

    found = re.search(r'data-answer="([^"]+)"', page)
    assert found is not None, "the gallery page carries its answer identity"
    return found.group(1)


def _slugs(client) -> list[str]:
    from db import naming

    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        named = [naming.entity_slug(conn, fid) for (fid,) in conn.execute("SELECT id FROM file ORDER BY id")]
        assert all(one is not None for one in named)
        return [one[1] for one in named if one is not None]
    finally:
        connect.close(conn)


def test_a_person_says_where_and_the_library_holds_it(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        client.post("/jobs/ingest")
        client.post("/jobs/context")
        _drain(client)
        a, b, c = _slugs(client)
        before = client.get(f"/i/{a}", headers=AS_MACHINE).json()
        assert before["where"] is None
        assert before["places"] == []
        page = client.get(f"/i/{a}", headers={"accept": "text/html"}).text
        assert "data-where-missing" in page
        assert "data-place-form" in page

        said = client.post(f"/i/{a}/place", json={"name": "  Lisbon ", "kind": "city"})
        assert said.status_code in (200, 201), said.text
        where = said.json()["where"]
        assert (where["name"], where["kind"], where["basis"]) == ("Lisbon", "city", "authored")
        assert where["qs"] == f"place.id%3Aeq%3A{where['id']}".replace("place.id%3Aeq", "f=place.id%3Aeq")
        told = client.get(f"/i/{a}", headers=AS_MACHINE).json()
        assert told["where"]["id"] == where["id"]
        assert told["places"] == [{"name": "Lisbon", "kind": "city"}]
        page = client.get(f"/i/{a}", headers={"accept": "text/html"}).text
        assert f'data-where="{where["id"]}"' in page
        assert "said by a person" in page

        # the same name is the same place, whatever the spelling's case
        again = client.post(f"/i/{b}/place", json={"name": "lisbon", "kind": "city"}).json()["where"]
        assert again["id"] == where["id"], "one Lisbon"
        # and the link opens exactly the pictures there
        link = client.get(f"/g?{where['qs']}", headers={"accept": "text/html"}).text
        assert 'data-total="2"' in link
        assert "place Lisbon" in link, "the chip says the place's name, never its id"
        assert f"place #{where['id']}" not in link
        shelf_with_span = client.get("/places", headers=AS_MACHINE).json()[0]
        assert shelf_with_span["first_seen"] is not None
        assert shelf_with_span["last_seen"] >= shelf_with_span["first_seen"]
        assert "data-seen" in client.get("/places", headers={"accept": "text/html"}).text

        # the shelf lists the place with its count and link
        shelf = client.get("/places", headers=AS_MACHINE).json()
        assert [(p["name"], p["kind"], p["pictures"]) for p in shelf] == [("Lisbon", "city", 2)]
        assert shelf[0]["qs"] == where["qs"]
        page = client.get("/places", headers={"accept": "text/html"}).text
        assert f'data-place="{shelf[0]["slug"]}"' in page
        assert shelf[0]["timeline"] == f"/timeline?f=place.id%3Aeq%3A{where['id']}"
        assert f'data-place-timeline="{shelf[0]["slug"]}"' in page
        assert told["where"]["timeline"] == shelf[0]["timeline"]
        assert "data-where-timeline" in client.get(f"/i/{a}", headers={"accept": "text/html"}).text
        assert (
            sum(
                b["pictures"]
                for b in client.get(
                    "/timeline/density", params={"bin": "day", "f": f"place.id:eq:{where['id']}"}, headers=AS_MACHINE
                ).json()["bins"]
            )
            == 2
        )

        # the folder's and an album's page say where and when their pictures are
        folder = client.get("/f/lib", headers=AS_MACHINE).json()
        assert [(p["name"], p["pictures"]) for p in folder["places"]] == [("Lisbon", 2)]
        assert folder["first_seen"] is not None
        assert "folder=lib" in folder["places"][0]["qs"]
        assert "data-places-line" in client.get("/f/lib", headers={"accept": "text/html"}).text
        made = client.post("/albums", json={"name": "Trip", "kind": "album"})
        assert made.status_code == 201, made.text
        conn = connect.connect(client.app.state.db_path)
        try:
            from db import collections, naming

            resolved = naming.resolve(conn, "collection", made.json()["slug"])
            assert resolved is not None
            collection_id = resolved[0]
            for name in ("p0.png", "p1.png"):
                fid = conn.execute("SELECT id FROM file WHERE name = ?", (name,)).fetchone()[0]
                collections.set_membership(conn, collection_id, fid, True, 5.0)
            conn.commit()
        finally:
            connect.close(conn)
        album = client.get(f"/t/{made.json()['slug']}", headers=AS_MACHINE).json()
        assert [(p["name"], p["pictures"]) for p in album["places"]] == [("Lisbon", 2)]
        assert album["first_seen"] is not None
        assert "data-places-line" in client.get(f"/t/{made.json()['slug']}", headers={"accept": "text/html"}).text

        # a rebuild keeps a person's word: the claim is authored state
        assert client.post("/jobs/context", params={"everything": "true"}).status_code == 201
        _drain(client)
        assert client.get(f"/i/{a}", headers=AS_MACHINE).json()["where"]["basis"] == "authored"

        # a withdrawal that names a "within" mints nothing
        before_places = len(client.get("/places", headers=AS_MACHINE).json())
        assert client.post(f"/i/{c}/place", json={"name": None, "within": "Atlantis"}).status_code in (200, 201)
        assert len(client.get("/places", headers=AS_MACHINE).json()) == before_places
        # withdrawn: nowhere said again; a bad kind is refused
        gone = client.post(f"/i/{a}/place", json={"name": None})
        assert gone.status_code in (200, 201)
        assert gone.json()["where"] is None
        assert client.get(f"/i/{a}", headers=AS_MACHINE).json()["where"] is None
        assert client.post(f"/i/{c}/place", json={"name": "Mars", "kind": "planet"}).status_code == 400


def test_a_session_is_somewhere_when_its_placed_members_agree(tmp_path):
    """A session's place is the one place its placed members agree on;
    members nobody placed do not veto, two places do."""
    import os

    root = tmp_path / "lib"
    root.mkdir()
    at = 1_686_355_200.0 + 14 * 3600
    for i in range(3):
        path = root / f"Screenshot 2023-06-10 at 14.{i * 5:02d}.00.png"
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(path)
        os.utime(path, (at + i * 300, at + i * 300))
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        client.post("/jobs/ingest")
        client.post("/jobs/context")
        _drain(client)
        a, b, c = _slugs(client)
        client.post(f"/i/{a}/place", json={"name": "Lisbon", "kind": "city"})
        client.post(f"/i/{b}/place", json={"name": "Lisbon", "kind": "city"})
        client.post("/jobs/events")
        _drain(client)
        sessions = client.get("/timeline/density", params={"bin": "day"}, headers=AS_MACHINE).json()["sessions"]
        assert len(sessions) == 1, sessions
        assert sessions[0]["place"]["name"] == "Lisbon"
        assert sessions[0]["place"]["qs"].startswith("f=place.id%3Aeq%3A")
        lisbon = sessions[0]["place"]["id"]
        narrowed = client.get("/timeline/density", params={"bin": "day", "kind": "image"}, headers=AS_MACHINE).json()
        assert "f=place.id%3Aeq%3A" in narrowed["sessions"][0]["place"]["qs"], "place link carries the scope"
        assert "kind=image" in narrowed["sessions"][0]["place"]["qs"]
        assert narrowed["coverage"]["present"] == 3, "coverage counts the scope"
        assert (
            narrowed["coverage"]["contested_qs"].endswith("kind=image")
            or "kind=image" in narrowed["coverage"]["contested_qs"]
        )
        assert lisbon

        client.post(f"/i/{c}/place", json={"name": "Porto", "kind": "city"})
        client.post("/jobs/events")
        _drain(client)
        sessions = client.get("/timeline/density", params={"bin": "day"}, headers=AS_MACHINE).json()["sessions"]
        assert sessions[0]["place"] is None, "two places: the session is not in one"
        # a member gone missing leaves the card's count and its link agreeing
        conn = connect.connect(client.app.state.db_path)
        try:
            gone = conn.execute("SELECT id FROM file WHERE name LIKE 'Screenshot 2023-06-10 at 14.10%'").fetchone()[0]
            conn.execute("UPDATE file SET missing_since = 1.0 WHERE id = ?", (gone,))
            conn.commit()
        finally:
            connect.close(conn)
        [session] = client.get("/timeline/density", params={"bin": "day"}, headers=AS_MACHINE).json()["sessions"]
        assert session["pictures"] == 2 == session["in_scope"]
        link = client.get(f"/g?{session['qs']}", headers={"accept": "text/html"}).text
        assert 'data-total="2"' in link


def test_the_lightbox_says_where_too(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "p.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        [slug] = _slugs(client)
        assert "data-lightbox-where" not in client.get(f"/i/{slug}", headers={"hx-request": "true"}).text
        client.post(f"/i/{slug}/place", json={"name": "Porto", "kind": "city"})
        part = client.get(f"/i/{slug}", headers={"hx-request": "true"}).text
        assert "data-lightbox-where" in part
        assert ">in Porto</a>" in part


def test_the_timeline_takes_any_gallery_question_as_its_scope(tmp_path):
    """`/timeline?folder=lib`, `?kind=image`, `?person=...`: the same
    question the gallery answers, as the surface's scope; its links are
    that question plus a moment; a rule-defined album is refused and a
    slug nothing lives at is a 404."""
    root = tmp_path / "lib"
    (root / "deep").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(root / (f"p{i}.png" if i < 2 else "deep/p2.png"))
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        client.post("/jobs/ingest")
        client.post("/jobs/context")
        _drain(client)
        whole = client.get("/timeline/density", params={"bin": "day"}, headers=AS_MACHINE).json()
        assert sum(b["pictures"] for b in whole["bins"]) == 3
        assert whole["scope"] is None
        deep = client.get("/timeline/density", params={"bin": "day", "folder": "deep"}, headers=AS_MACHINE).json()
        assert sum(b["pictures"] for b in deep["bins"]) == 1
        assert deep["scope"]["qs"].startswith("folder=deep")
        assert all(b["qs"].startswith("folder=deep&") for b in deep["bins"]), "every link is the question plus a moment"
        page = client.get("/timeline", params={"folder": "deep"}, headers={"accept": "text/html"}).text
        assert 'data-timeline-scope="folder=deep"' in page
        assert client.get("/timeline/density", params={"bin": "day", "folder": "nowhere"}).status_code == 404
        # a scope's value is its own address, so it carries no spelling --
        # the field is there and null rather than absent, because a reader
        # should not have to branch on which keys a part happens to have
        assert client.get("/timeline/density", params={"bin": "day", "kind": "image"}, headers=AS_MACHINE).json()[
            "scope"
        ]["parts"] == [{"key": "kind", "value": "image", "spelled": None}]
        smart = client.post("/albums/smart", json={"name": "Deep Ones", "folder": "deep"})
        assert smart.status_code == 201, smart.text
        ruled = client.get(
            "/timeline/density", params={"bin": "day", "album": smart.json()["slug"]}, headers=AS_MACHINE
        ).json()
        assert sum(b["pictures"] for b in ruled["bins"]) == 1, "a rule-defined album scopes through the one engine"
        assert ruled["scope"]["qs"] == f"album={smart.json()['slug']}"
        assert (
            client.get(f"/t/{smart.json()['slug']}", headers=AS_MACHINE)
            .json()["timeline"]
            .endswith(smart.json()["slug"])
        )
        assert client.get("/f/deep", headers=AS_MACHINE).json()["timeline"] == "/timeline?folder=deep"
        assert "data-folder-timeline" in client.get("/f/deep", headers={"accept": "text/html"}).text


def test_a_place_can_be_within_another(tmp_path):
    """ "Lisbon within Portugal": the parent is found or minted the same
    way, a bare Lisbon said later to be within Portugal gains the parent
    rather than a twin, the page spells the chain, the shelf says
    "in Portugal", and a story freezes the whole chain."""
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        a, b = _slugs(client)
        bare = client.post(f"/i/{a}/place", json={"name": "Lisbon", "kind": "city"}).json()["where"]
        assert bare["chain"] == ["Lisbon"]
        nested = client.post(
            f"/i/{b}/place", json={"name": "lisbon", "kind": "city", "within": "Portugal", "within_kind": "country"}
        ).json()["where"]
        assert nested["id"] == bare["id"], "the bare Lisbon gained its parent; no twin"
        assert nested["chain"] == ["Lisbon", "Portugal"]
        assert client.get(f"/i/{a}", headers=AS_MACHINE).json()["where"]["chain"] == ["Lisbon", "Portugal"]
        page = client.get(f"/i/{a}", headers={"accept": "text/html"}).text
        assert "Lisbon</a>, Portugal" in page
        shelf = {p["name"]: p for p in client.get("/places", headers=AS_MACHINE).json()}
        assert shelf["Lisbon"]["within"] == "Portugal"
        assert shelf["Portugal"]["within"] is None
        assert "in Portugal" in client.get("/places", headers={"accept": "text/html"}).text
        # a Lisbon already within somewhere else is another Lisbon
        other = client.post(
            f"/i/{b}/place", json={"name": "Lisbon", "kind": "city", "within": "Iowa", "within_kind": "region"}
        ).json()["where"]
        assert other["id"] != bare["id"]
        assert other["chain"] == ["Lisbon", "Iowa"]


def test_two_lisbons_cannot_be_minted(tmp_path):
    """The index is the word: a second place of the same kind, name and
    parent is refused by the database itself, whatever the minter
    believed; and `named` finds the parented one before adopting a bare
    one."""
    import sqlite3

    from db import places
    from tests.staging import fresh_schema

    conn = fresh_schema()
    lisbon = places.named(conn, "Lisbon", "city", 1.0)
    with pytest.raises(sqlite3.IntegrityError):
        places.place(conn, "lisbon", "city", 1.0)
    portugal = places.named(conn, "Portugal", "country", 1.0)
    assert places.named(conn, "Lisbon", "city", 1.0, within=portugal) == lisbon, "the bare one gained its parent"
    with pytest.raises(sqlite3.IntegrityError):
        places.place(conn, "Lisbon", "city", 1.0, parent_id=portugal)
    bare_again = places.named(conn, "Lisbon", "city", 1.0)
    assert bare_again == lisbon, "a bare ask finds the one Lisbon, parented or not"
    assert conn.execute("SELECT count(*) FROM place WHERE name = 'Lisbon'").fetchone()[0] == 1


def test_placing_a_selection_re_interprets_in_one_pass(tmp_path, monkeypatch):
    """The bulk write re-interprets every selected file in ONE pass -- one
    read of the folder tree -- not one rebuild per file inside the
    writer lane."""
    from db import context

    calls: list = []
    real = context._folder_names

    def counted(conn):
        calls.append(1)
        return real(conn)

    monkeypatch.setattr(context, "_folder_names", counted)
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(4):
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        page = client.get("/g", params={"folder": "lib"}).text
        import re

        answer = _answer_of(page)
        keys = re.findall(r'data-selection-key="([0-9a-f]{32})"', page)
        assert len(keys) == 4
        calls.clear()
        told = client.post(
            "/g/selection/place",
            params={"folder": "lib"},
            json={"answer": answer, "items": keys, "name": "Porto", "kind": "city"},
        )
        assert told.status_code < 300, told.text
        assert told.json()["targets"] == 4
        assert len(calls) == 1, f"the folder tree was read {len(calls)} times for 4 files"
        for slug in _slugs(client):
            assert client.get(f"/i/{slug}", headers=AS_MACHINE).json()["where"]["name"] == "Porto"


def test_a_faceted_gallery_can_be_curated_and_walked(tmp_path):
    """The links this product grew carry facets; the bulk writes and the
    picture page must prove and walk THAT question, or a selection made
    on a place's link 409s forever and a picture opened from it walks
    the whole library."""
    import re

    root = tmp_path / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        client.post("/jobs/ingest")
        client.post("/jobs/context")
        _drain(client)
        a, b, c = _slugs(client)
        where = client.post(f"/i/{a}/place", json={"name": "Lisbon", "kind": "city"}).json()["where"]
        client.post(f"/i/{b}/place", json={"name": "Lisbon", "kind": "city"})
        spelled = f"place.id:eq:{where['id']}"
        page = client.get("/g", params={"f": spelled}).text
        answer = _answer_of(page)
        keys = re.findall(r'data-selection-key="([0-9a-f]{32})"', page)
        assert len(keys) == 2
        told = client.post(
            "/g/selection/favorite", params={"f": spelled}, json={"answer": answer, "items": keys, "value": True}
        )
        assert told.status_code < 300, f"the bulk write must prove against the faceted question: {told.text}"
        assert told.json()["targets"] == 2
        placed = client.post(
            "/g/selection/place",
            params={"f": spelled},
            json={"answer": answer, "items": keys, "name": "Porto", "kind": "city"},
        )
        assert placed.status_code < 300, placed.text
        opened = client.get(f"/i/{c}", params={"f": spelled}, headers=AS_MACHINE).json()
        assert opened["context"]["in_answer"] is False, "c was never in Lisbon: not in the faceted answer"
        # Porto now: a and b moved; open a under the Porto link and walk it
        porto = client.get(f"/i/{a}", headers=AS_MACHINE).json()["where"]
        walked = client.get(f"/i/{a}", params={"f": f"place.id:eq:{porto['id']}"}, headers=AS_MACHINE).json()
        assert walked["context"]["total"] == 2, "the page walks the faceted question it was opened from"
        assert "place.id" in walked["context"]["qs"]


def test_a_sweep_takes_its_weights_from_the_setting_never_the_body(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "p.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        for route in ("/jobs/faces", "/jobs/annotate"):
            # Naming weights in the body is refused outright -- the sweep's
            # body contract is `everything` and nothing else.
            assert client.post(route, json={"models_dir": "Z:/somewhere/else", "everything": True}).status_code == 400
            asked = client.post(route, json={"everything": True})
            assert asked.status_code in (201, 400), asked.text  # 400 only when the setting itself is refused
        conn = connect.connect(client.app.state.db_path, read_only=True)
        try:
            dirs = [
                row[0]
                for row in conn.execute("SELECT json_extract(payload, '$.models_dir') FROM job")
                if row[0] is not None
            ]
        finally:
            connect.close(conn)
        assert dirs
        assert all("somewhere" not in one for one in dirs), dirs


def test_a_place_within_itself_is_refused_not_a_500(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(root / "p.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        (a,) = _slugs(client)
        told = client.post(
            f"/i/{a}/place", json={"name": "Lisbon", "kind": "city", "within": "Lisbon", "within_kind": "city"}
        )
        assert told.status_code == 400, told.text
        assert "within itself" in told.json()["detail"]
        client.post(
            f"/i/{a}/place", json={"name": "Lisbon", "kind": "city", "within": "Portugal", "within_kind": "country"}
        )
        # Portugal inside Lisbon would close the loop
        told = client.post(
            f"/i/{a}/place", json={"name": "Portugal", "kind": "country", "within": "Lisbon", "within_kind": "city"}
        )
        assert told.status_code == 400, told.text
        shelf = client.get("/places", headers=AS_MACHINE).json()
        assert sorted(p["name"] for p in shelf) == ["Lisbon", "Portugal"], "a refusal mints nothing"


def test_a_bare_mention_of_two_placed_twins_is_ambiguous(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (8, 8), (30 * i, 40, 50)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        a, b = _slugs(client)
        client.post(
            f"/i/{a}/place", json={"name": "Lisbon", "kind": "city", "within": "Portugal", "within_kind": "country"}
        )
        client.post(f"/i/{b}/place", json={"name": "Lisbon", "kind": "city", "within": "Iowa", "within_kind": "region"})
        told = client.post(f"/i/{b}/place", json={"name": "Lisbon", "kind": "city"})
        assert told.status_code == 400, told.text
        assert "Portugal" in told.json()["detail"]
        assert "Iowa" in told.json()["detail"]
        # said which: found, not minted
        told = client.post(
            f"/i/{b}/place", json={"name": "lisbon", "kind": "city", "within": "Iowa", "within_kind": "region"}
        )
        assert told.status_code < 300, told.text
        assert told.json()["where"]["chain"] == ["Lisbon", "Iowa"]
