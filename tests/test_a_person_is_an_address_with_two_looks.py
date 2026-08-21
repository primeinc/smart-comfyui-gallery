"""One person address, a page and a drawer, never a second resource.

`/p/{slug}` answers as the PersonView for machines, as a drawer
fragment for the mounted People index, and as the full profile for a
browser -- the same negotiation contract the media address carries,
with a drawer instead of a lightbox because a person is an entity with
a collection, not a piece of media. `/people` renders for a browser
and stays the historical JSON list for everything else.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, derived, library, naming, scan
from sg_web.app import build_app

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}
AS_OVERLAY = {"hx-request": "true"}


def _clustered_library(tmp: pathlib.Path) -> tuple:
    """Real files, fake-but-consistent face embeddings, one clustered
    person named Ana -- the served-fixture recipe, on disk."""
    root = tmp / "lib"
    root.mkdir()
    for name in ("ana_1.png", "ana_2.png", "ben_1.png"):
        Image.new("RGB", (16, 16), (200, 90, 40)).save(root / name)

    burrow = tmp / "run"
    burrow.mkdir()
    db_path = burrow / "gallery.db"
    conn = connect.connect(db_path)
    conn.executescript(connect.schema_sql())
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)

    files = {name: file_id for file_id, name in conn.execute("SELECT id, name FROM file")}
    rng = np.random.default_rng(5)
    ana = rng.standard_normal(32).astype(np.float32)
    for name, vector in (("ana_1.png", ana), ("ana_2.png", ana), ("ben_1.png", -ana)):
        derived.record_faces(
            conn,
            files[name],
            "test/embedder",
            "1",
            "aa",
            0.0,
            [
                {
                    "region": derived.region(conn, 0.1, 0.1, 0.2, 0.2),
                    "embedding": (vector + 0.01 * rng.standard_normal(32).astype(np.float32)).tobytes(),
                }
            ],
        )
    made = derived.cluster(conn, "test/embedder", "1", 0.0, threshold=0.55)
    assert len(made) == 1, "ana's pair must cluster; ben stays a singleton"
    run_id = derived.run_for(conn, "test/embedder", "1", "chinese-whispers", 0.55, 0.0)
    derived.make_primary(conn, run_id)
    person_id = naming.claim(conn, "person", "Ana")
    conn.execute("INSERT INTO person(id,name,created_at) VALUES(?, 'Ana', 0)", (person_id,))
    # The cluster carries the person, as the clustering job leaves it --
    # naming writes its durable assertions FROM this attachment.
    conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE id = ?", (person_id, made[0]))
    for name in ("ana_1.png", "ana_2.png"):
        derived.attribute(conn, files[name], person_id, run_id, "test/embedder", "1")
    conn.commit()
    conn.close()
    return burrow, root


@pytest.fixture(scope="module")
def faces(tmp_path_factory):
    burrow, _ = _clustered_library(tmp_path_factory.mktemp("person"))
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        yield client


def test_the_person_address_has_three_faces_and_declares_vary(faces):
    told = faces.get("/p/ana", headers=AS_MACHINE)
    page = faces.get("/p/ana", headers=AS_BROWSER)
    part = faces.get("/p/ana", headers=AS_OVERLAY)
    for answer in (told, page, part):
        assert answer.status_code == 200
        assert answer.headers["vary"] == "Accept, HX-Request"
    body = told.json()
    assert body["name"] == "Ana"
    assert body["count"] == 2
    assert sorted(p["name"] for p in body["pictures"]) == ["ana_1.png", "ana_2.png"]
    assert '<link rel="canonical" href="/p/ana">' in page.text
    assert "<html" in page.text
    assert "<html" not in part.text, "a drawer mounts into a page, never a page into a page"
    assert "data-drawer" in part.text
    assert "open full profile" in part.text, "the drawer hands off to the page it stands in for"
    assert "data-nav" not in part.text, "a person is not media: the drawer has no lightbox arrows"


def test_the_wildcard_machine_caller_keeps_its_json(faces):
    told = faces.get("/p/ana")
    assert told.headers["content-type"].startswith("application/json")
    assert told.json() == faces.get("/p/ana", headers=AS_MACHINE).json()
    index = faces.get("/people")
    assert index.headers["content-type"].startswith("application/json")
    assert index.json() == [{"name": "Ana", "slug": "ana", "pictures": 2}]


def test_the_people_index_renders_for_a_browser(faces):
    page = faces.get("/people", headers=AS_BROWSER)
    assert page.status_code == 200
    assert 'data-person="ana"' in page.text
    assert "/avatar/ana" in page.text
    assert "2 pictures" in page.text


def test_naming_still_mints_the_new_address(faces):
    told = faces.post("/p/ana/name", json={"name": "Ana Torres"})
    assert told.status_code < 300
    assert told.json()["slug"] == "ana-torres"
    moved = faces.get("/p/ana", headers=AS_BROWSER, follow_redirects=False)
    assert moved.status_code == 301
    assert moved.headers["location"] == "/p/ana-torres"
    assert '<link rel="canonical" href="/p/ana-torres">' in faces.get("/p/ana-torres", headers=AS_BROWSER).text


def test_the_person_path_owns_no_sql_and_the_drawer_is_not_a_lightbox():
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "sg_web" / "person_view.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"execute", "executemany", "executescript", "cursor"}, (
        "the person routes ran their own statement; the queries live in db/pages.py"
    )
    spoken = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "db":
            spoken |= {alias.name for alias in node.names}
    assert spoken <= {"authored", "connect", "naming", "pages", "resultset", "settings"}, (
        f"unexpected db vocabulary: {sorted(spoken)}"
    )


def test_a_commit_mid_assembly_cannot_mix_person_generations(tmp_path, monkeypatch):
    """The MediaView invariant, held here too: a naming commit landing
    between person_files and the remaining reads must not hand back
    pictures from one generation under the name of another. The writer
    renames the person exactly inside that window -- the response is
    wholly before, and the next request wholly after."""
    from db import pages

    burrow, _ = _clustered_library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        real = pages.person_files

        def files_then_commit(conn, person_id, run_id=None):
            rows = real(conn, person_id, run_id)
            writer = connect.connect(client.app.state.db_path)
            writer.execute("UPDATE person SET name = 'Renamed Mid-Read' WHERE id = ?", (person_id,))
            writer.commit()
            connect.close(writer)
            return rows

        monkeypatch.setattr(pages, "person_files", files_then_commit)
        raced = client.get("/p/ana").json()
        assert raced["name"] == "Ana", (
            "the name read a newer generation than the pictures: one response mixed two library states"
        )
        monkeypatch.setattr(pages, "person_files", real)
        assert client.get("/p/ana").json()["name"] == "Renamed Mid-Read", (
            "the NEXT response must see the commit; the snapshot is per-request, not a cache"
        )


def test_a_person_is_a_resultset_scope(tmp_path):
    """The bridge WI-38 exists for: person membership is the primary
    run's attribution, ordered and paged like every other scope, and
    the profile's grid is ONE ResultSet page whose links carry the
    person context so the arrows walk the person."""
    from db import resultset

    burrow, _ = _clustered_library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        db_path = client.app.state.db_path
        conn = connect.connect(db_path)
        told = resultset.describe(conn, "", resultset.parse(person="ana"), 0.0)
        assert told["total"] == 2, "membership is the attribution, not the library"
        with pytest.raises(LookupError):
            resultset.describe(conn, "", resultset.parse(person="nobody"), 0.0)

        # Person is a COMPOSABLE membership facet, not a third exclusive
        # scope: eligibility is an intersection of predicates.
        both = resultset.describe(conn, "", resultset.parse(person="ana", folder="lib"), 0.0)
        assert both["total"] == 2, "person AND folder is an intersection"
        narrowed = resultset.describe(conn, "", resultset.parse(person="ana", kind="video"), 0.0)
        assert narrowed["total"] == 0, "person AND kind is an intersection"
        from db import authored

        mixed = authored.collection(conn, "Mixed", 0.0)
        for name in ("ana_1.png", "ben_1.png"):
            file_id = conn.execute("SELECT id FROM file WHERE name = ?", (name,)).fetchone()[0]
            authored.add_to_collection(conn, mixed, file_id, 0.0)
        conn.commit()
        crossed = resultset.describe(conn, "", resultset.parse(person="ana", album="mixed"), 0.0)
        assert crossed["total"] == 1, "person AND album keeps only the attributed member of the album"
        connect.close(conn)

        # /g serves the person scope like any other question.
        grid = client.get("/g/grid", params={"person": "ana"}).text
        assert grid.count('data-slug="ana-') == 2
        assert "ben-1" not in grid

        # The profile's own grid is the ResultSet page, links in context.
        page = client.get("/p/ana", headers=AS_BROWSER).text
        assert page.count("?person=ana") >= 2, "profile media links must carry the person context"
        body = client.get("/p/ana").json()
        assert body["gallery"]["total"] == 2
        assert body["gallery"]["qs"] == "person=ana"
        assert [row["slug"] for row in body["gallery"]["items"]] == [
            row["slug"]
            for row in resultset.page(connect.connect(db_path), "", resultset.parse(person="ana"), 1, 0.0)["items"]
        ]

        # The item address under the person context walks the person.
        first = body["gallery"]["items"][0]["slug"]
        second = body["gallery"]["items"][1]["slug"]
        walked = client.get(f"/i/{first}", params={"person": "ana"}).json()
        assert walked["context"]["total"] == 2
        assert walked["next"] == second
        assert walked["previous"] is None
        assert walked["context"]["return_url"] == "/g?person=ana"
        outside = client.get("/i/ben-1", params={"person": "ana"}).json()
        assert outside["context"]["in_answer"] is False


def test_a_person_phrase_constrains_each_space_before_fusion(tmp_path, monkeypatch):
    """person + q composes through the SAME constrained-RRF door every
    scope uses: the person's membership reaches retrieval as the
    allowed set, applied per space before the fusion."""
    from db import resultset, retrieval

    burrow, _ = _clustered_library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        conn = connect.connect(client.app.state.db_path)
        anas = {
            row[0]
            for row in conn.execute(
                "SELECT fp.file_id FROM derived_file_person fp JOIN person p ON p.id = fp.person_id"
            )
        }
        seen: dict = {}

        def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
            seen.update({"allowed": allowed, "k": k})
            return {"results": [], "participants": [], "contributors": [], "missing": {}}

        monkeypatch.setattr(retrieval, "query", fused)
        resultset.describe(conn, "", resultset.parse(person="ana", text="beach"), 0.0)
        assert seen["allowed"] == anas, "the person facet must constrain retrieval before RRF"
        assert seen["k"] == len(anas)
        connect.close(conn)


def test_a_renamed_person_is_one_cached_question_and_context_heals(tmp_path):
    """Slugs are presentation; the projection keys on the bound entity.
    Both spellings of a renamed person are ONE question, and every
    answer re-spells the context with the LIVE slug so stale bookmarks
    heal as they are navigated."""
    from db import naming, resultset

    burrow, _ = _clustered_library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        conn = connect.connect(client.app.state.db_path)
        found = naming.resolve(conn, "person", "ana")
        assert found is not None
        naming.rename(conn, found[0], "Ana Torres", 5.0)
        conn.commit()

        old_spelling = resultset.describe(conn, "", resultset.parse(person="ana"), 0.0)
        new_spelling = resultset.describe(conn, "", resultset.parse(person="ana-torres"), 0.0)
        assert old_spelling["total"] == new_spelling["total"] == 2
        assert old_spelling["fingerprint"] == new_spelling["fingerprint"], (
            "two spellings of one person forked the projection cache"
        )
        assert old_spelling["qs"] == "person=ana-torres", "the answer re-spells the context live"
        connect.close(conn)

        walked = client.get("/i/ana-1", params={"person": "ana"}).json()
        assert walked["context"]["qs"] == "person=ana-torres"
        assert walked["context"]["return_url"] == "/g?person=ana-torres"


def test_switching_the_primary_run_is_a_different_question(tmp_path):
    """Person membership means THE PRIMARY run's attribution; promoting
    another run changes both the answer and its identity -- never a
    silently reused projection."""
    from db import derived, resultset

    burrow, _ = _clustered_library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        conn = connect.connect(client.app.state.db_path)
        before = resultset.describe(conn, "", resultset.parse(person="ana"), 0.0)
        assert before["total"] == 2

        other = derived.run_for(conn, "other/embedder", "2", "chinese-whispers", 0.4, 1.0)
        derived.make_primary(conn, other)
        conn.commit()

        after = resultset.describe(conn, "", resultset.parse(person="ana"), 1.0)
        assert after["total"] == 0, "the new primary attributes ana nothing; the membership must say so"
        assert after["fingerprint"] != before["fingerprint"], "the bound run is part of the question"
        connect.close(conn)


def test_the_browser_path_never_enumerates_the_whole_collection(tmp_path, monkeypatch):
    """The bounded profile must be bounded on the SERVER, not only in
    the DOM: the unbounded legacy list is the JSON Adapter's cost, and
    the HTML and drawer paths never pay it."""
    from db import pages

    burrow, _ = _clustered_library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        walked: list[int] = []
        real = pages.person_files

        def counted(conn, person_id, run_id=None):
            walked.append(person_id)
            return real(conn, person_id, run_id)

        monkeypatch.setattr(pages, "person_files", counted)
        assert client.get("/p/ana", headers=AS_BROWSER).status_code == 200
        assert client.get("/p/ana", headers=AS_OVERLAY).status_code == 200
        assert walked == [], "a rendered profile enumerated the person's whole photographic existence"
        body = client.get("/p/ana").json()
        assert len(walked) == 1, "the machine Adapter still carries the legacy list"
        assert sorted(p["name"] for p in body["pictures"]) == ["ana_1.png", "ana_2.png"]
