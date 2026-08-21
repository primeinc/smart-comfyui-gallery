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
    assert spoken <= {"authored", "connect", "naming", "pages", "resultset"}, (
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
