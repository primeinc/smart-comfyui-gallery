"""The application answers over HTTP from nothing but the schema.

One real library on disk, one database file, and every claim below is a
request: people are addressed by slug and survive a rename, jobs are
submitted, budgeted, cancelled and finished through the routes, and the
People page shows what the clustering run put there. This is the seam the
plan calls Phase 2 -- if any of it needs code outside `sg_web` and `db`,
the schema was not an application schema.
"""

from __future__ import annotations

import numpy as np
import pytest
from litestar.testing import TestClient

from db import connect, derived, library, naming, scan
from sg_web.app import build_app


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A real library on disk, clustered, behind a running application."""
    # The backend is not under test here; the numpy path is exact and needs
    # no hardware. SG_SIMILARITY_BACKEND is the production knob for this.
    monkeypatch.setenv("SG_SIMILARITY_BACKEND", "numpy")

    root = tmp_path / "lib"
    root.mkdir()
    for name in ("ana_1.png", "ana_2.png", "ben_1.png"):
        (root / name).write_bytes(b"\x89PNG-of-" + name.encode())

    db_path = tmp_path / "gallery.db"
    conn = connect.connect(db_path)
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)

    files = {name: file_id for file_id, name in conn.execute("SELECT id, name FROM file")}
    rng = np.random.default_rng(5)
    ana = rng.standard_normal(32).astype(np.float32)
    ben = -ana
    for name, vector in (("ana_1.png", ana), ("ana_2.png", ana), ("ben_1.png", ben)):
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
    assert len(made) == 1  # ana's pair; ben is a singleton and stays a face
    run_id = derived.run_for(conn, "test/embedder", "1", "chinese-whispers", 0.55, 0.0)
    derived.make_primary(conn, run_id)

    person_id = naming.claim(conn, "person", "Ana")
    conn.execute("INSERT INTO person(id,name,created_at) VALUES(?, 'Ana', 0)", (person_id,))
    for name in ("ana_1.png", "ana_2.png"):
        derived.attribute(conn, files[name], person_id, run_id, "test/embedder", "1")
    conn.commit()
    conn.close()

    with TestClient(app=build_app(str(db_path))) as client:
        yield client, person_id, root


def test_the_people_page_counts_pictures_not_detections(served):
    client, _, _ = served
    answer = client.get("/people")
    assert answer.status_code == 200
    assert answer.json() == [{"name": "Ana", "slug": "ana", "pictures": 2}]


def test_a_person_is_addressed_by_slug_and_shows_the_cross_axis_view(served):
    client, _, _ = served
    answer = client.get("/p/ana")
    assert answer.status_code == 200
    page = answer.json()
    assert page["name"] == "Ana"
    assert sorted(p["name"] for p in page["pictures"]) == ["ana_1.png", "ana_2.png"]
    assert page["across_folders"][0]["pictures"] == 2


def test_a_renamed_person_still_answers_at_the_old_address(served):
    client, person_id, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    fresh = naming.rename(conn, person_id, "Ana Torres", 1.0)
    conn.commit()
    conn.close()
    assert fresh == "ana-torres"
    assert client.get("/p/ana-torres").status_code == 200
    moved = client.get("/p/ana", follow_redirects=False)
    assert moved.status_code == 301
    assert moved.headers["location"] == "/p/ana-torres"
    assert client.get("/p/ana").status_code == 200  # and following it lands
    assert client.get("/p/nobody").status_code == 404


def test_clusterings_are_public_and_the_primary_is_marked(served):
    client, _, _ = served
    runs = client.get("/clusterings").json()
    assert len(runs) == 1
    assert runs[0]["is_primary"] == 1
    assert runs[0]["method"] == "chinese-whispers"


def test_the_whole_job_lifecycle_happens_over_requests(served):
    """Submit, budgeted progress, snapshot from cold, cancel, resubmit,
    completion -- every step a request, the row the only state."""
    client, _, root = served

    submitted = client.post("/jobs/verify").json()
    assert (submitted["state"], submitted["total"]) == ("queued", 3)
    job_id = submitted["id"]

    turn = client.post("/worker/turn", json={"budget": 1}).json()
    assert (turn["state"], turn["did"]) == ("running", 1)
    snapshot = client.get(f"/jobs/{job_id}").json()
    assert (snapshot["state"], snapshot["done_count"]) == ("running", 1)

    cancelled = client.post(f"/jobs/{job_id}/cancel").json()
    assert cancelled["cancel_requested"] == 1
    turn = client.post("/worker/turn").json()
    assert turn["state"] == "cancelled"

    # The library changes behind the application's back; a fresh sweep says so.
    (root / "ana_2.png").write_bytes(b"\x89PNG-TAMPERED")
    job_id = client.post("/jobs/verify").json()["id"]
    turn = client.post("/worker/turn").json()
    assert (turn["state"], turn["failed"]) == ("done", 1)
    snapshot = client.get(f"/jobs/{job_id}").json()
    assert (snapshot["state"], snapshot["done_count"]) == ("done", 3)

    assert client.post("/worker/turn").json() == {"state": "idle"}
    assert client.get("/jobs/999").status_code == 404


def test_choose_primary_is_an_action_the_application_offers(served):
    client, _, _ = served
    chosen = client.post("/clusterings/choose").json()
    assert chosen["primary_run"] is not None
    assert client.get("/clusterings").json()[0]["id"] == chosen["primary_run"]


def test_a_whole_run_is_contained_in_one_redirectable_directory(tmp_path, monkeypatch):
    """SMARTGALLERY_HOME moves everything a run owns -- database, models --
    in one setting. Nothing lands in OS application-data folders, and a
    first run needs nothing but the command that starts it."""
    from sg_web import home

    burrow = tmp_path / "elsewhere"
    monkeypatch.setenv("SMARTGALLERY_HOME", str(burrow))
    monkeypatch.delenv("SMARTGALLERY_MODELS", raising=False)
    assert home.home() == burrow
    assert home.db_path() == burrow / "gallery.db"
    assert home.models_dir() == burrow / "models"

    shared = tmp_path / "shared-weights"
    monkeypatch.setenv("SMARTGALLERY_MODELS", str(shared))
    assert home.models_dir() == shared, "a shared model dir is an option"

    with TestClient(app=build_app()) as client:
        assert client.get("/health").text == "ok"
        assert client.get("/people").json() == []
    assert (burrow / "gallery.db").exists(), "the run did not live in its home"


def test_media_roots_are_rows_managed_through_the_application(tmp_path, monkeypatch):
    """Any number of media directories, anywhere, registered and scanned
    over requests -- the pictures never live inside the run's home."""
    monkeypatch.setenv("SMARTGALLERY_HOME", str(tmp_path / "run"))
    box_one = tmp_path / "comfy-output"
    box_two = tmp_path / "phone-camera"
    for box in (box_one, box_two):
        box.mkdir()
        (box / "shot.png").write_bytes(b"\x89PNG-" + box.name.encode())

    with TestClient(app=build_app()) as client:
        first = client.post("/roots", json={"path": str(box_one)}).json()
        second = client.post("/roots", json={"path": str(box_two)}).json()
        assert first["id"] != second["id"]

        listed = client.get("/roots").json()
        assert [r["online"] for r in listed] == [True, True]

        swept = client.post(f"/roots/{first['id']}/scan").json()
        assert swept["added"] == 1
        swept = client.post(f"/roots/{second['id']}/scan").json()
        assert swept["added"] == 1
        assert client.post("/roots/999/scan").status_code == 404
