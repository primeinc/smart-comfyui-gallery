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

from db import connect, derived, library, naming, scan, settings
from sg_web.app import build_app


@pytest.fixture
def served(tmp_path):
    """A real library on disk, clustered, behind a running application."""
    root = tmp_path / "lib"
    root.mkdir()
    for name in ("ana_1.png", "ana_2.png", "ben_1.png"):
        (root / name).write_bytes(b"\x89PNG-of-" + name.encode())

    burrow = tmp_path / "run"
    burrow.mkdir()
    db_path = burrow / "gallery.db"
    conn = connect.connect(db_path)
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    # The backend is not under test here; the numpy path is exact and needs
    # no hardware. The similarity_backend setting is the production knob.
    settings.put(conn, "similarity_backend", "numpy")
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

    with TestClient(app=build_app(str(burrow))) as client:
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


def test_job_progress_is_pushed_over_the_socket_not_polled(served):
    """Submit a sweep and watch it happen: the socket sends the persisted
    snapshot first, then a delta per observable change, ending in the
    terminal state -- and the row agrees with everything it said."""
    client, _, _ = served
    with client.websocket_connect("/ws/jobs") as feed:
        first = feed.receive_json(timeout=10)
        assert first["type"] == "snapshot"
        assert first["jobs"] == []

        submitted = client.post("/jobs/verify").json()
        assert (submitted["state"], submitted["total"]) == ("queued", 3)
        job_id = submitted["id"]

        seen, state = [], None
        while state not in ("done", "failed", "cancelled"):
            delta = feed.receive_json(timeout=10)
            assert delta["job"] == job_id
            assert delta["total"] == 3
            seen.append(delta["done"])
            state = delta["state"]
        assert state == "done"
        assert seen == sorted(seen), "progress went backwards on the wire"
        assert seen[-1] == 3

    snapshot = client.get(f"/jobs/{job_id}").json()
    assert (snapshot["state"], snapshot["done_count"], snapshot["failed_count"]) == ("done", 3, 0)
    assert client.get("/jobs/999").status_code == 404


def test_verification_catches_tampered_bytes_while_you_watch(served):
    """The library changes behind the application's back; the sweep runs
    live and the cold row still knows how many files it could not vouch
    for."""
    client, _, root = served
    (root / "ana_2.png").write_bytes(b"\x89PNG-TAMPERED")
    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job_id = client.post("/jobs/verify").json()["id"]
        state = None
        while state not in ("done", "failed", "cancelled"):
            state = feed.receive_json(timeout=10)["state"]
        assert state == "done"
    snapshot = client.get(f"/jobs/{job_id}").json()
    assert (snapshot["done_count"], snapshot["failed_count"]) == (3, 1)


def test_the_worker_obeys_its_setting_changed_over_http(served):
    """Turning the worker off is a settings row, effective live: jobs
    queue and wait, cancellation is honoured the moment it returns."""
    import time as clock

    from sg_web import worker as worker_module

    client, _, _ = served
    client.post("/settings/worker", json={"value": "off"})
    job_id = client.post("/jobs/verify").json()["id"]
    clock.sleep(worker_module.IDLE_WAIT * 2.5)
    assert client.get(f"/jobs/{job_id}").json()["state"] == "queued", "the worker ignored its off switch"

    assert client.post(f"/jobs/{job_id}/cancel").json()["cancel_requested"] == 1
    client.post("/settings/worker", json={"value": "on"})
    deadline = clock.time() + 10
    state = None
    while clock.time() < deadline and state != "cancelled":
        state = client.get(f"/jobs/{job_id}").json()["state"]
        clock.sleep(0.05)
    assert state == "cancelled"


def test_choose_primary_is_an_action_the_application_offers(served):
    client, _, _ = served
    chosen = client.post("/clusterings/choose").json()
    assert chosen["primary_run"] is not None
    assert client.get("/clusterings").json()[0]["id"] == chosen["primary_run"]


def test_a_whole_run_is_contained_in_one_redirectable_directory(tmp_path):
    """One --home argument moves everything a run owns -- database, models,
    caches. Nothing lands in OS application-data folders, and a first run
    needs nothing but the command that starts it."""
    from sg_web import home

    burrow = tmp_path / "elsewhere"
    assert home.home(burrow) == burrow
    assert home.db_path(burrow) == burrow / "gallery.db"
    assert home.models_dir(burrow) == burrow / "models"
    assert home.thumbs_dir(burrow) == burrow / "thumbs"

    shared = tmp_path / "shared-weights"
    assert home.models_dir(burrow, str(shared)) == shared, "a shared model dir is an option"

    with TestClient(app=build_app(str(burrow))) as client:
        assert client.get("/health").text == "ok"
        assert client.get("/people").json() == []
    assert (burrow / "gallery.db").exists(), "the run did not live in its home"


def test_settings_are_rows_changed_while_the_application_runs(tmp_path):
    """Configuration is settings rows, not environment variables: listed,
    changed and validated over requests, effective without a restart."""
    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        listed = {row["key"]: row for row in client.get("/settings").json()}
        assert listed["similarity_backend"]["value"] == "auto"
        assert "numpy" in listed["similarity_backend"]["choices"]

        changed = client.post("/settings/similarity_backend", json={"value": "numpy"}).json()
        assert changed == {"key": "similarity_backend", "value": "numpy"}
        listed = {row["key"]: row for row in client.get("/settings").json()}
        assert listed["similarity_backend"]["value"] == "numpy"

        assert client.post("/settings/similarity_backend", json={"value": "cuda-magic"}).status_code == 400
        assert client.post("/settings/not_a_setting", json={"value": "x"}).status_code == 400


def test_a_bodyless_or_pathless_root_request_is_a_400_not_a_500(tmp_path):
    """The body is typed, so a missing or mistyped `path` is refused by
    the signature model with a 400 -- never a KeyError escaping as 500
    from a write route."""
    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        assert client.post("/roots", json={}).status_code == 400
        assert client.post("/roots", json={"kind": "library"}).status_code == 400
        assert client.post("/roots", json={"path": 123}).status_code == 400
        assert client.post("/roots").status_code == 400


def test_shutdown_stops_the_worker_before_the_channel_it_publishes_to(tmp_path, caplog):
    """Lifespan managers exit in reverse, so the worker must be REGISTERED
    after the channel: stopped and joined while the channel it publishes
    to is still alive. Ordered wrongly, a shutdown mid-drain logged
    "Plugin not yet initialized" from the bridge -- and the docstring's
    "no thread mid-write" was a lie. The job it was draining stays rows,
    picked up by the next run."""
    import logging

    from litestar.channels import ChannelsPlugin

    root = tmp_path / "lib"
    root.mkdir()
    for n in range(300):
        (root / f"frame_{n:03}.png").write_bytes(b"\x89PNG-" + f"{n:03}".encode() * 64)
    burrow = tmp_path / "run"
    burrow.mkdir()
    conn = connect.connect(burrow / "gallery.db")
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    conn.commit()
    conn.close()

    with caplog.at_level(logging.ERROR), TestClient(app=build_app(str(burrow))) as client:
        managers = client.app._lifespan_managers
        channel_at = next(i for i, m in enumerate(managers) if isinstance(m, ChannelsPlugin))
        worker_at = next(i for i, m in enumerate(managers) if getattr(m, "__name__", "") == "working")
        assert channel_at < worker_at, "the channel would tear down under a live worker"
        job_id = client.post("/jobs/verify").json()["id"]
        # leave immediately: the worker is mid-drain as the app exits

    said = " ".join(record.getMessage() for record in caplog.records)
    assert "Plugin not yet initialized" not in said, "a publish landed on a torn-down channel"
    assert "worker turn died" not in said

    import time as clock

    with TestClient(app=build_app(str(burrow))) as client:
        deadline = clock.time() + 30
        state = None
        while state != "done":
            assert clock.time() < deadline, "the interrupted job was never picked back up"
            state = client.get(f"/jobs/{job_id}").json()["state"]
            clock.sleep(0.1)


def test_media_roots_are_rows_managed_through_the_application(tmp_path):
    """Any number of media directories, anywhere, registered and scanned
    over requests -- the pictures never live inside the run's home."""
    box_one = tmp_path / "comfy-output"
    box_two = tmp_path / "phone-camera"
    for box in (box_one, box_two):
        box.mkdir()
        (box / "shot.png").write_bytes(b"\x89PNG-" + box.name.encode())

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
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
