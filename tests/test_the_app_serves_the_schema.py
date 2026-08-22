"""The application answers over HTTP from nothing but the schema.

One real library on disk, one database file, and every claim below is a
request: people are addressed by slug and survive a rename, jobs are
submitted, budgeted, cancelled and finished through the routes, and the
People page shows what the clustering run put there. This is the seam the
plan calls Phase 2 -- if any of it needs code outside `sg_web` and `db`,
the schema was not an application schema.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from db import authored, collection_rules, collections, connect, derived, naming
from tests.staging import Stage, staged


def _library(root: pathlib.Path) -> None:
    for name in ("ana_1.png", "ana_2.png", "ben_1.png"):
        (root / name).write_bytes(b"\x89PNG-of-" + name.encode())


def _clustered(stage: Stage) -> None:
    """Faces recorded and clustered, Ana named -- once per module."""
    conn = stage.conn()
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
    connect.close(conn)
    stage.held["person"] = person_id


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "served", _library, _clustered, worker=True) as stage:
        yield stage


@pytest.fixture
def served(_stage):
    """A real library on disk, clustered, behind a running application."""
    _stage.restore()
    return _stage.client, _stage.held["person"], _stage.root


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


@pytest.mark.slow
def test_the_worker_obeys_its_setting_changed_over_http(served):
    """Turning the worker off is a settings row, effective live: jobs
    queue and wait, cancellation is honoured the moment it returns.

    No sleeps and no polling: the delta feed is the wait mechanism the
    application itself offers. "Nothing happens" is a bounded receive
    that must time out; "it settles" is the next delta on the wire."""
    from sg_web import worker as worker_module

    client, _, _ = served
    client.post("/settings/worker", json={"value": "off"})
    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job_id = client.post("/jobs/verify").json()["id"]
        # The submit announces its committed row (sg_web/submitting.py);
        # after that, a worker that is off says nothing at all.
        born = feed.receive_json(timeout=10)
        assert (born["job"], born["state"]) == (job_id, "queued")

        import queue

        with pytest.raises((TimeoutError, queue.Empty)):
            # long enough that an ignored off-switch would have spoken.
            # Both exceptions: the installed litestar's test session times
            # out with queue.Empty; upstream HEAD moved to anyio streams
            # and TimeoutError (litestar-org/litestar@64cd7da
            # litestar/testing/websocket_test_session.py:388).
            feed.receive_json(timeout=worker_module.IDLE_WAIT * 2.5)
        assert client.get(f"/jobs/{job_id}").json()["state"] == "queued", "the worker ignored its off switch"

        assert client.post(f"/jobs/{job_id}/cancel").json()["cancel_requested"] == 1
        client.post("/settings/worker", json={"value": "on"})
        delta = feed.receive_json(timeout=10)
        while delta["state"] not in ("done", "failed", "cancelled"):
            delta = feed.receive_json(timeout=10)
        assert (delta["job"], delta["state"]) == (job_id, "cancelled")
    assert client.get(f"/jobs/{job_id}").json()["state"] == "cancelled"


def test_clustering_people_is_a_job_the_application_offers(served):
    """The People page's data is produced BY the application: the cluster
    job groups the embedded faces, mints an addressable person for every
    unnamed group, and naming one is a POST that retires the old address
    with a 301. Nothing here reaches into the database."""
    client, _, _ = served
    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job = client.post("/jobs/cluster").json()
        assert (job["kind"], job["total"]) == ("cluster_faces", 1)
        state = job["state"]
        while state not in ("done", "failed", "cancelled"):
            state = feed.receive_json(timeout=10)["state"]
        assert state == "done"

    people = client.get("/people").json()
    assert len(people) == 1, "one group of two faces; the singleton stays a face"
    minted = people[0]
    assert minted["name"] == "(unnamed)"
    assert minted["slug"].startswith("person-"), "an unnamed person is still addressable"
    assert minted["pictures"] == 2

    named = client.post(f"/p/{minted['slug']}/name", json={"name": "Ana Torres"}).json()
    assert named == {"slug": "ana-torres", "name": "Ana Torres", "asserted": 2}
    assert client.get("/people").json()[0]["name"] == "Ana Torres"
    moved = client.get(f"/p/{minted['slug']}", follow_redirects=False)
    assert moved.status_code == 301
    assert moved.headers["location"] == "/p/ana-torres"
    assert client.post("/p/ana-torres/name", json={"name": "   "}).status_code == 400
    assert client.post("/p/nobody/name", json={"name": "X"}).status_code == 404
    assert client.post("/p/ana-torres/name").status_code == 400


def test_a_name_survives_the_apps_own_recluster(served):
    """Naming through the application is durable AGAINST the application:
    the naming writes the assertion record, and a re-cluster re-applies
    the name from that record -- never loses it with a dissolved cluster."""
    client, _, _ = served

    def drained_cluster_job():
        with client.websocket_connect("/ws/jobs") as feed:
            assert feed.receive_json(timeout=10)["type"] == "snapshot"
            job_id = client.post("/jobs/cluster").json()["id"]
            state = None
            while state not in ("done", "failed", "cancelled"):
                state = feed.receive_json(timeout=10)["state"]
        assert client.get(f"/jobs/{job_id}").json()["state"] == "done"

    drained_cluster_job()
    minted = client.get("/people").json()[0]
    assert client.post(f"/p/{minted['slug']}/name", json={"name": "Ana Torres"}).json()["slug"] == "ana-torres"

    drained_cluster_job()
    people = client.get("/people").json()
    assert people == [{"name": "Ana Torres", "slug": "ana-torres", "pictures": 2}], (
        f"the application's own re-cluster lost the name the application accepted: {people}"
    )
    assert len(client.get("/p/ana-torres").json()["pictures"]) == 2


def test_a_recluster_replaces_its_own_placeholders(served):
    """Running the job twice leaves one person per group, not one per run:
    a placeholder whose group dissolved is deleted with its address, while
    a NAMED person keeps their entity -- a name is a human's word."""
    client, _, _ = served
    for _ in range(2):
        with client.websocket_connect("/ws/jobs") as feed:
            assert feed.receive_json(timeout=10)["type"] == "snapshot"
            job_id = client.post("/jobs/cluster").json()["id"]
            state = None
            while state not in ("done", "failed", "cancelled"):
                state = feed.receive_json(timeout=10)["state"]
        assert client.get(f"/jobs/{job_id}").json()["state"] == "done"
    people = client.get("/people").json()
    assert len(people) == 1, f"a second run must replace its placeholders, not add more: {people}"


def _drained_cluster_job(client) -> None:
    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job_id = client.post("/jobs/cluster").json()["id"]
        state = None
        while state not in ("done", "failed", "cancelled"):
            state = feed.receive_json(timeout=10)["state"]
    assert client.get(f"/jobs/{job_id}").json()["state"] == "done"


def test_a_name_in_a_second_embedding_space_is_kept_not_lost(served):
    """The cluster job mints addressable people for EVERY embedding
    space's run, but only one run is primary. Naming a person the primary
    pages do not show must still write the record that keeps the name --
    the application never accepts a name it cannot keep -- and a name it
    genuinely cannot keep is refused, not swallowed."""
    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    rng = np.random.default_rng(9)
    one = rng.standard_normal(16).astype(np.float32)
    files = {name: fid for fid, name in conn.execute("SELECT id, name FROM file")}
    for name in ("ana_1.png", "ana_2.png"):
        derived.record_faces(
            conn,
            files[name],
            "other/embedder",
            "1",
            "aa",
            0.0,
            [
                {
                    "region": derived.region(conn, 0.6, 0.6, 0.2, 0.2),
                    "embedding": (one + 0.01 * rng.standard_normal(16).astype(np.float32)).tobytes(),
                }
            ],
        )
    conn.commit()
    conn.close()

    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job = client.post("/jobs/cluster").json()
        assert job["total"] == 2, "two embedding spaces, two items"
        state = job["state"]
        while state not in ("done", "failed", "cancelled"):
            state = feed.receive_json(timeout=10)["state"]
        assert state == "done"

    conn = connect.connect(db_path)
    minted = conn.execute(
        "SELECT e.slug FROM derived_face_cluster c JOIN derived_face_run r ON r.id = c.run_id"
        " JOIN person p ON p.id = c.person_id JOIN entity e ON e.id = p.id"
        " WHERE r.is_primary = 0 AND p.name IS NULL"
    ).fetchone()
    conn.close()
    assert minted is not None, "the second space's group has no addressable person"

    answered = client.post(f"/p/{minted[0]}/name", json={"name": "Beata"})
    assert answered.status_code < 300, answered.text
    assert answered.json()["asserted"] == 2, "an accepted name must be written down"

    _drained_cluster_job(client)
    conn = connect.connect(db_path)
    still = conn.execute(
        "SELECT count(*) FROM derived_face_cluster c JOIN person p ON p.id = c.person_id WHERE p.name = 'Beata'"
    ).fetchone()[0]
    conn.close()
    assert still == 1, "the app's own recluster lost a name it accepted"
    assert client.get("/p/beata").status_code == 200

    # And the refusal: the fixture's hand-attributed person owns no
    # cluster and no assertion, so their name has nothing to be kept by.
    assert client.post("/p/ana/name", json={"name": "Ana R"}).status_code == 400


def test_renaming_asserts_only_what_the_human_addressed(served):
    """Durability may READ every run; authorship is WRITTEN only for the
    cluster the human actually addressed, and a system write never
    overwrites a row a human signed. Otherwise a rename launders model
    inference into the authored ground truth the run rankings judge
    against -- a feedback loop wearing a person's name."""
    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    rng = np.random.default_rng(11)
    one = rng.standard_normal(16).astype(np.float32)
    files = {name: fid for fid, name in conn.execute("SELECT id, name FROM file")}
    # In THIS space ben clusters with the anas, same box coordinates --
    # the run disagreement the multi-run design exists to hold.
    for name in ("ana_1.png", "ana_2.png", "ben_1.png"):
        derived.record_faces(
            conn,
            files[name],
            "other/embedder",
            "1",
            "aa",
            0.0,
            [
                {
                    "region": derived.region(conn, 0.1, 0.1, 0.2, 0.2),
                    "embedding": (one + 0.01 * rng.standard_normal(16).astype(np.float32)).tobytes(),
                }
            ],
        )
    conn.commit()
    conn.close()

    _drained_cluster_job(client)
    named = client.get("/people").json()[0]
    assert client.post(f"/p/{named['slug']}/name", json={"name": "Ana Torres"}).status_code < 300
    _drained_cluster_job(client)  # the seed spreads her onto the other run's 3-file cluster

    conn = connect.connect(db_path)
    user_id = authored.add_user(conn, "will", "hash", "ADMIN", 2.0)
    their_box = derived.region(conn, 0.05, 0.05, 0.3, 0.3)
    resolved = naming.resolve(conn, "person", "ana-torres")
    assert resolved is not None
    person_id = resolved[0]
    authored.assert_person(conn, person_id, files["ana_1.png"], user_id, 2.0, region_id=their_box)
    conn.commit()
    conn.close()

    assert client.post("/p/ana-torres/name", json={"name": "Ana T"}).status_code < 300

    conn = connect.connect(db_path)
    asserted = {
        row[0]
        for row in conn.execute(
            "SELECT f.name FROM person_assertion pa JOIN file f ON f.id = pa.file_id WHERE pa.person_id = ?",
            (person_id,),
        )
    }
    kept = conn.execute(
        "SELECT user_id, region_id FROM person_assertion WHERE person_id = ? AND file_id = ?",
        (person_id, files["ana_1.png"]),
    ).fetchone()
    conn.close()
    assert asserted == {"ana_1.png", "ana_2.png"}, f"a rename asserted files only a model inferred: {asserted}"
    assert kept == (user_id, their_box), "a system write overwrote a human-authored assertion"


def test_feedback_on_a_placeholder_outlives_the_recluster(served):
    """The pruning spares a person a feedback verdict points at: the
    judgement is authored, and it must keep its subject."""
    client, _, _ = served
    _drained_cluster_job(client)
    minted_slug = client.get("/people").json()[0]["slug"]
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    resolved = naming.resolve(conn, "person", minted_slug)
    assert resolved is not None
    person_id = resolved[0]
    authored.feedback(conn, "person", "wrong", 1.0, person_id=person_id)
    conn.commit()
    conn.close()

    _drained_cluster_job(client)
    conn = connect.connect(db_path)
    survived = conn.execute("SELECT count(*) FROM person WHERE id = ?", (person_id,)).fetchone()[0]
    judged = conn.execute("SELECT person_id FROM feedback").fetchone()[0]
    conn.close()
    assert survived == 1, "the pruning deleted a person a human's verdict points at"
    assert judged == person_id, "the verdict lost its subject"


def test_choose_primary_is_an_action_the_application_offers(served):
    client, _, _ = served
    chosen = client.post("/clusterings/choose").json()
    assert chosen["primary_run"] is not None
    assert client.get("/clusterings").json()[0]["id"] == chosen["primary_run"]


def test_the_front_door_is_the_gallery(served):
    """A browser at `/` lands in the gallery; a machine gets the compact
    library summary with a newest strip. The building entrance stopped
    pointing at JSON."""
    client, _, _ = served
    landed = client.get("/", headers={"accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert (landed.status_code, landed.headers["location"]) == (302, "/g")

    front = client.get("/").json()
    assert front["files"] == 3
    for fact in ("folders", "people", "collections", "artifacts"):
        assert isinstance(front[fact], int), f"the summary counts {fact}"
    assert {row["name"] for row in front["newest"]} == {"ana_1.png", "ana_2.png", "ben_1.png"}
    assert all(row["slug"] for row in front["newest"])


def test_every_new_address_survives_a_rename(served):
    """The addressing contract on every kind the entity layer added: a
    retired slug 301s within its own prefix, and a retired slug on the
    WRONG shelf lands home in ONE hop -- the canonical address is
    computed once from entity, live slug and kind, never a chain."""
    from db import ingest as ingest_module

    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    found = naming.resolve(conn, "file", "ana-1")
    assert found is not None
    naming.rename(conn, found[0], "ana prime", 5.0)
    found = naming.resolve(conn, "folder", "lib")
    assert found is not None
    naming.rename(conn, found[0], "library prime", 5.0)
    lora_id = ingest_module.artifact(conn, "lora", "detailTweaker", 5.0)
    naming.rename(conn, lora_id, "detail tweaker xl", 5.0)
    conn.commit()
    conn.close()

    moved = client.get("/i/ana-1", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/i/ana-prime")
    moved = client.get("/f/lib", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/f/library-prime")
    moved = client.get("/l/lora-detailtweaker", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/l/detail-tweaker-xl")

    first = client.get("/m/lora-detailtweaker", follow_redirects=False)
    assert (first.status_code, first.headers["location"]) == (301, "/l/detail-tweaker-xl"), (
        "wrong shelf + retired slug heals in ONE 301, never a chain"
    )
    second = client.get("/m/detail-tweaker-xl", follow_redirects=False)
    assert (second.status_code, second.headers["location"]) == (301, "/l/detail-tweaker-xl")
    assert client.get("/l/detail-tweaker-xl").json()["name"] == "detailTweaker"

    assert client.post("/albums", json={"name": "Trip"}).json()["slug"] == "trip"
    conn = connect.connect(db_path)
    found = naming.resolve(conn, "collection", "trip")
    assert found is not None
    naming.rename(conn, found[0], "Trip 2026", 6.0)
    conn.commit()
    conn.close()
    moved = client.get("/t/trip", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/t/trip-2026")


def test_albums_are_made_and_served_through_the_application(served):
    """An album is authored state with a full application surface: made,
    filled, emptied and read over routes, addressed by slug."""
    client, _, root = served
    made = client.post("/albums", json={"name": "Keepers"}).json()
    # The answer is the authoritative CollectionView, born at revision 1.
    assert (made["name"], made["slug"], made["kind"]) == ("Keepers", "keepers", "album")
    assert (made["definition_rev"], made["archived"]) == (1, False)
    assert client.post("/t/keepers/add", json={"file": "ana-1"}).json()["pictures"] == 1
    assert client.post("/t/keepers/add", json={"file": "ana-1"}).json()["pictures"] == 1, "adding twice is once"
    assert client.post("/t/keepers/add", json={"file": "ben-1"}).json()["pictures"] == 2

    listed = client.get("/albums").json()
    assert listed == [{"name": "Keepers", "slug": "keepers", "kind": "album", "pictures": 2}]
    page = client.get("/t/keepers").json()
    assert sorted(f["name"] for f in page["files"]) == ["ana_1.png", "ben_1.png"]

    assert client.post("/t/keepers/remove", json={"file": "ben-1"}).json()["pictures"] == 1
    assert [f["name"] for f in client.get("/t/keepers").json()["files"]] == ["ana_1.png"]

    assert client.post("/albums", json={"name": "   "}).status_code == 400
    assert client.post("/albums", json={"name": "Q", "kind": "smart"}).status_code == 400
    turned_away = client.post("/t/keepers/add", json={"file": "nope"})
    assert turned_away.status_code == 404
    assert "/i/nope" in turned_away.json()["detail"], (
        "the 404 names the file at its own address -- /t/keepers/add/nope is an address nothing lives at"
    )
    assert client.get("/t/lost").status_code == 404

    # A name collision suffixes, never steals.
    assert client.post("/albums", json={"name": "Keepers"}).json()["slug"] == "keepers-2"

    # A missing file drops out of every count the same way.
    client.post("/t/keepers/add", json={"file": "ben-1"})
    (root / "ben_1.png").unlink()
    assert client.post("/roots/1/scan").json()["missing"] == 1
    assert client.get("/albums").json()[0]["pictures"] == 1
    assert client.post("/t/keepers/add", json={"file": "ana-1"}).json()["pictures"] == 1


def test_a_smart_collection_refuses_filing_over_http(served):
    """The rule-defined kind refuses stored members at the route too, as
    a 400 with the reason -- never a 500 from the trigger beneath."""
    client, _, _ = served
    conn = connect.connect(client.app.state.db_path)
    seeds = collections.collection(conn, "Big seeds", 3.0, kind="smart")
    collection_rules.keep_prose(conn, seeds, sql="SELECT 1", now=3.0)
    conn.commit()
    conn.close()
    refused = client.post("/t/big-seeds/add", json={"file": "ana-1"})
    assert refused.status_code == 400
    assert "rule" in refused.json()["detail"]
    assert client.post("/t/big-seeds/remove", json={"file": "ana-1"}).status_code == 400, (
        "removing from a rule-defined collection must refuse like adding does"
    )
    served = client.get("/t/big-seeds").json()
    assert served["rule"] == {"sql": "SELECT 1", "nl": None}, "the rule is served; evaluation is declared deferred"
    assert served["files"] == []


def test_search_answers_by_meaning_from_the_joint_space(served, monkeypatch):
    """The CLIP trick over HTTP: stored image vectors and a typed phrase
    meet in one space, and /search returns the nearest pictures with
    scores -- no tags or captions anywhere in the loop. The encoder is
    faked; what is under test is the whole path from setting to space to
    resident index to ranked answer."""
    from db import similarity
    from vision import semantic

    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    conn.execute("UPDATE file SET content_sha256 = 'aa'")
    files = dict(conn.execute("SELECT name, id FROM file"))
    spec = similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)
    ana = np.array([1, 0, 0, 0], dtype=np.float32)
    ben = np.array([0, 1, 0, 0], dtype=np.float32)
    derived.record_embedding(conn, files["ana_1.png"], spec, ana, "aa", 0.0)
    derived.record_embedding(conn, files["ana_2.png"], spec, ana * 0.9 + ben * 0.1, "aa", 0.0)
    derived.record_embedding(conn, files["ben_1.png"], spec, ben, "aa", 0.0)
    conn.commit()
    connect.close(conn)

    class FakeText:
        def encode_query(self, phrase):
            assert phrase == "a woman smiling"
            return ana

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: FakeText())
    answer = client.get("/search", params={"q": "a woman smiling", "k": 3})
    assert answer.status_code == 200
    body = answer.json()
    assert body["participants"] == ["semantic.openclip.ViT-B-32.laion2b_s34b_b79k"]
    assert body["contributors"] == body["participants"]
    assert body["missing"] == {}
    told = body["results"]
    assert [row["name"] for row in told][:2] == ["ana_1.png", "ana_2.png"]
    assert told[0]["score"] > told[1]["score"] > told[-1]["score"]
    assert all(set(row) == {"slug", "name", "score", "sources"} for row in told)
    assert all("semantic.openclip.ViT-B-32.laion2b_s34b_b79k" in row["sources"] for row in told)
    ranks = [row["sources"]["semantic.openclip.ViT-B-32.laion2b_s34b_b79k"]["rank"] for row in told]
    assert ranks == sorted(ranks)

    # a bad model setting is a refused request, never a 500
    conn = connect.connect(db_path)
    from db import settings as settings_module

    settings_module.put(conn, "semantic_model", "broken")
    conn.commit()
    connect.close(conn)
    assert client.get("/search", params={"q": "x"}).status_code == 400


def test_search_fuses_two_spaces_by_rank_never_by_raw_score(served, monkeypatch):
    """Two participating spaces with WILDLY different score scales: the
    fused order can only come from ranks. The file both models agree on
    outranks each model's private favourite, and per-space provenance
    survives into the response."""
    from db import similarity
    from vision import semantic

    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    conn.execute("UPDATE file SET content_sha256 = 'aa'")
    from db import settings as settings_module

    settings_module.put(conn, "semantic_model", "ViT-B-32/one,ViT-B-32/two")
    files = dict(conn.execute("SELECT name, id FROM file"))
    one = similarity.semantic_space("ViT-B-32", "one", 4)
    two = similarity.semantic_space("ViT-B-32", "two", 4)
    agreed = np.array([1, 0, 0, 0], dtype=np.float32)
    private = np.array([0, 1, 0, 0], dtype=np.float32)
    away = np.array([0, 0, 1, 0], dtype=np.float32)
    # space one loves ana_1 then ana_2; space two loves ana_1 then ben_1.
    derived.record_embedding(conn, files["ana_1.png"], one, agreed, "aa", 0.0)
    derived.record_embedding(conn, files["ana_2.png"], one, agreed * 0.8 + private * 0.2, "aa", 0.0)
    derived.record_embedding(conn, files["ben_1.png"], one, away, "aa", 0.0)
    derived.record_embedding(conn, files["ana_1.png"], two, agreed, "aa", 0.0)
    derived.record_embedding(conn, files["ana_2.png"], two, away, "aa", 0.0)
    derived.record_embedding(conn, files["ben_1.png"], two, agreed * 0.8 + private * 0.2, "aa", 0.0)
    conn.commit()
    connect.close(conn)

    class PerSpace:
        """Each space's query lands at a different cosine LEVEL: space one
        answers near 1.0, space two near 0.35 -- raw magnitudes that mean
        nothing across spaces, which is why only ranks may fuse."""

        def __init__(self, checkpoint):
            self.query = {
                "one": agreed,
                "two": (0.35 * agreed + 0.9 * np.array([0, 0, 0, 1], dtype=np.float32)),
            }[checkpoint]

        def encode_query(self, phrase):
            return self.query

    monkeypatch.setattr(semantic, "encoder", lambda _provider, _dir, _model, checkpoint, **kw: PerSpace(checkpoint))
    body = client.get("/search", params={"q": "the agreed picture", "k": 3}).json()
    assert body["contributors"] == [
        "semantic.openclip.ViT-B-32.one",
        "semantic.openclip.ViT-B-32.two",
    ], "both configured spaces must be said to have answered"
    told = body["results"]
    assert told[0]["name"] == "ana_1.png", "the file both spaces agree on must fuse to the top"
    assert set(told[0]["sources"]) == {
        "semantic.openclip.ViT-B-32.one",
        "semantic.openclip.ViT-B-32.two",
    }
    raw_one = told[0]["sources"]["semantic.openclip.ViT-B-32.one"]["score"]
    raw_two = told[0]["sources"]["semantic.openclip.ViT-B-32.two"]["score"]
    assert raw_one > 0.9, "space one answers near the top of its own scale"
    assert raw_two < 0.5, "space two answers far down its own -- the scales are incomparable"
    assert told[0]["sources"]["semantic.openclip.ViT-B-32.one"]["rank"] == 1
    assert told[0]["sources"]["semantic.openclip.ViT-B-32.two"]["rank"] == 1


def test_a_replaced_file_stops_answering_by_its_old_picture(served, monkeypatch):
    """The scanner keeps a file's identity through an in-place byte
    replacement; its old embedding must NOT keep retrieving it until the
    re-embed happens -- the stale row is excluded the moment the bytes
    changed, not whenever the embed job gets around to it."""
    from db import similarity
    from vision import semantic

    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    conn.execute("UPDATE file SET content_sha256 = 'aa'")
    files = dict(conn.execute("SELECT name, id FROM file"))
    spec = similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)
    ana = np.array([1, 0, 0, 0], dtype=np.float32)
    derived.record_embedding(conn, files["ana_1.png"], spec, ana, "aa", 0.0)
    # the bytes change behind the embedding's back
    conn.execute("UPDATE file SET content_sha256 = 'REPLACED' WHERE id = ?", (files["ana_1.png"],))
    conn.commit()
    connect.close(conn)

    class FakeText:
        def encode_query(self, phrase):
            return ana

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: FakeText())
    body = client.get("/search", params={"q": "x", "k": 3}).json()
    assert body["results"] == [], "a known-stale representation answered for the replaced bytes"
    assert "semantic.openclip.ViT-B-32.laion2b_s34b_b79k" in body["missing"], (
        "a space that could not answer must be named, not silently absent"
    )


@pytest.mark.slow
def test_search_never_downloads_a_model(served):
    """Embeddings exist but the model cache is empty: the request is
    refused with the fix named, and no acquisition begins -- weights
    belong to /jobs/embed, not to a GET."""
    from db import similarity

    client, _, _ = served
    db_path = client.app.state.db_path
    conn = connect.connect(db_path)
    conn.execute("UPDATE file SET content_sha256 = 'aa'")
    files = dict(conn.execute("SELECT name, id FROM file"))
    spec = similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)
    derived.record_embedding(conn, files["ana_1.png"], spec, np.array([1, 0, 0, 0], dtype=np.float32), "aa", 0.0)
    conn.commit()
    connect.close(conn)

    answer = client.get("/search", params={"q": "banana"})
    assert answer.status_code == 400
    assert "provisioned" in answer.json()["detail"]
