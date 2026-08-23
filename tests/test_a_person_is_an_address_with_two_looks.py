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
from PIL import Image

from db import authored, collections, connect, derived, naming
from tests.staging import Stage, staged

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}
AS_OVERLAY = {"hx-request": "true"}


def _library(root: pathlib.Path) -> None:
    for name in ("ana_1.png", "ana_2.png", "ben_1.png"):
        Image.new("RGB", (16, 16), (200, 90, 40)).save(root / name)


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
    # The cluster carries the person, as the clustering job leaves it --
    # naming writes its durable assertions FROM this attachment.
    conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE id = ?", (person_id, made[0]))
    for name in ("ana_1.png", "ana_2.png"):
        derived.attribute(conn, files[name], person_id, run_id, "test/embedder", "1")
    conn.commit()
    connect.close(conn)
    stage.held["person"] = person_id


def _drained_cluster_job(client) -> None:
    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job_id = client.post("/jobs/cluster").json()["id"]
        state = None
        while state not in ("done", "failed", "cancelled"):
            state = feed.receive_json(timeout=10)["state"]
    assert client.get(f"/jobs/{job_id}").json()["state"] == "done"


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    """The clustered library behind the running application, with its
    worker: the cluster and verify jobs below run through it."""
    with staged(tmp_path_factory, "person", _library, _clustered, worker=True) as stage:
        yield stage


@pytest.fixture
def faces(_stage):
    _stage.restore()
    return _stage.client


def _second_space(client, names, *, seed: int, box: float) -> None:
    """Faces for `names` in a second embedding space, one tight group."""
    conn = connect.connect(client.app.state.db_path)
    rng = np.random.default_rng(seed)
    one = rng.standard_normal(16).astype(np.float32)
    files = {name: fid for fid, name in conn.execute("SELECT id, name FROM file")}
    for name in names:
        derived.record_faces(
            conn,
            files[name],
            "other/embedder",
            "1",
            "aa",
            0.0,
            [
                {
                    "region": derived.region(conn, box, box, 0.2, 0.2),
                    "embedding": (one + 0.01 * rng.standard_normal(16).astype(np.float32)).tobytes(),
                }
            ],
        )
    conn.commit()
    conn.close()


@pytest.fixture
def minted(faces) -> dict:
    """The cluster job run once through the application: the unnamed,
    addressable person it minted for ana's pair."""
    _drained_cluster_job(faces)
    people = faces.get("/people").json()
    assert len(people) == 1, people
    return people[0]


@pytest.fixture
def named_placeholder(minted, faces) -> str:
    """The minted person named Ana Torres; the value is the retired slug."""
    told = faces.post(f"/p/{minted['slug']}/name", json={"name": "Ana Torres"})
    assert told.status_code < 300, told.text
    return minted["slug"]


@pytest.fixture
def second_space_placeholder(faces) -> str:
    """Ana's pair embedded in a second space and the job run over both:
    the slug of the unnamed person the NON-primary run's group carries."""
    _second_space(faces, ("ana_1.png", "ana_2.png"), seed=9, box=0.6)
    _drained_cluster_job(faces)
    conn = connect.connect(faces.app.state.db_path)
    minted = conn.execute(
        "SELECT e.slug FROM derived_face_cluster c JOIN derived_face_run r ON r.id = c.run_id"
        " JOIN person p ON p.id = c.person_id JOIN entity e ON e.id = p.id"
        " WHERE r.is_primary = 0 AND p.name IS NULL"
    ).fetchone()
    conn.close()
    assert minted is not None, "the second space's group has no addressable person"
    return minted[0]


@pytest.fixture
def renamed_ana(faces) -> None:
    """Ana renamed through the Module: the slug `ana` is retired."""
    conn = connect.connect(faces.app.state.db_path)
    found = naming.resolve(conn, "person", "ana")
    assert found is not None
    naming.rename(conn, found[0], "Ana Torres", 5.0)
    conn.commit()
    connect.close(conn)


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
    assert index.json() == [{"name": "Ana", "slug": "ana", "pictures": 2, "first_seen": None, "last_seen": None}]


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


def test_a_commit_mid_assembly_cannot_mix_person_generations(faces, monkeypatch):
    """The MediaView invariant, held here too: a naming commit landing
    between person_files and the remaining reads must not hand back
    pictures from one generation under the name of another. The writer
    renames the person exactly inside that window -- the response is
    wholly before, and the next request wholly after."""
    from db import pages

    client = faces
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


# --- a person is a ResultSet scope (WI-38) ----------------------------------


def test_person_membership_is_the_primary_runs_attribution(faces):
    from db import resultset

    conn = connect.connect(faces.app.state.db_path)
    try:
        told = resultset.describe(conn, "", resultset.parse(person="ana"), 0.0)
    finally:
        connect.close(conn)
    assert told["total"] == 2, "membership is the attribution, not the library"


def test_an_unknown_person_is_a_lookup_error(faces):
    from db import resultset

    conn = connect.connect(faces.app.state.db_path)
    try:
        with pytest.raises(LookupError):
            resultset.describe(conn, "", resultset.parse(person="nobody"), 0.0)
    finally:
        connect.close(conn)


@pytest.mark.parametrize(
    ("facet", "total"),
    [({"folder": "lib"}, 2), ({"kind": "video"}, 0)],
    ids=["folder", "kind"],
)
def test_the_person_facet_composes_as_an_intersection(faces, facet, total):
    """Person is a COMPOSABLE membership facet, not a third exclusive
    scope: eligibility is an intersection of predicates."""
    from db import resultset

    conn = connect.connect(faces.app.state.db_path)
    try:
        told = resultset.describe(conn, "", resultset.parse(person="ana", **facet), 0.0)
    finally:
        connect.close(conn)
    assert told["total"] == total


def test_the_person_facet_intersects_an_album(faces):
    from db import resultset

    conn = connect.connect(faces.app.state.db_path)
    try:
        mixed = collections.collection(conn, "Mixed", 0.0)
        for name in ("ana_1.png", "ben_1.png"):
            file_id = conn.execute("SELECT id FROM file WHERE name = ?", (name,)).fetchone()[0]
            collections.set_membership(conn, mixed, file_id, True, 0.0)
        conn.commit()

        crossed = resultset.describe(conn, "", resultset.parse(person="ana", album="mixed"), 0.0)
    finally:
        connect.close(conn)
    assert crossed["total"] == 1, "person AND album keeps only the attributed member of the album"


def test_the_grid_serves_the_person_scope(faces):
    grid = faces.get("/g/grid", params={"person": "ana"}).text

    assert grid.count('data-slug="ana-') == 2
    assert "ben-1" not in grid


def test_the_profile_grid_is_the_resultset_page_in_context(faces):
    from db import resultset

    page = faces.get("/p/ana", headers=AS_BROWSER).text
    body = faces.get("/p/ana").json()

    assert page.count("?person=ana") >= 2, "profile media links must carry the person context"
    assert body["gallery"]["total"] == 2
    assert body["gallery"]["qs"] == "person=ana"
    conn = connect.connect(faces.app.state.db_path)
    try:
        expected = resultset.page(conn, "", resultset.parse(person="ana"), 1, 0.0)["items"]
    finally:
        connect.close(conn)
    assert [row["slug"] for row in body["gallery"]["items"]] == [row["slug"] for row in expected]


def test_an_item_under_the_person_context_walks_the_person(faces):
    items = faces.get("/p/ana").json()["gallery"]["items"]
    first, second = items[0]["slug"], items[1]["slug"]

    walked = faces.get(f"/i/{first}", params={"person": "ana"}).json()

    assert walked["context"]["total"] == 2
    assert walked["next"] == second
    assert walked["previous"] is None
    assert walked["context"]["return_url"] == "/g?person=ana"


def test_an_item_outside_the_person_answer_says_so(faces):
    outside = faces.get("/i/ben-1", params={"person": "ana"}).json()

    assert outside["context"]["in_answer"] is False


def test_a_person_phrase_constrains_each_space_before_fusion(faces, monkeypatch):
    """person + q composes through the SAME constrained-RRF door every
    scope uses: the person's membership reaches retrieval as the
    allowed set, applied per space before the fusion."""
    from db import resultset, retrieval

    conn = connect.connect(faces.app.state.db_path)
    anas = {
        row[0]
        for row in conn.execute("SELECT fp.file_id FROM derived_file_person fp JOIN person p ON p.id = fp.person_id")
    }
    seen: dict = {}

    def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        seen.update({"allowed": allowed, "k": k})
        return {"results": [], "participants": [], "contributors": [], "missing": {}}

    monkeypatch.setattr(retrieval, "query", fused)
    try:
        resultset.describe(conn, "", resultset.parse(person="ana", text="beach"), 0.0)
    finally:
        connect.close(conn)
    assert seen["allowed"] == anas, "the person facet must constrain retrieval before RRF"
    assert seen["k"] == len(anas)


def test_two_spellings_of_a_renamed_person_are_one_question(renamed_ana, faces):
    """Slugs are presentation; the projection keys on the bound entity."""
    from db import resultset

    conn = connect.connect(faces.app.state.db_path)
    try:
        old_spelling = resultset.describe(conn, "", resultset.parse(person="ana"), 0.0)
        new_spelling = resultset.describe(conn, "", resultset.parse(person="ana-torres"), 0.0)
    finally:
        connect.close(conn)

    assert old_spelling["total"] == new_spelling["total"] == 2
    assert old_spelling["fingerprint"] == new_spelling["fingerprint"], (
        "two spellings of one person forked the projection cache"
    )
    assert old_spelling["qs"] == "person=ana-torres", "the answer re-spells the context live"


def test_the_item_context_heals_to_the_live_slug(renamed_ana, faces):
    """A stale bookmark heals as it is navigated: every answer re-spells
    the context with the LIVE slug."""
    walked = faces.get("/i/ana-1", params={"person": "ana"}).json()

    assert walked["context"]["qs"] == "person=ana-torres"
    assert walked["context"]["return_url"] == "/g?person=ana-torres"


def test_switching_the_primary_run_is_a_different_question(faces):
    """Person membership means THE PRIMARY run's attribution; promoting
    another run changes both the answer and its identity -- never a
    silently reused projection."""
    from db import resultset

    conn = connect.connect(faces.app.state.db_path)
    try:
        before = resultset.describe(conn, "", resultset.parse(person="ana"), 0.0)
        assert before["total"] == 2

        other = derived.run_for(conn, "other/embedder", "2", "chinese-whispers", 0.4, 1.0)
        derived.make_primary(conn, other)
        conn.commit()

        after = resultset.describe(conn, "", resultset.parse(person="ana"), 1.0)
    finally:
        connect.close(conn)
    assert after["total"] == 0, "the new primary attributes ana nothing; the membership must say so"
    assert after["fingerprint"] != before["fingerprint"], "the bound run is part of the question"


def test_the_browser_path_never_enumerates_the_whole_collection(faces, monkeypatch):
    """The bounded profile must be bounded on the SERVER, not only in
    the DOM: the unbounded legacy list is the JSON Adapter's cost, and
    the HTML and drawer paths never pay it -- the JSON call is the
    control that the counter sees the enumeration at all."""
    from db import pages

    walked: list[int] = []
    real = pages.person_files

    def counted(conn, person_id, run_id=None):
        walked.append(person_id)
        return real(conn, person_id, run_id)

    monkeypatch.setattr(pages, "person_files", counted)
    assert faces.get("/p/ana", headers=AS_BROWSER).status_code == 200
    assert faces.get("/p/ana", headers=AS_OVERLAY).status_code == 200
    assert walked == [], "a rendered profile enumerated the person's whole photographic existence"
    body = faces.get("/p/ana").json()
    assert len(walked) == 1, "the machine Adapter still carries the legacy list"
    assert sorted(p["name"] for p in body["pictures"]) == ["ana_1.png", "ana_2.png"]


# --- the application over the clustered library ------------------------------------------


def test_a_person_is_addressed_by_slug_and_shows_the_cross_axis_view(faces):
    client = faces
    answer = client.get("/p/ana")
    assert answer.status_code == 200
    page = answer.json()
    assert page["name"] == "Ana"
    assert sorted(p["name"] for p in page["pictures"]) == ["ana_1.png", "ana_2.png"]
    assert page["across_folders"][0]["pictures"] == 2


def test_clusterings_are_public_and_the_primary_is_marked(faces):
    client = faces
    runs = client.get("/clusterings").json()
    assert len(runs) == 1
    assert runs[0]["is_primary"] == 1
    assert runs[0]["method"] == "chinese-whispers"


def test_job_progress_is_pushed_over_the_socket_not_polled(faces):
    """Submit a sweep and watch it happen: the socket sends the persisted
    snapshot first, then a delta per observable change, ending in the
    terminal state -- and the row agrees with everything it said."""
    client = faces
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


def test_an_unknown_job_is_404(faces):
    assert faces.get("/jobs/999").status_code == 404


# --- the cluster job and naming, through the application only ---------------


def test_the_cluster_job_mints_a_person_for_an_unnamed_group(faces):
    """The one end-to-end run: the People page's data is produced BY the
    application. The cluster job groups the embedded faces and mints an
    addressable person for every unnamed group; the singleton stays a
    face. Nothing here reaches into the database."""
    client = faces
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


def test_a_second_run_replaces_its_placeholder_instead_of_adding_one(minted, faces):
    """One person per group, not one per run: a placeholder whose group
    dissolved is deleted with its address."""
    _drained_cluster_job(faces)

    people = faces.get("/people").json()
    assert len(people) == 1, f"a second run must replace its placeholders, not add more: {people}"


def test_naming_a_placeholder_answers_the_new_address(minted, faces):
    named = faces.post(f"/p/{minted['slug']}/name", json={"name": "Ana Torres"})

    assert named.status_code < 300, named.text
    assert named.json() == {"slug": "ana-torres", "name": "Ana Torres", "asserted": 2}
    assert faces.get("/people").json()[0]["name"] == "Ana Torres"


def test_naming_retires_the_old_address_with_a_301(named_placeholder, faces):
    moved = faces.get(f"/p/{named_placeholder}", follow_redirects=False)

    assert moved.status_code == 301
    assert moved.headers["location"] == "/p/ana-torres"


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
def test_renaming_refuses_a_blank_name(faces, blank):
    assert faces.post("/p/ana/name", json={"name": blank}).status_code == 400


def test_renaming_an_unknown_person_is_404(faces):
    assert faces.post("/p/nobody/name", json={"name": "X"}).status_code == 404


def test_renaming_requires_a_body(faces):
    assert faces.post("/p/ana/name").status_code == 400


def test_a_human_name_survives_the_apps_own_recluster(named_placeholder, faces):
    """TEMPORAL SCENARIO: naming writes the assertion record, and a
    re-cluster re-applies the name from that record. A name is a human's
    word; the application never loses one it accepted."""
    _drained_cluster_job(faces)

    people = faces.get("/people").json()
    assert people == [
        {"name": "Ana Torres", "slug": "ana-torres", "pictures": 2, "first_seen": None, "last_seen": None}
    ], f"the application's own re-cluster lost the name the application accepted: {people}"
    assert len(faces.get("/p/ana-torres").json()["pictures"]) == 2


# --- a second embedding space ----------------------------------------------


def test_the_cluster_job_runs_every_embedding_space(faces):
    """The job mints addressable people for EVERY space's run, though
    only one run is primary."""
    _second_space(faces, ("ana_1.png", "ana_2.png"), seed=9, box=0.6)

    with faces.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job = faces.post("/jobs/cluster").json()
        state = job["state"]
        while state not in ("done", "failed", "cancelled"):
            state = feed.receive_json(timeout=10)["state"]

    assert job["total"] == 2, "two embedding spaces, two items"
    assert state == "done"
    conn = connect.connect(faces.app.state.db_path)
    try:
        minted = conn.execute(
            "SELECT count(*) FROM derived_face_cluster c JOIN derived_face_run r ON r.id = c.run_id"
            " JOIN person p ON p.id = c.person_id WHERE r.is_primary = 0 AND p.name IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    assert minted == 1, "the second space's group has no addressable person"


def test_naming_a_person_the_primary_pages_do_not_show_is_written_down(second_space_placeholder, faces):
    """The application never accepts a name it cannot keep: the record
    that keeps it is written even for a non-primary run's person."""
    answered = faces.post(f"/p/{second_space_placeholder}/name", json={"name": "Beata"})

    assert answered.status_code < 300, answered.text
    assert answered.json()["asserted"] == 2, "an accepted name must be written down"


def test_a_name_in_a_second_space_survives_a_recluster(second_space_placeholder, faces):
    """TEMPORAL SCENARIO: a name on a non-primary run's person is
    re-applied by the application's own re-cluster."""
    assert faces.post(f"/p/{second_space_placeholder}/name", json={"name": "Beata"}).status_code < 300

    _drained_cluster_job(faces)

    conn = connect.connect(faces.app.state.db_path)
    try:
        still = conn.execute(
            "SELECT count(*) FROM derived_face_cluster c JOIN person p ON p.id = c.person_id WHERE p.name = 'Beata'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert still == 1, "the app's own recluster lost a name it accepted"
    assert faces.get("/p/beata").status_code == 200


def test_a_name_nothing_can_keep_is_refused(faces):
    """A person who owns no cluster and no assertion has nothing a name
    could be kept by: refused, not swallowed."""
    conn = connect.connect(faces.app.state.db_path)
    loner = naming.claim(conn, "person", "Loner")
    conn.execute("INSERT INTO person(id,name,created_at) VALUES(?, 'Loner', 0)", (loner,))
    conn.commit()
    conn.close()

    assert faces.post("/p/loner/name", json={"name": "Loner R"}).status_code == 400


def test_renaming_asserts_only_what_the_human_addressed(faces):
    """Durability may READ every run; authorship is WRITTEN only for the
    cluster the human actually addressed, and a system write never
    overwrites a row a human signed. Otherwise a rename launders model
    inference into the authored ground truth the run rankings judge
    against -- a feedback loop wearing a person's name."""
    client = faces
    db_path = client.app.state.db_path
    # In THIS space ben clusters with the anas, same box coordinates --
    # the run disagreement the multi-run design exists to hold.
    _second_space(client, ("ana_1.png", "ana_2.png", "ben_1.png"), seed=11, box=0.1)
    conn = connect.connect(db_path)
    files = {name: fid for fid, name in conn.execute("SELECT id, name FROM file")}
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


def test_feedback_on_a_placeholder_outlives_the_recluster(faces):
    """The pruning spares a person a feedback verdict points at: the
    judgement is authored, and it must keep its subject."""
    client = faces
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


def test_choose_primary_is_an_action_the_application_offers(faces):
    client = faces
    chosen = client.post("/clusterings/choose").json()
    assert chosen["primary_run"] is not None
    assert client.get("/clusterings").json()[0]["id"] == chosen["primary_run"]


@pytest.fixture
def renamed_everything(faces) -> None:
    """A file, a folder, an artifact and a collection, each renamed
    through the Module so its first slug is retired."""
    from db import ingest as ingest_module

    assert faces.post("/albums", json={"name": "Trip"}).json()["slug"] == "trip"
    conn = connect.connect(faces.app.state.db_path)
    for kind, slug, new_name in (("file", "ana-1", "ana prime"), ("folder", "lib", "library prime")):
        found = naming.resolve(conn, kind, slug)
        assert found is not None
        naming.rename(conn, found[0], new_name, 5.0)
    lora_id = ingest_module.artifact(conn, "lora", "detailTweaker", 5.0)
    naming.rename(conn, lora_id, "detail tweaker xl", 5.0)
    found = naming.resolve(conn, "collection", "trip")
    assert found is not None
    naming.rename(conn, found[0], "Trip 2026", 6.0)
    conn.commit()
    conn.close()


@pytest.mark.parametrize(
    ("retired", "live"),
    [
        ("/i/ana-1", "/i/ana-prime"),
        ("/f/lib", "/f/library-prime"),
        ("/l/lora-detailtweaker", "/l/detail-tweaker-xl"),
        ("/t/trip", "/t/trip-2026"),
    ],
    ids=["file", "folder", "artifact", "collection"],
)
def test_a_retired_slug_301s_within_its_own_prefix(renamed_everything, faces, retired, live):
    """The addressing contract on every kind the entity layer added."""
    moved = faces.get(retired, follow_redirects=False)

    assert (moved.status_code, moved.headers["location"]) == (301, live)


@pytest.mark.parametrize("wrong_shelf", ["/m/lora-detailtweaker", "/m/detail-tweaker-xl"], ids=["retired", "live"])
def test_a_wrong_shelf_heals_in_one_hop(renamed_everything, faces, wrong_shelf):
    """The canonical address is computed once from entity, live slug and
    kind -- never a chain of 301s."""
    moved = faces.get(wrong_shelf, follow_redirects=False)

    assert (moved.status_code, moved.headers["location"]) == (301, "/l/detail-tweaker-xl")


def test_the_live_address_answers_the_renamed_entity(renamed_everything, faces):
    assert faces.get("/l/detail-tweaker-xl").json()["name"] == "detailTweaker"


_ANA = np.array([1, 0, 0, 0], dtype=np.float32)
_BEN = np.array([0, 1, 0, 0], dtype=np.float32)


def _embedded_in_the_default_space(client) -> None:
    """Image vectors for the three files in the default semantic space:
    ana_1 is ana, ana_2 is nearly ana, ben_1 is ben."""
    from db import similarity

    conn = connect.connect(client.app.state.db_path)
    conn.execute("UPDATE file SET content_sha256 = 'aa'")
    files = dict(conn.execute("SELECT name, id FROM file"))
    spec = similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)
    derived.record_embedding(conn, files["ana_1.png"], spec, _ANA, "aa", 0.0)
    derived.record_embedding(conn, files["ana_2.png"], spec, _ANA * 0.9 + _BEN * 0.1, "aa", 0.0)
    derived.record_embedding(conn, files["ben_1.png"], spec, _BEN, "aa", 0.0)
    conn.commit()
    connect.close(conn)


def test_search_answers_by_meaning_from_the_joint_space(faces, monkeypatch):
    """The CLIP trick over HTTP: stored image vectors and a typed phrase
    meet in one space, and /search returns the nearest pictures with
    scores -- no tags or captions anywhere in the loop. The encoder is
    faked; what is under test is the whole path from setting to space to
    resident index to ranked answer."""
    from vision import semantic

    _embedded_in_the_default_space(faces)

    class FakeText:
        def encode_query(self, phrase):
            assert phrase == "a woman smiling"
            return _ANA

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: FakeText())
    answer = faces.get("/search", params={"q": "a woman smiling", "k": 3})

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


def test_a_bad_model_setting_is_a_refused_search_never_a_500(faces):
    from db import settings as settings_module

    _embedded_in_the_default_space(faces)
    conn = connect.connect(faces.app.state.db_path)
    settings_module.put(conn, "semantic_model", "broken")
    conn.commit()
    connect.close(conn)

    assert faces.get("/search", params={"q": "x"}).status_code == 400


def test_search_fuses_two_spaces_by_rank_never_by_raw_score(faces, monkeypatch):
    """Two participating spaces with WILDLY different score scales: the
    fused order can only come from ranks. The file both models agree on
    outranks each model's private favourite, and per-space provenance
    survives into the response."""
    from db import similarity
    from vision import semantic

    client = faces
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


def test_a_replaced_file_stops_answering_by_its_old_picture(faces, monkeypatch):
    """The scanner keeps a file's identity through an in-place byte
    replacement; its old embedding must NOT keep retrieving it until the
    re-embed happens -- the stale row is excluded the moment the bytes
    changed, not whenever the embed job gets around to it."""
    from db import similarity
    from vision import semantic

    client = faces
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
def test_search_never_downloads_a_model(faces, monkeypatch):
    """Embeddings exist but the model cache is empty: the request is
    refused with the fix named, and no acquisition begins -- weights
    belong to /jobs/embed, not to a GET. The cache is EMPTIED for the
    test: the guard consults the run's models_dir and then the machine's
    shared Hugging Face cache, and a developer's box may hold the
    weights the test's premise denies."""
    from db import similarity
    from vision import weights

    monkeypatch.setattr(weights, "hub_cached", lambda repo, name, models_dir: None)
    client = faces
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


def test_the_cluster_job_says_what_it_grouped(faces, caplog):
    """One line per space on the console: run, threshold, faces, groups,
    how many groups a human's word named and how many were minted."""
    import logging

    with caplog.at_level(logging.INFO, logger="db.runner"):
        _drained_cluster_job(faces)

    said = [r.getMessage() for r in caplog.records if r.name == "db.runner" and r.getMessage().startswith("cluster ")]
    assert said == [
        "cluster test/embedder 1: run #1, threshold 0.55, 3 faces -> 1 groups (0 named by a human, 1 minted unnamed)",
        "cluster test/embedder 1: run #1 is the People page's default",
    ]


def test_an_empty_people_page_says_where_every_run_stands(faces):
    """A run that exists but is not the default is named with its
    standing -- never "nobody clustered yet" over a library that did."""
    conn = connect.connect(faces.app.state.db_path)
    conn.execute("UPDATE derived_face_run SET is_primary = 0")
    conn.commit()
    connect.close(conn)

    page = faces.get("/people", headers=AS_BROWSER)

    assert page.status_code == 200
    assert "nobody clustered yet" not in page.text
    assert 'data-run="1"' in page.text
    assert "run #1 (test/embedder): 3 faces in 1 groups" in page.text
    assert "sound but not adopted unasked; no run is the default" in page.text
    assert faces.get("/people").json() == [], "the machine list stays the primary run's answer"


def test_a_person_page_says_when_they_were_seen(faces):
    """The timeline's answer for one face: every current session holding
    one of their pictures, newest first, each a door onto THEIR pictures
    in it (the person scope composed with the session facet), and the
    story told of it when one was."""
    from tests.staging import settled

    client = faces
    assert client.get("/p/ana").json()["sessions"] == [], "no interpretation yet, no sessions"
    assert settled(client, client.post("/jobs/context").json()["id"]) == "done"
    assert settled(client, client.post("/jobs/events").json()["id"]) == "done"
    told = client.get("/p/ana").json()
    assert len(told["sessions"]) == 1, told["sessions"]
    session = told["sessions"][0]
    assert session["kind"] == "file_session"
    assert (session["theirs"], session["pictures"]) == (2, 3), "two of the three files are Ana's"
    assert "person=ana" in session["qs"]
    assert "event.id%3Aeq%3A" in session["qs"]
    import re

    page = client.get(f"/g?{session['qs']}").text
    assert int(re.search(r'data-total="(\d+)"', page).group(1)) == 2, "the door opens on Ana's pictures in that session"
    html = client.get("/p/ana", headers={"accept": "text/html"}).text
    assert "data-person-sessions" in html
    assert f'data-person-session="{session["id"]}"' in html
    assert session["timeline"].startswith("/timeline?bin=hour&start=")
    assert "data-person-session-tell" in html, "no story yet: the page opens the session's window on the timeline"


def test_search_refuses_rows_of_another_width_instead_of_crashing(faces, monkeypatch):
    """Vectors recorded 4 wide and an encoder that answers 512 wide are
    another build's rows: the space is reported missing with the fix,
    never searched into an assertion. With nothing else to answer from
    the request is a 400, not a 500."""
    from db import similarity
    from vision import semantic

    class Wide:
        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 512)

        def encode_query(self, text):
            raise AssertionError("a mismatched space must never be queried")

    monkeypatch.setattr(semantic, "encoder", lambda *a, **k: Wide())
    client = faces
    conn = connect.connect(client.app.state.db_path)
    conn.execute("UPDATE file SET content_sha256 = 'aa'")
    files = dict(conn.execute("SELECT name, id FROM file"))
    spec = similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)
    derived.record_embedding(conn, files["ana_1.png"], spec, np.array([1, 0, 0, 0], dtype=np.float32), "aa", 0.0)
    conn.commit()
    connect.close(conn)
    answer = client.get("/search", params={"q": "banana"})
    assert answer.status_code == 400, answer.text
    assert "4-dimensional" in answer.json()["detail"]
    assert "/jobs/embed" in answer.json()["detail"]


def test_the_gallery_chips_the_person_scope_too(faces):
    """A scope carried as its own parameter is as much a part of the
    question as a facet: the results page names it and lets it go."""
    page = faces.get("/g?person=ana&kind=image", headers={"accept": "text/html"}).text
    assert 'data-chip="person=ana"' in page
    assert "person ana" in page
    assert 'data-chip="kind=image"' in page
    import re

    removes = dict(re.findall(r'data-chip="([^"]+)">[^<]*<a href="([^"]+)"', page))
    assert removes["person=ana"] == "/g?kind=image"
    assert removes["kind=image"] == "/g?person=ana"


def test_the_people_index_says_when_each_was_seen(faces):
    """Beside the count, the span of a person's pictures on the human
    timeline -- absent, not guessed, until the library is interpreted;
    and the drawer lists their sessions."""
    from tests.staging import settled

    client = faces
    before = client.get("/people").json()
    assert before[0]["first_seen"] is None
    assert before[0]["last_seen"] is None
    assert settled(client, client.post("/jobs/context").json()["id"]) == "done"
    assert settled(client, client.post("/jobs/events").json()["id"]) == "done"
    after = client.get("/people").json()
    ana = next(p for p in after if p["slug"] == "ana")
    assert ana["first_seen"] is not None
    assert ana["first_seen"] <= ana["last_seen"]
    page = client.get("/people", headers={"accept": "text/html"}).text
    assert "data-person-seen" in page
    drawer = client.get("/p/ana", headers={"hx-request": "true"}).text
    assert "data-drawer-sessions" in drawer
