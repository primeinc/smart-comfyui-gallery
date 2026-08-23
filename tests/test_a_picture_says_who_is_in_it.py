"""A picture's page says who is in it and whether anyone looked.

The people come from the primary clustering by address; the looking
comes from the detector's pass record (derived_face_scan), so "nobody
here" and "nobody looked" are different sentences.
"""

from __future__ import annotations

import pytest

from db import connect, derived, naming
from tests.staging import staged
from tests.test_a_person_is_an_address_with_two_looks import _clustered, _library

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "who", _library, _clustered) as stage:
        yield stage


@pytest.fixture
def client(_stage):
    _stage.restore()
    return _stage.client


def _slug(client, name: str) -> tuple[int, str]:
    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        file_id = conn.execute("SELECT id FROM file WHERE name = ?", (name,)).fetchone()[0]
        return file_id, naming.entity_slug(conn, file_id)[1]
    finally:
        connect.close(conn)


def test_the_page_names_the_people_and_says_nobody_looked(client):
    _, slug = _slug(client, "ana_1.png")
    told = client.get(f"/i/{slug}", headers=AS_MACHINE).json()
    assert [p["href"] for p in told["faces"]["people"]] == ["/p/ana"]
    assert told["faces"]["looked"] == [], "faces were recorded directly; no detector pass is on record"
    page = client.get(f"/i/{slug}", headers=AS_BROWSER).text
    assert 'data-person="ana"' in page
    assert "data-faces-missing" in page


def test_a_recorded_pass_says_who_looked_and_what_it_found(client):
    file_id, slug = _slug(client, "ben_1.png")
    conn = connect.connect(client.app.state.db_path)
    try:
        sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
        derived.record_face_scan(conn, file_id, "test/embedder", "1", sha, 7.0, 1)
        conn.commit()
    finally:
        connect.close(conn)
    told = client.get(f"/i/{slug}", headers=AS_MACHINE).json()
    assert told["faces"]["people"] == [], "ben is a singleton face, no person"
    assert told["faces"]["looked"] == [{"model_id": "test/embedder", "model_version": "1", "faces": 1, "at": 7.0}]
    page = client.get(f"/i/{slug}", headers=AS_BROWSER).text
    assert "test/embedder found 1 face" in page
    assert "data-faces-missing" not in page
    # the pass covers THESE bytes only: new bytes, nobody looked
    conn = connect.connect(client.app.state.db_path)
    try:
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("f" * 64, file_id))
        conn.commit()
    finally:
        connect.close(conn)
    assert client.get(f"/i/{slug}", headers=AS_MACHINE).json()["faces"]["looked"] == []


def test_the_lightbox_says_who_is_with_you(client):
    _, slug = _slug(client, "ana_1.png")
    part = client.get(f"/i/{slug}", headers={"hx-request": "true"}).text
    assert "data-lightbox-people" in part
    assert 'href="/p/ana"' in part


def test_the_page_says_whether_its_metadata_was_read_from_these_bytes(client):
    """The record of the read, shown: never, current, or stale once a
    scan has recorded new bytes -- each with the ingest job named."""
    from db import ingest

    file_id, slug = _slug(client, "ana_2.png")
    assert client.get(f"/i/{slug}", headers=AS_MACHINE).json()["read"] == "never"
    page = client.get(f"/i/{slug}", headers=AS_BROWSER).text
    assert 'data-read="never"' in page
    conn = connect.connect(client.app.state.db_path)
    try:
        from db import detect

        ingest.one(conn, file_id, detect.path_of(conn, file_id), 5.0)
        conn.commit()
        assert client.get(f"/i/{slug}", headers=AS_MACHINE).json()["read"] == "current"
        assert 'data-read="current"' in client.get(f"/i/{slug}", headers=AS_BROWSER).text
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("d" * 64, file_id))
        conn.commit()
    finally:
        connect.close(conn)
    assert client.get(f"/i/{slug}", headers=AS_MACHINE).json()["read"] == "stale"
    assert "read from older bytes" in client.get(f"/i/{slug}", headers=AS_BROWSER).text


def test_a_person_page_says_where_they_were_seen(client):
    """Ana's pictures placed in Lisbon: her page lists Lisbon with how
    many of her pictures are there and a door to exactly those."""
    from db import authored, context, places

    ids = [_slug(client, name)[0] for name in ("ana_1.png", "ana_2.png")]
    conn = connect.connect(client.app.state.db_path)
    try:
        lisbon = places.named(conn, "Lisbon", "city", 1.0)
        for file_id in ids:
            authored.set_place(conn, file_id, client.app.state.actor_id, lisbon, 1.0)
            context.rebuild_one(conn, file_id, 1.0)
        conn.commit()
    finally:
        connect.close(conn)
    told = client.get("/p/ana", headers=AS_MACHINE).json()
    assert [(p["name"], p["kind"], p["pictures"]) for p in told["places"]] == [("Lisbon", "city", 2)]
    assert "person=ana" in told["places"][0]["qs"]
    assert "place.id%3Aeq%3A" in told["places"][0]["qs"]
    page = client.get("/p/ana", headers=AS_BROWSER).text
    assert "data-person-places" in page
    assert "where they were seen" in page
