"""The most expensive thing this application learns, made portable.

Ten years of a stable corpus can answer questions nothing else can --
which faces did the 2026 model confuse, how did one person's embedding
move from 30 to 40 -- and none of it survives if the numbers can only
be read by the program that made them.

So a naked 512-float vector is not the export. A vector without the
producer that made it, the preprocessing that fed it, the metric it is
compared under and how many floats it has recreates exactly the opaque
dependency this is supposed to escape: it cannot be reproduced, checked,
or compared with anything. `similarity_space` is that identity, it is
immutable by trigger, and it rides beside every group.

Grouped BY SPACE, because a vector is comparable only to another from
the same one. A library re-detected under a new model holds two
representations of one person, and flattening them into one list would
invite a comparison that means nothing.

These are BIOMETRIC TEMPLATES. This is a local library answering on the
machine that holds it, and the surface offering it says so.
"""

from __future__ import annotations

import numpy as np
import pytest
from litestar.testing import TestClient
from PIL import Image

from db import authored, connect, derived
from sg_web.app import build_app

pytestmark = pytest.mark.slow

DETECTOR, VERSION = "opencv/yunet+sface", "1"


def _clustered(conn, person_id: int, face_ids, *, dim: int = 4):
    run = derived.run_for(conn, DETECTOR, VERSION, derived.DEFAULT_METHOD, 0.55, 0.0)
    conn.execute("UPDATE derived_face_run SET is_primary = 1 WHERE id = ?", (run,))
    cluster = int(
        conn.execute(
            "INSERT INTO derived_face_cluster(run_id, person_id, centroid, dim, model_id, model_version, updated_at)"
            " VALUES(?, ?, ?, ?, ?, ?, 0) RETURNING id",
            (run, person_id, (np.ones(dim, np.float32) * 1.5).tobytes(), dim, DETECTOR, VERSION),
        ).fetchone()[0]
    )
    for face_id in face_ids:
        conn.execute("INSERT INTO derived_face_membership(cluster_id, face_id) VALUES(?, ?)", (cluster, face_id))
    return run, cluster


@pytest.fixture
def known(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        ids = [one for (one,) in conn.execute("SELECT id FROM file ORDER BY name")]
        for n, file_id in enumerate(ids):
            derived.record_faces(
                conn,
                file_id,
                DETECTOR,
                VERSION,
                f"{n:064d}",
                0.0,
                [
                    {
                        "region": derived.region(conn, 0.1, 0.2, 0.3, 0.4),
                        "embedding": (np.ones(4, np.float32) * (n + 1)).tobytes(),
                    }
                ],
            )
        sarah = authored.person(conn, "Sarah", 0.0)
        faces = [one for (one,) in conn.execute("SELECT id FROM derived_face_instance ORDER BY id")]
        _clustered(conn, sarah, faces)
        # a capture time on the first two only, so a range can exclude
        for file_id, when in zip(ids[:2], (1_600_000_000.0, 1_700_000_000.0), strict=True):
            conn.execute("INSERT INTO capture(file_id, captured_at, parsed_at) VALUES(?, ?, 0)", (file_id, when))
        conn.commit()
        yield client, conn, sarah, ids
        connect.close(conn)


def _exported(client, slug="sarah", **params) -> dict:
    told = client.get(f"/operations/export/faces/{slug}.json", params=params)
    assert told.status_code == 200, told.text
    return told.json()


def test_a_vector_arrives_with_what_makes_it_mean_something(known):
    """The whole entry: not a naked vector. Everything needed to
    reproduce, check or compare it comes with it."""
    client, _conn, _who, _ids = known
    told = _exported(client)

    assert told["person"] == "sarah"
    assert told["name"] == "Sarah"
    assert len(told["spaces"]) == 1
    space = told["spaces"][0]
    for named in (
        "space",
        "representation",
        "dimensions",
        "metric",
        "producer",
        "producer_version",
        "preprocess",
        "preprocess_version",
        "spec_hash",
    ):
        assert space[named], f"{named} is missing, and without it the numbers are opaque"
    assert space["producer"] == DETECTOR
    assert space["representation"] == "float32"
    assert space["metric"] == "cosine"
    assert space["centroid"] == [1.5, 1.5, 1.5, 1.5]

    assert len(space["faces"]) == 3
    first = space["faces"][0]
    assert first["embedding"] == [1.0, 1.0, 1.0, 1.0]
    assert first["region"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    assert len(first["sha256"]) == 64


def test_a_face_names_the_bytes_it_was_found_in_and_no_path(known):
    """The join back to a photograph is the content hash, which means
    the same picture in any library that holds it. A path is a fact
    about one machine and does not travel."""
    client, _conn, _who, _ids = known
    body = client.get("/operations/export/faces/sarah.json").text
    assert "p0.png" not in body
    assert "lib" not in body.replace("similarity", "")  # no directory names ride along
    for face in _exported(client)["spaces"][0]["faces"]:
        assert len(face["sha256"]) == 64
        assert "path" not in face
        assert "file_id" not in face


def test_two_spaces_are_never_flattened_into_one_list(known):
    """A vector is comparable only to another from the SAME space. A
    library re-detected under a new model holds two representations of
    one person, and one flat list would invite a comparison that means
    nothing."""
    client, conn, sarah, ids = known
    # a second detector over the same pictures, and a second primary run
    for n, file_id in enumerate(ids):
        derived.record_faces(
            conn,
            file_id,
            "insightface/arcface",
            "2",
            f"{n:064d}",
            1.0,
            [
                {
                    "region": derived.region(conn, 0.5, 0.5, 0.2, 0.2),
                    "embedding": (np.ones(8, np.float32) * (n + 10)).tobytes(),
                }
            ],
        )
    newer = [
        one
        for (one,) in conn.execute(
            "SELECT id FROM derived_face_instance WHERE model_id = 'insightface/arcface' ORDER BY id"
        )
    ]
    conn.execute("UPDATE derived_face_run SET is_primary = 0")
    run = derived.run_for(conn, "insightface/arcface", "2", derived.DEFAULT_METHOD, 0.55, 1.0)
    conn.execute("UPDATE derived_face_run SET is_primary = 1 WHERE id = ?", (run,))
    cluster = int(
        conn.execute(
            "INSERT INTO derived_face_cluster(run_id, person_id, centroid, dim, model_id, model_version, updated_at)"
            " VALUES(?, ?, ?, 8, 'insightface/arcface', '2', 1) RETURNING id",
            (run, sarah, (np.ones(8, np.float32) * 11).tobytes()),
        ).fetchone()[0]
    )
    for face_id in newer:
        conn.execute("INSERT INTO derived_face_membership(cluster_id, face_id) VALUES(?, ?)", (cluster, face_id))
    conn.commit()

    told = _exported(client)
    assert len(told["spaces"]) == 1, "only the PRIMARY run answers, and there is one"
    space = told["spaces"][0]
    assert space["producer"] == "insightface/arcface"
    assert space["dimensions"] == 8
    assert all(len(face["embedding"]) == 8 for face in space["faces"])


def test_a_date_range_is_over_capture_and_says_so(known):
    """ "Their faces from 2019" is a question about when the picture was
    TAKEN, so a photograph whose camera never said excludes itself the
    moment a range is given. That is the honest reading and the one the
    surface states."""
    client, _conn, _who, _ids = known
    whole = _exported(client)["spaces"][0]["faces"]
    assert len(whole) == 3

    narrowed = _exported(client, since=1_650_000_000.0)["spaces"][0]["faces"]
    assert len(narrowed) == 1, "the range took the one captured after it, and not the undated"
    assert narrowed[0]["captured_at"] == 1_700_000_000.0


def test_a_person_nobody_has_clustered_exports_nothing_rather_than_failing(known):
    """A person with a name and no faces is an ordinary state -- somebody
    typed a name before the clustering ran."""
    client, conn, _who, _ids = known
    authored.person(conn, "Nobody Yet", 0.0)
    conn.commit()
    told = _exported(client, slug="nobody-yet")
    assert told["name"] == "Nobody Yet"
    assert told["spaces"] == []


def test_no_such_person_is_a_404(known):
    client, _conn, _who, _ids = known
    assert client.get("/operations/export/faces/never-existed.json").status_code == 404


def test_it_is_offered_on_the_person_it_is_about(known):
    """Per person, so it belongs on their page rather than in a console
    -- and the link says what the numbers are before somebody puts them
    in a file."""
    client, _conn, _who, _ids = known
    page = client.get("/p/sarah", headers={"accept": "text/html"}).text
    assert "data-export-faces" in page
    assert 'href="/operations/export/faces/sarah.json"' in page
    assert "biometric" in page, "the link does not say what it hands over"
