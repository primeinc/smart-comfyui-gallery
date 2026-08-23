"""A person says where a picture happened, and the library holds it.

`place_id` had no producer: GPS alone names no place, and no resolver
ships. A person's word does -- POST /i/{slug}/place finds or mints the
place by name and kind, records the claim as authored desired state
(file_place), and the file's context is re-interpreted at once with
`location_basis = 'authored'`. The claim survives every rebuild and
opens a gallery door through the `place.id` facet.
"""

from __future__ import annotations

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


def _slugs(client) -> list[str]:
    from db import naming

    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        return [naming.entity_slug(conn, fid)[1] for (fid,) in conn.execute("SELECT id FROM file ORDER BY id")]
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
        # and the door opens exactly the pictures there
        door = client.get(f"/g?{where['qs']}", headers={"accept": "text/html"}).text
        assert 'data-total="2"' in door
        assert "place =" in door or "place" in door

        # a rebuild keeps a person's word: the claim is authored state
        assert client.post("/jobs/context", params={"everything": "true"}).status_code == 201
        _drain(client)
        assert client.get(f"/i/{a}", headers=AS_MACHINE).json()["where"]["basis"] == "authored"

        # withdrawn: nowhere said again; a bad kind is refused
        gone = client.post(f"/i/{a}/place", json={"name": None})
        assert gone.status_code in (200, 201)
        assert gone.json()["where"] is None
        assert client.get(f"/i/{a}", headers=AS_MACHINE).json()["where"] is None
        assert client.post(f"/i/{c}/place", json={"name": "Mars", "kind": "planet"}).status_code == 400
