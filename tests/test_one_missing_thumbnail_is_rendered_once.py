"""Sixty cells asking for one picture is one picture's worth of work.

A miss on `/thumbs/<shard>/<sha>.webp` renders, which is what gives a
fresh library a slow grid instead of a broken one. But nothing stopped
two askers from rendering the SAME missing file at the same time, and a
grid asks for sixty at once and gets reloaded while they are in flight.
Measured before this: four concurrent requests for one missing
thumbnail decoded, resized and encoded it four times, each on its own
database connection, all four competing with the precache job for the
same CPU.

Not a correctness bug -- the bytes land through a staging name and
`os.replace` (vision/thumbs.py), so racing writers produce identical
bytes. Purely waste, which is why the gate is per process and there is
no lock file: a cross-process lock would buy a correctness that is
already held and leave something to leak.

The tests below are written so that the failing direction cannot be
reached by a slow machine. Where two things must happen AT ONCE, they
meet at a `threading.Barrier` and the assertion is that they met -- not
that a stopwatch said so.
"""

from __future__ import annotations

import pathlib
import threading

import pytest
from litestar.testing import TestClient
from PIL import Image

import sg_web.app as app_module
from db import connect
from sg_web.app import build_app

pytestmark = pytest.mark.slow


def _library(tmp_path: pathlib.Path, pictures: int):
    root = tmp_path / "lib"
    root.mkdir()
    for i in range(pictures):
        Image.new("RGB", (64, 48), (30 + 40 * i, 120, 200)).save(root / f"p{i}.png")
    return root


def _served(tmp_path: pathlib.Path, pictures: int):
    root = _library(tmp_path, pictures)
    client = TestClient(app=build_app(str(tmp_path / "run"), worker=False))
    return client, root


def _asset_urls(client) -> list[str]:
    conn = connect.connect(client.app.state.db_path)
    try:
        shas = [row[0] for row in conn.execute("SELECT content_sha256 FROM file ORDER BY id")]
    finally:
        connect.close(conn)
    assert all(shas), "the scan did not hash the pictures, so there is no asset address"
    return [f"/thumbs/{sha[:2]}/{sha}.webp" for sha in shas]


def test_four_cells_wanting_one_picture_render_it_once(tmp_path, monkeypatch):
    """The defect. Four askers, one render, and all four still served."""
    client, root = _served(tmp_path, 1)
    renders: list[float] = []
    real = app_module._render_asset

    def counted(state, sha, variant, target):
        renders.append(0.0)
        # long enough that the other three are certainly queued behind the
        # gate; if it were somehow not, they would find the file already
        # rendered and the count would still be one -- slowness cannot
        # make this fail, only make it prove less
        threading.Event().wait(0.5)
        return real(state, sha, variant, target)

    monkeypatch.setattr(app_module, "_render_asset", counted)

    with client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        url = _asset_urls(client)[0]

        together = threading.Barrier(4)
        answers: dict[int, int] = {}

        def ask(n: int) -> None:
            together.wait(timeout=10)
            answers[n] = client.get(url).status_code

        askers = [threading.Thread(target=ask, args=(i,)) for i in range(4)]
        for one in askers:
            one.start()
        for one in askers:
            one.join(timeout=30)

    assert answers == dict.fromkeys(range(4), 200), "somebody waiting on the gate was not served"
    assert len(renders) == 1, f"one missing thumbnail was rendered {len(renders)} times"


def test_two_different_pictures_are_not_made_to_queue(tmp_path, monkeypatch):
    """The other half, and what a single global lock would break.

    Coalescing must be per FILE. A lock over all rendering would also
    render each file once and would serialise a fresh library's whole
    grid behind one picture at a time. So two different missing
    thumbnails have to be able to be inside the renderer together --
    asserted by making them meet, which they cannot do if one is waiting
    for the other.
    """
    client, root = _served(tmp_path, 2)
    met = threading.Barrier(2)
    failed_to_meet: list[str] = []
    real = app_module._render_asset

    def rendezvous(state, sha, variant, target):
        try:
            met.wait(timeout=10)
        except threading.BrokenBarrierError:
            failed_to_meet.append(sha)
        return real(state, sha, variant, target)

    monkeypatch.setattr(app_module, "_render_asset", rendezvous)

    with client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        urls = _asset_urls(client)
        assert len(urls) == 2
        assert urls[0] != urls[1], "both cells wanted the same bytes; that is the other test"

        answers: dict[int, int] = {}

        def ask(n: int) -> None:
            answers[n] = client.get(urls[n]).status_code

        askers = [threading.Thread(target=ask, args=(i,)) for i in range(2)]
        for one in askers:
            one.start()
        for one in askers:
            one.join(timeout=30)

    assert not failed_to_meet, "two different pictures could not be rendered at the same time"
    assert answers == {0: 200, 1: 200}


def test_the_gate_is_let_go_of(tmp_path):
    """It holds what is rendering NOW, not everything ever asked for."""
    client, root = _served(tmp_path, 2)
    with client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        for url in _asset_urls(client):
            assert client.get(url).status_code == 200
    assert app_module._RENDERS == {}, "a gate outlived the render it was for"


def test_a_gate_is_let_go_of_even_when_the_render_fails(tmp_path, monkeypatch):
    """A raised render must not leave the file permanently gated: the
    next asker has to be able to try, not block for ever."""
    client, root = _served(tmp_path, 1)

    def refuses(state, sha, variant, target):
        raise ValueError("nothing decodable here")

    monkeypatch.setattr(app_module, "_render_asset", refuses)

    with client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        url = _asset_urls(client)[0]
        assert client.get(url).status_code == 404
        assert client.get(url).status_code == 404, "the second ask did not even reach the renderer"

    assert app_module._RENDERS == {}, "a failed render kept its gate"


def test_the_slug_route_coalesces_too(tmp_path, monkeypatch):
    """`/thumb/<slug>` renders on a miss by its own path, and a person
    following a link hits it the same way a grid hits the asset URL."""
    client, root = _served(tmp_path, 1)
    renders: list[str] = []
    from vision import derive

    real = derive.put_one

    def counted(cache, sha, path, kind, orientation, variant):
        renders.append(sha)
        threading.Event().wait(0.5)
        return real(cache, sha, path, kind, orientation, variant)

    monkeypatch.setattr(derive, "put_one", counted)

    with client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            slug = conn.execute("SELECT slug FROM entity WHERE kind = 'file'").fetchone()[0]
        finally:
            connect.close(conn)

        together = threading.Barrier(3)
        answers: dict[int, int] = {}

        def ask(n: int) -> None:
            together.wait(timeout=10)
            answers[n] = client.get(f"/thumb/{slug}").status_code

        askers = [threading.Thread(target=ask, args=(i,)) for i in range(3)]
        for one in askers:
            one.start()
        for one in askers:
            one.join(timeout=30)

    assert answers == dict.fromkeys(range(3), 200)
    assert len(renders) == 1, f"one missing thumbnail was rendered {len(renders)} times"
