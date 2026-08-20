"""The real server, as a process: `python -m sg_web` answers on a socket.

TestClient exercises the same ASGI app in-process, which proves the
routes but not the entry point. This starts the actual command a person
runs -- uvicorn, the lifespan, the worker thread -- and proves the whole
thing over real HTTP (httpx.Client, encode/httpx httpx/_client.py:594):
first run from nothing, media with a range, a setting changed live, a
job drained by the worker with nobody stepping it.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.spawns


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_python_m_sg_web_serves_from_nothing(tmp_path):
    import httpx

    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (640, 360), (180, 40, 40)).save(root / "shot.png")
    import av

    with av.open(str(root / "clip.mp4"), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), (0, 0, 255), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    port = _free_port()
    # The child's stdout is its access log, one line per request; a pipe
    # nobody drains blocks the server at the OS buffer. A file has no
    # such ceiling, and holds the log for a post-mortem.
    server_log = (tmp_path / "server.log").open("wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "sg_web", "--home", str(tmp_path / "run"), "--port", str(port)],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as web:
            deadline = time.time() + 30
            while True:
                try:
                    if web.get("/health").text == "ok":
                        break
                except httpx.TransportError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.2)

            made = web.post("/roots", json={"path": str(root)}).json()
            swept = web.post(f"/roots/{made['id']}/scan").json()
            assert swept["added"] == 2

            # slugs are minted from name stems; the clip's is "clip"
            part = web.get("/media/clip", headers={"range": "bytes=0-7"})
            assert part.status_code == 206
            assert part.headers["content-range"].startswith("bytes 0-7/")
            assert part.content == (root / "clip.mp4").read_bytes()[:8]

            changed = web.post("/settings/similarity_backend", json={"value": "numpy"}).json()
            assert changed["value"] == "numpy"

            job = web.post("/jobs/verify").json()
            assert job["total"] == 2
            # Bounded HTTP polling, deliberately: this client is a bare
            # subprocess + httpx with no WebSocket library, and the test's
            # subject is the entry point. The realtime path is proven by
            # the app suite and the browser suite over /ws/jobs.
            deadline = time.time() + 30
            state = job["state"]
            while state not in ("done", "failed", "cancelled"):
                assert time.time() < deadline, "the worker never drained the job"
                time.sleep(0.2)
                state = web.get(f"/jobs/{job['id']}").json()["state"]
            assert state == "done", "the in-process worker did not finish an honest sweep"

            small = web.get("/thumb/shot")
            assert small.status_code == 200
            assert small.headers["content-type"].startswith("image/webp")

            # A real session is hundreds of requests, and every one is an
            # access-log line on the child's stdout. A spawner that hands
            # the child a pipe nobody drains freezes it at the OS buffer
            # (~4KB -- 64 access lines, probed on this host): the request
            # in flight hangs forever with no error. Volume is therefore
            # part of the entry-point contract.
            for _ in range(200):
                assert web.get("/health").text == "ok"
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=15)
        server_log.close()
    assert (tmp_path / "run" / "gallery.db").exists(), "the run did not live in its --home"
