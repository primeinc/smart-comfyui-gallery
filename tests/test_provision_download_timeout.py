"""A model download that stalls must give up, not wait for ever.

Every other outbound call in the program passes a timeout. The model
download did not:

    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:

so a server that accepts the connection and then says nothing held the
thread indefinitely. That is not a rare shape -- a stalled mirror, a
captive portal intercepting the request, a firewall that black-holes
instead of refusing, all behave that way. Auto-provisioning runs in the
background, so the result was an AI layer that never arrived, with nothing
reporting an error, because nothing had failed yet.

The timeout is per-read rather than a budget for the transfer, so a slow
connection still finishes a multi-gigabyte model as long as bytes keep
arriving. The test drives a socket that accepts and then goes quiet, which
is exactly the case a total-transfer budget would not distinguish from a
slow download.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from typing import ClassVar

import pytest

from smartgallery_ai import provision


@pytest.fixture
def silent_server():
    """A listener that accepts a connection and then never answers."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    accepted = []

    def _serve():
        try:
            conn, _addr = sock.accept()
            accepted.append(conn)
            # Hold it open, saying nothing, until the test tears down.
            time.sleep(30)
        except OSError:
            pass

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    yield port

    for conn in accepted:
        with contextlib.suppress(OSError):
            conn.close()
    sock.close()


def test_a_stalled_download_gives_up(silent_server, tmp_path, monkeypatch):
    """The regression: this never returned."""
    monkeypatch.setattr(provision, "DOWNLOAD_STALL_TIMEOUT", 1)
    dest = str(tmp_path / "model.bin")

    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        provision._download_url(f"http://127.0.0.1:{silent_server}/model.bin", dest)
    elapsed = time.monotonic() - started

    assert elapsed < 15, f"the download took {elapsed:.1f}s to give up"
    assert not os.path.exists(dest), "a failed download left a file behind"
    assert "timed out" in str(excinfo.value).lower() or "timeout" in type(excinfo.value).__name__.lower(), (
        f"gave up for some other reason: {type(excinfo.value).__name__}: {excinfo.value}"
    )


def test_the_timeout_is_actually_passed_through(tmp_path, monkeypatch):
    """The bug was one missing keyword, so this pins the keyword itself:
    the behavioural test above would also pass if a caller further out
    happened to impose a timeout of its own."""
    seen = {}

    class _FakeResponse:
        headers: ClassVar = {"Content-Length": "4"}

        def read(self, _n=None):
            return seen.pop("body", b"")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        seen["body"] = b"data"
        return _FakeResponse()

    monkeypatch.setattr(provision, "open_url", _fake_urlopen)
    dest = str(tmp_path / "model.bin")

    provision._download_url("http://example.invalid/model.bin", dest)

    assert seen["timeout"] == provision.DOWNLOAD_STALL_TIMEOUT, f"urlopen was called with timeout={seen['timeout']!r}"
    assert os.path.exists(dest), "the successful path stopped working"


def test_a_normal_download_still_completes(tmp_path, monkeypatch):
    """The counterpart: a timeout that aborted healthy downloads would be
    worse than the hang it replaced."""
    payload = b"x" * (3 << 20)  # 3 MB, several chunks

    class _Response:
        def __init__(self):
            self._pos = 0
            self.headers = {"Content-Length": str(len(payload))}

        def read(self, n=None):
            chunk = payload[self._pos : self._pos + (n or len(payload))]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(provision, "open_url", lambda url, timeout=None: _Response())
    dest = str(tmp_path / "model.bin")
    seen_progress = []

    provision._download_url(
        "http://example.invalid/m.bin", dest, progress=lambda done, total: seen_progress.append((done, total))
    )

    assert os.path.getsize(dest) == len(payload)
    assert seen_progress[-1] == (len(payload), len(payload))
