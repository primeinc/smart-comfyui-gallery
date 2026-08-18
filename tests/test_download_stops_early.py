"""A download that stops early must say so, and leave nothing behind.

The weights range from a few hundred megabytes upwards and arrive over
whatever connection somebody has, so a transfer ending early is the
ordinary failure, not the exotic one. It does not raise. CPython says so
in http/client.py:

    n = self.fp.readinto(b)
    if not n and b:
        # Ideally, we would raise IncompleteRead if the content-length
        # wasn't satisfied, but it might break compatibility.
        self._close_conn()

So a dropped connection ends the copy loop exactly the way a finished one
does, and whatever arrived was moved into place as the artifact.

Of the eleven artifacts, four pin a SHA-256 and four more come through
huggingface_hub, which checks its own. Three do not: the insightface pack
fetched from a direct URL. Those
three are zips, so today truncation does surface -- as BadZipFile, which
describes the wrong thing. The file is not a broken zip, it is part of a
zip, and the difference between those two messages is the difference
between retrying and going to look for a bad URL.

The .part file is the download in progress. On any failure it was left in
the models directory, where nothing ever looks for it again.
"""

from __future__ import annotations

import io
import os

import pytest

from smartgallery_ai import provision


class _Stream:
    """An HTTP response that declares one size and delivers another."""

    def __init__(self, body, declared=None):
        self._body = io.BytesIO(body)
        self.headers = {"Content-Length": str(len(body) if declared is None else declared)}

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NoLength(_Stream):
    """A chunked response: no Content-Length at all."""

    def __init__(self, body):
        super().__init__(body)
        self.headers = {}


@pytest.fixture
def serve(monkeypatch):
    """Answer the next urlopen with a given stream."""

    def _serve(stream):
        monkeypatch.setattr(provision.urllib.request, "urlopen", lambda url, timeout=None: stream)

    return _serve


def test_a_short_download_is_refused(serve, tmp_path):
    """The bug: 400 bytes of a declared 1000 were kept as the artifact."""
    serve(_Stream(b"x" * 400, declared=1000))
    dest = tmp_path / "weights.bin"

    with pytest.raises(provision.ProvisionError) as refused:
        provision._download_url("https://example.invalid/weights.bin", str(dest))

    message = str(refused.value)
    assert "400" in message, message
    assert "1,000" in message, message
    assert not dest.exists(), "a partial download was kept as the artifact"


def test_nothing_is_left_behind_after_a_short_download(serve, tmp_path):
    """The .part file is the download; abandoning it leaves the bytes in
    the models directory with nothing to find them again."""
    serve(_Stream(b"x" * 400, declared=1000))
    dest = tmp_path / "weights.bin"

    with pytest.raises(provision.ProvisionError):
        provision._download_url("https://example.invalid/weights.bin", str(dest))

    assert os.listdir(str(tmp_path)) == [], f"left behind: {os.listdir(str(tmp_path))}"


def test_nothing_is_left_behind_when_the_connection_raises(serve, tmp_path):
    """The other way a download ends: an error part way through."""

    class _Breaks(_Stream):
        def read(self, size=-1):
            chunk = super().read(size)
            if chunk:
                return chunk
            raise ConnectionResetError("the connection went away")

    serve(_Breaks(b"x" * 400, declared=1000))
    dest = tmp_path / "weights.bin"

    with pytest.raises(ConnectionResetError):
        provision._download_url("https://example.invalid/weights.bin", str(dest))

    assert os.listdir(str(tmp_path)) == [], f"left behind: {os.listdir(str(tmp_path))}"


def test_a_complete_download_is_kept(serve, tmp_path):
    """Over-reach guard, and every download that works. Refusing
    everything would satisfy the checks above."""
    body = b"y" * 4096
    serve(_Stream(body))
    dest = tmp_path / "weights.bin"

    provision._download_url("https://example.invalid/weights.bin", str(dest))

    assert dest.read_bytes() == body
    assert os.listdir(str(tmp_path)) == ["weights.bin"]


def test_a_server_that_declares_no_length_is_still_accepted(serve, tmp_path):
    """Over-reach guard: a chunked response has no length to check
    against, and refusing those would refuse a legitimate transfer.
    Chunked bodies raise on truncation on their own."""
    body = b"z" * 2048
    serve(_NoLength(body))
    dest = tmp_path / "weights.bin"

    provision._download_url("https://example.invalid/weights.bin", str(dest))

    assert dest.read_bytes() == body


def test_progress_still_reports_while_it_downloads(serve, tmp_path):
    """Over-reach guard: the byte count is now the return value as well as
    the thing the progress callback is fed, and both have to work."""
    serve(_Stream(b"w" * (3 << 20)))
    seen = []
    dest = tmp_path / "weights.bin"

    provision._download_url(
        "https://example.invalid/weights.bin", str(dest), progress=lambda done, total: seen.append((done, total))
    )

    assert seen, "no progress was reported"
    assert seen[-1] == (3 << 20, 3 << 20), seen[-1]


def test_a_truncated_stream_raises_nothing_by_itself():
    """Control. Everything here rests on the claim that a short body is
    not an error anywhere below this code -- if it were, there would be
    nothing to add."""
    stream = _Stream(b"x" * 400, declared=1000)
    sink = io.BytesIO()

    # Deliberately does not look at the return value: this has to hold
    # against the build that had none, or it is not a control.
    provision._copy_with_progress(stream, sink, 1000, None)

    assert sink.getvalue() == b"x" * 400, (
        "copying a body shorter than its declared length now fails on its own, so there is nothing here to add"
    )


def test_every_artifact_has_something_checking_it():
    """The sweep this came from, kept: an artifact fetched from a direct
    URL with no digest has only this length check between it and a silent
    half-download. A new one is fine -- it just has to be a deliberate
    choice, so the count is pinned."""
    unchecked = []
    for group in provision.GROUPS:
        unchecked.extend(
            artifact.dest
            for artifact in group.artifacts
            if artifact.sha256 is None and artifact.url and not artifact.hf_repo
        )

    assert sorted(unchecked) == sorted(
        [
            "insightface/models/antelopev2",
        ]
    ), (
        f"the set of direct-URL artifacts with no pinned digest changed: "
        f"{sorted(unchecked)}. Each one has only the declared length "
        f"standing between it and a half-download; pin a sha256 where you "
        f"can, and update this list where you cannot."
    )
