"""A gallery with no ffmpeg should get one, and never a broken one.

Without ffprobe the gallery quietly loses video duration and dimensions,
video thumbnails, waveforms and video metadata stripping. Telling somebody
to go and install ffmpeg is asking them to do work the gallery can do.

The pin was taken from the release rather than remembered. Verified at the
time of writing, by HEAD against both URLs:

    linux   HTTP 200  declared 124917816  pinned 124917816  match True
    win32   HTTP 200  declared 167405723  pinned 167405723  match True

Pinned to a month-end auto-build, not the `latest` tag: `latest` is
rolling, so the file behind a name changes and a digest recorded against
it goes stale on the next rebuild. BtbN keeps one build per month --
checked, an unbroken monthly series back to 2024-09 -- so a month-end tag
stays fetchable.

GPL rather than LGPL is forced by what the gallery does with it: the video
stream route runs `-vcodec libx264`, and x264 is GPL, so an LGPL build
would download fine and then fail every playback.

Nothing here touches the network. The download is driven through a fake
response so the failure paths -- a short transfer, a wrong digest -- can
be checked, which is the half that matters: a half-downloaded ffmpeg that
gets kept is worse than none.
"""

from __future__ import annotations

import ast
import hashlib
import io
import os
import tarfile
import zipfile

import pytest

import smartgallery


def _programs_here():
    build = smartgallery.ffmpeg_build_for_this_machine()
    if not build:
        pytest.skip("no pinned ffmpeg build for this platform")
    return build


def _make_zip(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _make_tar_xz(path, members):
    with tarfile.open(path, "w:xz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


# --- the pin itself -------------------------------------------------------


def test_the_tag_is_a_fixed_one_not_the_rolling_latest():
    """`latest` would make the recorded digests meaningless: the file
    behind a name there changes on every rebuild."""
    assert smartgallery.FFMPEG_RELEASE_TAG != "latest"
    assert smartgallery.FFMPEG_RELEASE_TAG.startswith("autobuild-")
    assert smartgallery.FFMPEG_RELEASE_TAG in smartgallery.FFMPEG_DOWNLOAD_BASE


@pytest.mark.parametrize("platform", sorted(smartgallery.FFMPEG_BUILDS))
def test_every_pinned_build_is_completely_specified(platform):
    """A size and a digest that are not both there is a download nothing
    can check."""
    build = smartgallery.FFMPEG_BUILDS[platform]

    assert build["bytes"] > 1_000_000, build
    assert len(build["sha256"]) == 64, build["sha256"]
    assert all(c in "0123456789abcdef" for c in build["sha256"]), build["sha256"]
    assert build["programs"][0].lower().startswith("ffprobe"), build["programs"]
    assert any(p.lower().startswith("ffmpeg") for p in build["programs"]), build


@pytest.mark.parametrize("platform", sorted(smartgallery.FFMPEG_BUILDS))
def test_the_build_is_gpl_because_the_stream_route_needs_x264(platform):
    """Not a preference. smartgallery streams video with libx264, which is
    GPL; an LGPL build has no x264 and would fail every playback."""
    assert "-gpl" in smartgallery.FFMPEG_BUILDS[platform]["asset"], (
        "the pinned asset is not a GPL build, so libx264 is missing and video playback cannot work"
    )
    assert "lgpl" not in smartgallery.FFMPEG_BUILDS[platform]["asset"]


# --- taking the programs out of the archive -------------------------------


def test_only_the_programs_come_out_and_they_come_out_flat(tmp_path):
    """The archives nest everything under a versioned folder; the gallery
    wants two files, not a tree."""
    build = _programs_here()
    probe, ffmpeg = build["programs"]
    archive = tmp_path / "build.zip"
    _make_zip(
        archive,
        {
            f"ffmpeg-n8.1.2/bin/{probe}": b"probe-bytes",
            f"ffmpeg-n8.1.2/bin/{ffmpeg}": b"ffmpeg-bytes",
            "ffmpeg-n8.1.2/doc/ffmpeg.html": b"docs",
            "ffmpeg-n8.1.2/LICENSE.txt": b"licence",
        },
    )
    out = tmp_path / "dest"

    taken = smartgallery._extract_ffmpeg_programs(str(archive), str(out), build["programs"])

    assert sorted(os.listdir(out)) == sorted([probe, ffmpeg])
    assert (out / probe).read_bytes() == b"probe-bytes"
    assert set(taken) == {probe.lower(), ffmpeg.lower()}


def test_a_tar_member_cannot_choose_where_it_lands(tmp_path):
    """tarfile honours a member called ../../ if you let it. Members are
    matched by basename and written to a path this code builds, so the
    archive has no say."""
    build = _programs_here()
    probe = build["programs"][0]
    archive = tmp_path / "build.tar.xz"
    _make_tar_xz(
        archive,
        {
            f"ffmpeg/bin/{probe}": b"probe-bytes",
            f"ffmpeg/bin/{build['programs'][1]}": b"ffmpeg-bytes",
            f"../../../{probe}": b"escaped",
        },
    )
    out = tmp_path / "dest"

    smartgallery._extract_ffmpeg_programs(str(archive), str(out), build["programs"])

    escaped = tmp_path.parent.parent / probe
    assert not escaped.exists(), f"a member escaped to {escaped}"
    assert (out / probe).read_bytes() == b"probe-bytes", "the first matching member should win, not the last"


def test_an_archive_without_the_programs_is_an_error(tmp_path):
    """Silently producing nothing would leave the gallery announcing
    success with no ffmpeg."""
    build = _programs_here()
    archive = tmp_path / "build.zip"
    _make_zip(archive, {"ffmpeg-n8.1.2/README.txt": b"nothing useful"})

    with pytest.raises(OSError, match="did not contain"):
        smartgallery._extract_ffmpeg_programs(str(archive), str(tmp_path / "d"), build["programs"])


# --- the download, driven through a fake response -------------------------


class _Response:
    def __init__(self, payload, declared=None):
        self._body = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload) if declared is None else declared)}

    def read(self, size=-1):
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """A pinned build whose archive is served from memory."""
    build = dict(_programs_here())
    probe, ffmpeg = build["programs"]

    archive = tmp_path / "payload.zip"
    _make_zip(archive, {f"ffmpeg-n8.1.2/bin/{probe}": b"probe-bytes", f"ffmpeg-n8.1.2/bin/{ffmpeg}": b"ffmpeg-bytes"})
    payload = archive.read_bytes()

    build["asset"] = "payload.zip"
    build["bytes"] = len(payload)
    build["sha256"] = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(smartgallery, "ffmpeg_build_for_this_machine", lambda: build)
    monkeypatch.setattr(smartgallery, "BASE_SMARTGALLERY_PATH", str(tmp_path))
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: True)
    return build, payload


def test_a_good_download_is_kept(offline, tmp_path, monkeypatch):
    """Over-reach guard, and the whole point: when it works, it works.

    Through monkeypatch, not a bare assignment. urllib.request.urlopen is
    the real module's attribute, shared by everything in the process --
    setting it directly left every later test in the session downloading
    through this stub, and broke the AI provisioning timeout test six
    files away."""
    _build, payload = offline
    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", lambda url, timeout=None: _Response(payload))

    found = smartgallery.fetch_ffmpeg()

    assert found is not None
    assert os.path.isfile(found)
    assert os.path.basename(found).lower().startswith("ffprobe")


def test_a_short_download_is_thrown_away(offline, tmp_path, monkeypatch):
    """The half that matters. A truncated ffmpeg that gets kept is worse
    than no ffmpeg: it is a program that starts and then fails."""
    _build, payload = offline
    monkeypatch.setattr(
        smartgallery.urllib.request,
        "urlopen",
        lambda url, timeout=None: _Response(payload[: len(payload) // 2], declared=len(payload)),
    )

    assert smartgallery.fetch_ffmpeg() is None
    assert os.listdir(smartgallery.bundled_ffmpeg_dir()) == [], "a partial download was left behind"


def test_a_download_with_the_wrong_digest_is_thrown_away(offline, monkeypatch):
    """The digest is the reason a pinned tag was worth finding."""
    build, payload = offline
    build["sha256"] = "0" * 64
    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", lambda url, timeout=None: _Response(payload))

    assert smartgallery.fetch_ffmpeg() is None
    assert os.listdir(smartgallery.bundled_ffmpeg_dir()) == []


def test_a_program_that_is_not_ffprobe_is_thrown_away(offline, monkeypatch):
    """Last check of the three: whatever came out has to answer as
    ffprobe. Size and digest only say the bytes are the expected bytes."""
    _build, payload = offline
    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", lambda url, timeout=None: _Response(payload))
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: False)

    assert smartgallery.fetch_ffmpeg() is None


def test_a_connection_that_dies_is_not_an_exception(offline, monkeypatch):
    """No ffmpeg is a gallery without video features, not a gallery that
    will not start."""

    def _explode(url, timeout=None):
        raise ConnectionResetError("the connection went away")

    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", _explode)

    assert smartgallery.fetch_ffmpeg() is None


# --- how it fits into resolution ------------------------------------------


def test_an_already_fetched_ffmpeg_is_used_without_downloading_again(offline, monkeypatch):
    """Fetching costs a large download once. Every later start has to find
    it sitting there."""
    _build, payload = offline
    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", lambda url, timeout=None: _Response(payload))
    first = smartgallery.fetch_ffmpeg()
    assert first is not None

    def _refuse(url, timeout=None):
        raise AssertionError("downloaded again instead of using what was there")

    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(smartgallery, "FFPROBE_MANUAL_PATH", "")
    # Nothing on PATH -- only the copy that was fetched. The bare name is
    # rejected on purpose: a real ffprobe on PATH SHOULD win over a
    # fetched one, so accepting it here would be testing the wrong thing.
    fetched_dir = os.path.normcase(smartgallery.bundled_ffmpeg_dir())
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: os.path.normcase(str(path)).startswith(fetched_dir))

    assert smartgallery.bundled_ffprobe_path() is not None
    assert smartgallery.find_ffprobe_path() == first


def test_turning_it_off_downloads_nothing(offline, monkeypatch):
    """Over-reach guard for anybody on a metered or closed connection."""

    def _refuse(url, timeout=None):
        raise AssertionError("downloaded despite FFMPEG_AUTO_DOWNLOAD=false")

    monkeypatch.setattr(smartgallery.urllib.request, "urlopen", _refuse)
    monkeypatch.setattr(smartgallery, "FFMPEG_AUTO_DOWNLOAD", False)
    monkeypatch.setattr(smartgallery, "FFPROBE_MANUAL_PATH", "")
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: False)

    assert smartgallery.find_ffprobe_path() is None


# --- saying something while it downloads -----------------------------------


class _Sink(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def _report_into(tty, steps=11):
    sink = _Sink(tty)
    real = smartgallery.sys.stdout
    smartgallery.sys.stdout = sink
    try:
        report = smartgallery.download_progress_reporter(interval=0.0)
        total = 170 * 1024 * 1024
        for step in range(steps):
            report(total * step // (steps - 1), total)
    finally:
        smartgallery.sys.stdout = real
    return sink.getvalue()


def test_a_download_says_how_it_is_going():
    """The gallery blocks on this at startup. 170 MB with nothing on the
    console is a startup that looks hung, which is the one thing that
    makes people kill it."""
    written = _report_into(tty=True)

    assert written.strip(), "nothing was reported at all"
    assert "%" in written
    assert "170" in written


def test_a_log_file_does_not_get_one_enormous_line():
    """Where output is redirected -- a launcher keeping a log, a service
    manager, ComfyUI starting the gallery -- a rewriting line turns into a
    single line thousands of characters long."""
    written = _report_into(tty=False)

    assert "\r" not in written, "carriage returns went into a redirected stream"
    assert written.count("\n") >= 5, written


def test_a_terminal_gets_one_line_that_rewrites_itself():
    """The other half: on a terminal this must not scroll a screenful."""
    written = _report_into(tty=True)

    assert "\r" in written
    assert written.count("\n") <= 1, f"printed {written.count(chr(10))} lines instead of rewriting one"


def test_the_reporter_is_actually_passed_to_the_download(gallery_tree):
    """The reporter existed before this and was not wired up, which is how
    the silent version shipped. Checked in the source because the call is
    on the path that downloads."""

    fn = next(
        (
            node
            for node in ast.walk(gallery_tree)
            if isinstance(node, ast.FunctionDef) and node.name == "find_ffprobe_path"
        ),
        None,
    )
    assert fn is not None, "find_ffprobe_path is gone"

    calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fetch_ffmpeg"
    ]
    assert calls, "nothing fetches ffmpeg any more"
    for call in calls:
        assert any(kw.arg == "progress" for kw in call.keywords), (
            "fetch_ffmpeg is called without a progress reporter, so a "
            "170 MB download says nothing between starting and finishing"
        )
