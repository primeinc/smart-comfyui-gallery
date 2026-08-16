"""Playing a video must not be able to cost a server thread for good.

Every other call to ffmpeg and ffprobe was given a timeout, and the sweep
that keeps them honest looks at subprocess.run, check_output and call. The
stream route uses Popen, so the sweep stepped straight over the one call
site where a stall costs the most -- and it is the same call site the
timeouts were written for: "on a truncated or malformed file, on a path
that goes away mid-read, on network storage that stops answering."

There, a stuck ffmpeg does not fail the request. It writes nothing, the
read blocks, and the thread blocked in it is the one serving the request,
so nobody is left to notice. The clean-up runs when the generator is
closed, which a blocked thread never reaches. Measured against the shipped
loop: six seconds in, nothing yielded, thread still alive, child still
running. A handful of such files and the gallery stops answering at all.

A total timeout would be the wrong instrument -- a two hour film
legitimately streams for two hours -- so the bound is on silence. Output
resets the clock; only a gap ends the stream. test_a_slow_stream_is_left
_alone is the half that matters: it runs far longer than the bound and
must survive.

The shipped loop lived inside a closure inside the route, where nothing
could reach it, so the tests below take hold of the reader by name. That
means against the old build they report the missing seam rather than the
hang itself -- so the hang is proved separately, by
test_the_shipped_loop_would_have_blocked_for_ever, which rebuilds that
exact loop and must fail to finish. It passes on both builds by design:
it is the control that makes "the stream ended" mean something.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

import smartgallery


def _child(body):
    return [sys.executable, "-c", body]


SILENT = "import time; time.sleep(60)"

# 60 chunks over ~6s. Small chunks on purpose: read() waits for a full 16KB
# before returning anything, so a stream like this reads as silent to any
# clock watching the reader rather than the pipe.
TRICKLE = ("import sys, time\n"
           "for _ in range(60):\n"
           "    sys.stdout.buffer.write(b'x' * 64)\n"
           "    sys.stdout.buffer.flush()\n"
           "    time.sleep(0.1)\n")

BURST = ("import sys\n"
         "sys.stdout.buffer.write(b'y' * 200000)\n"
         "sys.stdout.buffer.flush()\n")


def _spawn(body):
    return subprocess.Popen(_child(body), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)


def _drain(generator, limit):
    """Consume a generator on another thread. Returns (finished, bytes)."""
    collected = bytearray()
    done = threading.Event()

    def run():
        try:
            for chunk in generator:
                collected.extend(chunk)
        finally:
            done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    finished = done.wait(timeout=limit)
    return finished, bytes(collected)


@pytest.fixture()
def helper():
    streamer = getattr(smartgallery, "stream_media_process", None)
    assert streamer is not None, (
        "smartgallery has no stream_media_process; the stream route reads "
        "the transcoder inline with nothing bounding a silent child")
    return streamer


def test_the_shipped_loop_would_have_blocked_for_ever():
    """Control, and the reason for the file.

    This is the loop as it was, run against a child that writes nothing and
    does not exit. It has to fail to finish -- otherwise 'the generator
    ended' proves nothing about the tests below, because a plain read loop
    would have satisfied them too."""
    process = _spawn(SILENT)

    def shipped():
        try:
            while True:
                data = process.stdout.read(16384)
                if not data:
                    break
                yield data
        finally:
            process.terminate()

    try:
        finished, body = _drain(shipped(), limit=4)

        assert not finished, (
            "the unguarded loop returned on its own; the child is supposed "
            "to sit there silently, so this test is not measuring a stall")
        assert body == b""
        assert process.poll() is None, "the child exited by itself"
    finally:
        process.kill()
        process.wait(timeout=5)


def test_a_stalled_transcode_is_given_up_on(helper):
    """The fix. Same silent child, and the stream has to end by itself."""
    process = _spawn(SILENT)

    try:
        started = time.monotonic()
        finished, body = _drain(helper(process, "clip.mp4", stall_timeout=1),
                                limit=15)
        elapsed = time.monotonic() - started

        assert finished, (
            f"the stream had not ended {elapsed:.1f}s after the child went "
            f"silent; the thread serving that request is still blocked")
        assert body == b""
        assert elapsed < 10, f"it took {elapsed:.1f}s to give up on a 1s bound"
    finally:
        process.kill()
        process.wait(timeout=5)


def test_the_child_is_not_left_running(helper):
    """Giving up on the stream has to end the transcode too, or every
    stalled play leaves an ffmpeg behind holding the file open."""
    process = _spawn(SILENT)

    try:
        _drain(helper(process, "clip.mp4", stall_timeout=1), limit=15)

        for _ in range(50):
            if process.poll() is not None:
                break
            time.sleep(0.1)

        assert process.poll() is not None, "ffmpeg is still running"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_a_slow_stream_is_left_alone(helper):
    """Over-reach guard, and the one that decides whether the fix is safe
    to ship: this child runs about six seconds, six times its own stall
    bound, in chunks far smaller than one read. A bound on length rather
    than on silence cuts it off, and so does a clock that watches the
    reader instead of the pipe."""
    process = _spawn(TRICKLE)

    try:
        finished, body = _drain(helper(process, "slow.mp4", stall_timeout=1),
                                limit=30)

        assert finished, "the slow stream never ended"
        assert body == b"x" * (64 * 60), (
            f"a stream that kept producing was cut off after "
            f"{len(body)} of {64 * 60} bytes")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_an_ordinary_stream_arrives_whole(helper):
    """Over-reach guard: more than one read's worth, delivered intact."""
    process = _spawn(BURST)

    try:
        finished, body = _drain(helper(process, "clip.mp4", stall_timeout=5),
                                limit=30)

        assert finished
        assert body == b"y" * 200000, f"got {len(body)} bytes of 200000"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_the_default_bound_is_generous():
    """Silence long enough to be a fault, not a slow first frame. A value
    small enough to interrupt a real transcode would be worse than none."""
    assert smartgallery.MEDIA_STREAM_STALL_TIMEOUT >= 30
    assert smartgallery.MEDIA_STREAM_STALL_TIMEOUT <= 600


def test_the_stream_route_reads_through_the_helper():
    """The loop was inline once and can be again. Whoever puts a read loop
    back into the route gets told here rather than in a bug report about
    the gallery hanging."""
    import ast
    import io
    import pathlib

    source = pathlib.Path(smartgallery.__file__)
    tree = ast.parse(io.open(source, encoding="utf-8").read())

    route = next((node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "stream_video"), None)
    assert route is not None, "stream_video is gone; this check is stale"

    called = {call.func.id for call in ast.walk(route)
              if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}
    assert "stream_media_process" in called, (
        "stream_video no longer streams through stream_media_process, so "
        "nothing bounds a transcoder that goes silent")


def test_every_pipe_the_gallery_reads_is_bounded():
    """The sweep the original one should have had.

    subprocess.run carries its own timeout, so the existing check covers
    it. Popen does not: whoever reads the pipe is responsible, and there is
    exactly one place doing that. A second one is a second way to hang."""
    import ast
    import io
    import pathlib

    source = pathlib.Path(smartgallery.__file__)
    tree = ast.parse(io.open(source, encoding="utf-8").read())

    piped = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Popen"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"):
            continue
        keywords = {kw.arg for kw in node.keywords if kw.arg}
        if "stdout" in keywords:
            piped.append(node.lineno)

    assert len(piped) == 1, (
        f"subprocess.Popen reading a pipe at lines {piped}. Each one needs a "
        f"clock on it -- a read from a child that stops producing blocks the "
        f"thread that serves the request. Route it through "
        f"stream_media_process or give it its own bound, then update this "
        f"count.")
