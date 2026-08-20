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
loop: nothing yielded, thread still alive, child still running. A handful
of such files and the gallery stops answering at all.

A total timeout would be the wrong instrument -- a two hour film
legitimately streams for two hours -- so the bound is on silence. Output
resets the clock; only a gap ends the stream. test_a_slow_stream_is_left
_alone is the half that matters: it runs far longer than the bound and
must survive.

Nothing here spawns a child or sleeps. The reader needs two things from a
transcode -- `stdout.read1()` and `kill()` -- and it reads its clock from
this module's `time`, so both are supplied directly. pytest's own guidance
is to "prefer patching the reference that your code uses instead of
patching the original object in the standard library"
(pytest doc/en/how-to/monkeypatch.rst:243-247), which is what the clock
fixture below does. Simulated seconds cost nothing, so a six-second
trickle and a sixty-second silence are both instant, and the timings the
assertions rest on are exact rather than raced for.
"""

from __future__ import annotations

import ast
import collections
import threading

import pytest

import smartgallery

# The bound the fake transcodes are given. Small on purpose: the reader's
# watchdog polls at min(1.0, max(stall_timeout / 4, 0.05)) seconds, and that
# poll is the only real time this file spends. The bound itself is measured
# on the fake clock, so its value costs nothing.
STALL = 0.2


class _Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _FakeStdout:
    """Hands out queued chunks, then blocks the way a silent pipe does.

    Each chunk carries how much simulated time passed before it arrived, so
    a stream that trickles for minutes costs the suite nothing.
    """

    def __init__(self, clock):
        self._clock = clock
        self._chunks = collections.deque()
        self.killed = threading.Event()
        self.blocked = threading.Event()

    def feed(self, data, after=0.0):
        self._chunks.append((data, after))

    def read1(self, _size):
        if self._chunks:
            data, after = self._chunks.popleft()
            self._clock.advance(after)
            return data
        # Nothing left to say. A real pipe blocks here until the child is
        # killed and the descriptor closes; waiting on the event returns the
        # instant that happens rather than after any fixed delay.
        self.blocked.set()
        self.killed.wait(10)
        return b""

    read = read1  # the shipped loop used read(); the guarded one uses read1()


class _FakeProcess:
    def __init__(self, clock):
        self.stdout = _FakeStdout(clock)
        self._returncode = None

    def kill(self):
        self._returncode = -9
        self.stdout.killed.set()

    terminate = kill

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        return self._returncode


@pytest.fixture
def clock(monkeypatch):
    """Replace the clock the reader itself reads, not time.monotonic."""
    fake = _Clock()
    monkeypatch.setattr(smartgallery.time, "monotonic", fake.monotonic)
    return fake


@pytest.fixture
def helper():
    streamer = getattr(smartgallery, "stream_media_process", None)
    assert streamer is not None, (
        "smartgallery has no stream_media_process; the stream route reads "
        "the transcoder inline with nothing bounding a silent child"
    )
    return streamer


def _drain(generator):
    """Consume a generator on another thread. Returns (done_event, sink)."""
    collected = bytearray()
    done = threading.Event()

    def run():
        try:
            for chunk in generator:
                collected.extend(chunk)
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    return done, collected


def test_the_shipped_loop_would_have_blocked_for_ever(clock):
    """Control, and the reason for the file.

    This is the loop as it was, run against a transcode that writes nothing
    and does not exit. It has to fail to finish -- otherwise 'the generator
    ended' proves nothing about the tests below, because a plain read loop
    would have satisfied them too."""
    process = _FakeProcess(clock)

    def shipped():
        while True:
            data = process.stdout.read(16384)
            if not data:
                break
            yield data

    done, body = _drain(shipped())

    # Returns the moment the loop reaches its blocking read; no fixed wait.
    assert process.stdout.blocked.wait(10), "the loop never reached a read"
    assert not done.is_set(), (
        "the unguarded loop returned on its own; the transcode is supposed "
        "to sit there silently, so this test is not measuring a stall"
    )
    assert bytes(body) == b""
    assert process.poll() is None, "nothing killed the transcode"

    process.kill()
    assert done.wait(10)


def test_a_stalled_transcode_is_given_up_on(helper, clock):
    """The fix. Same silent transcode, and the stream ends by itself."""
    process = _FakeProcess(clock)

    done, body = _drain(helper(process, "clip.mp4", stall_timeout=STALL))

    assert process.stdout.blocked.wait(10), "the reader never reached a read"
    clock.advance(STALL * 2)

    assert done.wait(10), (
        "the stream never ended after the transcode went silent for longer "
        "than its bound; the thread serving that request is still blocked"
    )
    assert bytes(body) == b""


def test_the_child_is_not_left_running(helper, clock):
    """Giving up on the stream has to end the transcode too, or every
    stalled play leaves an ffmpeg behind holding the file open."""
    process = _FakeProcess(clock)

    done, _body = _drain(helper(process, "clip.mp4", stall_timeout=STALL))

    assert process.stdout.blocked.wait(10)
    clock.advance(STALL * 2)
    assert done.wait(10)

    assert process.poll() is not None, "ffmpeg is still running"


def test_a_slow_stream_is_left_alone(helper, clock):
    """Over-reach guard, and the one that decides whether the fix is safe to
    ship: this transcode runs six times its own stall bound, in chunks far
    smaller than one read. A bound on length rather than on silence cuts it
    off, and so does a clock that watches the reader instead of the pipe."""
    process = _FakeProcess(clock)
    chunks, gap, bound = 60, 0.1, STALL
    for _ in range(chunks):
        process.stdout.feed(b"x" * 64, after=gap)

    done, body = _drain(helper(process, "slow.mp4", stall_timeout=bound))

    assert process.stdout.blocked.wait(10), "the stream never ran dry"
    assert chunks * gap > bound * 5, "the trickle no longer outlasts its bound"
    clock.advance(bound + 1)
    assert done.wait(10), "the slow stream never ended"

    assert bytes(body) == b"x" * (64 * chunks), (
        f"a stream that kept producing was cut off after {len(body)} of {64 * chunks} bytes"
    )


def test_an_ordinary_stream_arrives_whole(helper, clock):
    """Over-reach guard: more than one read's worth, delivered intact."""
    process = _FakeProcess(clock)
    process.stdout.feed(b"y" * 200000)

    done, body = _drain(helper(process, "clip.mp4", stall_timeout=STALL))

    assert process.stdout.blocked.wait(10)
    clock.advance(STALL * 2)
    assert done.wait(10)

    assert bytes(body) == b"y" * 200000, f"got {len(body)} bytes of 200000"


def test_the_default_bound_is_generous():
    """Silence long enough to be a fault, not a slow first frame. A value
    small enough to interrupt a real transcode would be worse than none."""
    assert smartgallery.MEDIA_STREAM_STALL_TIMEOUT >= 30
    assert smartgallery.MEDIA_STREAM_STALL_TIMEOUT <= 600


def test_the_stream_route_reads_through_the_helper(gallery_tree):
    """The loop was inline once and can be again. Whoever puts a read loop
    back into the route gets told here rather than in a bug report about
    the gallery hanging."""

    tree = gallery_tree

    route = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "stream_video"), None
    )
    assert route is not None, "stream_video is gone; this check is stale"

    called = {
        call.func.id for call in ast.walk(route) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "stream_media_process" in called, (
        "stream_video no longer streams through stream_media_process, so nothing bounds a transcoder that goes silent"
    )


def test_every_pipe_the_gallery_reads_is_bounded(gallery_tree):
    """The sweep the original one should have had.

    subprocess.run carries its own timeout, so the existing check covers
    it. Popen does not: whoever reads the pipe is responsible, and there is
    exactly one place doing that. A second one is a second way to hang."""

    tree = gallery_tree

    piped = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            continue
        keywords = {kw.arg for kw in node.keywords if kw.arg}
        if "stdout" in keywords:
            piped.append(node.lineno)

    assert len(piped) == 1, (
        f"subprocess.Popen reading a pipe at lines {piped}. Each one needs a "
        f"clock on it -- a read from a child that stops producing blocks the "
        f"thread that serves the request. Route it through "
        f"stream_media_process or give it its own bound, then update this "
        f"count."
    )
