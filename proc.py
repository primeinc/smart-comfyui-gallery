"""The one place this repository spawns a process.

stdout and stderr go to temporary FILES, never pipes. A file has no bounded
buffer, so the child cannot block writing to it, no reader thread is needed,
and nothing here waits on a handle a descendant still holds.

That last part is the reason this module exists. `subprocess.run(argv,
capture_output=True, timeout=N)` kills the direct child when the timeout fires
and then calls `communicate()` again, with no timeout, to drain the pipes; that
call waits for every handle on the write end to close, and a grandchild
inherited them. The bound on the wait is real and the drain after it is not.

A timeout here kills the whole process TREE and returns `TIMED_OUT` with
whatever was written by then, which is evidence about the stall. stdin, when
given, is written whole and closed before the wait -- safe for the same reason
the rest is: there is no output pipe to deadlock on.

`LOCAL_SECONDS` is measured. Git plumbing against a clone on this disk runs in
about 25 ms across the 22 consumer clones, and the slowest whole-repository
index -- deepinsight/insightface, 9,478 objects through one `cat-file --batch`
-- takes 749 ms.

`proc_attack.py` holds both claims to a runtime control.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: The exit code a timed-out call reports. 124 is what `timeout(1)` uses, and
#: the callers already read it as "did not finish" rather than as git's own.
TIMED_OUT: Final[int] = 124

#: How long the post-kill wait may take before this gives up on reaping. A
#: bound rather than a blocking wait: the point of this module is that nothing
#: here waits on a process it cannot kill.
REAP_SECONDS: Final[float] = 5.0

#: How long ONE local command may take: a git plumbing call against a clone on
#: this disk, or a probe interpreter. Forty times the slowest measured call.
LOCAL_SECONDS: Final[float] = 30.0


def _kill_tree(child: subprocess.Popen[bytes]) -> None:
    """Kill the child AND everything it spawned.

    `Popen.kill()` reaches the direct child only, which is what leaves an
    orphaned grandchild behind. On Windows `taskkill /T` walks the tree; on
    POSIX the child was given its own process group and the group is signalled.
    """
    if sys.platform == "win32":
        # The system's own taskkill, by absolute path. Resolving it off PATH
        # would let anything earlier on PATH be what kills our subprocesses.
        killer = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32" / "taskkill.exe"
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [str(killer), "/T", "/F", "/PID", str(child.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=REAP_SECONDS,
            )
    else:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(child.pid), 9)
    with contextlib.suppress(OSError, ProcessLookupError):
        child.kill()


def run(
    argv: list[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    stdin: bytes | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes]:
    """One command, its exit code, and its two streams as bytes.

    Bytes, never text: several callers read blobs, and a decode that raises
    inside a reader thread is the other way this call fails to return. A
    command that ran out of time reports `TIMED_OUT` and whatever it had
    written by then.
    """
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        child = subprocess.Popen(
            argv,
            stdout=out,
            stderr=err,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            start_new_session=sys.platform != "win32",
        )
        code = TIMED_OUT
        try:
            if stdin is not None and child.stdin is not None:
                # Whole, then closed. Safe because stdout is a file: the child
                # cannot be blocked writing output while we are writing input.
                try:
                    child.stdin.write(stdin)
                finally:
                    child.stdin.close()
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(child)
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                child.wait(timeout=REAP_SECONDS)
        except OSError:
            _kill_tree(child)
        out.seek(0)
        err.seek(0)
        return code, out.read(), err.read()


@dataclass
class Running:
    """A process still going, and the only way to stop it.

    `stop()` kills the TREE, which is what separates this from holding a
    `Popen`: a server that spawned a worker leaves that worker holding the
    port, and the next run finds the address in use.
    """

    child: subprocess.Popen[bytes]

    @property
    def pid(self) -> int:
        return self.child.pid

    def exited(self) -> int | None:
        """The exit code if it has finished, None while it runs."""
        return self.child.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.child.wait(timeout=timeout)

    def stop(self) -> None:
        if self.child.poll() is None:
            _kill_tree(self.child)
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            self.child.wait(timeout=REAP_SECONDS)


@contextlib.contextmanager
def background(
    argv: list[str],
    *,
    log: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    inherit_streams: bool = False,
) -> Generator[Running]:
    """A long-lived process, stopped with its whole tree on the way out.

    `log` names a file both streams are written to. Never a pipe: a server
    writes one access-log line per request and a pipe nobody drains fills its
    OS buffer and blocks the server mid-run.

    `inherit_streams` hands the child this process's own console instead, for
    a launcher that must stay interactive and forward the terminal's signals.
    """
    stack = contextlib.ExitStack()
    with stack:
        if inherit_streams:
            out: Any = None
        elif log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            out = stack.enter_context(log.open("wb"))
        else:
            out = subprocess.DEVNULL
        child = subprocess.Popen(
            argv,
            stdout=out,
            stderr=subprocess.STDOUT if out is not None else None,
            stdin=subprocess.DEVNULL if not inherit_streams else None,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            start_new_session=sys.platform != "win32" and not inherit_streams,
        )
        running = Running(child)
        try:
            yield running
        finally:
            running.stop()


def text(
    argv: list[str], *, timeout: float, cwd: Path | None = None, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """The same call, decoded. `surrogateescape` so no byte can raise."""
    code, out, err = run(argv, timeout=timeout, cwd=cwd, env=env)
    return (
        code,
        out.decode("utf-8", errors="surrogateescape"),
        err.decode("utf-8", errors="surrogateescape"),
    )
