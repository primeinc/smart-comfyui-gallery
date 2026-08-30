"""Attack the one process runner. Presence of `_kill_tree` is not proof.

`proc.py` is the single exemption from the repository's ban on spawning, and
every other module now depends on one claim it makes:

    a timeout returns, bounded, and kills the whole process tree

That claim is an ORACLE, and an unattacked oracle is a caption. So this builds
the exact topology that produced the original forty-minute stall --

    this process  ->  child  ->  grandchild holding the inherited output handle

-- and runs BOTH mechanisms against it.

The negative control is the load-bearing half. `subprocess.run(...,
capture_output=True, timeout=N)` must be shown to HANG on this topology here,
on this machine, in this build of CPython. Without that, a green result from
`proc.run` proves only that the fixture is easy: it would pass equally if the
grandchild had never inherited anything and there had never been a defect.

Six things are established, in order:

    1 old_shape_hangs        the negative control -- subprocess.run does NOT return
    2 returns_bounded        proc.run returns, well inside the deadline
    3 reports_timed_out      it reports TIMED_OUT rather than an exit code
    4 child_is_dead          the direct child is gone
    5 grandchild_is_dead     the descendant holding the handle is gone
    6 streams_still_work     success and ordinary nonzero exit still carry bytes

Run through `just proc-attack`.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Final

import proc

REPO: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent

#: How long the child and grandchild sleep. Long enough that neither can exit
#: on its own inside this run, so a process found dead was KILLED.
HOLD_SECONDS: Final[int] = 300

#: The timeout under attack, and the ceiling a bounded return must beat. The
#: ceiling is generous on purpose: the claim is "returns", not "returns fast",
#: and a tight bound would make this a flaky timing test instead of a proof.
ATTACK_TIMEOUT: Final[float] = 2.0
MUST_RETURN_WITHIN: Final[float] = 60.0

#: How long the negative control is given to prove it hangs. It must exceed
#: `ATTACK_TIMEOUT` by a wide margin, or "did not return yet" would just mean
#: "was not asked to yet".
CONTROL_PATIENCE: Final[float] = 25.0


@dataclass
class Probe:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _topology(where: pathlib.Path) -> list[str]:
    """argv for a child that spawns a grandchild INHERITING its streams.

    The grandchild is given no `stdout`/`stderr` of its own, which is what
    makes it inherit the handles the parent was given -- the whole mechanism
    of the original defect. Both write their pid where this process can read
    it, then sleep past the end of the run.
    """
    grandchild = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8'); "
        "sys.stdout.write('grandchild alive\\n'); sys.stdout.flush(); "
        f"time.sleep({HOLD_SECONDS})"
    )
    child = (
        "import os, pathlib, subprocess, sys, time; "
        "here = pathlib.Path(sys.argv[1]); "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], str(here / 'grandchild.pid')]); "
        "(here / 'child.pid').write_text(str(os.getpid()), encoding='utf-8'); "
        "sys.stdout.write('child alive\\n'); sys.stdout.flush(); "
        f"time.sleep({HOLD_SECONDS})"
    )
    return [sys.executable, "-c", child, str(where), grandchild]


def _pid_from(where: pathlib.Path, name: str, patience: float = 15.0) -> int | None:
    """The pid a spawned process recorded, once it has recorded it."""
    target = where / name
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if target.is_file():
            held = target.read_text(encoding="utf-8").strip()
            if held.isdigit():
                return int(held)
        time.sleep(0.05)
    return None


def _alive(pid: int) -> bool:
    """Is this pid still a running process?

    Asked of the OS rather than of a handle we hold: the point is whether
    anything survived, including a process this run no longer has a handle to.
    """
    if sys.platform == "win32":
        killer = pathlib.Path("C:/Windows/System32/tasklist.exe")
        code, out, _ = proc.text([str(killer), "/FI", f"PID eq {pid}", "/NH"], timeout=proc.LOCAL_SECONDS)
        return code == 0 and str(pid) in out
    try:
        import os

        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _reap(pids: list[int | None]) -> None:
    """Leave nothing behind, whatever the probes concluded."""
    for pid in pids:
        if pid is None:
            continue
        if sys.platform == "win32":
            proc.run(
                [str(pathlib.Path("C:/Windows/System32/taskkill.exe")), "/T", "/F", "/PID", str(pid)],
                timeout=proc.LOCAL_SECONDS,
            )
        else:
            import contextlib
            import os
            import signal

            with contextlib.suppress(OSError, ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def control_old_shape_hangs(where: pathlib.Path) -> tuple[Probe, list[int | None]]:
    """THE NEGATIVE CONTROL. `subprocess.run` must not return on this topology.

    The banned shape runs in a CHILD interpreter, reached through `proc.run`.
    If the claim holds that call never comes back, so running it here would
    hang the attack; and a hang this process owns is one it cannot kill.

    The child prints a line only after the call returns. `proc.run` gives it
    ten times the timeout it passes on, so an empty stream is the drain still
    waiting on the grandchild's handle -- the defect, observed.
    """
    scratch = where / "control"
    scratch.mkdir(parents=True, exist_ok=True)
    argv = _topology(scratch)
    old_shape = (
        "import json, subprocess, sys, time;"
        "argv = json.loads(sys.argv[1]);"
        "started = time.monotonic();"
        "exec('try:\\n"
        "    subprocess.run(argv, capture_output=True, text=True, check=False, timeout="
        f"{ATTACK_TIMEOUT})\\n"
        "except Exception:\\n"
        "    pass');"
        "sys.stdout.write('RETURNED after %.1fs' % (time.monotonic() - started))"
    )
    code, out, _ = proc.run([sys.executable, "-c", old_shape, json.dumps(argv)], timeout=CONTROL_PATIENCE)
    child = _pid_from(scratch, "child.pid", patience=1.0)
    grandchild = _pid_from(scratch, "grandchild.pid", patience=1.0)
    hung = code == proc.TIMED_OUT and not out.strip()
    return (
        Probe(
            "1 old_shape_hangs",
            hung,
            f"the banned shape had not returned after {CONTROL_PATIENCE}s"
            if hung
            else f"exit {code}, stdout {out!r}: this topology does not reproduce the defect, "
            f"so a pass from proc.run below proves nothing",
        ),
        [child, grandchild],
    )


def attack_the_runner(where: pathlib.Path) -> tuple[list[Probe], list[int | None]]:
    """`proc.run` against the same topology: bounded, TIMED_OUT, tree dead."""
    scratch = where / "runner"
    scratch.mkdir(parents=True, exist_ok=True)

    seen: dict[str, Any] = {}

    def call() -> None:
        started = time.monotonic()
        seen["code"], seen["out"], _ = proc.run(_topology(scratch), timeout=ATTACK_TIMEOUT)
        seen["seconds"] = time.monotonic() - started

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    child = _pid_from(scratch, "child.pid")
    grandchild = _pid_from(scratch, "grandchild.pid")
    worker.join(timeout=MUST_RETURN_WITHIN)

    if worker.is_alive():
        return (
            [
                Probe("2 returns_bounded", False, f"proc.run HAS NOT RETURNED after {MUST_RETURN_WITHIN}s"),
                Probe("3 reports_timed_out", False, "no result: the call never returned"),
                Probe("4 child_is_dead", False, "no result: the call never returned"),
                Probe("5 grandchild_is_dead", False, "no result: the call never returned"),
            ],
            [child, grandchild],
        )

    # Killing is asynchronous: the handles close when the OS finishes, so a
    # bounded settle here measures "did it die", not "had it died by the
    # instant the call returned".
    deadline = time.monotonic() + proc.REAP_SECONDS + 5.0
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in (child, grandchild) if pid is not None):
            break
        time.sleep(0.1)

    took = float(seen.get("seconds", -1.0))
    return (
        [
            Probe(
                "2 returns_bounded",
                0 <= took < MUST_RETURN_WITHIN,
                f"returned after {took:.1f}s against a {MUST_RETURN_WITHIN}s ceiling",
            ),
            Probe(
                "3 reports_timed_out",
                seen.get("code") == proc.TIMED_OUT,
                f"exit code {seen.get('code')} (TIMED_OUT is {proc.TIMED_OUT})",
            ),
            Probe(
                "4 child_is_dead",
                child is not None and not _alive(child),
                f"child pid {child}: alive={None if child is None else _alive(child)}",
            ),
            Probe(
                "5 grandchild_is_dead",
                grandchild is not None and not _alive(grandchild),
                f"grandchild pid {grandchild}: alive={None if grandchild is None else _alive(grandchild)}",
            ),
        ],
        [child, grandchild],
    )


def probe_streams_still_work() -> Probe:
    """A runner that only survives timeouts is not a runner.

    Ordinary success and ordinary FAILURE both have to carry their bytes back,
    on the stream they were written to, with the real exit code -- otherwise
    every lane reading `err` on a nonzero exit silently reads nothing.
    """
    ok_code, ok_out, ok_err = proc.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('to-stdout'); sys.stderr.write('to-stderr')"],
        timeout=proc.LOCAL_SECONDS,
    )
    bad_code, _, bad_err = proc.run(
        [sys.executable, "-c", "import sys; sys.stderr.write('the reason'); sys.exit(3)"],
        timeout=proc.LOCAL_SECONDS,
    )
    held = (
        ok_code == 0
        and ok_out == b"to-stdout"
        and ok_err == b"to-stderr"
        and bad_code == 3
        and bad_err == b"the reason"
    )
    return Probe(
        "6 streams_still_work",
        held,
        f"success=({ok_code}, {ok_out!r}, {ok_err!r}); failure=({bad_code}, {bad_err!r})",
    )


def run_all() -> list[Probe]:
    leftovers: list[int | None] = []
    with tempfile.TemporaryDirectory(prefix="proc_attack_") as raw:
        where = pathlib.Path(raw)
        try:
            control, pids = control_old_shape_hangs(where)
            leftovers += pids
            attacks, pids = attack_the_runner(where)
            leftovers += pids
            return [control, *attacks, probe_streams_still_work()]
        finally:
            # The control's tree is still alive by construction -- that IS the
            # control -- so it is killed here rather than left holding handles
            # on a directory this is about to remove.
            _reap(leftovers)
            time.sleep(0.5)


def main() -> int:
    probes = run_all()
    print("process runner controls\n")
    for one in probes:
        print(f"{one.mark} {one.name}")
        print(f"       {one.detail}")

    failing = [one.name for one in probes if not one.held]
    print(f"\n{len(probes)} probes, {len(failing)} failing: {failing or 'none'}")
    if failing:
        print("proc.run is NOT shown to defeat the descendant-held-handle deadlock")

    where = REPO / "compat" / "generated"
    where.mkdir(parents=True, exist_ok=True)
    target = where / "proc_controls.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps({"probes": [vars(one) for one in probes], "failing": failing}, indent=2, sort_keys=True)
        )
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
