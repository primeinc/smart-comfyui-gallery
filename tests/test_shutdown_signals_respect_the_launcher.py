"""Starting the gallery so it survives the terminal closing has to work.

The gallery installs a shutdown handler for SIGINT, SIGTERM and SIGHUP.
That handler is not a graceful stop -- it kills the whole process group
with SIGKILL and exits -- and it was installed unconditionally.

nohup exists to make a process survive its terminal, and the way it does
that is to ignore the hangup before starting it:

    /* Windows does not support SIGHUP.  */
    #ifdef SIGHUP
      signal (SIGHUP, SIG_IGN);
    #endif
      char **cmd = argv + optind;
      execvp (*cmd, cmd);
                                    -- coreutils/src/nohup.c

The gallery inherits that ignore and then replaces it. So
`nohup python smartgallery.py &`, which is how a gallery is kept running
over SSH, died the moment the session ended -- killed by the handler it
had installed over the protection it was given. Measured, using SIGINT
because Windows has no SIGHUP:

    SIGINT   launcher set it to        : 1
    SIGINT   after installing a handler: <function a_handler ...>
    SIGINT   with the guard            : 1

A shell backgrounding a job without job control ignores SIGINT for it for
the same reason, and got the same treatment.

getsignal reports SIG_IGN for a signal that is already ignored, so the
ones to leave alone can be asked for rather than guessed at. Nothing here
decides which signals matter; whoever started the gallery did.

Not changed here: the handler itself still kills the process group rather
than stopping in an orderly way, and its stated reason -- releasing the
port -- is something both servers already handle by asking for
SO_REUSEADDR before they bind (see test_port_check_matches_the_server).
Making it graceful is a separate change with its own risks, and is not
this one.
"""

from __future__ import annotations

import os
import signal

import pytest

import smartgallery


def _handler(signum, frame):  # never actually delivered in these tests
    raise AssertionError("the test handler ran")


@pytest.fixture()
def restore_signals():
    """Put every handler back, whatever the test did."""
    names = [n for n in ("SIGINT", "SIGTERM", "SIGHUP")
             if getattr(signal, n, None) is not None]
    saved = {n: signal.getsignal(getattr(signal, n)) for n in names}
    yield names
    for name, previous in saved.items():
        signal.signal(getattr(signal, name), previous)


def test_a_signal_the_launcher_ignores_is_left_ignored(restore_signals):
    """The bug: nohup's protection was overwritten by the thing it was
    protecting."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    installed, left_alone = smartgallery.install_shutdown_signals(
        _handler, names=["SIGINT"])

    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN, (
        "the gallery took over a signal the launcher had ignored, so "
        "nohup no longer keeps it running")
    assert left_alone == ["SIGINT"]
    assert installed == []


def test_a_signal_nobody_touched_is_taken(restore_signals):
    """Over-reach guard, and the ordinary case: run from a terminal,
    Ctrl+C must still stop the gallery."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    installed, left_alone = smartgallery.install_shutdown_signals(
        _handler, names=["SIGINT"])

    assert signal.getsignal(signal.SIGINT) is _handler
    assert installed == ["SIGINT"]
    assert left_alone == []


def test_each_signal_is_decided_on_its_own(restore_signals):
    """One ignored signal must not stop the others being taken -- a
    backgrounded job ignores SIGINT while SIGTERM still has to stop it."""
    if "SIGTERM" not in restore_signals:
        pytest.skip("no SIGTERM on this platform")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    installed, left_alone = smartgallery.install_shutdown_signals(
        _handler, names=["SIGINT", "SIGTERM"])

    assert installed == ["SIGTERM"]
    assert left_alone == ["SIGINT"]
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is _handler


def test_a_signal_this_platform_lacks_is_skipped(restore_signals):
    """Windows has no SIGHUP; asking for it must not stop the rest being
    installed, and must not raise."""
    installed, left_alone = smartgallery.install_shutdown_signals(
        _handler, names=["SIGHUP", "SIGTERM"])

    expected = [n for n in ("SIGHUP", "SIGTERM")
                if getattr(signal, n, None) is not None]
    assert installed == expected, (installed, left_alone)


def test_the_default_set_covers_what_it_used_to(restore_signals):
    """The signals it takes must not have quietly shrunk: SIGINT and
    SIGTERM everywhere, and SIGHUP where it exists."""
    installed, left_alone = smartgallery.install_shutdown_signals(_handler)

    taken = set(installed) | set(left_alone)
    assert {"SIGINT", "SIGTERM"} <= taken, taken
    if hasattr(signal, "SIGHUP"):
        assert "SIGHUP" in taken, taken


def test_installing_a_handler_replaces_an_ignore(restore_signals):
    """Control. Everything above rests on the claim that installing a
    handler destroys an inherited SIG_IGN; if that stopped being true
    there would be nothing here to guard."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN

    signal.signal(signal.SIGINT, _handler)

    assert signal.getsignal(signal.SIGINT) is _handler, (
        "signal.signal no longer replaces SIG_IGN, so nohup's protection "
        "was never at risk and these checks guard nothing")


def test_the_startup_path_goes_through_the_guard():
    """The handler is installed under `if __name__ == '__main__'`, which
    no test runs, so the wiring is checked in the source."""
    import ast
    import io
    import pathlib

    source = pathlib.Path(smartgallery.__file__)
    text = io.open(source, encoding="utf-8").read()
    tree = ast.parse(text)

    installs = [node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "signal"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "signal"]

    outside = []
    for call in installs:
        enclosing = [f.name for f in ast.walk(tree)
                     if isinstance(f, ast.FunctionDef)
                     and any(n is call for n in ast.walk(f))]
        if "install_shutdown_signals" not in enclosing:
            outside.append(call.lineno)

    assert outside == [], (
        f"signal.signal called at lines {outside} outside "
        f"install_shutdown_signals, so a signal the launcher chose to "
        f"ignore can be taken over again")
