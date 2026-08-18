"""The port check must not refuse a port the server would have taken.

Refusing is not a warning here. The gallery prints CRITICAL ERROR: PORT
ALREADY IN USE, tells you to pick another port, and waits for a keypress
before exiting. So a check stricter than the server does not degrade the
start; it prevents it, over a port that would have worked.

The check bound plain. Both servers ask for SO_REUSEADDR before they bind
-- waitress calls set_reuse_addr() in server.py before
bind_server_socket(), werkzeug's sets allow_reuse_address -- and on POSIX
a port still holding connections from a crashed run refuses a plain bind
and accepts that one. Restart after a crash is exactly when someone is
already annoyed.

Measured on win32, where the two agree in all three cases, which is why
this went unseen on the machine it was written on:

    case 1  free port                 plain: ok        reuse: ok
    case 2  another program listening  plain: refused   reuse: refused
    case 3  crashed with a connection  plain: ok        reuse: ok

Rather than encode what SO_REUSEADDR means on each system -- which is
where this sort of fix goes wrong -- the check now asks for exactly what
the server asks for. Then the two cannot disagree, whatever it means
locally. The tests below hold that as a differential: for each case, the
check and a bind performed the way the server performs it must give the
same answer.

There was also a block just before serve() titled FORCE SOCKET REUSE
(LINUX/TMUX FIX) which built a socket, set the options on it, and closed
it without binding. Socket options belong to a socket, so it did nothing
to the one waitress then opened. It is gone; the last test keeps it gone.
"""

from __future__ import annotations

import ast
import contextlib
import socket

import pytest

import smartgallery


def _server_style_bind(port):
    """A bind performed the way waitress performs its own."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with contextlib.suppress(OSError):
            s.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
            )
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


@pytest.fixture
def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("0.0.0.0", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.fixture
def busy_port():
    """A port with something actually listening on it."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", 0))
    holder.listen(1)
    yield holder.getsockname()[1]
    holder.close()


@pytest.fixture
def crashed_port():
    """A port the gallery was serving on when it died with a connection
    still open -- the case the removed block was aiming at."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    accepted, _ = server.accept()
    accepted.sendall(b"hello")
    client.recv(5)
    accepted.close()  # the server side closing first is what leaves it
    client.close()
    server.close()
    return port


def test_a_free_port_is_free(free_port):
    """Control. Without this the check could refuse everything and the
    agreement tests below would still hold."""
    assert smartgallery.check_port_available(free_port) is True


def test_a_port_in_use_is_refused(busy_port):
    """The other control, and the one that matters most: asking for
    SO_REUSEADDR must not turn the check into a rubber stamp. If this ever
    passes, the check has stopped checking and the gallery will start on
    top of something else."""
    assert smartgallery.check_port_available(busy_port) is False


@pytest.mark.parametrize("case", ["free_port", "busy_port", "crashed_port"])
def test_the_check_agrees_with_the_server(case, request):
    """The property, stated as a differential so it holds on any system
    without this file having to know what SO_REUSEADDR means there."""
    port = request.getfixturevalue(case)

    checked = smartgallery.check_port_available(port)
    server_would_bind = _server_style_bind(port)

    assert checked == server_would_bind, (
        f"on a {case.replace('_', ' ')} the check says "
        f"{'available' if checked else 'IN USE'} while the server would "
        f"{'bind' if server_would_bind else 'fail'}. Where the check is the "
        f"stricter one the gallery refuses to start on a working port."
    )


def test_the_check_asks_for_what_the_server_asks_for(gallery_tree):
    """Source-level, because the differential above can only prove
    agreement in the cases this machine can build. On a system where the
    two would part company, the option is the reason."""
    tree = gallery_tree

    function = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "check_port_available"),
        None,
    )
    assert function is not None, "check_port_available is gone"

    attributes = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    assert "SO_REUSEADDR" in attributes, (
        "check_port_available binds without SO_REUSEADDR while both servers "
        "bind with it, so it can refuse a port they would accept"
    )
    assert "bind" in attributes, "it no longer binds anything"


def test_no_socket_options_are_set_on_a_socket_that_is_thrown_away(gallery_tree):
    """The block that was removed: a socket built, configured and closed
    without ever being bound or handed to anything. It read as a fix for
    the restart problem and was not one, so the problem stayed while the
    code said otherwise."""
    tree = gallery_tree

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setsockopt"):
            continue
        target = node.func.value
        if not isinstance(target, ast.Name):
            continue
        # Inside check_port_available the socket is bound straight after,
        # which is the whole point; anywhere else, say where it goes.
        enclosing = [
            f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and any(n is node for n in ast.walk(f))
        ]
        if "check_port_available" in enclosing:
            continue
        offenders.append(node.lineno)

    assert offenders == [], (
        f"setsockopt at lines {offenders} on a socket outside the port "
        f"check. Options belong to the socket they are set on -- if it is "
        f"not the one that gets bound, the call does nothing."
    )
