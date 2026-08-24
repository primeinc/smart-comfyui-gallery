"""The harness's own contract: every connection a test opens is closed.

conftest's `_close_memory_databases` closes what a test left behind, and
`staging.keep` is how a fixture says "this one outlives the test, I close
it myself". That exemption is the only way a connection legitimately
survives its test, so the exemption has to name the connection it means.
"""

from __future__ import annotations

import gc

from db import connect
from tests import staging


def test_a_kept_connection_is_held_rather_than_its_address():
    """An `id()` is unique only among LIVE objects.

    `keep` recorded `id(conn)` in a set of ints. A kept connection is
    closed and dropped by its owner in the ordinary course of things --
    tests/test_the_pages_are_answerable.py rebuilds its master inside a
    test and closes it again at module teardown -- and the moment that
    object dies its address is free. CPython hands the very next
    connection the same address (measured: reused on the first
    allocation), and conftest then read a live, unrelated connection as
    long-lived and never closed it. That connection later reached the
    collector still open, as `ResourceWarning: unclosed database`, blamed
    on whichever test happened to be running at the time.

    So the mark has to hold the connection, not its address: while the
    mark stands the object cannot die, and its address cannot be handed
    to anything else.
    """
    marked = staging.keep(connect.memory())
    address = id(marked)
    marked.close()
    del marked
    gc.collect()

    assert any(id(one) == address for one in staging.LONG_LIVED), (
        "keep() recorded an address rather than the connection: the owner has dropped it, "
        "so that address is now free for the next connection, which conftest will then "
        "refuse to close"
    )


def test_the_exemption_answers_for_a_connection_it_was_never_given():
    """A connection nobody marked is nobody's exemption."""
    conn = connect.memory()
    try:
        assert not staging.is_kept(conn)
    finally:
        conn.close()


def test_a_kept_connection_is_its_own_exemption():
    marked = staging.keep(connect.memory())
    try:
        assert staging.is_kept(marked)
    finally:
        marked.close()


def test_the_currency_monitors_are_closed_by_the_process_that_owns_them(tmp_path):
    """A process-lifetime cache still has to give its handles up.

    `resultset._MONITORS` holds one read-only connection per database
    file, and that is deliberate: `PRAGMA data_version` is comparable
    only across reads on the SAME connection, so nothing may close a
    monitor per request. But "held for the process" is a lifetime with an
    END, and left to the interpreter those globals are torn down and
    every monitor is deleted without close() -- one `unclosed database`
    warning per open database file.
    """
    import warnings

    from db import build, resultset

    path = tmp_path / "gallery.db"
    conn = connect.connect(path)
    try:
        conn.executescript(build.schema_sql())
        conn.commit()
        before = len(resultset._MONITORS)
        resultset.currency(conn)
        assert len(resultset._MONITORS) == before + 1, "the read did not mint a monitor"
    finally:
        conn.close()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        assert resultset.close_monitors() >= 1
    assert resultset._MONITORS == {}, "a closed monitor is forgotten, so the next read mints a fresh one"
