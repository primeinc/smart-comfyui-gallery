"""The library scan must index files even where multiprocessing cannot run.

`full_sync_database` fans work out to a ProcessPoolExecutor. When that pool
dies -- a corrupted file segfaulting a worker, or an environment that
forbids spawning at all (frozen builds, restricted containers, memory
pressure) -- every pending future raises BrokenProcessPool. Previously that
was reported as "likely due to a corrupted file", nothing was inserted, and
the run still announced "Full scan completed": an empty gallery with no
actionable message. Whatever the pool does not finish is now processed in
this process instead.

The executor is always substituted here: these tests must never spawn real
processes (slow, and on Windows the child re-imports the test runner).

A scan only reaches the pool when the batch is at least
PARALLEL_SCAN_MIN_FILES, so the fixture below lowers that bound to 1. Three
probe files are otherwise under it, and every check here would pass on the
in-process path without the pool being touched at all -- including the two
that exist to prove what happens when it collapses.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image


class _DeadPoolExecutor(InlineExecutor):
    """Every future fails the way a collapsed pool reports itself."""

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        future.set_exception(
            concurrent.futures.process.BrokenProcessPool("A process in the process pool was terminated abruptly")
        )
        return future


@pytest.fixture(autouse=True)
def always_reach_the_pool(smartgallery_app, monkeypatch):
    """Three files is below the threshold that keeps a small scan in this
    process, and every test in this file is about the pool."""
    monkeypatch.setattr(smartgallery_app, "PARALLEL_SCAN_MIN_FILES", 1)


def _purge_probe_rows(smartgallery_app):
    """The gallery database is session-scoped; a scan only processes files
    it does not already know, so each test must start with these absent."""
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE 'scanprobe_%'")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def gallery_files(smartgallery_app):
    """Three uniquely named images inside the gallery root, cleaned up after."""
    root = smartgallery_app.BASE_OUTPUT_PATH
    os.makedirs(os.path.join(root, "scanprobe"), exist_ok=True)
    made = []
    for name, size, colour in (
        ("scanprobe_a.png", (64, 48), (200, 30, 30)),
        ("scanprobe_b.png", (48, 64), (30, 200, 30)),
        (os.path.join("scanprobe", "scanprobe_c.png"), (32, 32), (30, 30, 200)),
    ):
        path = os.path.join(root, name)
        Image.new("RGB", size, colour).save(path)
        made.append(path)
    _purge_probe_rows(smartgallery_app)
    yield made
    for path in made:
        with contextlib.suppress(OSError):
            os.remove(path)
    _purge_probe_rows(smartgallery_app)


def _indexed_probe_names(conn):
    return sorted(r[0] for r in conn.execute("SELECT name FROM files WHERE name LIKE 'scanprobe_%'").fetchall())


def _scan(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
        return _indexed_probe_names(conn)
    finally:
        conn.close()


def test_scan_indexes_files_when_the_pool_works(smartgallery_app, gallery_files, monkeypatch):
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    assert _scan(smartgallery_app) == ["scanprobe_a.png", "scanprobe_b.png", "scanprobe_c.png"]


def test_scan_falls_back_to_sequential_when_the_pool_is_dead(smartgallery_app, gallery_files, monkeypatch, capsys):
    """The regression: this environment used to index nothing at all."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", _DeadPoolExecutor)

    indexed = _scan(smartgallery_app)

    assert indexed == ["scanprobe_a.png", "scanprobe_b.png", "scanprobe_c.png"], (
        "files were skipped when the process pool was unavailable"
    )
    out = capsys.readouterr().out
    assert "one at a time" in out, "no explanation of the fallback was printed"
    assert "corrupted file" not in out, "a pool-level failure is still being blamed on file corruption"


def test_scan_survives_pool_creation_failing_outright(smartgallery_app, gallery_files, monkeypatch):
    """Some environments refuse to create the pool at all, rather than
    failing per-future."""

    def refuse(*_a, **_k):
        raise OSError("spawning is not permitted here")

    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", refuse)
    assert _scan(smartgallery_app) == ["scanprobe_a.png", "scanprobe_b.png", "scanprobe_c.png"]


def test_one_unreadable_file_does_not_cost_the_others(smartgallery_app, gallery_files, monkeypatch):
    """A per-file error is attributed to that file and the scan continues --
    the original fault-tolerance intent, preserved."""
    real_process = smartgallery_app.process_single_file

    def explode_on_b(path, *args, **kwargs):
        if os.path.basename(path) == "scanprobe_b.png":
            raise ValueError("simulated decode failure")
        return real_process(path, *args, **kwargs)

    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "process_single_file", explode_on_b)

    assert _scan(smartgallery_app) == ["scanprobe_a.png", "scanprobe_c.png"]


def test_a_small_scan_never_starts_a_pool(smartgallery_app, gallery_files, monkeypatch):
    """The threshold, and the reason it exists: a pool worker has to import
    this module before it can do anything, which costs about a second
    whether the batch is three files or three thousand.

    The bound is restored to its real value here -- the autouse fixture
    lowers it for every other test in this file -- and any attempt to build
    an executor fails the test rather than being quietly tolerated."""
    monkeypatch.setattr(smartgallery_app, "PARALLEL_SCAN_MIN_FILES", 12)

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "a three-file scan reached for a process pool; each worker would "
            "import the whole module before touching a single picture"
        )

    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", forbidden)

    assert _scan(smartgallery_app) == ["scanprobe_a.png", "scanprobe_b.png", "scanprobe_c.png"], (
        "the in-process path indexed a different set than the pool does"
    )


def test_a_large_enough_scan_still_uses_the_pool(smartgallery_app, gallery_files, monkeypatch):
    """Control for the test above. A threshold that swallowed every scan
    would satisfy it while quietly making the gallery single-threaded."""
    monkeypatch.setattr(smartgallery_app, "PARALLEL_SCAN_MIN_FILES", 2)
    built = []

    class _Counted(InlineExecutor):
        def __init__(self, max_workers=None):
            super().__init__(max_workers)
            built.append(max_workers)

    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", _Counted)

    assert _scan(smartgallery_app) == ["scanprobe_a.png", "scanprobe_b.png", "scanprobe_c.png"]
    assert built, (
        "three files with a bound of two did not reach the pool, so the "
        "threshold has turned scanning sequential at every size"
    )
