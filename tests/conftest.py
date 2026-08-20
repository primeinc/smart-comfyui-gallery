"""Shared pytest fixtures, and the environment the suite runs in.

The gallery resolves its settings while it is being imported: BASE_OUTPUT_PATH
and friends are read at module scope, cache directories are created, and argv
is parsed. So the environment has to be right BEFORE anything imports it, and
that ordering is the whole reason this file is shaped the way it is.

It is done in `pytest_configure`, which pytest calls once per initial conftest
"after command line options have been parsed" (hookspec.py:138-152) and before
collection -- so every test module still imports the gallery normally at its
own top level. Doing it in this module's body instead made the order a
property of where the statements happened to sit: hoisting conftest's own
`import smartgallery` above them silently resolved BASE_OUTPUT_PATH to
C:/ComfyUI/output, and the suite would have run against a real library.

`pytest.MonkeyPatch` rather than assigning to os.environ: usable outside a
fixture since 6.2 (monkeypatch.py:120-124), and it undoes the whole lot at
the end of the session instead of leaving the process changed.

The one import that still cannot move is the gallery itself, in the fixtures
below -- this module's body runs before pytest_configure does.
"""

import os
import tempfile

import pytest

_ENVIRONMENT = pytest.MonkeyPatch()
_SESSION_TMP_DIR = tempfile.mkdtemp(prefix="smartgallery_test_")


def pytest_configure(config):
    """Point the gallery at a throwaway directory before anything loads it."""
    # The AI layer is opt-OUT in production; the suite runs the monolith with
    # the explicit opt-out so browsing-path tests exercise the disabled
    # contract, and with auto-provisioning off so no test can ever download.
    # (Default-enabled itself is pinned in tests/test_provision.py.)
    for name, value in (
        ("ENABLE_AI_DAM", "false"),
        ("AI_DAM_AUTO_PROVISION", "false"),
        # A machine with no ffmpeg would otherwise pull a
        # ~170 MB build the first time anything resolved
        # ffprobe. The fetch has its own tests, which drive
        # it through a fake response.
        ("FFMPEG_AUTO_DOWNLOAD", "false"),
        # 'auto' would import torch into this process,
        # breaking the browsing-never-imports-torch guard, and
        # would make clustering tests depend on whatever
        # accelerators the host has. Backend-specific tests
        # call their implementations directly.
        ("AI_DAM_FACE_GRAPH_BACKEND", "numpy"),
    ):
        if not os.environ.get(name):
            _ENVIRONMENT.setenv(name, value)

    # FORCED to this session's temp directory, never merely defaulted. Anyone
    # who runs the gallery has BASE_OUTPUT_PATH exported (run_smartgallery.bat
    # sets it), and defaulting meant the suite inherited it and operated on
    # their real library: tests create files in the gallery root, scan it end
    # to end, and delete rows. A test run must never be able to touch a real
    # collection, whatever the shell says.
    for var, leaf in (
        ("BASE_OUTPUT_PATH", "output"),
        ("BASE_SMARTGALLERY_PATH", "gallery"),
        ("BASE_INPUT_PATH", "input"),
    ):
        _ENVIRONMENT.setenv(var, os.path.join(_SESSION_TMP_DIR, leaf))
        os.makedirs(os.environ[var], exist_ok=True)

    # DELETE_TO decides whether deletions are recoverable; inheriting a real
    # one would scatter test files through the developer's trash folder.
    _ENVIRONMENT.delenv("DELETE_TO", raising=False)

    # Belt and braces: if a future edit ever reintroduces an inherited path,
    # fail before collection with an explanation rather than quietly writing
    # into somebody's library.
    tmp_root = os.path.realpath(tempfile.gettempdir())
    for var in ("BASE_OUTPUT_PATH", "BASE_SMARTGALLERY_PATH", "BASE_INPUT_PATH"):
        resolved = os.path.realpath(os.environ[var])
        if not resolved.startswith(tmp_root):
            raise RuntimeError(
                f"refusing to run the test suite against {var}={resolved!r}: it "
                f"is outside the temp directory ({tmp_root}). Tests create, "
                f"scan and delete files under these paths."
            )


def pytest_unconfigure(config):
    """Leave the process's environment as it was found."""
    _ENVIRONMENT.undo()


def pytest_collection_modifyitems(config, items):
    """Hold back the tests that start a child process.

    A handful of checks can only be answered by another program: is this
    committed file CRLF (git), does this inline script parse (node), is the
    container entrypoint valid shell (bash), does a filename survive a
    redirected pipe on a machine that has not opted into UTF-8 (a second
    interpreter). They are real checks and they stay in the tree, but a
    process start costs more than everything they assert, and a suite
    nobody runs catches nothing.

    So a plain `pytest` starts no child at all, and `RUN_TOOL_TESTS=1`
    runs them -- the same shape as the opt-in in tests/test_real_backends.py.
    Marked rather than deleted, so `-m spawns` still lists exactly which
    checks are being traded away.
    """
    if os.environ.get("RUN_TOOL_TESTS") == "1":
        return

    held_back = pytest.mark.skip(reason="starts a child process; run with RUN_TOOL_TESTS=1")
    for item in items:
        if "spawns" in item.keywords:
            item.add_marker(held_back)


@pytest.fixture(scope="session")
def smartgallery_app():
    """Import smartgallery, initialize its database, and return the module."""
    import smartgallery

    # init_db() opens DATABASE_FILE directly; normally the SQLITE_CACHE_DIR
    # parent is created by initialize_gallery() before init_db() ever runs.
    os.makedirs(smartgallery.SQLITE_CACHE_DIR, exist_ok=True)
    # Same reason: without it every scan in the suite logs a thumbnail
    # failure, which is noise a real thumbnail regression would hide in.
    os.makedirs(smartgallery.THUMBNAIL_CACHE_DIR, exist_ok=True)
    smartgallery.init_db()
    return smartgallery
