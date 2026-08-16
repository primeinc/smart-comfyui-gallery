"""Shared pytest fixtures.

Sets BASE_OUTPUT_PATH / BASE_SMARTGALLERY_PATH / BASE_INPUT_PATH to a
session-scoped temp directory at *conftest import time* -- before any test
module has a chance to `import smartgallery` -- so importing the monolith
(which reads these env vars, creates cache directories, and parses CLI args
at module scope) is safe everywhere in the suite.
"""

import os
import tempfile

import pytest

_SESSION_TMP_DIR = tempfile.mkdtemp(prefix="smartgallery_test_")

# The AI layer is opt-OUT in production; the suite runs the monolith with
# the explicit opt-out so browsing-path tests exercise the disabled
# contract, and with auto-provisioning off so no test can ever download.
# (Default-enabled itself is pinned in tests/test_provision.py.)
os.environ.setdefault('ENABLE_AI_DAM', 'false')
os.environ.setdefault('AI_DAM_AUTO_PROVISION', 'false')

# Pin the face-graph backend: 'auto' would import torch into this process,
# breaking the browsing-never-imports-torch guard, and would make clustering
# tests depend on whatever accelerators the host has. Backend-specific tests
# call their implementations directly (the torch one in a subprocess).
os.environ.setdefault('AI_DAM_FACE_GRAPH_BACKEND', 'numpy')

# The gallery paths are FORCED to this session's temp directory, never
# merely defaulted. Anyone who runs the gallery has BASE_OUTPUT_PATH
# exported (run_smartgallery.bat sets it), and with setdefault the suite
# inherited it and operated on their real library: tests create files in
# the gallery root, scan it end to end, and delete rows. A test run must
# never be able to touch a real collection, whatever the shell says.
os.environ['BASE_OUTPUT_PATH'] = os.path.join(_SESSION_TMP_DIR, 'output')
os.environ['BASE_SMARTGALLERY_PATH'] = os.path.join(_SESSION_TMP_DIR, 'gallery')
os.environ['BASE_INPUT_PATH'] = os.path.join(_SESSION_TMP_DIR, 'input')
# DELETE_TO decides whether deletions are recoverable; inheriting a real
# one would scatter test files through the developer's trash folder.
os.environ.pop('DELETE_TO', None)

for _var in ('BASE_OUTPUT_PATH', 'BASE_SMARTGALLERY_PATH', 'BASE_INPUT_PATH'):
    os.makedirs(os.environ[_var], exist_ok=True)

# Belt and braces: if a future edit ever reintroduces an inherited path,
# fail at collection with an explanation rather than quietly writing into
# somebody's library.
_TMP_ROOT = os.path.realpath(tempfile.gettempdir())
for _var in ('BASE_OUTPUT_PATH', 'BASE_SMARTGALLERY_PATH', 'BASE_INPUT_PATH'):
    _resolved = os.path.realpath(os.environ[_var])
    if not _resolved.startswith(_TMP_ROOT):
        raise RuntimeError(
            f"refusing to run the test suite against {_var}={_resolved!r}: it is "
            f"outside the temp directory ({_TMP_ROOT}). Tests create, scan and "
            f"delete files under these paths.")


@pytest.fixture(autouse=True)
def _isolate_vector_generations():
    """The vector-store generation registry is process-global (shared by
    service and worker instances); tests use throwaway DBs, so leak a
    generation across tests and a same-stamp collision serves another
    test's vectors."""
    from smartgallery_ai import vectors

    with vectors._GEN_LOCK:
        vectors._GENERATIONS.clear()
    vectors.set_writer_active(False)
    yield
    with vectors._GEN_LOCK:
        vectors._GENERATIONS.clear()
    vectors.set_writer_active(False)


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
