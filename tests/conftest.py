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

os.environ.setdefault('BASE_OUTPUT_PATH', os.path.join(_SESSION_TMP_DIR, 'output'))
os.environ.setdefault('BASE_SMARTGALLERY_PATH', os.path.join(_SESSION_TMP_DIR, 'gallery'))
os.environ.setdefault('BASE_INPUT_PATH', os.path.join(_SESSION_TMP_DIR, 'input'))

for _var in ('BASE_OUTPUT_PATH', 'BASE_SMARTGALLERY_PATH', 'BASE_INPUT_PATH'):
    os.makedirs(os.environ[_var], exist_ok=True)


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
    smartgallery.init_db()
    return smartgallery
