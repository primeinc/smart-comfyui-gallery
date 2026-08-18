"""Run a snippet of the shipped JavaScript and read back what it produced.

Several checks here only mean anything if the template's own function is
executed rather than read: an escaper that misses one character looks
identical to a correct one in the source. So the function is lifted out of
the template and handed to node.

Every one of those files had its own copy of the same three steps -- find
node or skip, run it with a JSON argument, assert it exited cleanly and
parse stdout -- differing only in the skip message. A bug in any of those
steps had to be fixed five times, and the timeout only in the copy someone
happened to open.

pytest puts a test file's own directory on sys.path by default
(doc/en/explanation/pythonpath.rst), which is what lets a sibling test
module import this without making tests/ a package.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

# Long enough that a loaded machine does not fail the run, short enough that
# a node that never returns does not hang the suite.
NODE_TIMEOUT = 300


def node_path(reason: str = "node is not on PATH; the shipped JavaScript cannot be run here"):
    """The node executable, or skip the test saying why it matters."""
    found = shutil.which("node")
    if found is None:
        pytest.skip(reason)
    return found


def run_node(script: str, payload):
    """Run `script` with `payload` as argv[1] JSON; return its parsed stdout.

    `check=False` is deliberate: the assertion below reports node's stderr,
    which says what actually went wrong, where CalledProcessError would only
    say that something did.
    """
    done = subprocess.run(
        [node_path(), "-e", script, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=NODE_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)
