"""An executor that runs submissions in this process.

The scan fans work out to a ProcessPoolExecutor. A test that let it do
that would start real interpreters, and on Windows each one re-imports the
test runner -- so every test touching a scan substitutes this instead.

It was copy-pasted into twenty-eight test files, identical in all but the
docstring. One definition means a fix reaches all of them; before this, it
reached whichever copy someone happened to edit.

pytest puts a test file's own directory on sys.path under the default
prepend import mode (doc/en/explanation/pythonpath.rst), which is what
makes `from inline_executor import InlineExecutor` work from a sibling
test module without making tests/ a package.
"""

from __future__ import annotations

import concurrent.futures
import logging

_logger = logging.getLogger(__name__)


class InlineExecutor:
    """Runs each submission immediately, in-process."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            # Mirrors the executor contract: whatever the work raised is
            # carried on the future and re-raised by .result(), so nothing
            # is swallowed here -- it is handed to the caller instead.
            # Logged as well, because concurrent.futures says nothing about
            # a future whose exception is never retrieved.
            _logger.debug("an inline submission raised", exc_info=True)
            future.set_exception(exc)
        return future
