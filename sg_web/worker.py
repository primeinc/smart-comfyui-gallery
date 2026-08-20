"""The in-process worker: one thread, draining the job table it is told to.

Realtime by construction. Every observable change a turn makes -- claim,
item, terminal state -- is pushed through `publish` the moment it commits,
so a subscribed page renders progress as it happens and nothing anywhere
polls. The job row stays the truth: a reload, a dropped socket or a
restart recovers by reading it back, never by replaying events.

The thread crosses two boundaries that are easy to get wrong:

- sqlite: connections refuse cross-thread use, so the worker owns one,
  opened inside the thread, per-item commits making its progress visible
  to every request connection.
- the event loop: `ChannelsPlugin.publish` is synchronous but puts onto
  an `asyncio.Queue` owned by the loop (litestar-org/litestar@64cd7da
  litestar/channels/plugin.py:139-156), which is not thread safe. The
  `publish` callable handed in here is already bridged with
  `loop.call_soon_threadsafe`; this module never touches the loop.

The `worker` setting is read every pass, so turning it off over HTTP
idles the thread from the next boundary without a restart; jobs are rows
and wait. A crash inside one turn is logged and survived -- the lease
expires and the job is reclaimed -- because a worker that dies on the
first broken job turns one bad file into a stopped library.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time

from db import runner, settings

_logger = logging.getLogger(__name__)

#: Seconds an idle worker waits before looking again, unless woken by a
#: submit. Bounds how stale the `worker` setting can be, not job latency:
#: submits set the wake event and are picked up immediately.
IDLE_WAIT = 1.0


def run(db_path: str, publish, stop: threading.Event, wake: threading.Event) -> None:
    """The thread body: turns until told to stop."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    owner = f"worker-{os.getpid()}"
    try:
        while not stop.is_set():
            turn = None
            if settings.flag(conn, "worker"):
                try:
                    turn = runner.run_next(conn, owner=owner, now=time.time(), clock=time.time, on_progress=publish)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    _logger.exception("a worker turn died; the job's lease will be reclaimed")
            if turn is None:
                wake.wait(IDLE_WAIT)
                wake.clear()
    finally:
        conn.close()
