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
import threading
import time

from db import connect, runner, settings

_logger = logging.getLogger(__name__)

#: Seconds an idle worker waits before looking again, unless woken by a
#: submit. Bounds how stale the `worker` setting can be, not job latency:
#: submits set the wake event and are picked up immediately.
IDLE_WAIT = 1.0


def run(db_path: str, publish, stop: threading.Event, wake: threading.Event) -> None:
    """The thread body: turns until told to stop."""
    conn = connect.connect(db_path)
    owner = f"worker-{os.getpid()}"
    try:
        # Hot similarity spaces become resident at boot -- restored from
        # snapshots when they match, rebuilt once when they don't -- so
        # producers upsert into live indexes instead of paying a build
        # on the first job. A failed warm is a slower first job, never a
        # worker that refuses to start.
        try:
            runner.warm_similarity(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            _logger.exception("similarity warm failed; spaces will build on first use")
        while not stop.is_set():
            turn = None
            # The flag read is economy -- skip the claim entirely while
            # off. The GUARANTEE is the gate inside the claim itself: a
            # flag read here goes stale in the gap before the claim, and
            # an off-switch committed in that gap must still win.
            if settings.flag(conn, "worker"):
                try:
                    turn = runner.run_next(
                        conn,
                        owner=owner,
                        now=time.time(),
                        gate=("worker", settings.REGISTRY["worker"][0]),
                        clock=time.time,
                        on_progress=publish,
                        should_stop=stop.is_set,
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    _logger.exception("a worker turn died; the job's lease will be reclaimed")
            if turn is None:
                wake.wait(IDLE_WAIT)
                wake.clear()
    finally:
        try:
            # The shutdown sweep: dirty spaces to their CPU snapshots,
            # so the next boot restores instead of rebuilding.
            from db import similarity

            similarity.manager_for(conn).checkpoint_all()
        except Exception:
            _logger.exception("index checkpoint on shutdown failed")
        connect.close(conn)
