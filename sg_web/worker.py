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
import pathlib
import threading
import time

from db import connect, runner, settings
from sg_web import home

_logger = logging.getLogger(__name__)

#: Seconds an idle worker waits before looking again, unless woken by a
#: submit. Bounds how stale the `worker` setting can be, not job latency:
#: submits set the wake event and are picked up immediately.
IDLE_WAIT = 1.0

#: Seconds between two looks at the schedule table. Not the resolution of
#: a schedule -- those are hours -- but the cost of asking: an idle
#: worker wakes every second, and a query per second for a row that
#: changes twice a year is a poll nobody is waiting for.
SCHEDULE_EVERY = 60.0


def run(db_path: str, publish, stop: threading.Event, wake: threading.Event, publish_event=None) -> None:
    """The thread body: turns until told to stop. `publish` carries the
    progress deltas; `publish_event` carries the ledger (db/ledger.py) --
    every committed row, and the pending reports between them."""
    conn = connect.connect(db_path)
    owner = f"worker-{os.getpid()}"
    try:
        # Hot similarity spaces become resident at boot -- restored from
        # snapshots when they match, rebuilt once when they don't -- so
        # producers upsert into live indexes instead of paying a build
        # on the first job. A failed warm is a slower first job, never a
        # worker that refuses to start.
        try:
            runner.warm_similarity(conn, time.time())
            conn.commit()
        except Exception:
            conn.rollback()
            _logger.exception("similarity warm failed; spaces will build on first use")
        # The home is where the database sits: the worker is handed a
        # path to the file rather than the burrow, and `models_dir`
        # wants the burrow.
        burrow = pathlib.Path(db_path).parent
        looked = 0.0
        while not stop.is_set():
            turn = None
            # What runs without being asked, started on the worker's own
            # turn. Not a timer of its own: a second scheduler is a
            # second thing that can be running while nobody thinks
            # anything is, and the runner is already the only thing that
            # runs jobs. A worker that is off starts nothing, which is
            # what off should mean.
            if settings.flag(conn, "worker") and time.time() - looked >= SCHEDULE_EVERY:
                looked = time.time()
                try:
                    runner.run_schedules(
                        conn,
                        time.time(),
                        models_dir=str(home.models_dir(burrow, settings.value(conn, "models_dir"))),
                        thumbs_dir=str(home.thumbs_dir(burrow)) if settings.flag(conn, "thumbnail_precache") else None,
                    )
                    conn.commit()
                except Exception:
                    # A schedule that cannot start must not take the
                    # worker down with it: the jobs somebody asked for by
                    # hand are the ones that matter more.
                    conn.rollback()
                    _logger.exception("a schedule could not be started")
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
                        on_event=publish_event,
                        should_stop=stop.is_set,
                    )
                    conn.commit()
                except Exception:
                    # run_next already committed a `worker.turn_failed`
                    # row with the traceback before letting this propagate.
                    #
                    # A BUSY database does not arrive here: `run_next`
                    # answers None when it cannot get the writer to claim,
                    # because that is backpressure and not a defect.
                    conn.rollback()
                    from db import similarity

                    similarity.discard_pending(conn)
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
