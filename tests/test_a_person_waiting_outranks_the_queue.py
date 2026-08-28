"""A browser blocked on a cell outranks a queue guessing at the future.

The precache renders several pictures at a time now, which fills every
core -- while GUESSING at what will be wanted later. A person looking at
a page is not guessing. Measured on a grid of 30 misses served while a
precache ran, 4000x3000 throughout:

    precache 1 in flight              1.25s   42 ms/cell
    precache 8, not standing aside    1.81s   60 ms/cell
    precache 8, standing aside        1.53s   51 ms/cell

Half the regression, and the shortfall is honest rather than a bug: a
render already running cannot be interrupted, because there is nothing
to interrupt it with. So a person arriving mid-batch pays the tail of
whatever is in flight and no more.

Two rules the mechanism must never break, which is what is tested here:
the person NEVER waits -- they are the work being prioritised -- and the
queue is only ever slowed, never stopped.
"""

from __future__ import annotations

import threading

import pytest

from vision import derive

pytestmark = pytest.mark.slow


def test_the_queue_holds_off_while_somebody_is_waiting():
    """The whole thing, as an event rather than a stopwatch: a
    speculative render must not BEGIN while a person is blocked."""
    began = threading.Event()
    released = threading.Event()

    def speculative():
        derive.stand_aside()
        began.set()
        released.wait(timeout=10)

    with derive.waited_on():
        worker = threading.Thread(target=speculative, daemon=True)
        worker.start()
        # It is held, not slow: given a person waiting, it cannot start
        # at all, so a slow machine cannot turn this green by accident.
        # A BROKEN gate fires within scheduler latency (single-digit ms);
        # 100ms is margin over noise, not part of the claim.
        assert not began.wait(0.1), "a speculative render started while a person was waiting"

    assert began.wait(timeout=10), "the queue never resumed after the person was served"
    released.set()
    worker.join(timeout=10)


def test_the_person_never_waits():
    """The other direction, and the one that would be a deadlock. The
    queue standing aside must not become the person standing aside."""
    done = threading.Event()

    def person():
        with derive.waited_on():
            pass
        with derive.waited_on():
            pass
        done.set()

    # somebody already waiting, so the gate is closed when this one arrives
    with derive.waited_on():
        who = threading.Thread(target=person, daemon=True)
        who.start()
        assert done.wait(timeout=10), "a person waiting was made to wait for another person"
    who.join(timeout=10)


def test_two_people_are_both_served_before_the_queue_resumes():
    """Counted rather than flagged: a grid is thirty of these at once,
    and the LAST to finish decides when the queue may carry on."""
    began = threading.Event()
    first = derive.waited_on()
    second = derive.waited_on()
    first.__enter__()
    second.__enter__()
    try:
        worker = threading.Thread(target=lambda: (derive.stand_aside(), began.set()), daemon=True)
        worker.start()
        first.__exit__(None, None, None)
        # 100ms of margin over scheduler noise; a broken count fires in ms.
        assert not began.wait(0.1), "the queue resumed while somebody was still waiting"
    finally:
        second.__exit__(None, None, None)
    assert began.wait(timeout=10)
    worker.join(timeout=10)


def test_a_marker_that_leaks_costs_a_pause_and_not_the_queue():
    """`waited_on` is a context manager and should always be balanced,
    but a queue that can be stopped for ever by one lost decrement is a
    queue with a way to be stopped for ever. Past `patience` the picture
    renders anyway, which is the behaviour there was before any of
    this."""
    held = derive.waited_on()
    held.__enter__()
    try:
        began = threading.Event()

        def speculative():
            # The patience is the test's own dial: 50ms proves "renders
            # anyway once patience runs out" exactly as 200ms did.
            derive.stand_aside(patience=0.05)
            began.set()

        worker = threading.Thread(target=speculative, daemon=True)
        worker.start()
        assert began.wait(timeout=10), "a leaked marker stopped the queue rather than slowing it"
        worker.join(timeout=10)
    finally:
        held.__exit__(None, None, None)


def test_nobody_waiting_costs_nothing():
    """The ordinary case is every render when no page is open, so it has
    to be a set Event and not a sleep."""
    began = threading.Event()
    worker = threading.Thread(target=lambda: (derive.stand_aside(), began.set()), daemon=True)
    worker.start()
    assert began.wait(timeout=5), "the queue stood aside for nobody"
    worker.join(timeout=5)


def test_the_gate_is_left_open_afterwards():
    """Module state: a test or a request that has finished must not leave
    the queue holding its breath."""
    with derive.waited_on():
        pass
    assert derive._WAITING.nobody.is_set()
    assert derive._WAITING.count == 0
