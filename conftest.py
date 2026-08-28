"""One module asked for is one module's worth of workers.

`pytest.ini` runs the suite on four workers grouped by module, and that
is right for the suite: `--dist loadscope` keeps a module's tests on one
worker because its server is module-scoped, and four measured 194.8s ->
65.8s.

It is exactly wrong for one module. `loadscope` cannot split a module, so
a run naming a single file has ONE scope group: worker gw0 does all of it
while gw1..gw3 start, idle and exit. Every test line in such a run says
`[gw0]` -- which is the tell, and it was on screen for a dozen runs
before anybody read it. The three spare workers are a fresh interpreter
each, importing the whole application to answer nothing.

So a run that names one file drops to in-process. Nothing about how the
tests execute changes -- one worker was already running all of them --
except that three interpreters are no longer built to watch.

Done in `pytest_cmdline_main` rather than `pytest_load_initial_conftests`,
which is where this was written first and did nothing: conftest files are
loaded BY that hook, so this file's own implementation of it is
registered after pluggy has already assembled the callers and is never
invoked. By `pytest_cmdline_main` the conftest is a plugin like any
other -- registered after xdist, so among `tryfirst` implementations
pluggy calls this one before xdist's, which then reads the count and
turns itself off (xdist/plugin.py:326).
"""

from __future__ import annotations

import pytest


def _one_module(args: list[str]) -> bool:
    """True when the arguments name tests in a single file.

    Anything that is not a plain path leaves the four workers alone: this
    only ever answers True for the one shape it is about.
    """
    named = {arg.split("::", 1)[0] for arg in args if not arg.startswith("-")}
    return len(named) == 1 and next(iter(named)).endswith(".py")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    if _one_module(list(config.args)):
        config.option.numprocesses = 0
