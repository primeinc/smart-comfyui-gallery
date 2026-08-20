"""Every consumer in the new pairing opens the database through db/connect.py.

`connect()` is where foreign keys, IMMEDIATE writers, busy_timeout, WAL
and the cache are set -- all per-connection, all silently absent on a raw
sqlite3.connect. A consumer that bypasses it runs with sixty-one foreign
keys inert and DEFERRED writers that fail mid-transaction (db/connect.py).

Scope is the new application pairing and its libraries -- `db/`,
`sg_web/`, `metaparse/` and `vision/` -- everything the new stack would
import. The legacy Flask application is being cut over, not policed, and
tooling builds throwaway state on purpose. Three files hold raw connects
by decision:

  connect.py  it IS the one place
  migrate.py  read-only probes and migration targets with deliberate
              isolation choices the live settings would fight
  build.py    an in-memory scratch build for drift checking; no gallery
              file is ever opened
"""

from __future__ import annotations

import ast

from source_tree import REPO_ROOT, parsed, shipped

_NEW_PAIRING = {"db", "sg_web", "metaparse", "vision"}
_DECIDED = {"connect.py", "migrate.py", "build.py"}


def _raw_connects(tree):
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "connect"
            and isinstance(func.value, ast.Name)
            and func.value.id == "sqlite3"
        ):
            calls.append(node)
    return calls


def _new_pairing_sources():
    return [source for source in shipped() if source.relative_to(REPO_ROOT).parts[0] in _NEW_PAIRING]


def test_the_sweep_sees_the_connects_that_are_decided():
    """Control: the decided files hold raw connects. A sweep finding none
    anywhere would report the same clean answer while understanding
    nothing."""
    decided = [s for s in _new_pairing_sources() if s.name in _DECIDED]
    assert len(decided) == len(_DECIDED), f"a decided file is missing from the tree: {decided}"
    assert any(_raw_connects(parsed(s)) for s in decided)


def test_every_consumer_goes_through_connect():
    offenders = {}
    for source in _new_pairing_sources():
        if source.name in _DECIDED:
            continue
        for call in _raw_connects(parsed(source)):
            offenders[f"{source.parent.name}/{source.name}:{call.lineno}"] = "sqlite3.connect"

    assert not offenders, f"{offenders} open the database without db/connect.py's settings"
