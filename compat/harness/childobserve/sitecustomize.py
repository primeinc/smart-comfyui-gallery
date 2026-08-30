"""Record what a CHILD python process opens, and hand it back to the parent.

An audit hook is per-process. The parent's hook cannot see a child's opens, so
every artifact a subprocess resolved was invisible -- and this suite runs its
whole case population in six child processes.

CPython imports `sitecustomize` automatically at startup if it is importable
(Doc/library/site.rst), before any user code, which is the only moment early
enough to catch a load that happens at import time. `observe.child_env` puts
this directory on `PYTHONPATH` and names an output file; a child started any
other way imports this, finds no output file named, and does nothing.

Deliberately standalone: it may not import `compat`, because the child may be
any interpreter with any working directory, and a `sitecustomize` that raises
breaks every process it is attached to.
"""

from __future__ import annotations

import atexit
import json
import os
import sys

#: Where to write, and what counts. Both come from the parent, so a child that
#: was not asked to record cannot accidentally start.
_OUT = os.environ.get("COMPAT_OBSERVE_CHILD_OUT", "")
_ROOTS = [one for one in os.environ.get("COMPAT_OBSERVE_CHILD_ROOTS", "").split(os.pathsep) if one]
_SUFFIX = {".onnx", ".pth", ".pt", ".bin", ".safetensors", ".task", ".ckpt", ".npy", ".npz", ".zip"}

_seen: dict[str, str] = {}


def _under_root(name: str) -> bool:
    try:
        here = os.path.realpath(name)
    except (OSError, ValueError):
        return False
    return any(here.lower().startswith(os.path.realpath(root).lower()) for root in _ROOTS)


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event != "open" or not args:
        return
    try:
        name = str(args[0])
        wanted = os.path.splitext(name)[1].lower() in _SUFFIX and _under_root(name)
        # Written on every NEW artifact, not only at exit: a child killed by
        # the timeout tree-kill never runs atexit, and losing those loads is
        # indistinguishable from opening nothing.
        if wanted and _seen.setdefault(os.path.basename(name), name) is name:
            _write()
    except (OSError, ValueError, TypeError):
        # Never raise out of an audit hook: it fires inside the interpreter,
        # and a hook that throws takes the child down with it.
        return


def _write() -> None:
    try:
        with open(_OUT, "w", encoding="utf-8", newline="") as handle:
            json.dump(
                [{"loader": "open", "identity": name, "path": path} for name, path in sorted(_seen.items())],
                handle,
            )
    except OSError:
        return


if _OUT:
    sys.addaudithook(_audit)
    atexit.register(_write)
