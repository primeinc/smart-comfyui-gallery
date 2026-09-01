from __future__ import annotations

import atexit
import json
import os
import sys

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

        if wanted and _seen.setdefault(os.path.basename(name), name) is name:
            _write()
    except (OSError, ValueError, TypeError):
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
