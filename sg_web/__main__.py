"""`python -m sg_web`: the whole application, one command.

The home directory is created, the database is built from the schema, the
worker starts with the server and stops with it. Every other choice is a
settings row changed over HTTP while it runs (db/settings.py); the flags
here are only what cannot live inside the database they locate: where it
is, and what socket to answer.

It does not build the browser bundles and it refuses to start without
them. `sg_web/static/build` is esbuild's output and IS committed
(.gitignore carries the exception and the reason), so the bundles are
there on a plain clone and the refusal is not the normal path. It stays
because "there is a .js for every template" is a property of the tree,
not of the clone: an entry point added without committing what it built,
or a build directory somebody deleted, produces a server whose pages
render and whose every script 404s -- a picture that would not zoom, keys
that answered nothing, an activity panel that never connected. Legal
according to each part, broken according to the person looking at it.

Nor does it require a particular interpreter. Any python that can read
this file runs the application: one that cannot import a server hands the
command to `.venv` and waits on it (`handover`). There is no environment
to activate and no wrapper to remember, because a launch instruction that
depends on which shell is open is not an instruction.

uvicorn is the server (encode/uvicorn uvicorn/main.py:494-501,
`uvicorn.run(app, host=..., port=...)`); the app is the same object the
test suite exercises through Litestar's TestClient.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

#: How a template asks for one of esbuild's bundles.
_LOADED = re.compile(r"/static/build/([\w.-]+)\.js")

#: What to run to make them exist, named in the refusal.
BUILD_COMMAND = "npm ci && npm run build-web    (or: just web build)"

#: What to run when there is no environment to hand the command to.
RUN_COMMAND = "uv sync    then    uv run python -m sg_web"

#: Set across the handover below so the child never repeats it. A child
#: that still cannot import has a broken environment, not the wrong one.
_HANDED_OVER = "SG_WEB_REEXEC"


def unbuilt(templates: pathlib.Path, static: pathlib.Path) -> list[str]:
    """The bundles the templates load that esbuild has not written.

    Read OUT OF the templates rather than listed here. The entry points
    live in frontend/build.ts, and a second copy of that list in Python
    would be one rename away from passing while the page 404s -- which is
    the exact failure this exists to catch, reintroduced one layer up.
    """
    wanted = sorted(
        {
            found.group(1)
            for page in sorted(templates.glob("*.html"))
            for found in _LOADED.finditer(page.read_text(encoding="utf-8"))
        }
    )
    return [name for name in wanted if not (static / "build" / f"{name}.js").is_file()]


def missing() -> str | None:
    """The first of this command's dependencies this interpreter cannot import.

    Asked before the imports rather than by attempting them, so the answer
    is a name this module can act on instead of a traceback whose last
    line is `import uvicorn`.
    """
    for package in ("uvicorn", "litestar"):
        if importlib.util.find_spec(package) is None:
            return package
    return None


def interpreter() -> pathlib.Path | None:
    """The project environment's python, if `uv sync` has made one.

    A checkout's dependencies live in `.venv` beside pyproject.toml and
    nowhere else, so this is a lookup and not a search.
    """
    venv = HERE.parent / ".venv"
    for relative in ("Scripts/python.exe", "bin/python"):
        found = venv / relative
        if found.is_file():
            return found
    return None


def handover() -> None:
    """Run this command again under the environment that can serve it.

    Which python a person ends up invoking is not a thing this repository
    controls -- an IDE's run button, a shell whose PATH grew a shim, a
    shortcut made months ago -- and every one of them arrives here able to
    read this file and unable to import a server. The environment that can
    is one known path away.

    So it is handed the command instead of the person being handed a rule:
    no venv to activate, no wrapper to remember, no answer that depends on
    which shell is open. `python -m sg_web` is the whole instruction, and
    it is true from anywhere, because the interpreter that cannot serve
    replaces itself with the one that can and keeps the argv it was given.

    Returns only when it cannot: nothing is missing, there is no `.venv`,
    or this process is already a handover's child. The flag is why the
    last case terminates -- a child that still cannot import has a broken
    environment, not the wrong one, and spawning again would only loop.

    A child this process waits on, NOT `os.execv`. execv replaces nothing
    on Windows -- the CRT spawns a second process and destroys this one,
    so the shell is handed exit 0 by a parent that did no work while a
    detached server holds the port with its output going nowhere, Ctrl-C
    reaches nothing, and the next start cannot bind. Measured before this
    line was written: `python -m sg_web --port 8791` returned 0 and
    silent, and PID 26348 was still LISTENING on 8791 afterwards.

    `Popen` and not `run` because a server has no timeout to give -- it
    runs until told to stop, which is the case `run` has no spelling for
    and sglint SG003 correctly refuses. Streams are inherited rather than
    piped, so the child's log is this command's log (SG004: a pipe here
    would be one nobody drains, and uvicorn would block on a full one).
    """
    if missing() is None or os.environ.get(_HANDED_OVER):
        return
    python = interpreter()
    if python is None:
        return
    serving = subprocess.Popen(
        [str(python), "-m", "sg_web", *sys.argv[1:]],
        env={**os.environ, _HANDED_OVER: "1"},
    )
    try:
        ended = serving.wait()
    except KeyboardInterrupt:
        # The console delivered that ^C to the child too, and wait() has
        # already given it a moment to act on it before re-raising here
        # (cpython Lib/subprocess.py:1444-1461). This second wait is for
        # its real exit status, so Ctrl-C reports what the server did.
        ended = serving.wait()
    raise SystemExit(ended)


def main() -> None:
    handover()
    absent_package = missing()
    if absent_package is not None:
        print(
            f"this python has no {absent_package}: {sys.executable}\n"
            f"and {HERE.parent / '.venv'} is not an environment that has it either.\n"
            f"build one:\n\n    {RUN_COMMAND}\n",
            file=sys.stderr,
        )
        raise SystemExit(2)

    import uvicorn

    from sg_web.app import build_app

    parser = argparse.ArgumentParser(prog="sg_web", description="Serve the gallery.")
    parser.add_argument("--home", default=None, help="the run's directory (default ~/.smartgallery)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8777, help="bind port (default 8777)")
    asked = parser.parse_args()
    absent = unbuilt(HERE / "templates", HERE / "static")
    if absent:
        print(
            f"the browser bundles are not built: {', '.join(absent)}\n"
            "every page would load a script that is not there, and the surface would be inert.\n"
            f"build them first:\n\n    {BUILD_COMMAND}\n",
            file=sys.stderr,
        )
        raise SystemExit(2)
    uvicorn.run(build_app(asked.home), host=asked.host, port=asked.port)


if __name__ == "__main__":
    main()
