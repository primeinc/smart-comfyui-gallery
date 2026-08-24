"""`python -m sg_web`: the whole application, one command.

The home directory is created, the database is built from the schema, the
worker starts with the server and stops with it. Every other choice is a
settings row changed over HTTP while it runs (db/settings.py); the flags
here are only what cannot live inside the database they locate: where it
is, and what socket to answer.

What it does NOT do is build the browser bundles, and it now refuses to
start without them. `sg_web/static/build` is esbuild's output and is not
committed (.gitignore says so on purpose), so a checkout that ran only
the Python half served every template asking for a script that was not
there. The server came up, the pages rendered, and each one was a shell
with nothing behind it: a picture that would not zoom, keys that answered
nothing, an activity panel that never connected. Legal according to each
part, broken according to the person looking at it -- and invisible to a
test suite whose every lane depends on `web::build` having already run.

Two ways to make that impossible: build the assets here, which needs the
node and npm `uv sync` never promised, or refuse to serve half an
application and say exactly what is missing. This is the second.

uvicorn is the server (encode/uvicorn uvicorn/main.py:494-501,
`uvicorn.run(app, host=..., port=...)`); the app is the same object the
test suite exercises through Litestar's TestClient.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import uvicorn

from sg_web.app import build_app

HERE = pathlib.Path(__file__).resolve().parent

#: How a template asks for one of esbuild's bundles.
_LOADED = re.compile(r"/static/build/([\w.-]+)\.js")

#: What to run to make them exist, named in the refusal.
BUILD_COMMAND = "npm ci && npm run build-web    (or: just web build)"


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


def main() -> None:
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
