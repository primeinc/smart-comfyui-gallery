"""`python -m sg_web`: the whole application, one command.

A first run needs nothing else -- the home directory is created, the
database is built from the schema, the worker starts with the server and
stops with it. Every other choice is a settings row changed over HTTP
while it runs (db/settings.py); the flags here are only what cannot live
inside the database they locate: where it is, and what socket to answer.

uvicorn is the server (encode/uvicorn uvicorn/main.py:494-501,
`uvicorn.run(app, host=..., port=...)`); the app is the same object the
test suite exercises through Litestar's TestClient.
"""

from __future__ import annotations

import argparse

import uvicorn

from sg_web.app import build_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="sg_web", description="Serve the gallery.")
    parser.add_argument("--home", default=None, help="the run's directory (default ~/.smartgallery)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8777, help="bind port (default 8777)")
    asked = parser.parse_args()
    uvicorn.run(build_app(asked.home), host=asked.host, port=asked.port)


if __name__ == "__main__":
    main()
