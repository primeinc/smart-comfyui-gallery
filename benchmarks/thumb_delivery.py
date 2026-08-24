"""What one gallery page costs to deliver, per thumbnail.

The derivative cache is already content-addressed and immutable on disk
(`vision/thumbs.py path_for`: `<sha[:2]>/<sha>.webp`, keyed on
`content_sha256`, safe to delete because nothing in it cannot be
recomputed). That is the same shape PhotoPrism stores thumbnails in --
`thumbPath/h[0]/h[1]/h[2]/<hash>_<w>x<h>_<method>.<fmt>`,
refs/photoprism/photoprism/internal/thumb/create.go:24-50.

What was NOT the same shape was the delivery. Every 64-pixel cell went
back through the semantic application: open a SQLite connection, resolve
a slug to an entity, check whether the slug is retired, read the file's
kind and content hash, build the cache path, stat it, then read the whole
file into memory and hand it back with no cache headers at all -- so the
browser asked again on the next page view, and the one after that.

This measures that, so the change can be judged rather than asserted:

    app requests      how many reach the application at all
    connections       SQLite connections opened per page view
    bytes             what crosses the wire
    cacheable         whether a second view can cost nothing

Run through `just bench thumbs-delivery`.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EVIDENCE = REPO / "benchmarks" / "results" / "thumb_delivery.json"

#: One page of a real gallery. The default page size is 60; this is what
#: a person actually loads when they open the application.
PICTURES = 60

_SRC = re.compile(r'<img src="([^"]+)"')


def _library(root: pathlib.Path) -> None:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    root.mkdir(parents=True, exist_ok=True)
    for i in range(PICTURES):
        info = PngInfo()
        info.add_text("parameters", f"a picture number {i}\nSteps: 20, Seed: {i}, Model: bench")
        # Big enough that the derivative is a real resize rather than a
        # copy, so the bytes measured are the bytes a person receives.
        Image.new("RGB", (1600, 1200), (20 + i % 200, 90, 140)).save(root / f"p{i:03d}.png", pnginfo=info)


def _counted_connections():
    """Count every SQLite connection the application opens.

    Patched at `db.connect.connect`, which every route reaches through --
    so the number is what the application really did, not what this file
    believes about it.
    """
    from db import connect

    seen = {"opened": 0}
    real = connect.connect

    def counting(*args, **kwargs):
        seen["opened"] += 1
        return real(*args, **kwargs)

    connect.connect = counting
    return seen, lambda: setattr(connect, "connect", real)


def _measure(client, page_html: str) -> dict:
    """Fetch every thumbnail the page asks for, as a browser would."""
    sources = [one for one in _SRC.findall(page_html) if "/thumb" in one or "/t/" in one]
    counted, restore = _counted_connections()
    started = time.perf_counter()
    total = 0
    cacheable = 0
    statuses: dict[int, int] = {}
    for src in sources:
        answered = client.get(src)
        statuses[answered.status_code] = statuses.get(answered.status_code, 0) + 1
        total += len(answered.content)
        control = answered.headers.get("cache-control", "")
        if "immutable" in control or "max-age" in control:
            cacheable += 1
    spent = time.perf_counter() - started
    restore()
    return {
        "thumbnails": len(sources),
        "app_requests": len(sources),
        "connections_opened": counted["opened"],
        "bytes": total,
        "cacheable": cacheable,
        "ms": round(spent * 1000, 1),
        "statuses": statuses,
        "sample_url": sources[0] if sources else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="What one gallery page costs to deliver.")
    parser.add_argument("--label", default="current", help="what this measurement is of")
    asked = parser.parse_args()

    # One line per request, sixty times a second while the jobs drain,
    # is not evidence -- it is 640KB of log around the number.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from litestar.testing import TestClient

    from sg_web.app import build_app

    scratch = pathlib.Path(tempfile.mkdtemp())
    root = scratch / "pics"
    _library(root)

    # The real worker, because the jobs that fill the derivative cache
    # are what makes this a measurement of DELIVERY rather than of
    # rendering. `worker=False` leaves them queued for ever.
    app = build_app(str(scratch / "run"))
    with TestClient(app=app) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        swept = client.post(f"/roots/{made['id']}/scan").json()
        print(f"scanned {swept['added']}")
        client.post("/jobs/ingest")
        _drained(client)
        # WARM. A cold cache measures rendering, which is the precache
        # job's business and not delivery's; the steady state a person
        # browses in is one where the derivatives exist.
        client.post("/jobs/thumbs")
        _drained(client)

        counted, restore = _counted_connections()
        started = time.perf_counter()
        page = client.get("/g").text
        shell_ms = round((time.perf_counter() - started) * 1000, 1)
        shell_connections = counted["opened"]
        restore()

        told = _measure(client, page)
        told["label"] = asked.label
        told["shell_ms"] = shell_ms
        told["shell_connections"] = shell_connections
        told["connections_per_page_view"] = shell_connections + told["connections_opened"]

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    held = []
    if EVIDENCE.exists():
        with EVIDENCE.open(encoding="utf-8") as sheet:
            held = json.load(sheet)
    held = [one for one in held if one.get("label") != asked.label]
    held.append(told)
    with EVIDENCE.open("w", encoding="utf-8", newline="\n") as sheet:
        json.dump(held, sheet, indent=2)

    print(json.dumps(told, indent=2))
    print(f"\n-> {EVIDENCE}")
    return 0


def _drained(client, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while True:
        running = [job["id"] for job in client.get("/jobs").json() if job["state"] in ("queued", "running")]
        if not running:
            return
        if time.time() > deadline:
            raise RuntimeError(f"jobs still running: {running}")
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
