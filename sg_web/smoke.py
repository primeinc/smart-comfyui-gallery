"""Walk the REAL run: every surface and a handful of real pictures, over
the database in the home directory, and say what answered what.

`python -m sg_web.smoke [--home DIR]` -- the check a green test lane over
fresh databases cannot make: the one file that matters, on this build.
Exit 1 on any 5xx, any refusal the page did not ask for, or a picture
route that does not answer bytes.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse

from litestar.testing import TestClient

from sg_web.app import build_app

SURFACES = (
    "/g",
    "/timeline",
    "/operations",
    "/operations/overview",
    "/places",
    "/people",
    "/albums",
    "/keywords",
    "/folders",
    "/stories",
)
PICTURES = 5


def walk(home: str | None) -> list[str]:
    """Every route walked with its status; failures are the lines starting with `FAIL`."""
    told: list[str] = []
    with TestClient(app=build_app(home, worker=False)) as client:
        for path in SURFACES:
            answer = client.get(path, headers={"accept": "text/html"})
            told.append(f"{'FAIL' if answer.status_code >= 500 else 'ok  '} {answer.status_code} {path}")
        grid = client.get("/g", headers={"accept": "text/html"}).text
        slugs = list(dict.fromkeys(re.findall(r'href="/i/([^"?]+)', grid)))[:PICTURES]
        if not slugs:
            told.append("note no pictures in the library; the picture routes were not walked")
        for slug in slugs:
            for path in (f"/i/{slug}", f"/thumb/{slug}", f"/preview/{slug}"):
                answer = client.get(path)
                bytes_route = not path.startswith("/i/")
                bad = answer.status_code >= 500 or (
                    bytes_route
                    and not (answer.status_code == 200 and answer.headers.get("content-type", "").startswith("image/"))
                )
                told.append(f"{'FAIL' if bad else 'ok  '} {answer.status_code} {path}")
        told.extend(crawl(client))
    return told


#: Links a page emits that are bytes or a machine's by design: not landings.
BYTES = ("/media/", "/thumb/", "/preview/", "/avatar/", "/static/")
#: How many links of one SHAPE the crawl follows: a library has thousands
#: of picture pages and they are one shape; the links that matter are
#: the shapes, and every shape is walked.
PER_SHAPE = 3


def _shape(path: str) -> str:
    """A link's shape: its route with the slug or id blanked, and the
    query reduced to its keys -- a facet by its key, never its value.
    `/i/abc?f=context.moment:gte:1&sort=moment` and `/i/def` are two
    shapes: the timeline's link onto a picture walks a different answer
    than the gallery's, and a 500 in one is not in the other."""
    route, _, query = path.partition("?")
    parts = route.split("/")
    blanked = "/".join(
        "*"
        if i > 1
        and (
            parts[i - 1] in ("i", "p", "t", "f", "m", "l", "w", "job", "plans", "renders", "prompts", "snapshots")
            or seg.isdigit()
        )
        else seg
        for i, seg in enumerate(parts)
    )
    keys = sorted(
        {
            f"{name}={urllib.parse.unquote(value).split(':', 1)[0]}" if name == "f" else name
            for name, value in urllib.parse.parse_qsl(query, keep_blank_values=True)
        }
    )
    return blanked + ("?" + "&".join(keys) if keys else "")


def crawl(client) -> list[str]:
    """Every `href` every page emits, followed as a browser, a few of each
    shape: a link that lands a person on JSON, a 4xx or a 5xx is a FAIL
    line. Every shape of link is walked; no shape is sampled away."""
    told: list[str] = []
    seen: set[str] = set()
    walked: dict[str, int] = {}
    queue: list[tuple[str, str]] = [
        (p, "the front")
        for p in ("/g", "/timeline", "/people", "/places", "/albums", "/keywords", "/folders", "/operations")
    ]
    while queue:
        path, emitter = queue.pop(0)
        shape = _shape(path)
        if path in seen or walked.get(shape, 0) >= PER_SHAPE:
            continue
        seen.add(path)
        walked[shape] = walked.get(shape, 0) + 1
        answer = client.get(path, headers={"accept": "text/html,application/xhtml+xml"}, follow_redirects=True)
        kind = answer.headers.get("content-type", "")
        if answer.status_code >= 400 or not kind.startswith("text/html"):
            told.append(
                f"FAIL {answer.status_code} {path} -> {kind.split(';')[0] or 'no content type'}"
                f" (a link onto no page, emitted by {emitter})"
            )
            continue
        for found in re.findall(r'href="([^"#]+)"', answer.text):
            link = found.replace("&amp;", "&")
            if link.startswith("/") and not link.startswith(BYTES) and link not in seen:
                queue.append((link, path))
    told.append(f"ok   crawled {len(seen)} links of {len(walked)} shapes")
    return told


def main() -> None:
    parser = argparse.ArgumentParser(prog="sg_web.smoke", description="Walk the real run's surfaces and pictures.")
    parser.add_argument("--home", default=None, help="the run's directory (default ~/.smartgallery)")
    asked = parser.parse_args()
    lines = walk(asked.home)
    for line in lines:
        print(line)
    failed = [line for line in lines if line.startswith("FAIL")]
    print(f"{len(lines) - len(failed)} ok, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
