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
