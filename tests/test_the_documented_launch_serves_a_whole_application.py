"""The boundary every other test in this suite sits above.

`just test`, `just test-slow`, `just smoke` and `just serve` all depend on
`web::build`, so by the time any of them runs, esbuild has already written
`sg_web/static/build`. They then prove the browser behaves -- truthfully,
and only after a prerequisite nothing in the Python asked for was silently
satisfied.

The README's Run section documents `uv sync` and `uv run python -m sg_web`
and nothing else. On a checkout that has never run npm, that path started
a server whose every page loaded a script returning 404: the pictures
rendered, and nothing about them worked. One missing bundle wearing as
many hats as there are surfaces.

So this module asks the two questions the others cannot:

    does the documented launcher refuse to serve a brainless application?
    does every asset a rendered page asks for actually resolve?
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest
from litestar.testing import TestClient
from PIL import Image

from sg_web import __main__ as launcher
from sg_web.app import build_app

REPO = pathlib.Path(__file__).resolve().parent.parent
#: Everything a rendered page pulls from the application itself.
_ASSET = re.compile(r'(?:src|href)="(/static/[^"?]+)')


def test_the_launcher_names_every_bundle_the_templates_ask_for(tmp_path):
    """Read out of the templates, never listed twice.

    The entry points live in frontend/build.ts. A second copy in Python
    would be one rename away from reporting "all built" while the page
    404s -- the same defect one layer up.
    """
    templates = REPO / "sg_web" / "templates"
    absent = launcher.unbuilt(templates, tmp_path)
    assert absent, "a static directory with no build/ has every bundle missing"
    assert "media" in absent, "media.html loads media.js"
    assert "gallery" in absent, "gallery.html loads gallery.js"
    # and it is the templates that decide, not a hand-kept list
    spelled = {
        found.group(1)
        for page in templates.glob("*.html")
        for found in launcher._LOADED.finditer(page.read_text(encoding="utf-8"))
    }
    assert set(absent) == spelled


def test_a_built_tree_satisfies_the_launcher(tmp_path):
    """The negative control. Without it the check above would pass for a
    function that called everything missing forever."""
    templates = REPO / "sg_web" / "templates"
    build = tmp_path / "build"
    build.mkdir()
    for name in launcher.unbuilt(templates, tmp_path):
        (build / f"{name}.js").write_bytes(b"// pretend\n")
    assert launcher.unbuilt(templates, tmp_path) == []


def test_the_documented_launcher_refuses_to_serve_without_its_bundles(tmp_path, monkeypatch, capsys):
    """`main` itself, with its HERE pointed at a tree that has no build/ --
    the state of any fresh checkout -- so this cannot pass merely because
    the developer's own tree happens to be built.

    The real entry point rather than a subprocess: it raises SystemExit
    with the exit code and prints the refusal, which is the whole of what
    a person meets, and this repository's own rule is that a test does
    not start a program (sglint SG006).
    """
    hollow = tmp_path / "sg_web"
    (hollow / "static").mkdir(parents=True)
    (hollow / "templates").mkdir()
    for page in (REPO / "sg_web" / "templates").glob("*.html"):
        (hollow / "templates" / page.name).write_bytes(page.read_bytes())

    monkeypatch.setattr(launcher, "HERE", hollow)
    monkeypatch.setattr(sys, "argv", ["sg_web"])
    # uvicorn.run would serve; reaching it at all is the failure
    monkeypatch.setattr(launcher.uvicorn, "run", _never_served)

    with pytest.raises(SystemExit) as refused:
        launcher.main()

    assert refused.value.code == 2
    said = capsys.readouterr().err
    assert "not built" in said
    assert "media" in said, "the refusal names what is missing"
    assert launcher.BUILD_COMMAND in said, "and what to run about it"


def _never_served(*args, **kwargs):
    raise AssertionError(f"the launcher served an application with no bundles: {args} {kwargs}")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A real library behind the real application."""
    tmp = tmp_path_factory.mktemp("launch")
    root = tmp / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (48, 36), (30 * i, 90, 160)).save(root / f"p_{i}.png")
    with TestClient(app=build_app(str(tmp / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 3
        yield client


def test_every_asset_a_page_asks_for_is_served(served):
    """The 404 that made the viewer inert, caught in HTTP alone.

    Each surface is rendered and every `/static/...` it names is fetched.
    A bundle a template loads and esbuild does not emit fails here, in
    milliseconds, without a browser -- and so does a stylesheet renamed
    out from under a page.
    """
    slug = served.get("/g/peek", params={"page": 1, "count": 1}).json()["items"][0]["slug"]
    surfaces = ["/g", f"/i/{slug}", "/people", "/places", "/albums", "/folders", "/timeline", "/operations"]

    asked: dict[str, set[str]] = {}
    for where in surfaces:
        page = served.get(where, headers={"accept": "text/html"})
        assert page.status_code == 200, f"{where} did not render: {page.status_code}"
        for asset in _ASSET.findall(page.text):
            asked.setdefault(asset, set()).add(where)
    assert asked, "the shell loads a stylesheet and htmx at the very least"

    broken = {}
    for asset, pages in sorted(asked.items()):
        got = served.get(asset).status_code
        if got != 200:
            broken[asset] = (got, sorted(pages))
    assert not broken, f"pages ask for assets the application does not serve: {broken}"


def test_the_media_page_loads_the_bundle_that_makes_it_a_viewer(served):
    """Named on purpose rather than left to the sweep above.

    `media.js` is where mountViewer runs -- the wheel, the keys, the
    inspector, the walk. When it 404s the page still renders a
    photograph, which is exactly why nobody noticed.
    """
    slug = served.get("/g/peek", params={"page": 1, "count": 1}).json()["items"][0]["slug"]
    page = served.get(f"/i/{slug}", headers={"accept": "text/html"})
    assert "/static/build/media.js" in page.text
    assert served.get("/static/build/media.js").status_code == 200
