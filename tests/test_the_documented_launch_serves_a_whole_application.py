"""The boundary every other test in this suite sits above.

`just test`, `just test-slow`, `just smoke` and `just serve` all depend on
`web::build`, so by the time any of them runs, esbuild has already written
`sg_web/static/build`. They then prove the browser behaves -- truthfully,
and only after a prerequisite nothing in the Python asked for was silently
satisfied.

The bundles are committed now, so the documented path no longer needs
npm and the refusal is no longer the fresh checkout's normal state. It is
still the tree's only guard: an entry point whose output was never
committed, or a build directory somebody deleted, serves pages that
render and scripts that 404 -- the pictures arrive and nothing about them
works. One missing bundle wearing as many hats as there are surfaces.

The launcher's other refusal is about the interpreter rather than the
tree, and has the opposite shape: it does not refuse, it hands the
command to `.venv` and waits, so `python -m sg_web` from a PATH whose
python is a shim does not die on `import uvicorn`.

So this module asks what the others cannot:

    does the launcher refuse to serve a brainless application?
    does an interpreter that cannot serve hand off to one that can?
    does every asset a rendered page asks for actually resolve?

What it does NOT do is prove the documented bootstrap on a cold
checkout -- no .venv, no build directory, README commands only, a real
process answering a real socket. That is a lane outside pytest, and this
suite must not be read as standing in for it.
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
    assert "app" in absent, "base.html loads app.js, and every page renders the shell"
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
    # uvicorn.run would serve; reaching it at all is the failure. Patched
    # by dotted path rather than through the launcher, which no longer
    # holds the name: `main` imports uvicorn itself, after the handover,
    # so that an interpreter without one gets handed to the environment
    # that has it instead of a traceback at module import.
    monkeypatch.setattr("uvicorn.run", _never_served)

    with pytest.raises(SystemExit) as refused:
        launcher.main()

    assert refused.value.code == 2
    said = capsys.readouterr().err
    assert "not built" in said
    assert "app" in said, "the refusal names what is missing"
    assert launcher.BUILD_COMMAND in said, "and what to run about it"


def _never_served(*args, **kwargs):
    raise AssertionError(f"the launcher served an application with no bundles: {args} {kwargs}")


def test_this_interpreter_serves_without_a_handover(monkeypatch):
    """The `uv run` path. Nothing is missing, so nothing is spawned.

    Named first because a handover that fires when it should not is the
    expensive failure: every start would pay for a second interpreter.
    """
    monkeypatch.setattr(launcher.subprocess, "Popen", _never_spawned)
    assert launcher.missing() is None, "the test environment is the one that serves"
    launcher.handover()  # returns, or _never_spawned raises


def test_an_interpreter_without_a_server_is_handed_to_the_one_that_has_it(monkeypatch):
    """The defect this exists for.

    `python -m sg_web` finds whatever python a PATH offers -- a shim, an
    IDE's run button, a shortcut made months ago. That interpreter could
    read the source and not import a server, so the documented command
    died on `import uvicorn` while the environment holding uvicorn sat
    one known path away, unused.

    Measured, before the handover existed:

        File "sg_web\\__main__.py", line 35, in <module>
            import uvicorn
        ModuleNotFoundError: No module named 'uvicorn'
    """
    monkeypatch.setattr(launcher, "missing", lambda: "uvicorn")
    monkeypatch.setattr(sys, "argv", ["sg_web", "--port", "8791"])
    monkeypatch.delenv(launcher._HANDED_OVER, raising=False)
    handed = {}

    def _spawn(argv, env):
        handed.update(argv=argv, env=env)
        return _Waited(7)

    monkeypatch.setattr(launcher.subprocess, "Popen", _spawn)

    with pytest.raises(SystemExit) as ended:
        launcher.handover()

    assert ended.value.code == 7, "the child's exit status is this command's"
    assert handed["argv"][0] == str(launcher.interpreter()), "the venv's python, not this one"
    assert handed["argv"][1:3] == ["-m", "sg_web"], "the same command"
    assert handed["argv"][3:] == ["--port", "8791"], "carrying the argv it was given"
    assert handed["env"][launcher._HANDED_OVER] == "1", "and marked, so the child cannot repeat it"


def test_a_handover_never_hands_over_again(monkeypatch):
    """The termination proof.

    A child that still cannot import has a broken environment, not the
    wrong one. Without the flag it would spawn a grandchild with the same
    complaint, forever, and the person would meet a fork bomb instead of
    an error message.
    """
    monkeypatch.setattr(launcher, "missing", lambda: "uvicorn")
    monkeypatch.setenv(launcher._HANDED_OVER, "1")
    monkeypatch.setattr(launcher.subprocess, "Popen", _never_spawned)
    launcher.handover()  # returns, so `main` goes on to refuse and say why


def test_without_an_environment_to_hand_to_it_refuses_and_names_the_command(monkeypatch, capsys):
    """No `.venv`: the one case where a person must be told something.

    The refusal names the interpreter, the directory that was not an
    environment, and `uv sync`. A bare ModuleNotFoundError names the
    package and nothing about the environment that would have had it.
    """
    monkeypatch.setattr(launcher, "missing", lambda: "uvicorn")
    monkeypatch.setattr(launcher, "interpreter", lambda: None)
    monkeypatch.setattr(sys, "argv", ["sg_web"])
    monkeypatch.setattr(launcher.subprocess, "Popen", _never_spawned)

    with pytest.raises(SystemExit) as refused:
        launcher.main()

    assert refused.value.code == 2
    said = capsys.readouterr().err
    assert "uvicorn" in said, "the refusal names what is missing"
    assert ".venv" in said, "and the environment that was not there"
    assert launcher.RUN_COMMAND in said, "and what to run about it"


def test_the_environment_this_suite_runs_in_is_the_one_the_handover_targets(monkeypatch):
    """The negative control for `interpreter()`.

    Without it the handover tests would pass against a function that
    returned a path to nothing. This asserts the lookup finds the very
    interpreter running these lines.
    """
    found = launcher.interpreter()
    assert found is not None, "`uv sync` made a .venv and this suite is running inside it"
    assert found.is_file()
    assert found.resolve() == pathlib.Path(sys.executable).resolve()

    monkeypatch.setattr(launcher, "HERE", pathlib.Path(REPO.anchor) / "nowhere" / "sg_web")
    assert launcher.interpreter() is None, "and reports absence rather than a path that is not there"


def _never_spawned(*args, **kwargs):
    raise AssertionError(f"the launcher spawned an interpreter it did not need: {args} {kwargs}")


class _Waited:
    """A child that has already finished, standing in for Popen.

    A real one would be a second interpreter serving a real socket, which
    is the thing a test does not start (sglint SG006). What the launcher
    owes its caller is the child's status, so that is what this carries.
    """

    def __init__(self, ended: int) -> None:
        self._ended = ended

    def wait(self) -> int:
        return self._ended


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

    `app.js` is where mountViewer runs -- the wheel, the keys, the
    inspector, the walk. When it 404s the page still renders a
    photograph, which is exactly why nobody noticed.
    """
    slug = served.get("/g/peek", params={"page": 1, "count": 1}).json()["items"][0]["slug"]
    page = served.get(f"/i/{slug}", headers={"accept": "text/html"})
    assert "/static/build/app.js" in page.text
    assert served.get("/static/build/app.js").status_code == 200


# --- where it binds ---------------------------------------------------------
#
# A media library with no sign-in must not arrive on the network because
# somebody forgot a flag, and must arrive on it when somebody asks. Both
# halves are asserted on the address actually handed to the server.


def _bound(monkeypatch, argv: list[str]) -> dict:
    """Run the launcher far enough to see where it would bind."""
    held: dict = {}

    def _remember(app, host, port, **kwargs):
        held["host"] = host
        held["port"] = port
        # present and False on the default (redacted) path, absent on
        # --log-user-paths: an access log's URLs spell what the library
        # holds, so it rides the same flag (sg_web/__main__.py)
        held["access_log"] = kwargs.get("access_log", "absent")

    monkeypatch.setattr(sys, "argv", ["sg_web", *argv])
    monkeypatch.setattr(launcher, "missing", lambda: None)
    monkeypatch.setattr(launcher, "unbuilt", lambda *_: [])
    monkeypatch.setattr(launcher, "build_app", lambda home: home, raising=False)
    monkeypatch.setattr("uvicorn.run", _remember)
    monkeypatch.setattr("sg_web.app.build_app", lambda home: home)
    launcher.main()
    return held


def test_by_default_it_binds_this_machine_and_nothing_else(monkeypatch):
    held = _bound(monkeypatch, [])
    assert held["host"] == launcher.LOCAL
    assert held["access_log"] is False, "by default the request log names nothing, because URLs do"


def test_log_user_paths_restores_the_access_log(monkeypatch):
    assert _bound(monkeypatch, ["--log-user-paths"])["access_log"] == "absent"


def test_public_binds_every_interface(monkeypatch):
    held = _bound(monkeypatch, ["--public", "--port", "9123"])
    assert held["host"] == launcher.PUBLIC
    assert held["port"] == 9123


def test_a_named_host_is_still_honoured(monkeypatch):
    assert _bound(monkeypatch, ["--host", "192.168.1.5"])["host"] == "192.168.1.5"


def test_asking_for_both_differently_is_refused_rather_than_resolved(monkeypatch, capsys):
    """Two ways to say one thing, said differently. A launcher that
    quietly picked one would bind somewhere nobody asked for -- and for
    this flag that is the difference between one machine and a whole
    network."""
    monkeypatch.setattr(sys, "argv", ["sg_web", "--public", "--host", "192.168.1.5"])
    monkeypatch.setattr(launcher, "missing", lambda: None)
    monkeypatch.setattr("uvicorn.run", _never_served)
    with pytest.raises(SystemExit) as refused:
        launcher.main()
    assert refused.value.code == 2
    said = capsys.readouterr().err
    assert "--public" in said
    assert "192.168.1.5" in said


def test_going_public_says_so_and_says_where(monkeypatch, capsys):
    """It is a real change in who can reach the library, and there is no
    password on any of it -- so it is said out loud. Not refused: it was
    asked for on purpose."""
    _bound(monkeypatch, ["--public", "--port", "9123"])
    said = capsys.readouterr().err
    assert "EVERY interface" in said
    assert "no sign-in" in said
    assert "9123" in said, "the addresses printed are ones a person can actually type"


def test_the_addresses_it_prints_are_real_ones(monkeypatch):
    found = launcher.reachable()
    assert found, "always at least one, so the notice never prints an empty list"
    for one in found:
        assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+", one), one
