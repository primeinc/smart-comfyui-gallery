"""One shell, every page; one activity surface, every page; one runtime
door. WI-51's contract, pinned at the seams a browser can see.

The shell is Jinja inheritance on the application's one template engine
(templates/base.html); a page that does not extend it is a page outside
the product. The activity surface is the persisted job rows rendered by
sg_web/activity.py -- the active ones and the recently settled ones --
mounted by the shell and kept live by /ws/jobs?as=html, the same
subscribe-before-snapshot feed the JSON consumers use in a second
representation. Operations is a Litestar Router under /operations: the
runtime's own page, posting url-encoded forms and receiving fragments,
its refusals rendered into the shell's notice with their real status.

The coverage contract at the bottom is the part that outlives this
change: every browser-facing capability names its owning view Module and
a sample request, the sample is made and must render through the shell;
every other route is requested as a browser and must NOT render HTML, or
be declared headless with a reason.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest
from litestar.testing import TestClient
from PIL import Image

from sg_web.app import build_app

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "sg_web" / "templates"
TERMINAL = ("done", "failed", "cancelled")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shell")
    root = tmp / "lib"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (12, 12), (30 * i, 90, 120)).save(root / f"s_{i}.png")
    with TestClient(app=build_app(str(tmp / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 3
        yield client, root


def _drained(client, job_ids) -> None:
    """Wait on the feed until every named job reaches a terminal state."""
    pending = set(job_ids)
    with client.websocket_connect("/ws/jobs") as feed:
        snap = feed.receive_json(timeout=10)
        pending &= {row["id"] for row in snap["jobs"]}
        while pending:
            delta = feed.receive_json(timeout=30)
            if delta["state"] in TERMINAL:
                pending.discard(delta["job"])


def _state_of(frame: str) -> str:
    found = re.search(r'data-state="(\w+)"', frame)
    assert found is not None, frame
    return found.group(1)


# --- the shell ---------------------------------------------------------------


def test_every_page_template_extends_the_shell():
    """Structural: a full page is a child of base.html, never its own
    document. Fragments (underscore-prefixed) are mounted, never served
    whole, and carry no document at all."""
    pages = sorted(p for p in TEMPLATES.glob("*.html") if not p.name.startswith("_") and p.name != "base.html")
    assert len(pages) >= 13, [p.name for p in pages]
    for page in pages:
        held = page.read_text(encoding="utf-8")
        assert held.lstrip().startswith('{% extends "base.html" %}'), f"{page.name} is not a child of the shell"
        assert "<!doctype" not in held.lower(), f"{page.name} carries its own document"
    for fragment in TEMPLATES.glob("_*.html"):
        held = fragment.read_text(encoding="utf-8").lower()
        assert "<html" not in held, f"{fragment.name} is a page, not a fragment"
        assert "<!doctype" not in held, f"{fragment.name} is a page, not a fragment"


def test_every_browser_page_carries_the_same_navigation(served):
    """The product areas are reachable from every rendered page, the
    operational door is there too, and the shell's notice and activity
    surface are mounted once each."""
    client, _ = served
    for where in ("/g", "/people", "/albums", "/folders", "/timeline", "/operations", "/i/s-0", "/f/lib"):
        page = client.get(where, headers=AS_BROWSER)
        assert page.status_code == 200, f"{where}: {page.status_code} {page.text[:300]}"
        shells = page.text.count('<nav class="shell"')
        assert shells == 1, f"{where} mounts the shell {shells} times"
        for door in ("/g", "/people", "/albums", "/folders", "/timeline", "/operations"):
            assert f'href="{door}"' in page.text, f"{where} does not reach {door}"
        assert page.text.count('id="shell-notice"') == 1, where
        assert page.text.count('id="activity-jobs"') == 1, where
        assert 'ws-connect="/ws/jobs?as=html"' in page.text, f"{where} is not wired to the feed"
    gallery = client.get("/g", headers=AS_BROWSER).text
    assert re.search(r'href="/g"[^>]*aria-current="page"', gallery)
    assert not re.search(r'href="/people"[^>]*aria-current="page"', gallery)


def test_machine_representations_carry_no_shell(served):
    """Negotiation is untouched: JSON callers and htmx fragments never
    receive the document. And the index routes negotiate through the one
    seam: a wildcard Accept is the machine default everywhere."""
    client, _ = served
    assert "shell" not in client.get("/i/s-0", headers={"accept": "application/json"}).text
    for part in (client.get("/i/s-0", headers={"hx-request": "true"}).text, client.get("/g/grid").text):
        assert "<html" not in part
        assert 'class="shell"' not in part
    for index in ("/people", "/albums", "/folders", "/timeline", "/f/lib"):
        machine = client.get(index)
        assert machine.headers["content-type"].startswith("application/json"), index
        assert machine.headers["vary"] == "Accept, HX-Request", index


def test_one_engine_renders_every_page_strictly(served):
    """There is one Jinja environment, and it is strict: a page that
    names a field its view did not supply is a 500 at render, never an
    empty string on screen. The story templates render through it too."""
    from jinja2 import StrictUndefined

    client, _ = served
    engine = client.app.template_engine.engine
    assert engine.undefined is StrictUndefined
    assert engine.autoescape is True
    assert "activity_jobs" in engine.globals
    # the story page's environment is this one, not a private copy
    from sg_web import story_view

    assert not hasattr(story_view, "_story_env")


# --- the activity surface ----------------------------------------------------


@pytest.mark.slow
def test_cold_load_renders_the_persisted_jobs(served):
    """A page served while a job is queued shows that job from the rows,
    before any socket opens: the worker is switched off so the row stays
    queued for the read. Cancelling it is announced at once -- the row
    shows `cancelling` on the next cold load and on the live feed, with
    no worker involved."""
    client, _ = served
    client.post("/settings/worker", json={"value": "off"})
    try:
        with client.websocket_connect("/ws/jobs") as feed:
            assert feed.receive_json(timeout=10)["type"] == "snapshot"
            job_id = client.post("/jobs/verify").json()["id"]
            born = feed.receive_json(timeout=10)
            assert (born["job"], born["state"], born["cancel_requested"]) == (job_id, "queued", 0)

            page = client.get("/g", headers=AS_BROWSER).text
            assert re.search(rf'id="job-{job_id}"[^>]*data-state="queued"', page)
            assert f'hx-post="/jobs/{job_id}/cancel"' in page, "a live job offers its cancel"

            assert client.post(f"/jobs/{job_id}/cancel").json()["cancel_requested"] == 1
            asked = feed.receive_json(timeout=10)
            assert (asked["job"], asked["state"], asked["cancel_requested"]) == (job_id, "queued", 1)
        later = client.get("/g", headers=AS_BROWSER).text
        assert re.search(rf'id="job-{job_id}"[^>]*data-cancelling', later), "the cancel asked for is not shown"
        assert f'hx-post="/jobs/{job_id}/cancel"' not in later, "a job already asked to stop is not asked again"
    finally:
        client.post("/settings/worker", json={"value": "on"})
    _drained(client, [job_id])
    # settled, it stays on the cold list as what it became
    settled = client.get("/g", headers=AS_BROWSER).text
    assert re.search(rf'id="job-{job_id}"[^>]*data-state="cancelled"', settled)


def test_the_html_feed_is_the_same_feed_rendered(served):
    """/ws/jobs?as=html: the list first (out-of-band, whole), then one
    fragment per delta -- an append for a job the connection has never
    seen, a replacement afterwards -- through to the terminal state. The
    JSON feed's contract, in htmx's grammar. A settled job stays on the
    next cold list as the row it became, so both moments agree."""
    client, _ = served
    with client.websocket_connect("/ws/jobs?as=html") as feed:
        first = feed.receive_text(timeout=10)
        assert 'id="activity-jobs"' in first, first
        assert 'hx-swap-oob="true"' in first, first
        job_id = client.post("/jobs/verify").json()["id"]
        born = feed.receive_text(timeout=10)
        assert 'hx-swap-oob="beforeend:#activity-jobs"' in born, "a new job must be appended, not swapped into nothing"
        assert f'id="job-{job_id}"' in born
        assert _state_of(born) == "queued"
        state = "queued"
        frames = 0
        while state not in TERMINAL:
            frame = feed.receive_text(timeout=10)
            frames += 1
            assert f'<li id="job-{job_id}"' in frame, frame
            assert 'hx-swap-oob="true"' in frame, frame
            assert "beforeend" not in frame, "a known job is replaced in place, never appended again"
            state = _state_of(frame)
        assert state == "done"
        assert frames >= 2
    later = client.get("/g", headers=AS_BROWSER).text
    assert re.search(rf'id="job-{job_id}"[^>]*data-state="done"', later), "the settled job left the cold list"
    assert f'hx-post="/jobs/{job_id}/cancel"' not in later, "a settled job offers no cancel"


def test_the_feed_renders_a_job_the_list_has_never_seen(served):
    """A job born before the socket opened but after the list was read
    is the gap the subscribe-first order closes; a job born on another
    page mid-connection is the gap `seen` closes. Both are appends."""
    client, _ = served
    with client.websocket_connect("/ws/jobs?as=html") as feed:
        feed.receive_text(timeout=10)
        first = client.post("/jobs/verify").json()["id"]
        second = client.post("/jobs/verify").json()["id"]
        appended = {first: False, second: False}
        # other jobs share the feed (the fixture's scan is still hashing);
        # only the first frame naming each new job is the append under test
        for _ in range(40):
            frame = feed.receive_text(timeout=10)
            for job_id in [j for j, seen in appended.items() if not seen and f'id="job-{j}"' in frame]:
                assert "beforeend" in frame, frame
                appended[job_id] = True
            if all(appended.values()):
                break
        assert all(appended.values()), appended
    _drained(client, [first, second])


# --- operations ------------------------------------------------------------


def test_operations_is_one_router_over_the_runtime(served):
    """The page lists roots, launchers, settings and clusterings; each form
    posts url-encoded and gets its section back, with a receipt swapped
    into the shell's notice. The JSON routes keep answering beside it."""
    client, root = served
    page = client.get("/operations", headers=AS_BROWSER)
    assert page.status_code == 200, page.text[:300]
    assert "data-operations-roots" in page.text
    assert str(root) in page.text
    for kind in ("ingest", "verify", "phash", "faces", "cluster", "context", "events"):
        assert f'hx-post="/operations/jobs/{kind}"' in page.text, kind
    assert 'data-setting="worker"' in page.text

    started = client.post("/operations/jobs/verify")
    assert started.status_code == 200, started.text
    assert "queued #" in started.text
    queued = int(started.text.rsplit("#", 1)[1].split("<", 1)[0])

    changed = client.post("/operations/settings/thumbnail_precache", data={"value": "off"})
    assert changed.status_code == 200, changed.text
    assert 'id="shell-notice"' in changed.text, "the receipt rides out-of-band into the shell's notice"
    assert "thumbnail_precache = off" in changed.text
    assert {row["key"]: row["value"] for row in client.get("/settings").json()}["thumbnail_precache"] == "off"
    client.post("/settings/thumbnail_precache", json={"value": "on"})

    scanned = client.post("/operations/roots/1/scan")
    assert scanned.status_code == 200, scanned.text
    assert "3 matched" in scanned.text

    elsewhere = root.parent / "elsewhere"
    elsewhere.mkdir()
    added = client.post("/operations/roots", data={"path": str(elsewhere), "kind": "library"})
    assert added.status_code == 200, added.text
    assert "elsewhere" in added.text

    chosen = client.post("/operations/clusterings/choose")
    assert chosen.status_code == 200, chosen.text
    assert "primary run" in chosen.text
    _drained(client, [queued])


def test_a_refusal_is_rendered_with_its_status_not_swallowed(served):
    """The error path is a rendered path. A refused form answers with the
    refusal's own status and the reason as the notice fragment -- htmx
    swaps it into #shell-notice (templates/base.html response handling)
    -- never a JSON body the page silently drops."""
    client, _ = served
    for where, data, status, reason in (
        ("/operations/jobs/nothing", None, 404, "nothing to start"),
        ("/operations/settings/thumbnail_precache", {"value": "sideways"}, 400, "must be one of"),
        ("/operations/settings/not_a_setting", {"value": "x"}, 400, "not a setting"),
        ("/operations/roots/99/scan", None, 404, "no root 99"),
        ("/operations/roots", {"path": "   "}, 400, "needs a path"),
    ):
        answer = client.post(where, data=data)
        assert answer.status_code == status, (where, answer.status_code, answer.text)
        assert answer.headers["content-type"].startswith("text/html"), where
        assert "data-error" in answer.text, (where, answer.text)
        assert reason in answer.text, (where, answer.text)
    # a refusal that never reaches the handler -- a missing form field --
    # is a 400 the same way
    bare = client.post("/operations/roots", data={})
    assert bare.status_code == 400
    assert "data-error" in bare.text, bare.text
    # the shell declares the response handling that lands these
    page = client.get("/operations", headers=AS_BROWSER).text
    assert '"code":"[45]..","swap":true,"error":true,"target":"#shell-notice"' in page


def test_the_gallery_header_offers_no_operations(served):
    """The stop condition: media browsing asks questions about media.
    Launching sweeps, registering roots and switching the worker live on
    /operations and nowhere else."""
    client, _ = served
    gallery = client.get("/g", headers=AS_BROWSER).text
    header = gallery.split('<header class="bar">', 1)[1].split("</header>", 1)[0]
    for forbidden in ("/operations/jobs/", "/roots", "/settings/"):
        assert forbidden not in header, f"the gallery header grew an operational control: {forbidden}"


# --- the coverage contract ---------------------------------------------------

#: Every user-facing capability: its owning view Module, the browser
#: affordance that reaches it, and a sample request that must render
#: through the shell. Adding a capability means naming its door here.
SURFACES = {
    "gallery.gallery": ("sg_web/gallery.py", "shell nav", ("GET", "/g", None)),
    "gallery.grid_fragment": ("sg_web/gallery.py", "pager hx-get in _grid.html", ("GET", "/g/grid", None)),
    "media_view.media_page": ("sg_web/media_view.py", "grid cell; lightbox fragment", ("GET", "/i/s-0", None)),
    "person_view.people_index": ("sg_web/person_view.py", "shell nav", ("GET", "/people", None)),
    "person_view.person_page": ("sg_web/person_view.py", "person card; drawer fragment", None),
    "collection_view.albums_index": ("sg_web/collection_view.py", "shell nav", ("GET", "/albums", None)),
    "collection_view.album_page": ("sg_web/collection_view.py", "album tree", None),
    "folder_view.folders_index": ("sg_web/folder_view.py", "shell nav", ("GET", "/folders", None)),
    "folder_view.folder_page": ("sg_web/folder_view.py", "folder card", ("GET", "/f/lib", None)),
    "artifact_view.model_page": ("sg_web/artifact_view.py", "media facts", None),
    "artifact_view.lora_page": ("sg_web/artifact_view.py", "media facts", None),
    "artifact_view.workflow_page": ("sg_web/artifact_view.py", "story heroes", None),
    "timeline_view.timeline": ("sg_web/timeline_view.py", "shell nav", ("GET", "/timeline", None)),
    "story_view.render_document": ("sg_web/story_view.py", "timeline session", None),
    "story_view.plan_evolution": ("sg_web/story_view.py", "story", None),
    "operations.operations_page": ("sg_web/operations.py", "shell nav", ("GET", "/operations", None)),
    "operations.launch": ("sg_web/operations.py", "sweep buttons", ("POST", "/operations/jobs/events", None)),
    "operations.add_root": ("sg_web/operations.py", "add-root form", None),
    "operations.scan_root": (
        "sg_web/operations.py",
        "scan button per root",
        ("POST", "/operations/roots/1/scan", None),
    ),
    "operations.change_setting": (
        "sg_web/operations.py",
        "one form per setting",
        ("POST", "/operations/settings/worker", {"value": "on"}),
    ),
    "operations.choose_primary": (
        "sg_web/operations.py",
        "choose button",
        ("POST", "/operations/clusterings/choose", None),
    ),
}

#: Browser-facing capabilities with no page of their own, and why.
HEADLESS = {
    "jobs_feed": "transport: the shell's activity surface is its page",
    "front": "redirects a browser to /g",
    "health": "liveness probe",
}


def _handlers(app):
    for route in app.routes:
        for handler in getattr(route, "route_handlers", [getattr(route, "route_handler", None)]):
            if handler is not None and handler.handler_name != "options_handler":
                yield route, handler


def _qualified(handler) -> str:
    fn = handler.fn
    while hasattr(fn, "func"):
        fn = fn.func
    module = getattr(fn, "__module__", "").rsplit(".", 1)[-1]
    return f"{module}.{handler.handler_name}"


def test_every_named_door_renders_through_the_shell(served):
    """Each sample request in SURFACES, made as a browser, answers HTML --
    a page through the shell, or a fragment with none of it."""
    client, _ = served
    for name, (_module, _door, sample) in SURFACES.items():
        if sample is None:
            continue
        method, where, data = sample
        answer = client.request(method, where, headers=AS_BROWSER, data=data)
        assert answer.status_code == 200, (name, answer.status_code, answer.text[:200])
        assert answer.headers["content-type"].startswith("text/html"), name
        is_page = "<html" in answer.text
        assert is_page == ('<nav class="shell"' in answer.text), f"{name}: a page carries the shell, a fragment none"
    _drained(client, [row["id"] for row in client.get("/jobs").json()])


def test_every_other_route_is_machine_shaped_or_declared_headless(served):
    """The sweep: every registered handler is a named door, declared
    headless, or -- requested as a browser -- answers something other
    than HTML. Routes with path parameters that nobody named are
    requested with a placeholder that must NOT render a page either.
    GET only: a sweep that POSTs every machine route starts every sweep
    (and downloads a model for the embed one); a POST route that renders
    a page is caught by the source check, since a Template is what such a
    handler returns. Litestar's own OpenAPI UI under /schema is the
    framework's, not the product's."""
    client, _ = served
    registered: set[str] = set()
    for route, handler in _handlers(client.app):
        name = _qualified(handler)
        registered.add(handler.handler_name)
        if name in SURFACES or handler.handler_name in HEADLESS or handler.handler_name == "static":
            continue
        if route.path.startswith("/schema"):
            continue
        methods = set(getattr(handler, "http_methods", ()) or ())
        if "GET" in methods:
            where = re.sub(r"\{(\w+):int\}", "1", route.path)
            where = re.sub(r"\{(\w+):str\}", "nobody", where)
            answer = client.get(where, headers=AS_BROWSER)
            assert not answer.headers.get("content-type", "").startswith("text/html"), (
                f"{name} (GET {where}) renders HTML but names no browser door"
            )
        fn = handler.fn
        while hasattr(fn, "func"):
            fn = fn.func
        if inspect.isfunction(fn):
            assert "Template(" not in inspect.getsource(fn), f"{name} renders a Template but names no browser door"
    stale = set(SURFACES) - {_qualified(handler) for _, handler in _handlers(client.app)}
    assert not stale, f"SURFACES names handlers that are not registered: {sorted(stale)}"
    assert set(HEADLESS) <= registered, sorted(set(HEADLESS) - registered)


# --- the runtime invariant ---------------------------------------------------


def test_the_feed_is_one_process_by_construction():
    """MemoryChannelsBackend fans out inside one process. The entry point
    starts exactly one, and nothing passes uvicorn a worker count; both
    are pinned so the day either changes, the channel backend changes
    with it."""
    import ast

    here = pathlib.Path(__file__).resolve().parent.parent / "sg_web"
    tree = ast.parse((here / "__main__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "run":
            assert "workers" not in {kw.arg for kw in node.keywords}, "uvicorn workers > 1 splits the feed"
    assert "MemoryChannelsBackend()" in (here / "app.py").read_text(encoding="utf-8")
