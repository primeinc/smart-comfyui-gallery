"""One shell, every page; one activity surface, every page; one runtime
door. WI-51's contract, pinned at the seams a browser can see.

The shell is Jinja inheritance on the application's one template engine
(templates/base.html); a page that does not extend it is a page outside
the product. The activity surface is the persisted job rows rendered by
sg_web/activity.py, mounted by the shell and kept live by /ws/jobs?as=html
-- the same subscribe-before-snapshot feed the JSON consumers use, in a
second representation. Operations is a Litestar Router under /operations:
the runtime's own page, posting url-encoded forms and receiving fragments.

The coverage contract at the bottom is the part that outlives this
change: every browser-facing capability names its owning view Module and
its affordance, or is declared headless with a reason, and the list is
checked against what the application actually registers.
"""

from __future__ import annotations

import pathlib
import re
import sys

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
        active = {row["id"] for row in snap["jobs"]}
        pending &= active
        while pending:
            delta = feed.receive_json(timeout=30)
            if delta["state"] in TERMINAL:
                pending.discard(delta["job"])


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
    """The product areas are reachable from every rendered page, and the
    operational door is there too, set apart from media browsing."""
    client, _ = served
    for where in ("/g", "/people", "/albums", "/folders", "/timeline", "/operations", "/i/s-0", "/f/lib"):
        page = client.get(where, headers=AS_BROWSER)
        assert page.status_code == 200, f"{where}: {page.status_code} {page.text[:300]}"
        shells = page.text.count('<nav class="shell"')
        assert shells == 1, f"{where} mounts the shell {shells} times"
        for door in ("/g", "/people", "/albums", "/folders", "/timeline", "/operations"):
            assert f'href="{door}"' in page.text, f"{where} does not reach {door}"
        assert "data-activity" in page.text, f"{where} has no activity surface"
        assert 'ws-connect="/ws/jobs?as=html"' in page.text, f"{where} is not wired to the feed"
    gallery = client.get("/g", headers=AS_BROWSER).text
    assert re.search(r'href="/g"[^>]*aria-current="page"', gallery)
    assert not re.search(r'href="/people"[^>]*aria-current="page"', gallery)


def test_machine_representations_carry_no_shell(served):
    """Negotiation is untouched: JSON callers and htmx fragments never
    receive the document."""
    client, _ = served
    assert "shell" not in client.get("/i/s-0", headers={"accept": "application/json"}).text
    for part in (client.get("/i/s-0", headers={"hx-request": "true"}).text, client.get("/g/grid").text):
        assert "<html" not in part
        assert 'class="shell"' not in part


def test_the_story_templates_are_children_of_the_shell():
    """The story renderer keeps its StrictUndefined environment and still
    inherits the shell -- the shell reads only what every render passes."""
    for name in ("story.html", "evolution.html"):
        assert (TEMPLATES / name).read_text(encoding="utf-8").startswith('{% extends "base.html" %}'), name


# --- the activity surface ----------------------------------------------------


def test_cold_load_renders_the_persisted_jobs(served):
    """A page served while a job is queued shows that job from the rows,
    before any socket opens: the worker is switched off so the row stays
    queued for the read."""
    client, _ = served
    client.post("/settings/worker", json={"value": "off"})
    try:
        job_id = client.post("/jobs/verify").json()["id"]
        page = client.get("/g", headers=AS_BROWSER).text
        assert f'id="job-{job_id}"' in page
        assert re.search(rf'id="job-{job_id}"[^>]*data-state="queued"', page)
        assert f'hx-post="/jobs/{job_id}/cancel"' in page, "a live job offers its cancel"
        assert client.post(f"/jobs/{job_id}/cancel").json()["cancel_requested"] == 1
    finally:
        client.post("/settings/worker", json={"value": "on"})
    _drained(client, [job_id])


def test_the_html_feed_is_the_same_feed_rendered(served):
    """/ws/jobs?as=html: the list first (out-of-band, whole), then one
    fragment per delta -- an append for a job the connection has never
    seen, a replacement afterwards -- through to the terminal state. The
    JSON feed's contract, in htmx's grammar."""
    client, _ = served
    with client.websocket_connect("/ws/jobs?as=html") as feed:
        first = feed.receive_text(timeout=10)
        assert 'id="activity-jobs"' in first, first
        assert 'hx-swap-oob="true"' in first, first
        job_id = client.post("/jobs/verify").json()["id"]
        born = feed.receive_text(timeout=10)
        assert 'hx-swap-oob="beforeend:#activity-jobs"' in born, "a new job must be appended, not swapped into nothing"
        assert f'id="job-{job_id}"' in born
        assert 'data-state="queued"' in born
        state = "queued"
        frames = 0
        while state not in TERMINAL:
            frame = feed.receive_text(timeout=10)
            frames += 1
            assert f'<li id="job-{job_id}"' in frame, frame
            assert 'hx-swap-oob="true"' in frame, frame
            assert "beforeend" not in frame, "a known job is replaced in place, never appended again"
            found = re.search(r'data-state="(\w+)"', frame)
            assert found is not None
            state = found.group(1)
        assert state == "done"
        assert frames >= 2
    later = client.get("/g", headers=AS_BROWSER).text
    assert f'id="job-{job_id}"' not in later, "a settled job is not an active row"


def test_the_json_feed_announces_a_queued_job(served):
    """The submit itself speaks on the feed now -- the first delta for a
    job is its committed `queued` row, not the worker's first claim."""
    client, _ = served
    with client.websocket_connect("/ws/jobs") as feed:
        assert feed.receive_json(timeout=10)["type"] == "snapshot"
        job_id = client.post("/jobs/verify").json()["id"]
        first = feed.receive_json(timeout=10)
        assert (first["job"], first["state"], first["done"], first["total"]) == (job_id, "queued", 0, 3)
        state = first["state"]
        while state not in TERMINAL:
            state = feed.receive_json(timeout=10)["state"]
        assert state == "done"


# --- operations ------------------------------------------------------------


def test_operations_is_one_router_over_the_runtime(served):
    """The page lists roots, launchers, settings and clusterings; each form
    posts url-encoded and gets its section back, with a notice swapped
    out-of-band. The JSON routes keep answering beside it."""
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
    assert client.post("/operations/jobs/nothing").status_code == 404

    changed = client.post("/operations/settings/thumbnail_precache", data={"value": "off"})
    assert changed.status_code == 200, changed.text
    assert "thumbnail_precache = off" in changed.text
    assert {row["key"]: row["value"] for row in client.get("/settings").json()}["thumbnail_precache"] == "off"
    client.post("/settings/thumbnail_precache", json={"value": "on"})
    assert client.post("/operations/settings/thumbnail_precache", data={"value": "sideways"}).status_code == 400

    scanned = client.post("/operations/roots/1/scan")
    assert scanned.status_code == 200, scanned.text
    assert "3 matched" in scanned.text
    assert client.post("/operations/roots/99/scan").status_code == 404

    elsewhere = root.parent / "elsewhere"
    elsewhere.mkdir()
    added = client.post("/operations/roots", data={"path": str(elsewhere), "kind": "library"})
    assert added.status_code == 200, added.text
    assert "elsewhere" in added.text
    assert client.post("/operations/roots", data={"path": "   "}).status_code == 400

    chosen = client.post("/operations/clusterings/choose")
    assert chosen.status_code == 200, chosen.text
    assert "primary run" in chosen.text
    _drained(client, [queued])


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

#: Every user-facing capability, its owning view Module, and the browser
#: affordance that reaches it. A handler that renders HTML and is absent
#: here fails the sweep below; adding a capability means naming its door.
SURFACES = {
    "gallery.gallery": ("sg_web/gallery.py", "shell nav -> /g; templates/gallery.html"),
    "gallery.grid_fragment": ("sg_web/gallery.py", "pager hx-get inside templates/_grid.html"),
    "media_view.media_page": ("sg_web/media_view.py", "grid cell -> /i/{slug}; lightbox fragment"),
    "person_view.people_index": ("sg_web/person_view.py", "shell nav -> /people"),
    "person_view.person_page": ("sg_web/person_view.py", "person card -> /p/{slug}; drawer fragment"),
    "collection_view.albums_index": ("sg_web/collection_view.py", "shell nav -> /albums"),
    "collection_view.album_page": ("sg_web/collection_view.py", "album tree -> /t/{slug}"),
    "folder_view.folders_index": ("sg_web/folder_view.py", "shell nav -> /folders"),
    "folder_view.folder_page": ("sg_web/folder_view.py", "folder card -> /f/{slug}"),
    "artifact_view.model_page": ("sg_web/artifact_view.py", "media facts -> /m/{slug}"),
    "artifact_view.lora_page": ("sg_web/artifact_view.py", "media facts -> /l/{slug}"),
    "artifact_view.workflow_page": ("sg_web/artifact_view.py", "story heroes -> /w/{slug}"),
    "timeline_view.timeline": ("sg_web/timeline_view.py", "shell nav -> /timeline"),
    "story_view.render_document": ("sg_web/story_view.py", "timeline session -> /stories/renders/{id}"),
    "story_view.plan_evolution": ("sg_web/story_view.py", "story -> /stories/plans/{id}/evolution"),
    "operations.operations_page": ("sg_web/operations.py", "shell nav -> /operations"),
    "operations.launch": ("sg_web/operations.py", "sweep buttons on /operations"),
    "operations.add_root": ("sg_web/operations.py", "add-root form on /operations"),
    "operations.scan_root": ("sg_web/operations.py", "scan button per root on /operations"),
    "operations.change_setting": ("sg_web/operations.py", "one form per setting on /operations"),
    "operations.choose_primary": ("sg_web/operations.py", "choose button on /operations"),
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
                yield handler


def test_every_html_handler_names_its_door(served):
    """The sweep: every registered handler that can render HTML is in
    SURFACES, every headless one in HEADLESS, and nothing in either list
    is stale."""
    import inspect

    client, _ = served
    rendered: set[str] = set()
    registered: set[str] = set()
    for handler in _handlers(client.app):
        # sync handlers are wrapped for the thread pool; the function is
        # `.func` (litestar-org/litestar@v2.24.0 litestar/utils/sync.py:35-45)
        fn = handler.fn
        while hasattr(fn, "func"):
            fn = fn.func
        registered.add(handler.handler_name)
        if not inspect.isfunction(fn):
            continue  # a bound method from a library (the static files router)
        module = fn.__module__.rsplit(".", 1)[-1]
        source = inspect.getsource(fn)
        # a handler may hand the rendering to a private helper in its own
        # module (artifact_view._artifact_page, collection_view._album_page)
        owner = sys.modules[fn.__module__]
        for helper in set(re.findall(r"\b(_[a-z_]+)\b", source)):
            held = getattr(owner, helper, None)
            if inspect.isfunction(held):
                source += inspect.getsource(held)
        # three ways a handler renders: a Template, the negotiation seam, or
        # a hand-rendered HTML response (sg_web/story_view.py)
        if "Template(" in source or "presented(" in source or 'media_type="text/html"' in source:
            rendered.add(f"{module}.{handler.handler_name}")
    missing = rendered - set(SURFACES)
    assert not missing, f"HTML-rendering handlers with no named browser door: {sorted(missing)}"
    stale = set(SURFACES) - rendered
    assert not stale, f"SURFACES names handlers that no longer render: {sorted(stale)}"
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
