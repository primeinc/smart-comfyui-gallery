"""One shell, every page; one activity surface, every page; one runtime
link. WI-51's contract, pinned at the seams a browser can see.

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

import pathlib
import re

import pytest
from litestar.testing import TestClient
from PIL import Image

from sg_web.app import build_app
from tests.staging import settled

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
        swept = client.post(f"/roots/{made['id']}/scan").json()
        assert swept["added"] == 3
        # The scan queued the thumbnail job; the module's tests read the
        # feed and the cold list expecting only the jobs they start.
        settled(client, swept["precache"])
        yield client, root


def _drained(client, job_ids) -> None:
    """Wait until every named job reaches a terminal state -- on the
    ROW (tests/staging.settled), not on delta frames: a paused-and-
    resumed job under a saturated runner outlives any fixed inter-frame
    timeout, and this helper only waits, it asserts nothing about the
    feed. The feed's own behaviour has its own tests below."""
    for job_id in job_ids:
        settled(client, job_id)


def _state_of(frame: str) -> str:
    found = re.search(r'data-state="(\w+)"', frame)
    assert found is not None, frame
    return found.group(1)


# --- the shell ---------------------------------------------------------------


def test_the_front_link_is_the_gallery(served):
    """A browser at `/` lands in the gallery; a machine gets the compact
    library summary with a newest strip. The building entrance stopped
    pointing at JSON."""
    client, _ = served
    landed = client.get("/", headers={"accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert (landed.status_code, landed.headers["location"]) == (302, "/g")

    front = client.get("/").json()
    assert front["files"] == 3
    for fact in ("folders", "people", "collections", "artifacts"):
        assert isinstance(front[fact], int), f"the summary counts {fact}"
    assert {row["name"] for row in front["newest"]} == {"s_0.png", "s_1.png", "s_2.png"}
    assert all(row["slug"] for row in front["newest"])


def test_every_browser_page_carries_the_same_navigation(served):
    """The product areas are reachable from every rendered page, the
    operational link is there too, and the shell's notice and activity
    surface are mounted once each."""
    client, _ = served
    for where in (
        "/g",
        "/people",
        "/places",
        "/albums",
        "/folders",
        "/timeline",
        "/stories",
        "/operations",
        "/i/s-0",
        "/f/lib",
    ):
        page = client.get(where, headers=AS_BROWSER)
        assert page.status_code == 200, f"{where}: {page.status_code} {page.text[:300]}"
        shells = page.text.count('<nav class="shell"')
        assert shells == 1, f"{where} mounts the shell {shells} times"
        for link in ("/g", "/people", "/places", "/albums", "/folders", "/timeline", "/operations"):
            assert f'href="{link}"' in page.text, f"{where} does not reach {link}"
        assert page.text.count('id="shell-notice"') == 1, where
        assert page.text.count('id="activity-jobs"') == 1, where
        assert 'ws-connect="/ws/jobs?as=html"' in page.text, f"{where} is not wired to the feed"
    gallery = client.get("/g", headers=AS_BROWSER).text
    assert re.search(r'href="/g"[^>]*aria-current="page"', gallery)
    assert not re.search(r'href="/people"[^>]*aria-current="page"', gallery)


def test_the_site_has_a_face(served):
    """Every rendered page links the icon, and the blind probe -- a
    client asking /favicon.ico without reading any HTML -- gets the
    rasterized one instead of a 404 traceback per visit in the log."""
    client, _ = served
    page = client.get("/g", headers=AS_BROWSER).text
    assert 'rel="icon"' in page
    assert "/static/favicon.svg" in page
    probed = client.get("/favicon.ico")
    assert probed.status_code == 200
    assert probed.headers["content-type"].startswith("image/x-icon")
    assert probed.content[:4] == b"\x00\x00\x01\x00", "an ICO container, not a mislabeled bitmap"
    linked = client.get("/static/favicon.svg")
    assert linked.status_code == 200


def test_machine_representations_carry_no_shell(served):
    """Negotiation is untouched: JSON callers and htmx fragments never
    receive the document. And the index routes negotiate through the one
    seam: a wildcard Accept is the machine default everywhere."""
    client, _ = served
    assert "shell" not in client.get("/i/s-0", headers={"accept": "application/json"}).text
    for part in (client.get("/i/s-0", headers={"hx-request": "true"}).text, client.get("/g/grid").text):
        assert "<html" not in part
        assert 'class="shell"' not in part
    for index in ("/people", "/places", "/albums", "/folders", "/timeline", "/stories", "/f/lib"):
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
    # settled, it stays on the cold list as what it became -- the page,
    # named so it cannot shadow the imported wait helper
    cooled = client.get("/g", headers=AS_BROWSER).text
    assert re.search(rf'id="job-{job_id}"[^>]*data-state="cancelled"', cooled)


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
    for kind in ("ingest", "verify", "phash", "faces", "cluster", "annotate", "context", "events"):
        assert f'hx-post="/operations/jobs/{kind}"' in page.text, kind
    for kind in ("ingest", "phash", "faces", "embed", "annotate", "context"):
        assert f'hx-post="/operations/jobs/{kind}?everything=true"' in page.text, f"{kind} has an 'again'"
    for kind in ("verify", "cluster", "events"):
        assert f'data-launch-again="{kind}"' not in page.text, f"{kind} already does all of it"
    again = client.post("/operations/jobs/phash", params={"everything": "true"})
    assert again.status_code == 200, again.text
    assert "all of it again: queued #" in again.text
    assert client.post("/operations/jobs/verify", params={"everything": "true"}).status_code == 404
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


# --- every link leads to a page ----------------------------------------------


def _links(page: str) -> list[str]:
    """Every same-site href a page emits that is a landing, not bytes."""
    return [
        href.replace("&amp;", "&")
        for href in re.findall(r'href="([^"#]+)"', page)
        if href.startswith("/")
        and not href.startswith(("/media/", "/thumb/", "/preview/", "/avatar/", "/static/", "/operations/export/"))
    ]


def test_every_link_every_page_emits_lands_on_a_page(served):
    """Walk as a person would: from the navigation, follow every link
    every page renders, and every one of them answers an HTML page --
    never JSON, never a 4xx, never a 5xx. Nothing is written down in
    advance; the pages themselves say where a person can go."""
    client, _ = served
    queue = ["/g", "/timeline", "/people", "/places", "/albums", "/folders", "/operations"]
    seen: set[str] = set()
    while queue:
        where = queue.pop(0)
        if where in seen:
            continue
        seen.add(where)
        answer = client.get(where, headers=AS_BROWSER, follow_redirects=True)
        assert answer.status_code == 200, f"{where}: {answer.status_code} {answer.text[:200]}"
        kind = answer.headers["content-type"]
        assert kind.startswith("text/html"), f"{where} lands a person on {kind}"
        assert '<nav class="shell"' in answer.text, f"{where} renders without the shell"
        queue.extend(link for link in _links(answer.text) if link not in seen)
    assert len(seen) >= 12, sorted(seen)
    _drained(client, [row["id"] for row in client.get("/jobs").json()])


# --- the runtime invariant ---------------------------------------------------


def test_the_feed_is_one_process_by_construction():
    """MemoryChannelsBackend fans out inside one process. The entry point
    starts exactly one, and nothing passes uvicorn a worker count; both
    are pinned so the day either changes, the channel backend changes
    with it."""
