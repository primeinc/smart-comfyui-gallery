"""The Operations Console, witnessed in a real browser.

A uvicorn server with the real worker, a Chromium page on /operations,
and the acceptance contract's browser clauses: a job appears and moves
to its terminal state without a reload; an item failure shows its exact
recorded error; a cancel progresses request -> cooperative stop ->
cancelled; a disconnect and reconnect reconstructs history with no gap
and no duplicate; the number of events the page holds equals the number
the ledger produced. Status cards alone do not pass these.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
from PIL import Image

pytestmark = pytest.mark.slow

FILES = 12
TYPES_OF = """(id) => [...document.querySelectorAll('[data-tape-rows] [data-event][data-job="' + id + '"]')]
    .map(li => li.dataset.type)"""
HAS_DONE = """(id) => document.querySelector(
    '[data-tape-rows] [data-event][data-type="job.done"][data-job="' + id + '"]') !== null"""


def _free_port() -> int:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        return held.getsockname()[1]


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    import httpx
    import uvicorn

    from sg_web.app import build_app

    tmp = tmp_path_factory.mktemp("browser")
    root = tmp / "lib"
    root.mkdir()
    for i in range(FILES):
        Image.new("RGB", (8, 8), (10 * i, 60, 120)).save(root / f"b_{i:02d}.png")
    port = _free_port()
    app = build_app(str(tmp / "run"), worker=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="websockets")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="sg-browser-test", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while True:
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, "the server did not come up"
        time.sleep(0.1)
    with httpx.Client(base_url=base, timeout=10) as api:
        made = api.post("/roots", json={"path": str(root)}).json()
        swept = api.post(f"/roots/{made['id']}/scan").json()
        assert swept["added"] == FILES
        if swept["precache"] is not None:
            _settled(api, swept["precache"])
        yield base, api, root
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        yield chromium
        chromium.close()


@pytest.fixture
def page(browser, served):
    base, _api, _root = served
    context = browser.new_context()
    page = context.new_page()
    page.goto(base + "/operations")
    page.wait_for_selector('[data-health-transport][data-transport="connected"]', timeout=10_000)
    yield page
    context.close()


def _settled(api, job_id, timeout=30.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        state = api.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state}"
        time.sleep(0.05)


def test_a_job_appears_moves_and_settles_without_a_reload(page, served):
    """Contract 14: start a multi-item job, watch it appear in the matrix,
    watch item-start rows and progress changes arrive on the feed, watch
    the terminal state -- one page load, no polling."""
    _base, api, _root = served
    api.post("/settings/worker", json={"value": "off"})
    page.click('[data-launch="verify"]')
    row = page.wait_for_selector('[data-matrix-job][data-state="queued"]', timeout=10_000)
    job_id = int(row.get_attribute("data-matrix-job"))
    page.wait_for_selector(
        f'[data-tape-rows] [data-event][data-type="job.submitted"][data-job="{job_id}"]', timeout=10_000
    )
    api.post("/settings/worker", json={"value": "on"})
    page.wait_for_selector(
        f'[data-tape-rows] [data-event][data-type="item.started"][data-job="{job_id}"]', timeout=20_000
    )
    page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-state="done"]', timeout=30_000)
    assert _settled(api, job_id) == "done"
    types = page.evaluate(TYPES_OF, job_id)
    assert types.count("item.started") >= 1, types
    assert types.count("item.done") >= 2, types
    assert types[-1] == "job.done"
    page.click(f'[data-matrix-job="{job_id}"]')
    page.wait_for_selector(f'[data-inspect-job="{job_id}"][data-state="done"]', timeout=10_000)
    body = page.inner_text("[data-inspector-body]")
    for word in ("attempt", "fence", "owner", "heartbeat", "lease", "checkpoint", "cancellation", "payload"):
        assert word in body, word


def test_a_per_item_failure_shows_its_exact_recorded_error(page, served):
    """Contract 15: break one file behind the library's back; the verify
    sweep records the exact error on that item, and the live console
    shows it as an item failure -- the job continues."""
    _base, api, root = served
    broken = root / "b_03.png"
    broken.write_bytes(broken.read_bytes() + b"tampered")
    try:
        job_id = api.post("/jobs/verify").json()["id"]
        # the tape paints a window; every item reports phases now, so the
        # failure sits above the painted tail -- filter to warnings (a
        # presentation filter: nothing held is touched) to bring it in view
        page.select_option("[data-tape-filter-severity]", "warning")
        failed_row = page.wait_for_selector(
            f'[data-tape-rows] [data-event][data-type="item.failed"][data-job="{job_id}"]', timeout=30_000
        )
        assert "bytes changed behind the library's back" in failed_row.inner_text()
        assert failed_row.get_attribute("data-condition") == "item-failure"
        page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-state="done"]', timeout=30_000)
        page.click(f'[data-matrix-job="{job_id}"]')
        page.wait_for_selector('[data-failures] [data-condition="item-failure"]', timeout=10_000)
        shown = page.inner_text('[data-failures] [data-condition="item-failure"]')
        assert "b_03.png" in shown
        assert "bytes changed behind the library's back" in shown
        assert "continues" in shown
        recorded = api.get(f"/operations/job/{job_id}", headers={"accept": "application/json"}).json()["failures"]
        assert len(recorded) == 1
        assert recorded[0]["error"] in shown
    finally:
        broken.write_bytes(broken.read_bytes()[: -len(b"tampered")])


def test_cancellation_is_witnessed_as_request_then_cooperative_stop_then_cancelled(page, served):
    """Contract 17: the badge moves from 'cancelling' (the request) to
    'cancelled' only when the runner stops at a boundary and says so."""
    _base, api, _root = served
    api.post("/settings/worker", json={"value": "off"})
    job_id = api.post("/jobs/verify").json()["id"]
    page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-state="queued"]', timeout=10_000)
    page.click(f'[data-matrix-job="{job_id}"]')
    page.wait_for_selector(f'[data-inspect-job="{job_id}"] .inspect-cancel', timeout=10_000)
    page.click(f'[data-inspect-job="{job_id}"] .inspect-cancel')
    page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-cancelling]', timeout=10_000)
    page.wait_for_selector(
        f'[data-tape-rows] [data-event][data-type="job.cancel_requested"][data-job="{job_id}"]', timeout=10_000
    )
    assert page.inner_text(f'[data-matrix-job="{job_id}"] .matrix-state') == "cancelling"
    api.post("/settings/worker", json={"value": "on"})
    page.wait_for_selector(
        f'[data-tape-rows] [data-event][data-type="job.cancelled"][data-job="{job_id}"]', timeout=20_000
    )
    page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-state="cancelled"]', timeout=10_000)
    page.wait_for_selector(f'[data-inspect-job="{job_id}"] [data-cancellation="cancelled"]', timeout=10_000)
    types = page.evaluate(TYPES_OF, job_id)
    assert types.index("job.cancel_requested") < types.index("job.claimed") < types.index("job.cancelled")


def test_a_reconnect_reconstructs_history_with_no_gap_and_no_duplicate(page, served):
    """Contract 16: the page goes offline while a job runs to completion,
    comes back, and the ledger it holds is contiguous and unrepeated."""
    _base, api, _root = served
    api.post("/settings/worker", json={"value": "off"})
    job_id = api.post("/jobs/verify").json()["id"]
    page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-state="queued"]', timeout=10_000)
    # Chromium's offline emulation refuses NEW connections; the open socket
    # is dropped by the operator's own control, so the page is blind while
    # the job runs and every retry fails until the network returns.
    page.context.set_offline(True)
    page.click("[data-transport-reconnect]")
    page.wait_for_selector('[data-health-transport]:not([data-transport="connected"])', timeout=10_000)
    api.post("/settings/worker", json={"value": "on"})
    assert _settled(api, job_id) == "done"
    assert page.query_selector(f'[data-matrix-job="{job_id}"][data-state="done"]') is None, "blind, as intended"
    page.context.set_offline(False)
    page.wait_for_selector('[data-health-transport][data-transport="connected"]', timeout=20_000)
    page.wait_for_selector(f'[data-matrix-job="{job_id}"][data-state="done"]', timeout=20_000)
    page.wait_for_function(HAS_DONE, arg=job_id, timeout=20_000)
    assert page.get_attribute("[data-console]", "data-gaps") == "0"
    last_held = int(page.get_attribute("[data-console]", "data-last-event-id"))
    assert last_held == api.get("/operations/events?after=0&limit=1").json()["last_id"]
    ids = page.evaluate(
        "() => [...document.querySelectorAll('[data-tape-rows] [data-event]')].map(li => li.dataset.event)"
    )
    assert len(ids) == len(set(ids)), "an id was painted twice"


def test_the_page_holds_every_event_the_ledger_produced(page, served):
    """Contract 18: after a job, the count held by the page equals the
    count reachable by paging the rows. DOM windows; nothing samples.
    Contract 12: pausing the tape pauses painting, not ingestion."""
    _base, api, _root = served
    api.post("/settings/worker", json={"value": "on"})
    job_id = api.post("/jobs/verify").json()["id"]
    assert _settled(api, job_id) == "done"
    page.wait_for_function(HAS_DONE, arg=job_id, timeout=20_000)
    produced, after = 0, 0
    while True:
        told = api.get(f"/operations/events?after={after}&limit=100").json()
        produced += len(told["events"])
        if told["next_after"] is None:
            break
        after = told["next_after"]
    assert int(page.get_attribute("[data-console]", "data-last-event-id")) == told["last_id"]
    held = int(page.get_attribute("[data-console]", "data-held"))
    # the cold page carried the newest rows and the feed the rest; the
    # "earlier" control pages the remainder until the page holds it all
    while held < produced:
        page.click("[data-tape-earlier]")
        page.wait_for_function(
            "(was) => Number(document.querySelector('[data-console]').dataset.held) > was", arg=held, timeout=10_000
        )
        held = int(page.get_attribute("[data-console]", "data-held"))
    assert held == produced
    assert page.get_attribute("[data-console]", "data-gaps") == "0"
    page.click("[data-tape-pause]")
    new_job = api.post("/jobs/verify").json()["id"]
    assert _settled(api, new_job) == "done"
    page.wait_for_function(
        "(was) => Number(document.querySelector('[data-console]').dataset.held) > was", arg=held, timeout=20_000
    )
    assert "paused" in page.inner_text("[data-tape-count]")
    page.click("[data-tape-pause]")
    page.wait_for_selector(f'[data-tape-rows] [data-event][data-type="job.done"][data-job="{new_job}"]', timeout=10_000)
