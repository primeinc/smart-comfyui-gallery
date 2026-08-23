"""The timeline is a primary way into the library, not a probe.

Every shelf and session is a door spelled by the Facet Interface; the
doors open galleries ordered by the human moment; the gallery says
which question it is answering and lets one chip go; a picture's page
says when it happened, on what evidence, and which sessions it is in;
the surface carries pictures, its weeks start on Monday, and it says how
much of the library it can see.
"""

from __future__ import annotations

import datetime
import re

import pytest

from db import facets, pages
from tests.staging import staged
from tests.test_the_timeline_is_a_surface import DAY, JUNE_10, _interpreted, _library


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "timeline-doors", _library, _interpreted) as stage:
        yield stage


@pytest.fixture
def doors(_stage):
    _stage.restore()
    return _stage.client


def _total(client, qs: str) -> int:
    opened = client.get(f"/g?{qs}")
    assert opened.status_code == 200, opened.text
    found = re.search(r'data-total="(\d+)"', opened.text)
    assert found is not None
    return int(found.group(1))


def _density(client, **params):
    told = client.get("/timeline/density", params=params, headers={"accept": "application/json"})
    assert told.status_code == 200, told.text
    return told.json()


def test_a_session_is_a_door_that_opens_exactly_its_members(doors):
    whole = _density(doors, bin="day")
    for s in whole["sessions"]:
        assert "event.id%3Aeq%3A" in s["qs"]
        assert s["qs"].endswith("sort=moment")
        assert _total(doors, s["qs"]) == s["pictures"], s
    assert facets.facet("event.id", "eq", "1").key == "event.id"
    with pytest.raises(ValueError, match="allows eq"):
        facets.facet("event.id", "gte", "1")


def test_a_stale_event_refuses_with_the_remedy(doors):
    """The facet answers only for runs proven over the current
    interpretation: after the contexts move on, yesterday's session id
    is refused with the job that regroups it -- never an empty room, and
    never whatever the id now means."""
    whole = _density(doors, bin="day")
    held = next(s for s in whole["sessions"] if s["domain"] == "wall")
    assert _total(doors, held["qs"]) == held["pictures"]
    doors.post("/jobs/context", params={"everything": "true"})  # re-interpreting advances the generation
    from tests.test_the_timeline_is_a_surface import _drain

    _drain(doors)
    refused = doors.get(f"/g?{held['qs']}")
    assert refused.status_code == 400, refused.text
    assert "events job" in refused.json()["detail"]
    assert doors.get("/g?f=event.id:eq:999999").status_code == 404, "no such session is not found, not stale"


def test_a_door_orders_by_the_moment_it_opened_on(doors):
    qs = "f=context.local_day%3Aeq%3A2023-06-10&sort=moment"
    page = doors.get(f"/g?{qs}").text
    names = re.findall(r'alt="([^"]+)"', page)
    assert names[0] == "scan-001.png", "09:00 (the folder's day claim) comes before the afternoon's screenshots"
    assert names[1:4] == [
        "Screenshot 2023-06-10 at 14.00.00.png",
        "Screenshot 2023-06-10 at 14.05.01.png",
        "Screenshot 2023-06-10 at 14.12.02.png",
    ]
    newest = re.findall(r'alt="([^"]+)"', doors.get(f"/g?{qs.replace('moment', 'moment-newest')}").text)
    assert newest == list(reversed(names))
    assert doors.get("/g?sort=sideways").status_code == 400


def test_the_gallery_shows_its_facets_as_chips_that_can_go(doors):
    page = doors.get("/g?f=context.local_day%3Aeq%3A2023-06-10&f=context.origin%3Aeq%3Aimported&sort=moment").text
    assert 'data-chip="context.local_day:eq:2023-06-10"' in page
    assert "day 2023-06-10" in page
    assert "origin imported" in page
    removes = re.findall(r'data-chip="([^"]+)">[^<]*<a href="([^"]+)"', page)
    assert removes == [
        ("context.local_day:eq:2023-06-10", "/g?f=context.origin%3Aeq%3Aimported&amp;sort=moment"),
        ("context.origin:eq:imported", "/g?f=context.local_day%3Aeq%3A2023-06-10&amp;sort=moment"),
    ]
    assert "data-chips" not in doors.get("/g").text


def test_every_bar_and_session_on_the_page_is_a_door(doors):
    """/timeline as a machine reads it is the surface: the bars of the
    opening window and the sessions touching it, each a door that opens
    exactly its pictures; the page renders those same doors."""
    body = doors.get("/timeline", headers={"accept": "application/json"}).json()
    assert body["bins"]
    for bar in body["bins"]:
        assert _total(doors, bar["qs"]) == bar["pictures"], bar
    for session in body["sessions"]:
        assert _total(doors, session["qs"]) == session["pictures"]
        assert session["domain"] in ("wall", "instant")
        assert session["start"] is not None
    page = doors.get("/timeline", headers={"accept": "text/html"}).text
    for marker in ("data-bin-at=", "data-session-open=", "data-samples", "data-overview", "data-preset="):
        assert marker in page, marker
    coverage = body["coverage"]
    assert (coverage["interpreted"], coverage["present"], coverage["complete"]) == (8, 8, True)
    assert coverage["policy_version"] == pages.context.POLICY_VERSION
    # the downloads' btime disputes nothing on a copy but may on a fresh
    # write: the count is the filesystem's, so only its shape is pinned
    assert isinstance(coverage["contested"], int)
    assert "8 pictures dated" in page


def test_the_surface_carries_pictures_origins_and_its_coverage(doors):
    view = _density(doors, bin="hour", start=JUNE_10, end=JUNE_10 + DAY)
    assert view["sampled"] is True
    for b in view["bins"]:
        assert 1 <= len(b["samples"]) <= pages.SAMPLES_PER_BIN
        assert sum(b["origin"].values()) == b["pictures"]
        assert b["origin"]["imported"] == b["pictures"], "screenshots carry neither camera nor generator"
    assert view["coverage"]["complete"] is True
    for s in view["sessions"]:
        assert 1 <= len(s["samples"]) <= pages.SAMPLES_PER_SESSION
        assert s["tellable"] is True
        assert s["planner"] == "file_history"


def test_the_pictures_of_a_window_come_with_their_place_on_it(doors):
    """/timeline/pictures is what draws pictures ON time: every picture
    of the window in moment order, each with its shape, its moment and
    precision, the sessions it is in, and its door -- the same members
    the window's door opens, the same sessions the surface lists."""
    view = _density(doors, bin="hour", start=JUNE_10, end=JUNE_10 + DAY)
    told = doors.get(
        "/timeline/pictures", params={"start": JUNE_10, "end": JUNE_10 + DAY}, headers={"accept": "application/json"}
    )
    assert told.status_code == 200, told.text
    body = told.json()
    pictures = body["pictures"]
    door = f"f=context.moment%3Agte%3A{int(JUNE_10)}&f=context.moment%3Alt%3A{int(JUNE_10 + DAY)}"
    assert body["total"] == len(pictures) == _total(doors, door)
    assert [p["moment"] for p in pictures] == sorted(p["moment"] for p in pictures)
    for p in pictures:
        assert JUNE_10 <= p["moment"] < JUNE_10 + DAY
        assert p["width"], "a picture knows its shape"
        assert p["height"]
        assert p["precision"] in ("day", "hour", "minute", "second", "subsecond")
        assert p["domain"] in ("wall", "instant")
        assert p["href"].startswith(f"/i/{p['slug']}?")
    listed = {s["id"] for s in view["sessions"]}
    named = {sid for p in pictures for sid in p["sessions"]}
    assert named == listed, "the sessions named are the ones the surface lists"
    for s in view["sessions"]:
        assert sum(s["id"] in p["sessions"] for p in pictures) == s["pictures"]
    asked = {"start": JUNE_10, "end": JUNE_10 + DAY, "limit": 2}
    capped = doors.get("/timeline/pictures", params=asked, headers={"accept": "application/json"}).json()
    assert (len(capped["pictures"]), capped["total"]) == (2, body["total"]), "a cap says how many more"
    assert doors.get("/timeline/pictures", params={"start": JUNE_10, "end": JUNE_10}).status_code == 400


def test_a_week_starts_on_a_monday(doors):
    week = _density(doors, bin="week", start=JUNE_10 - 3 * DAY, end=JUNE_10 + 4 * DAY)
    assert week["bins"], "the June pictures fall in a week"
    for b in week["bins"]:
        assert (b["at"] - pages.MONDAY) % 604_800 == 0
        assert datetime.datetime.fromtimestamp(b["at"], datetime.UTC).weekday() == 0
    assert week["bin_seconds"] == 604_800


def test_a_picture_says_when_and_in_which_sessions(doors):
    page = doors.get("/g?f=context.local_day%3Aeq%3A2023-06-10&sort=moment").text
    slugs = re.findall(r'data-slug="([^"]+)"', page)
    item = doors.get(f"/i/{slugs[1]}", headers={"accept": "application/json"}).json()
    when = item["when"]
    assert (when["domain"], when["basis"], when["precision"]) == ("wall", "filename", "second")
    assert when["local_day"] == "2023-06-10"
    assert when["day_qs"].startswith("f=context.local_day%3Aeq%3A2023-06-10")
    assert when["timeline"].startswith("/timeline?bin=hour&start=")
    assert [s["kind"] for s in when["sessions"]] == ["file_session"]
    assert _total(doors, when["sessions"][0]["qs"]) == 5
    fragment = doors.get(f"/i/{slugs[1]}", headers={"hx-request": "true"}).text
    assert 'data-lightbox-day href="/g?f=context.local_day%3Aeq%3A2023-06-10' in fragment, "the lightbox opens the day"
    html = doors.get(f"/i/{slugs[1]}", headers={"accept": "text/html"}).text
    for marker in ('data-when data-domain="wall"', "data-when-sessions", "data-when-day", "data-when-session-tell"):
        assert marker in html, marker
    assert when["sessions"][0]["timeline"].startswith("/timeline?bin=hour&start=")


def test_the_contested_count_is_a_door_onto_exactly_the_disputed(doors):
    body = doors.get("/timeline", headers={"accept": "application/json"}).json()
    coverage = body["coverage"]
    assert "context.disputed%3Aeq%3A1" in coverage["contested_qs"]
    assert _total(doors, coverage["contested_qs"]) == coverage["contested"]
    undisputed = _total(doors, "f=context.disputed%3Aeq%3A0")
    assert undisputed + coverage["contested"] == coverage["interpreted"], (
        "every interpreted picture is one or the other"
    )
    page = doors.get("/g?f=context.disputed%3Aeq%3A1").text
    assert "disputed 1" in page or coverage["contested"] == 0


def test_every_scope_page_opens_its_pictures_in_time_order(doors):
    """A folder, an album, a person: each page is a scope the gallery
    answers, and each offers that scope in the order the timeline
    speaks -- sort=moment on the same canonical question."""
    page = doors.get("/f/lib", headers={"accept": "text/html"}).text
    assert "data-in-time-order" in page
    href = re.search(r'href="(/g\?[^"]*sort=moment)" data-in-time-order', page)
    assert href is not None
    # `folder=` is the folder's OWN media: the scan lives one folder down
    assert _total(doors, href.group(1).replace("/g?", "").replace("&amp;", "&")) == 7


def test_the_session_list_is_bounded_and_says_so(doors, monkeypatch):
    """A range touching more sessions than the page lists carries a
    bounded head and the total, never a silent cut; past a smaller bound
    the cards drop their thumbnails and say to zoom in."""
    from sg_web import timeline_view

    whole = _density(doors, bin="day")
    assert whole["sessions_total"] == len(whole["sessions"]) == 2
    assert whole["sessions_sampled"] is True
    monkeypatch.setattr(timeline_view, "SESSIONS_MOST", 1)
    capped = _density(doors, bin="day")
    assert (capped["sessions_total"], len(capped["sessions"])) == (2, 1)
    assert capped["sessions"][0]["start"] == max(s["start"] for s in whole["sessions"]), "the one listed is the latest"
    monkeypatch.setattr(timeline_view, "SESSIONS_MOST", 200)
    monkeypatch.setattr(timeline_view, "SESSIONS_SAMPLED_MOST", 1)
    bare = _density(doors, bin="day")
    assert bare["sessions_sampled"] is False
    assert all(s["samples"] == [] for s in bare["sessions"]), "past the bound the cards carry no thumbnails"


def test_a_session_says_who_is_in_it(doors):
    """Two of a session's pictures carry Ana by the primary clustering:
    the session card names her with a door to her page and how many of
    its pictures are hers; a session nobody is in names nobody."""
    import numpy as np

    from db import connect, derived, naming

    whole = _density(doors, bin="day")
    assert all(s["people"] == [] for s in whole["sessions"]), "no faces recorded yet"
    assert all(s["people_total"] == 0 for s in whole["sessions"])
    target = whole["sessions"][0]
    conn = connect.connect(doors.app.state.db_path)
    try:
        members = [
            row[0]
            for row in conn.execute(
                "SELECT file_id FROM derived_event_file WHERE event_id = ? ORDER BY file_id LIMIT 2", (target["id"],)
            )
        ]
        assert len(members) == 2
        rng = np.random.default_rng(5)
        ana = rng.standard_normal(32).astype(np.float32)
        for file_id in members:
            derived.record_faces(
                conn,
                file_id,
                "test/embedder",
                "1",
                "aa",
                0.0,
                [
                    {
                        "region": derived.region(conn, 0.1, 0.1, 0.2, 0.2),
                        "embedding": (ana + 0.01 * rng.standard_normal(32).astype(np.float32)).tobytes(),
                    }
                ],
            )
        [cluster_id] = derived.cluster(conn, "test/embedder", "1", 0.0, threshold=0.55)
        run_id = derived.run_for(conn, "test/embedder", "1", "chinese-whispers", 0.55, 0.0)
        derived.make_primary(conn, run_id)
        person_id = naming.claim(conn, "person", "Ana")
        conn.execute("INSERT INTO person(id,name,created_at) VALUES(?, 'Ana', 0)", (person_id,))
        conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE id = ?", (person_id, cluster_id))
        for file_id in members:
            derived.attribute(conn, file_id, person_id, run_id, "test/embedder", "1")
        conn.commit()
    finally:
        connect.close(conn)
    again = _density(doors, bin="day")
    held = next(s for s in again["sessions"] if s["id"] == target["id"])
    assert held["people"] == [{"slug": "ana", "name": "Ana", "href": "/p/ana", "pictures": 2}]
    assert all(s["people"] == [] for s in again["sessions"] if s["id"] != target["id"])


def test_the_surface_can_be_scoped_by_the_gallerys_facets(doors):
    """`/timeline?f=place.id:eq:N` draws only the pictures in that place:
    the bins, the samples, the sessions and the page's shelves count the
    scope, and every door carries it, so what is drawn is what opens.
    A session is a door, not a scope; a bad facet is refused."""
    from db import authored, connect, context, places

    whole = _density(doors, bin="day")
    all_pictures = sum(b["pictures"] for b in whole["bins"])
    conn = connect.connect(doors.app.state.db_path)
    try:
        lisbon = places.named(conn, "Lisbon", "city", 1.0)
        ids = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id LIMIT 2")]
        for file_id in ids:
            authored.set_place(conn, file_id, doors.app.state.actor_id, lisbon, 1.0)
            context.rebuild_one(conn, file_id, 1.0)
        conn.commit()
    finally:
        connect.close(conn)
    spelled = f"place.id:eq:{lisbon}"
    scoped = _density(doors, bin="day", f=spelled)
    assert sum(b["pictures"] for b in scoped["bins"]) == 2 < all_pictures
    assert scoped["scope"]["parts"] == [{"key": "place.id", "value": lisbon, "spelled": spelled}]
    assert scoped["scope"]["qs"] == f"f=place.id%3Aeq%3A{lisbon}"
    assert all("place.id%3Aeq%3A" in b["qs"] for b in scoped["bins"]), "every door carries the scope"
    assert all("place.id%3Aeq%3A" in s["qs"] for s in scoped["sessions"])
    for s in scoped["sessions"]:
        assert _total(doors, s["qs"]) == s["in_scope"] >= 1, "a scoped session's door opens its in-scope members"
        assert s["in_scope"] <= s["pictures"]
    assert all(s["in_scope"] == s["pictures"] for s in whole["sessions"]), "unscoped, every member is in scope"
    assert len(scoped["sessions"]) <= len(whole["sessions"])
    page = doors.get("/timeline", params={"f": spelled}, headers={"accept": "text/html"}).text
    assert "data-timeline-scope" in page
    shelf = doors.get("/timeline", params={"f": spelled}, headers={"accept": "application/json"}).json()
    assert shelf["extent"]["pictures"] == 2
    assert all("place.id%3Aeq%3A" in b["qs"] for b in shelf["bins"])
    assert all("place.id%3Aeq%3A" in p["href"] for p in shelf["presets"]), "a preset keeps the scope"
    assert doors.get("/timeline/density", params={"bin": "day", "f": "event.id:eq:1"}).status_code == 400
    assert doors.get("/timeline/density", params={"bin": "day", "f": "vibe:eq:1"}).status_code == 400


def test_every_bars_door_opens_exactly_what_the_bar_counted(doors):
    """The finest zoom over a day that holds a day-precision claim: the
    scan's folder-day file sits at midnight inside the first hour's
    window but was never counted in that bar, and its door must not
    open it either -- the door carries the precision the count applied
    (`context.granule`)."""
    whole = _density(doors, bin="day")
    day = next(b for b in whole["bins"] if b["pictures"] >= 5)
    hours = _density(doors, bin="hour", start=day["at"], end=day["at"] + 86_400)
    assert hours["spans"], "the day-precision claim is drawn as a span, not counted in an hour"
    assert any(b["pictures"] for b in hours["bins"])
    for b in hours["bins"]:
        assert "context.granule%3Alte%3A3600" in b["qs"]
        assert _total(doors, b["qs"]) == b["pictures"], f"the bar says {b['pictures']}, the door opens more"
    for b in whole["bins"]:
        assert _total(doors, b["qs"]) == b["pictures"]
