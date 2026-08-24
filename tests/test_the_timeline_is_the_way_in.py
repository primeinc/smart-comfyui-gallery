"""The timeline is a primary way into the library, not a probe.

Every shelf and session is a link spelled by the Facet Interface; the
links open galleries ordered by the human moment; the gallery says
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
    with staged(tmp_path_factory, "timeline-links", _library, _interpreted) as stage:
        yield stage


@pytest.fixture
def links(_stage):
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


def test_a_session_is_a_link_that_opens_exactly_its_members(links):
    whole = _density(links, bin="day")
    for s in whole["sessions"]:
        assert "event.id%3Aeq%3A" in s["qs"]
        assert s["qs"].endswith("sort=moment")
        assert _total(links, s["qs"]) == s["pictures"], s
    assert facets.facet("event.id", "eq", "1").key == "event.id"
    with pytest.raises(ValueError, match="allows eq"):
        facets.facet("event.id", "gte", "1")


def test_a_stale_event_refuses_with_the_remedy(links):
    """The facet answers only for runs proven over the current
    interpretation: after the contexts move on, yesterday's session id
    is refused with the job that regroups it -- never an empty room, and
    never whatever the id now means."""
    whole = _density(links, bin="day")
    held = next(s for s in whole["sessions"] if s["domain"] == "wall")
    assert _total(links, held["qs"]) == held["pictures"]
    links.post("/jobs/context", params={"everything": "true"})  # re-interpreting advances the generation
    from tests.test_the_timeline_is_a_surface import _drain

    _drain(links)
    refused = links.get(f"/g?{held['qs']}")
    assert refused.status_code == 400, refused.text
    assert "events job" in refused.json()["detail"]
    assert links.get("/g?f=event.id:eq:999999").status_code == 404, "no such session is not found, not stale"


def test_a_link_orders_by_the_moment_it_opened_on(links):
    qs = "f=context.local_day%3Aeq%3A2023-06-10&sort=moment"
    page = links.get(f"/g?{qs}").text
    names = re.findall(r'alt="([^"]+)"', page)
    assert names[0] == "scan-001.png", "09:00 (the folder's day claim) comes before the afternoon's screenshots"
    assert names[1:4] == [
        "Screenshot 2023-06-10 at 14.00.00.png",
        "Screenshot 2023-06-10 at 14.05.01.png",
        "Screenshot 2023-06-10 at 14.12.02.png",
    ]
    newest = re.findall(r'alt="([^"]+)"', links.get(f"/g?{qs.replace('moment', 'moment-newest')}").text)
    assert newest == list(reversed(names))
    assert links.get("/g?sort=sideways").status_code == 400


def test_the_gallery_shows_its_facets_as_chips_that_can_go(links):
    page = links.get("/g?f=context.local_day%3Aeq%3A2023-06-10&f=context.origin%3Aeq%3Aimported&sort=moment").text
    assert 'data-chip="context.local_day:eq:2023-06-10"' in page
    assert "day 2023-06-10" in page
    assert "origin imported" in page
    removes = re.findall(r'data-chip="([^"]+)">[^<]*<a href="([^"]+)"', page)
    assert removes == [
        ("context.local_day:eq:2023-06-10", "/g?f=context.origin%3Aeq%3Aimported&amp;sort=moment"),
        ("context.origin:eq:imported", "/g?f=context.local_day%3Aeq%3A2023-06-10&amp;sort=moment"),
    ]
    assert "data-chips" not in links.get("/g").text


def test_every_bar_and_session_on_the_page_is_a_link(links):
    """/timeline as a machine reads it is the surface: the bars of the
    opening window and the sessions touching it, each a link that opens
    exactly its pictures; the page renders those same links."""
    body = links.get("/timeline", headers={"accept": "application/json"}).json()
    assert body["bins"]
    for bar in body["bins"]:
        assert _total(links, bar["qs"]) == bar["pictures"], bar
    for session in body["sessions"]:
        assert _total(links, session["qs"]) == session["pictures"]
        assert session["domain"] in ("wall", "instant")
        assert session["start"] is not None
    page = links.get("/timeline", headers={"accept": "text/html"}).text
    for marker in ("data-bin-at=", "data-session-open=", "data-samples", "data-overview", "data-preset="):
        assert marker in page, marker
    coverage = body["coverage"]
    assert (coverage["interpreted"], coverage["present"], coverage["complete"]) == (8, 8, True)
    assert coverage["policy_version"] == pages.context.POLICY_VERSION
    # the downloads' btime disputes nothing on a copy but may on a fresh
    # write: the count is the filesystem's, so only its shape is pinned
    assert isinstance(coverage["contested"], int)
    assert "8 pictures dated" in page


def test_the_surface_carries_pictures_origins_and_its_coverage(links):
    view = _density(links, bin="hour", start=JUNE_10, end=JUNE_10 + DAY)
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


def test_the_pictures_of_a_window_come_with_their_place_on_it(links):
    """/timeline/pictures is what draws pictures ON time: every picture
    of the window in moment order, each with its shape, its moment and
    precision, the sessions it is in, and its link -- the same members
    the window's link opens, the same sessions the surface lists."""
    view = _density(links, bin="hour", start=JUNE_10, end=JUNE_10 + DAY)
    told = links.get(
        "/timeline/pictures", params={"start": JUNE_10, "end": JUNE_10 + DAY}, headers={"accept": "application/json"}
    )
    assert told.status_code == 200, told.text
    body = told.json()
    pictures = body["pictures"]
    link = f"f=context.moment%3Agte%3A{int(JUNE_10)}&f=context.moment%3Alt%3A{int(JUNE_10 + DAY)}"
    assert body["total"] == len(pictures) == _total(links, link)
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
    capped = links.get("/timeline/pictures", params=asked, headers={"accept": "application/json"}).json()
    assert (len(capped["pictures"]), capped["total"]) == (2, body["total"]), "a cap says how many more"
    assert links.get("/timeline/pictures", params={"start": JUNE_10, "end": JUNE_10}).status_code == 400


def test_opening_a_session_is_telling_its_story(links):
    """/stories/sessions/{id} is the session's link: it freezes the
    session, asks for its plan and, once the plan exists, renders and
    sends the person to the story; while the plan is durable work it
    sends them back where they came from, on this site only."""
    from tests.test_the_timeline_is_a_surface import _drain

    whole = _density(links, bin="day")
    held = whole["sessions"][0]
    first = links.get(f"/stories/sessions/{held['id']}", params={"back": "/timeline?x=1"}, follow_redirects=False)
    assert first.status_code == 303, first.text
    where = first.headers["location"]
    assert where.startswith("/stories/renders/") or where == f"/timeline?x=1#session-{held['id']}", where
    if not where.startswith("/stories/renders/"):
        _drain(links)
        again = links.get(f"/stories/sessions/{held['id']}", follow_redirects=False)
        assert again.status_code == 303, again.text
        where = again.headers["location"]
    assert where.startswith("/stories/renders/")
    assert links.get(where, headers={"accept": "application/json"}).status_code == 200
    told = _density(links, bin="day")
    assert next(s for s in told["sessions"] if s["id"] == held["id"])["story"]["href"] == where, (
        "the story rides the session"
    )
    elsewhere = links.get(f"/stories/sessions/{held['id']}", params={"back": "//evil"}, follow_redirects=False)
    assert elsewhere.headers["location"].startswith("/stories/renders/")
    assert links.get("/stories/sessions/999999", follow_redirects=False).status_code == 404


def test_a_range_spreads_to_what_the_asker_can_show_and_a_moment_names_its_picture(links):
    """/timeline/spread answers n pictures of a range spread through it
    -- every k-th in moment order, never the first n -- and never more
    than the asker asked; /timeline/at answers the picture a moment
    points at: the nearest in time, either side."""
    whole = links.get("/timeline/pictures", params={"start": JUNE_10, "end": JUNE_10 + DAY}).json()["pictures"]
    moments = [p["moment"] for p in whole]
    asked = {"start": JUNE_10, "end": JUNE_10 + DAY}
    three = links.get("/timeline/spread", params={**asked, "n": 3}).json()["pictures"]
    assert len(three) == 3
    got = [p["moment"] for p in three]
    assert got == sorted(got)
    assert got[0] == moments[0], "the spread starts at the range's first picture"
    assert got != moments[:3], "and is not simply the first three"
    assert {p["slug"] for p in three} <= {p["slug"] for p in whole}
    everything = links.get("/timeline/spread", params={**asked, "n": 1_000}).json()["pictures"]
    assert [p["moment"] for p in everything] == moments, "asking for more than there is answers all of it, once"
    assert links.get("/timeline/spread", params={**asked, "n": 0}).json()["pictures"], "n is at least one"
    assert links.get("/timeline/spread", params={"start": JUNE_10, "end": JUNE_10, "n": 3}).status_code == 400
    at = links.get("/timeline/at", params={"t": moments[2] + 1}).json()
    assert at["slug"] == whole[2]["slug"], "a second after a picture: that picture"
    just_before = links.get("/timeline/at", params={"t": moments[2] - 1}).json()
    assert just_before["slug"] == whole[2]["slug"], (
        "a second before it: still that picture, not the one an hour earlier"
    )
    before_all = links.get("/timeline/at", params={"t": moments[0] - 10 * 365 * DAY}).json()
    assert before_all["moment"] == moments[0], "before everything: the oldest"
    # by rank: the k-th of n, so a burst spreads over a segment's whole height
    for k, want in ((0, whole[0]), (2, whole[2]), (len(whole) - 1, whole[-1]), (10_000, whole[-1]), (-3, whole[0])):
        told = links.get("/timeline/nth", params={**asked, "k": k}).json()
        assert (told["slug"], told["of"]) == (want["slug"], len(whole)), k
    assert links.get("/timeline/nth", params={"start": 0, "end": 1, "k": 0}).status_code == 404
    assert links.get("/timeline/at", params={"t": JUNE_10, "f": "place.id:eq:999999"}).status_code in (400, 404)


def test_a_week_starts_on_a_monday(links):
    week = _density(links, bin="week", start=JUNE_10 - 3 * DAY, end=JUNE_10 + 4 * DAY)
    assert week["bins"], "the June pictures fall in a week"
    for b in week["bins"]:
        assert (b["at"] - pages.MONDAY) % 604_800 == 0
        assert datetime.datetime.fromtimestamp(b["at"], datetime.UTC).weekday() == 0
    assert week["bin_seconds"] == 604_800


def test_a_picture_says_when_and_in_which_sessions(links):
    page = links.get("/g?f=context.local_day%3Aeq%3A2023-06-10&sort=moment").text
    slugs = re.findall(r'data-slug="([^"]+)"', page)
    item = links.get(f"/i/{slugs[1]}", headers={"accept": "application/json"}).json()
    when = item["when"]
    assert (when["domain"], when["basis"], when["precision"]) == ("wall", "filename", "second")
    assert when["local_day"] == "2023-06-10"
    assert when["day_qs"].startswith("f=context.local_day%3Aeq%3A2023-06-10")
    assert when["timeline"].startswith("/timeline?bin=hour&start=")
    assert [s["kind"] for s in when["sessions"]] == ["file_session"]
    assert _total(links, when["sessions"][0]["qs"]) == 5
    fragment = links.get(f"/i/{slugs[1]}", headers={"hx-request": "true"}).text
    # The overlay carries the day as a link into the gallery. It moved from
    # the old lightbox's stuffed label into the viewer's About panel when
    # one viewer replaced two presentations; what it must still be is an
    # address, not a client-side filter.
    assert 'href="/g?f=context.local_day%3Aeq%3A2023-06-10' in fragment, "the overlay opens the day"
    assert "data-when-day" in fragment
    html = links.get(f"/i/{slugs[1]}", headers={"accept": "text/html"}).text
    for marker in ('data-when data-domain="wall"', "data-when-sessions", "data-when-day", "data-when-session-tell"):
        assert marker in html, marker
    assert when["sessions"][0]["timeline"].startswith("/timeline?bin=hour&start=")


def test_the_contested_count_is_a_link_onto_exactly_the_disputed(links):
    body = links.get("/timeline", headers={"accept": "application/json"}).json()
    coverage = body["coverage"]
    assert "context.disputed%3Aeq%3A1" in coverage["contested_qs"]
    assert _total(links, coverage["contested_qs"]) == coverage["contested"]
    undisputed = _total(links, "f=context.disputed%3Aeq%3A0")
    assert undisputed + coverage["contested"] == coverage["interpreted"], (
        "every interpreted picture is one or the other"
    )
    page = links.get("/g?f=context.disputed%3Aeq%3A1").text
    assert "disputed 1" in page or coverage["contested"] == 0


def test_every_scope_page_opens_its_pictures_in_time_order(links):
    """A folder, an album, a person: each page is a scope the gallery
    answers, and each offers that scope in the order the timeline
    speaks -- sort=moment on the same canonical question."""
    page = links.get("/f/lib", headers={"accept": "text/html"}).text
    assert "data-in-time-order" in page
    href = re.search(r'href="(/g\?[^"]*sort=moment)" data-in-time-order', page)
    assert href is not None
    # `folder=` is the folder's OWN media: the scan lives one folder down
    assert _total(links, href.group(1).replace("/g?", "").replace("&amp;", "&")) == 7


def test_the_session_list_is_bounded_and_says_so(links, monkeypatch):
    """A range touching more sessions than the page lists carries a
    bounded head and the total, never a silent cut; past a smaller bound
    the cards drop their thumbnails and say to zoom in."""
    from sg_web import timeline_view

    whole = _density(links, bin="day")
    assert whole["sessions_total"] == len(whole["sessions"]) == 2
    assert whole["sessions_sampled"] is True
    monkeypatch.setattr(timeline_view, "SESSIONS_MOST", 1)
    capped = _density(links, bin="day")
    assert (capped["sessions_total"], len(capped["sessions"])) == (2, 1)
    assert capped["sessions"][0]["start"] == max(s["start"] for s in whole["sessions"]), "the one listed is the latest"
    monkeypatch.setattr(timeline_view, "SESSIONS_MOST", 200)
    monkeypatch.setattr(timeline_view, "SESSIONS_SAMPLED_MOST", 1)
    bare = _density(links, bin="day")
    assert bare["sessions_sampled"] is False
    assert all(s["samples"] == [] for s in bare["sessions"]), "past the bound the cards carry no thumbnails"


def test_a_session_says_who_is_in_it(links):
    """Two of a session's pictures carry Ana by the primary clustering:
    the session card names her with a link to her page and how many of
    its pictures are hers; a session nobody is in names nobody."""
    import numpy as np

    from db import connect, derived, naming

    whole = _density(links, bin="day")
    assert all(s["people"] == [] for s in whole["sessions"]), "no faces recorded yet"
    assert all(s["people_total"] == 0 for s in whole["sessions"])
    target = whole["sessions"][0]
    conn = connect.connect(links.app.state.db_path)
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
    again = _density(links, bin="day")
    held = next(s for s in again["sessions"] if s["id"] == target["id"])
    assert held["people"] == [{"slug": "ana", "name": "Ana", "href": "/p/ana", "pictures": 2}]
    assert all(s["people"] == [] for s in again["sessions"] if s["id"] != target["id"])


def test_the_surface_can_be_scoped_by_the_gallerys_facets(links):
    """`/timeline?f=place.id:eq:N` draws only the pictures in that place:
    the bins, the samples, the sessions and the page's shelves count the
    scope, and every link carries it, so what is drawn is what opens.
    A session is a link, not a scope; a bad facet is refused."""
    from db import authored, connect, context, places

    whole = _density(links, bin="day")
    all_pictures = sum(b["pictures"] for b in whole["bins"])
    conn = connect.connect(links.app.state.db_path)
    try:
        lisbon = places.named(conn, "Lisbon", "city", 1.0)
        ids = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id LIMIT 2")]
        for file_id in ids:
            authored.set_place(conn, file_id, links.app.state.actor_id, lisbon, 1.0)
            context.rebuild_one(conn, file_id, 1.0)
        conn.commit()
    finally:
        connect.close(conn)
    spelled = f"place.id:eq:{lisbon}"
    scoped = _density(links, bin="day", f=spelled)
    assert sum(b["pictures"] for b in scoped["bins"]) == 2 < all_pictures
    assert scoped["scope"]["parts"] == [{"key": "place.id", "value": lisbon, "spelled": spelled}]
    assert scoped["scope"]["qs"] == f"f=place.id%3Aeq%3A{lisbon}"
    assert all("place.id%3Aeq%3A" in b["qs"] for b in scoped["bins"]), "every link carries the scope"
    assert all("place.id%3Aeq%3A" in s["qs"] for s in scoped["sessions"])
    for s in scoped["sessions"]:
        assert _total(links, s["qs"]) == s["in_scope"] >= 1, "a scoped session's link opens its in-scope members"
        assert s["in_scope"] <= s["pictures"]
    assert all(s["in_scope"] == s["pictures"] for s in whole["sessions"]), "unscoped, every member is in scope"
    assert len(scoped["sessions"]) <= len(whole["sessions"])
    page = links.get("/timeline", params={"f": spelled}, headers={"accept": "text/html"}).text
    assert "data-timeline-scope" in page
    shelf = links.get("/timeline", params={"f": spelled}, headers={"accept": "application/json"}).json()
    assert shelf["extent"]["pictures"] == 2
    assert all("place.id%3Aeq%3A" in b["qs"] for b in shelf["bins"])
    assert all("place.id%3Aeq%3A" in p["href"] for p in shelf["presets"]), "a preset keeps the scope"
    assert links.get("/timeline/density", params={"bin": "day", "f": "event.id:eq:1"}).status_code == 400
    assert links.get("/timeline/density", params={"bin": "day", "f": "vibe:eq:1"}).status_code == 400


def test_every_bars_link_opens_exactly_what_the_bar_counted(links):
    """The finest zoom over a day that holds a day-precision claim: the
    scan's folder-day file sits at midnight inside the first hour's
    window but was never counted in that bar, and its link must not
    open it either -- the link carries the precision the count applied
    (`context.granule`)."""
    whole = _density(links, bin="day")
    day = next(b for b in whole["bins"] if b["pictures"] >= 5)
    hours = _density(links, bin="hour", start=day["at"], end=day["at"] + 86_400)
    assert hours["spans"], "the day-precision claim is drawn as a span, not counted in an hour"
    assert any(b["pictures"] for b in hours["bins"])
    for b in hours["bins"]:
        assert "context.granule%3Alte%3A3600" in b["qs"]
        assert _total(links, b["qs"]) == b["pictures"], f"the bar says {b['pictures']}, the link opens more"
    for b in whole["bins"]:
        assert _total(links, b["qs"]) == b["pictures"]


def test_a_session_filter_lands_on_the_sessions_range(links):
    """The gallery's "on the timeline" link carries the gallery's
    filters, a session among them. The timeline shows every session of a
    range, so a session filter is answered with the session's own hour
    range, the other filters kept -- never a 400 from a link a page
    emitted."""
    whole = _density(links, bin="day")
    held = whole["sessions"][0]
    answer = links.get(f"/timeline?{held['qs']}&f=context.disputed%3Aeq%3A0", follow_redirects=False)
    assert answer.status_code == 303, answer.text
    landed = answer.headers["location"]
    assert "event.id" not in landed
    assert "f=context.disputed%3Aeq%3A0" in landed
    assert re.search(r"start=\d+&end=\d+", landed)
    page = links.get(landed, headers={"accept": "application/json"})
    assert page.status_code == 200, page.text
    assert any(s["id"] == held["id"] for s in page.json()["sessions"]), "the session is on its own range"
    assert links.get("/timeline?f=event.id%3Aeq%3A999999").status_code == 404


def test_every_link_a_one_second_timeline_emits_lands_on_a_page(links):
    """A scope whose pictures all sit in one second: the scrubber must
    not append a gap after the last picture whose range clips to
    nothing. Every timeline link on the page answers 200."""
    whole = _density(links, bin="day")
    moment = whole["bins"][0]["at"]
    one = links.get(f"/g?f=context.moment%3Agte%3A{int(moment)}&f=context.moment%3Alt%3A{int(moment) + DAY:.0f}").text
    found = re.search(r'data-slug="([^"]+)"', one)
    assert found is not None, "a picture on the day"
    when = links.get(f"/i/{found.group(1)}", headers={"accept": "application/json"}).json()["when"]["moment"]
    lo, hi = int(when), int(when) + 1
    qs = f"f=context.moment%3Agte%3A{lo}&f=context.moment%3Alt%3A{hi}"
    page = links.get(f"/timeline?{qs}", headers={"accept": "text/html"})
    assert page.status_code == 200, page.text
    hrefs = [h.replace("&amp;", "&") for h in re.findall(r'href="(/timeline\?[^"]+)"', page.text)]
    assert hrefs, "the page links onto itself"
    for href in hrefs:
        found = re.search(r"start=(\d+)&end=(\d+)", href)
        if found:
            assert int(found.group(1)) < int(found.group(2)), href
        assert links.get(href, headers={"accept": "text/html"}).status_code == 200, href
