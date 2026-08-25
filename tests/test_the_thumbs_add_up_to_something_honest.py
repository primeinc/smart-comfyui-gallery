"""What the verdicts say, and what they must refuse to say.

Counting them is trivial. Everything worth testing here is the refusing,
because this is the kind of surface that lies confidently: a percentage
with an invisible error bar, computed from a sample nobody drew at
random, next to a model's name.

Three rules, one test each, and each is a thing the numbers would say
wrong if it were left out:

    a biased sample     people judge what they LOOK at, and reach for
                        `wrong` far sooner than `right`. So a raw error
                        rate is a statement about which pictures got
                        opened -- what survives is a comparison between
                        producers over the files BOTH were judged on.
    say the n           a model is not worse than another on four
                        verdicts. Below the floor: how many more are
                        needed, never a number.
    never a cause       an observation about a correlation is worth
                        showing and is not worth the word `because`.
"""

from __future__ import annotations

import uuid

import pytest

from db import authored, verdicts
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0
BLIP = ("Salesforce/blip-image-captioning-base", "main")
OTHER = ("other/captioner", "v2")


@pytest.fixture
def library():
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")
    conn.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'me','x','USER',0)")
    for at in range(2, 42):
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, uuid.uuid4().bytes, f"f{at}"))
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1,0,0,0)",
            (at, f"f{at}.png"),
        )
    conn.commit()
    yield conn
    conn.close()


def _judge(conn, producer, files, verdict):
    for file_id in files:
        authored.feedback(
            conn,
            "annotation",
            verdict,
            NOW,
            file_id=file_id,
            annotation_kind="caption",
            user_id=1,
            model_id=producer[0],
            model_version=producer[1],
        )
    conn.commit()


def test_a_producer_is_counted_by_what_it_was_told(library):
    _judge(library, BLIP, range(2, 14), "wrong")
    _judge(library, BLIP, range(14, 20), "right")
    told = verdicts.by_producer(library)
    assert len(told) == 1
    one = told[0]
    assert (one.model_id, one.wrong, one.right, one.judged) == (BLIP[0], 12, 6, 18)
    assert one.wrong_share == pytest.approx(12 / 18)


def test_below_the_floor_it_says_how_many_more_it_needs(library):
    """A model is not worse than another on four verdicts. `None`, never
    zero: a zero would read as "never wrong"."""
    _judge(library, BLIP, range(2, 6), "wrong")
    one = verdicts.by_producer(library)[0]
    assert one.judged == 4
    assert one.enough is False
    assert one.wrong_share is None, "a rate under the floor is not a rate"
    assert one.needs == verdicts.ENOUGH - 4


def test_the_comparison_is_over_the_files_both_were_judged_on(library):
    """The rule the whole module turns on.

    Over ALL verdicts these two look nothing alike: one was judged forty
    times and the other five. Over the files BOTH were judged on, the
    person, the day and the pictures are shared, and what is left is the
    difference between the models.
    """
    shared = list(range(2, 22))
    _judge(library, BLIP, shared, "wrong")
    _judge(library, OTHER, shared, "right")
    # and a tail only one of them was ever judged on, which must not
    # move the comparison at all
    _judge(library, BLIP, range(22, 42), "right")

    (held,) = verdicts.contests(library)
    assert held.shared == len(shared)
    assert held.wrong[BLIP] == len(shared)
    assert held.wrong[OTHER] == 0

    # the raw rates disagree with the contest, which is the point
    rates = {(one.model_id, one.model_version): one.wrong_share for one in verdicts.by_producer(library)}
    assert rates[BLIP] == pytest.approx(20 / 40), "over everything it looks half wrong"
    assert held.wrong[BLIP] / held.shared == 1.0, "over the shared files it was wrong every time"


def test_two_producers_never_judged_together_are_not_compared(library):
    """No shared files, no comparison. An empty intersection is an
    answer -- inventing one from two disjoint samples is the exact
    mistake the contest exists to avoid."""
    _judge(library, BLIP, range(2, 22), "wrong")
    _judge(library, OTHER, range(22, 42), "right")
    assert verdicts.contests(library) == []


def test_a_thin_contest_says_it_is_thin(library):
    _judge(library, BLIP, range(2, 5), "wrong")
    _judge(library, OTHER, range(2, 5), "right")
    (held,) = verdicts.contests(library)
    assert held.shared == 3
    assert held.enough is False


def test_a_rebuild_of_the_derived_layer_changes_no_number(library):
    """Read from `feedback` alone and never joined to the annotations:
    the judgement is the durable half and the annotation is the
    disposable one, so a join would make a re-run look like people
    changed their minds."""
    _judge(library, BLIP, range(2, 14), "wrong")
    before = verdicts.by_producer(library)
    library.execute("DELETE FROM derived_annotation")
    library.commit()
    assert verdicts.by_producer(library) == before


def test_a_verdict_with_no_producer_is_left_out_rather_than_lumped(library):
    """A judgement from before the producer columns existed is real and
    is not attributable. Counting it under some model would invent an
    attribution; counting it under "unknown" would put a bucket in a
    list of models. It is simply not in this answer."""
    authored.feedback(library, "annotation", "wrong", NOW, file_id=2, annotation_kind="caption", user_id=1)
    library.commit()
    assert verdicts.by_producer(library) == []
    assert authored.standing_verdict(library, 2, "caption", None, None, 1) == "wrong", "still recorded, still findable"


# --- and what the console shows of it ---------------------------------------


def test_the_console_shows_the_counts_and_refuses_the_rate(tmp_path):
    """End to end, and the refusal is the part that matters: under the
    floor the page says how many more are needed rather than printing a
    percentage a person would act on."""
    import time as clock

    from litestar.testing import TestClient
    from PIL import Image

    from db import connect, derived
    from sg_web.app import build_app

    root = tmp_path / "pics"
    root.mkdir()
    for i in range(4):
        Image.new("RGB", (16, 12), (30 * i, 90, 140)).save(root / f"p{i}.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            files = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id")]
            for file_id in files:
                sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
                derived.annotate(conn, file_id, "caption", "a thing", BLIP[0], BLIP[1], sha, clock.time())
            conn.commit()
        finally:
            connect.close(conn)

        # nothing judged yet: the section is absent, which is honest --
        # no verdicts is not a model with nothing wrong with it
        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert "data-operations-judged" not in page

        conn = connect.connect(client.app.state.db_path)
        try:
            for file_id in files[:3]:
                authored.feedback(
                    conn,
                    "annotation",
                    "wrong",
                    clock.time(),
                    file_id=file_id,
                    annotation_kind="caption",
                    model_id=BLIP[0],
                    model_version=BLIP[1],
                )
            conn.commit()
        finally:
            connect.close(conn)

        told = client.get("/operations/overview").json()["judged"]
        assert told["floor"] == verdicts.ENOUGH
        assert told["producers"][0]["judged"] == 3
        assert told["producers"][0]["wrong_share"] is None

        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert "data-operations-judged" in page
        assert "data-judged-needs" in page, "under the floor it says what it needs"
        assert "% wrong" not in page, "and does not print a rate it cannot support"
