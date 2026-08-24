"""A search answers with a SET, not the library in a different order.

Every semantic space scores every file it holds. A text encoder maps
any phrase somewhere and its nearest neighbours always exist, so a
ranking never ends on its own -- which is why asking this library for
`a cat` and for `xyzzy plugh frobnitz` both returned 2,995 files across
50 pages, the same 2,995 being every file that had ever been embedded.
That is not an answer to either question. It is the library, shuffled.

No absolute cosine says "related": measured over the real library the
nonsense phrase scored HIGHER on OpenCLIP (max .263) than `a
photograph of a mountain landscape` (max .218), so a floor would keep
the nonsense and discard the answer. What DOES carry between phrases is
the shape of one phrase's own distribution -- a head standing above its
own median, or no head at all. `retrieval.head` cuts there.

The consequence a person sees: the grid stops at the point the ranking
stops having anything to say, every cell carries how far above the
middle it stands, and `depth=all` is there for whoever wants the whole
ranked library back.
"""

from __future__ import annotations

import pathlib
import statistics

import numpy as np
import pytest
from PIL import Image

from db import connect, derived, resultset, retrieval, scan, settings
from vision import semantic

NOW = 1_700_000_000.0
SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
CLIP = ("openclip", "ViT-B-32", "laion2b_s34b_b79k")


class Asks:
    """One fixed probe, so a picture's recorded cosine IS its score."""

    def encode_query(self, phrase):
        probe = np.zeros(64, dtype=np.float32)
        probe[0] = 1.0
        return probe


@pytest.fixture
def asks(monkeypatch):
    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Asks())


def _shelf(tmp_path, cosines: list[float]):
    """A library whose files score exactly these cosines, in this order."""
    root = tmp_path / "pics"
    root.mkdir()
    for i in range(len(cosines)):
        Image.new("RGB", (8, 8), (i % 256, (i * 7) % 256, 60)).save(root / f"p{i:04d}.png")
    conn = connect.memory()
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    ids = {name: file_id for file_id, name in conn.execute("SELECT id, replace(name, '.png', '') FROM file")}
    shas = dict(conn.execute("SELECT id, content_sha256 FROM file"))
    space = semantic.space(*CLIP, 64)
    for i, cosine in enumerate(cosines):
        file_id = ids[f"p{i:04d}"]
        v = np.zeros(64, dtype=np.float32)
        v[0] = cosine
        # The remaining norm is spread over the other axes, so every
        # picture is a unit vector and the probe reads back `cosine`.
        v[1 + (i % 63)] = np.sqrt(max(0.0, 1.0 - cosine * cosine))
        derived.record_embedding(conn, file_id, space, v, shas[file_id], NOW)
    settings.put(conn, "semantic_model", "ViT-B-32/laion2b_s34b_b79k")
    conn.commit()
    return conn


#: A sharp head: eight pictures well clear of a body of ninety-two.
SHARP = [0.90] * 8 + [0.10 + (i % 7) * 0.001 for i in range(92)]
#: No head, the shape a real ranking makes: a normal density, sampled
#: at its own even quantiles so the fixture carries no randomness. This
#: is what a phrase nothing answers looks like -- the whole library
#: bunched on its own median, the best of it three deviations up. Half
#: that span is +1.5 sigma, which is the top ~7%: the same order as the
#: 8% the nonsense phrases actually kept over the real library.
BELL = [statistics.NormalDist(0.15, 0.03).inv_cdf((i + 0.5) / 601) for i in range(601)]
#: No head, the worst case the rule can be handed: a hundred scores in
#: a perfectly even ramp, so a quarter of them are in the top half of
#: the span whatever the cut. Real distributions are never this.
RAMP = [0.30 + i * 0.001 for i in range(100)]


# --- the cut itself ---------------------------------------------------------


def test_a_head_that_stands_above_its_own_middle_is_the_answer():
    assert retrieval.head(sorted(SHARP, reverse=True)) == 8


def test_a_ranking_with_no_head_answers_with_almost_nothing():
    """The library sitting on its own median. The rule must not mistake
    "all equally close" for "all relevant"."""
    kept = retrieval.head(sorted(BELL, reverse=True))
    assert kept <= len(BELL) // 10, f"a headless ranking kept {kept} of {len(BELL)}"


def test_the_most_a_headless_ranking_can_keep_is_a_quarter():
    """The rule's ceiling, asserted on the shape that reaches it: an
    even ramp puts a quarter of its scores in the top half of the span.
    Nothing can make this ranking answer -- the guarantee is only that
    the answer is never the whole library."""
    kept = retrieval.head(sorted(RAMP, reverse=True))
    assert len(RAMP) // 5 <= kept <= len(RAMP) // 4 + 1, kept


def test_a_ranking_nothing_could_answer_is_empty_not_everything():
    assert retrieval.head([]) == 0
    assert retrieval.head([0.2] * 40) == 0, "a perfectly flat ranking has no head at all"


# --- what the answer contains -----------------------------------------------


def test_a_search_returns_fewer_files_than_it_ranked(tmp_path, asks):
    """The defect, stated as an assertion: 100 files in, 8 out."""
    conn = _shelf(tmp_path, SHARP)
    shape = resultset.page(conn, str(tmp_path), resultset.parse(text="anything"), 1, NOW)
    assert conn.execute("SELECT count(*) FROM derived_embedding").fetchone()[0] == 100
    assert shape["total"] == 8, "the ranking stops where it stops answering"
    assert shape["pages"] == 1, "one page, not the whole library paginated"


def test_the_answer_says_how_much_it_ranked_to_get_there(tmp_path, asks):
    """A cut nobody can see is indistinguishable from a small library."""
    conn = _shelf(tmp_path, SHARP)
    shape = resultset.describe(conn, str(tmp_path), resultset.parse(text="anything"), NOW)
    told = shape["provenance"]
    assert told["ranked"] == 100
    assert told["answering"] == 8


def test_every_cell_carries_how_far_above_the_middle_it_stands(tmp_path, asks):
    """The UI gave no indication of relevance because the answer
    carried none. One number, and it is the SAME number the cut is
    made on -- so what a person sees explains where the page ended."""
    conn = _shelf(tmp_path, SHARP)
    shape = resultset.page(conn, str(tmp_path), resultset.parse(text="anything"), 1, NOW)
    for item in shape["items"]:
        assert 0.0 <= item["relevance"] <= 1.0, item
        assert item["relevance"] >= retrieval.HEAD_SPAN, "a member of the head stands at least that far up"


def test_a_timed_answer_has_no_relevance_to_report(tmp_path, asks):
    """Nothing was asked of any space, so there is no such quantity."""
    conn = _shelf(tmp_path, SHARP)
    shape = resultset.page(conn, str(tmp_path), resultset.parse(), 1, NOW)
    assert shape["total"] == 100
    assert all(item["relevance"] is None for item in shape["items"])


# --- the escape -------------------------------------------------------------


def test_depth_all_gives_back_the_whole_ranked_library(tmp_path, asks):
    conn = _shelf(tmp_path, SHARP)
    shape = resultset.page(conn, str(tmp_path), resultset.parse(text="anything", depth="all"), 1, NOW)
    assert shape["total"] == 100, "the old behaviour, asked for on purpose"


def test_depth_rides_the_url_so_the_answer_is_addressable(tmp_path, asks):
    assert "depth=all" in resultset.canonical(resultset.parse(text="cat", depth="all"))
    assert "depth" not in resultset.canonical(resultset.parse(text="cat")), "the default is not spelled"


def test_depth_without_a_phrase_is_refused(tmp_path):
    """A time sort ranks nothing, so there is no head to keep."""
    with pytest.raises(ValueError, match="depth"):
        resultset.parse(depth="all")


def test_an_unknown_depth_is_refused(tmp_path):
    with pytest.raises(ValueError, match="depth"):
        resultset.parse(text="cat", depth="deep")


def test_the_two_depths_are_two_questions(tmp_path, asks):
    """Same phrase, different answers -- so they must not share a
    projection, or one would be served under the other's key."""
    conn = _shelf(tmp_path, SHARP)
    head = resultset.describe(conn, str(tmp_path), resultset.parse(text="anything"), NOW)
    whole = resultset.describe(conn, str(tmp_path), resultset.parse(text="anything", depth="all"), NOW)
    assert head["fingerprint"] != whole["fingerprint"]
