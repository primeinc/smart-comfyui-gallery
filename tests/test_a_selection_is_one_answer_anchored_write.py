"""Bulk curation: one desired fact, one proven selection, one write.

A selection names the answer it was made against and the entity uuids
it chose. The server proves both against the authoritative projection
OUTSIDE any write transaction -- proving may cost a materialization,
and sqlite's one writer lane never pays for it -- then claims the lane
for a narrow revalidate-and-mutate: one currency comparison, the
authored write, one commit. A commit in the handoff earns one re-proof
outside the lane; a stale answer, a foreign file, or a vanished entity
writes zero rows. The response's after-state lets the client settle by
answer identity exactly as single writes do.
"""

from __future__ import annotations

import os
import pathlib
import re
import sqlite3

import pytest
from PIL import Image

from db import authored, collections, connect, resultset
from tests import retrieving
from tests.staging import Stage, staged

AS_MACHINE = {"accept": "application/json"}


def _library(root: pathlib.Path) -> None:
    (root / "side").mkdir(parents=True)
    stamped = 1_700_000_000
    for i in range(5):
        path = root / f"pic_{i}.png"
        Image.new("RGB", (12, 12), (50 + i * 30, 90, 140)).save(path)
        os.utime(path, (stamped + i * 60, stamped + i * 60))
    other = root / "side" / "aside_0.png"
    Image.new("RGB", (12, 12), (200, 90, 40)).save(other)
    os.utime(other, (stamped + 900, stamped + 900))


def _prepare(stage: Stage) -> None:
    client = stage.client
    client.post("/albums", json={"name": "Keep"})


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_selection_is_one_answer_anchored_write", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def chosen(_stage):
    _stage.restore()
    return _stage.client


def _grid(client, **params) -> tuple[str, dict[str, str]]:
    """(answer, {slug: uuid}) for the current grid of this question."""
    page = client.get("/g", params=params).text
    found = re.search(r'data-answer="([^"]+)"', page)
    assert found is not None
    answer = found.group(1)
    shells = re.findall(
        r'data-selection-key="([0-9a-f]{32})">\s*<input[^>]*>\s*<a class="cell"[^>]*data-slug="([^"]+)"', page
    )
    return answer, {slug: uuid for uuid, slug in shells}


def _favorites(client) -> set[str]:
    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        return {row[0] for row in conn.execute("SELECT e.slug FROM favorite fav JOIN entity e ON e.id = fav.file_id")}
    finally:
        connect.close(conn)


def test_one_desired_fact_lands_on_the_whole_selection_atomically(chosen):
    answer, keys = _grid(chosen, folder="lib")
    picked = [keys["pic-1"], keys["pic-3"], keys["pic-3"]]  # naming twice is naming once
    told = chosen.post(
        "/g/selection/favorite", params={"folder": "lib"}, json={"answer": answer, "items": picked, "value": True}
    )
    assert told.status_code < 300
    body = told.json()
    assert body["targets"] == 2
    assert "changed" not in body, "targets, not a transition count nothing computed"
    assert _favorites(chosen) == {"pic-1", "pic-3"}
    # The question does not depend on favorite: the answer is unchanged
    # and the currency moved -- the client keeps its selection.
    assert body["after"]["answer"] == answer
    # Desired state: the retry is harmless.
    again = chosen.post(
        "/g/selection/favorite",
        params={"folder": "lib"},
        json={"answer": body["after"]["answer"], "items": picked, "value": True},
    )
    assert again.status_code < 300
    assert _favorites(chosen) == {"pic-1", "pic-3"}


def test_a_stale_or_foreign_selection_writes_nothing(chosen):
    answer, keys = _grid(chosen, folder="lib")
    _, aside = _grid(chosen, folder="side")

    # Stale answer: 409, zero writes.
    refused = chosen.post(
        "/g/selection/favorite",
        params={"folder": "lib"},
        json={"answer": "not-the-answer", "items": [keys["pic-1"]], "value": True},
    )
    assert refused.status_code == 409
    assert _favorites(chosen) == set()

    # A file OUTSIDE this answer poisons the WHOLE batch: 409, zero writes.
    mixed = chosen.post(
        "/g/selection/favorite",
        params={"folder": "lib"},
        json={"answer": answer, "items": [keys["pic-1"], aside["aside-0"]], "value": True},
    )
    assert mixed.status_code == 409
    assert "not part of this answer" in mixed.json()["detail"]
    assert _favorites(chosen) == set()

    # A vanished entity likewise refuses the whole batch by count.
    gone = "0" * 32
    assert (
        chosen.post(
            "/g/selection/favorite",
            params={"folder": "lib"},
            json={"answer": answer, "items": [keys["pic-1"], gone], "value": True},
        ).status_code
        == 409
    )
    assert _favorites(chosen) == set()

    # Payloads that were never selections: refused as bad questions.
    for items in (["zz"], [], [keys["pic-1"][:8]]):
        refused = chosen.post(
            "/g/selection/favorite",
            params={"folder": "lib"},
            json={"answer": answer, "items": items, "value": True},
        )
        assert refused.status_code == 400
    heap = [keys["pic-1"]] * (resultset.SUBSET_MOST + 1)
    assert (
        chosen.post(
            "/g/selection/favorite", params={"folder": "lib"}, json={"answer": answer, "items": heap, "value": True}
        ).status_code
        == 400
    )


def test_bulk_rating_is_exact_and_desired_state(chosen):
    answer, keys = _grid(chosen)
    picked = [keys["pic-0"], keys["pic-2"]]
    told = chosen.post("/g/selection/rating", json={"answer": answer, "items": picked, "value": 4})
    assert told.status_code < 300
    for slug in ("pic-0", "pic-2"):
        assert chosen.get(f"/i/{slug}", headers=AS_MACHINE).json()["authored"]["rating"] == 4
    assert chosen.get("/i/pic-1", headers=AS_MACHINE).json()["authored"]["rating"] is None

    fresh = told.json()["after"]["answer"]
    assert chosen.post("/g/selection/rating", json={"answer": fresh, "items": picked, "value": 9}).status_code == 400
    assert chosen.post("/g/selection/rating", json={"answer": fresh, "items": picked, "value": True}).status_code == 400
    for slug in ("pic-0", "pic-2"):
        assert chosen.get(f"/i/{slug}", headers=AS_MACHINE).json()["authored"]["rating"] == 4, (
            "a refused bulk rating must write nothing"
        )
    cleared = chosen.post("/g/selection/rating", json={"answer": fresh, "items": picked, "value": None})
    assert cleared.status_code < 300
    assert chosen.get("/i/pic-0", headers=AS_MACHINE).json()["authored"]["rating"] is None


def test_bulk_membership_shares_the_smart_refusal(chosen):
    conn = connect.connect(chosen.app.state.db_path)
    collections.collection(conn, "Rules", 1.0, kind="smart")
    conn.commit()
    connect.close(conn)

    answer, keys = _grid(chosen)
    picked = [keys["pic-0"], keys["pic-4"]]
    told = chosen.post("/g/selection/collections/keep", json={"answer": answer, "items": picked, "value": True})
    assert told.status_code < 300
    assert chosen.get("/t/keep", headers=AS_MACHINE).json()["count"] == 2

    refused = chosen.post(
        "/g/selection/collections/rules",
        json={"answer": told.json()["after"]["answer"], "items": picked, "value": True},
    )
    assert refused.status_code == 400
    assert "rule" in refused.json()["detail"]
    conn = connect.connect(chosen.app.state.db_path, read_only=True)
    filed = conn.execute(
        "SELECT count(*) FROM collection_file cf JOIN collection c ON c.id = cf.collection_id WHERE c.kind = 'smart'"
    ).fetchone()[0]
    connect.close(conn)
    assert filed == 0

    out = chosen.post(
        "/g/selection/collections/keep",
        json={"answer": told.json()["after"]["answer"], "items": picked, "value": False},
    )
    assert out.status_code < 300
    assert chosen.get("/t/keep", headers=AS_MACHINE).json()["count"] == 0


def test_membership_writes_settle_by_answer_identity(chosen):
    """/g?album=keep with a bulk remove-from-keep: the after-answer must
    differ, because the selected files left the walked answer; a write
    the question does not depend on keeps the answer identical."""
    _, keys = _grid(chosen)
    first = chosen.post(
        "/g/selection/collections/keep",
        json={"answer": _grid(chosen)[0], "items": [keys["pic-1"], keys["pic-2"]], "value": True},
    )
    assert first.status_code < 300

    scoped_answer, scoped = _grid(chosen, album="keep")
    assert set(scoped) == {"pic-1", "pic-2"}
    told = chosen.post(
        "/g/selection/collections/keep",
        params={"album": "keep"},
        json={"answer": scoped_answer, "items": [scoped["pic-1"]], "value": False},
    )
    assert told.status_code < 300
    assert told.json()["after"]["answer"] != scoped_answer, "the removed file left the walked answer"
    assert told.json()["after"]["total"] == 1

    # The other direction of the same contract: raising ratings INSIDE
    # rating_min=4 keeps every member above the threshold -- the answer
    # identity is preserved and the client keeps its selection mounted.
    rate_answer, rated = _grid(chosen)
    lifted = chosen.post(
        "/g/selection/rating", json={"answer": rate_answer, "items": [rated["pic-0"], rated["pic-3"]], "value": 4}
    )
    assert lifted.status_code < 300
    starred_answer, starred = _grid(chosen, rating_min=4)
    assert set(starred) == {"pic-0", "pic-3"}
    raised = chosen.post(
        "/g/selection/rating",
        params={"rating_min": 4},
        json={"answer": starred_answer, "items": [starred["pic-0"], starred["pic-3"]], "value": 5},
    )
    assert raised.status_code < 300
    assert raised.json()["after"]["answer"] == starred_answer, "4 -> 5 must not move the rating_min=4 answer"
    dropped = chosen.post(
        "/g/selection/rating",
        params={"rating_min": 4},
        json={"answer": starred_answer, "items": [starred["pic-0"]], "value": 3},
    )
    assert dropped.status_code < 300
    assert dropped.json()["after"]["answer"] != starred_answer, "4 -> 3 leaves the walked answer"


def _second_writer_can_begin(db_path) -> bool:
    # The held case waits this out in full before refusing, so the window
    # is pure margin: the lane is held for the whole mutation, orders of
    # magnitude longer than a scheduler hiccup.
    other = connect.connect(db_path)
    other.execute("PRAGMA busy_timeout=20")
    try:
        other.execute("BEGIN IMMEDIATE")
        other.rollback()
    except sqlite3.OperationalError:
        return False
    else:
        return True
    finally:
        other.close()


def test_the_proof_never_holds_the_writer_lane(chosen, monkeypatch):
    """The invariant is NOT "nobody writes while we prove" -- proving may
    run a whole materialization, and holding sqlite's one writer lane
    through that starves every other write. While the proof runs, a
    second writer CAN take the lane; once the narrow mutation
    transaction begins, it cannot."""
    from db import authored as authored_module

    witnessed: list[str] = []
    real_prove = resultset.prove_subset

    def free_during_proof(conn, *args, **kwargs):
        witnessed.append("proof-free" if _second_writer_can_begin(chosen.app.state.db_path) else "proof-held")
        return real_prove(conn, *args, **kwargs)

    real_many = authored_module.set_favorite_many

    def held_during_write(conn, *args, **kwargs):
        witnessed.append("write-held" if not _second_writer_can_begin(chosen.app.state.db_path) else "write-free")
        return real_many(conn, *args, **kwargs)

    monkeypatch.setattr(resultset, "prove_subset", free_during_proof)
    monkeypatch.setattr(authored_module, "set_favorite_many", held_during_write)
    answer, keys = _grid(chosen)
    told = chosen.post("/g/selection/favorite", json={"answer": answer, "items": [keys["pic-0"]], "value": True})
    assert told.status_code < 300
    assert witnessed == ["proof-free", "write-held"]


def test_a_commit_in_the_handoff_is_reproved_not_trusted(chosen, monkeypatch):
    """A commit landing between a completed proof and the writer lane:
    the stale proof's currency is rejected by revalidation, ONE re-proof
    runs outside the lane, and -- the answer being unchanged by the
    unrelated commit -- the retry lands. Nothing is ever written from
    the first proof's generation."""
    real_prove = resultset.prove_subset
    proofs: list[str] = []

    def prove_then_racing_commit(conn, *args, **kwargs):
        proof = real_prove(conn, *args, **kwargs)
        proofs.append(proof.currency)
        if len(proofs) == 1:
            # An UNRELATED commit: rates a file outside the folder
            # question -- currency moves, the answer does not.
            writer = connect.connect(chosen.app.state.db_path)
            file_id = writer.execute("SELECT id FROM file WHERE name = 'aside_0.png'").fetchone()[0]
            authored.set_rating(writer, file_id, chosen.app.state.actor_id, 3, 0.0)
            writer.commit()
            connect.close(writer)
        return proof

    monkeypatch.setattr(resultset, "prove_subset", prove_then_racing_commit)
    answer, keys = _grid(chosen, folder="lib")
    told = chosen.post(
        "/g/selection/favorite",
        params={"folder": "lib"},
        json={"answer": answer, "items": [keys["pic-2"]], "value": True},
    )
    assert told.status_code < 300, told.text
    assert len(proofs) == 2, "the stale proof must be re-proved, not trusted"
    assert proofs[0] != proofs[1], "the re-proof must see the racing commit's generation"
    assert _favorites(chosen) == {"pic-2"}


def test_a_changed_answer_in_the_handoff_writes_nothing(chosen, monkeypatch):
    """The same race, but the commit CHANGES the walked answer: the
    re-proof raises, the response is 409, and zero rows moved."""
    real_prove = resultset.prove_subset
    raced: list[str] = []

    def prove_then_membership_commit(conn, *args, **kwargs):
        proof = real_prove(conn, *args, **kwargs)
        if not raced:
            raced.append("raced")
            writer = connect.connect(chosen.app.state.db_path)
            file_id = writer.execute("SELECT id FROM file WHERE name = 'pic_1.png'").fetchone()[0]
            keep = writer.execute("SELECT id FROM collection WHERE name = 'Keep'").fetchone()[0]
            collections.set_membership(writer, keep, file_id, False, 0.0)
            writer.commit()
            connect.close(writer)
        return proof

    _, keys = _grid(chosen)
    filed = chosen.post(
        "/g/selection/collections/keep",
        json={"answer": _grid(chosen)[0], "items": [keys["pic-1"], keys["pic-2"]], "value": True},
    )
    assert filed.status_code < 300

    monkeypatch.setattr(resultset, "prove_subset", prove_then_membership_commit)
    scoped_answer, scoped = _grid(chosen, album="keep")
    told = chosen.post(
        "/g/selection/favorite",
        params={"album": "keep"},
        json={"answer": scoped_answer, "items": [scoped["pic-1"]], "value": True},
    )
    assert told.status_code == 409
    assert _favorites(chosen) == set(), "a raced-away answer must write nothing"


def test_a_semantic_proof_runs_without_the_writer_lane(chosen, monkeypatch):
    """The expensive case named directly: a semantic projection cache
    miss reaches retrieval while the curation connection holds NO write
    transaction -- a second writer can take the lane mid-FAISS."""
    from db import retrieval

    conn = connect.connect(chosen.app.state.db_path, read_only=True)
    ranked = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id")]
    connect.close(conn)
    witnessed: list[str] = []

    def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
        witnessed.append("free" if _second_writer_can_begin(chosen.app.state.db_path) else "held")
        held = [i for i in ranked if allowed is None or i in allowed]
        return retrieving.answered(held)

    monkeypatch.setattr(retrieval, "query", fused)
    answer, keys = _grid(chosen, q="banana")
    told = chosen.post(
        "/g/selection/favorite",
        params={"q": "banana"},
        json={"answer": answer, "items": [keys["pic-0"]], "value": True},
    )
    assert told.status_code < 300, told.text
    assert witnessed, "the semantic path never ran"
    assert set(witnessed) == {"free"}, "FAISS ran while the writer lane was held"


def test_every_non_hex_spelling_is_refused_as_a_bad_question(chosen):
    """bytes.fromhex skips whitespace, so length alone is not the gate:
    a 34-character padded spelling, a 32-character one with spaces
    HIDING INSIDE it (30 hex digits, 15 bytes), and a non-hex character
    are all 400 -- malformed selections, never 409-shaped racing."""
    answer, keys = _grid(chosen)
    valid = keys["pic-0"]
    for rotten in (
        valid[:16] + " " + valid[16:] + " ",  # padded: raw length 34
        valid[:14] + "  " + valid[16:],  # spaces inside: raw length 32
        valid[:31] + "g",  # not hex at all
        valid.upper()[:31] + " ",  # trailing space at raw length 32
    ):
        refused = chosen.post("/g/selection/favorite", json={"answer": answer, "items": [rotten], "value": True})
        assert refused.status_code == 400, rotten
    assert _favorites(chosen) == set()
    # And the canonical spelling still lands.
    landed = chosen.post("/g/selection/favorite", json={"answer": answer, "items": [valid], "value": True})
    assert landed.status_code < 300
    assert _favorites(chosen) == {"pic-0"}


def _where(client, slug: str):
    return client.get(f"/i/{slug}", headers={"accept": "application/json"}).json()["where"]


def test_a_selection_can_be_placed_at_once(chosen):
    """Where these pictures happened, said for the whole selection: one
    place found or minted, one authored claim per file, every file
    re-interpreted -- and withdrawn the same way."""
    answer, keys = _grid(chosen, folder="lib")
    picked = [keys["pic-1"], keys["pic-3"]]
    told = chosen.post(
        "/g/selection/place",
        params={"folder": "lib"},
        json={"answer": answer, "items": picked, "name": "Porto", "kind": "city"},
    )
    assert told.status_code < 300, told.text
    assert told.json()["targets"] == 2
    first, third = _where(chosen, "pic-1"), _where(chosen, "pic-3")
    assert (first["name"], first["kind"], first["basis"]) == ("Porto", "city", "authored")
    assert third["id"] == first["id"], "one Porto for the whole selection"
    assert _where(chosen, "pic-2") is None or _where(chosen, "pic-2")["basis"] != "authored"
    gone = chosen.post(
        "/g/selection/place", params={"folder": "lib"}, json={"answer": answer, "items": picked, "name": None}
    )
    assert gone.status_code < 300, gone.text
    assert _where(chosen, "pic-1") is None
    refused = chosen.post(
        "/g/selection/place",
        params={"folder": "lib"},
        json={"answer": answer, "items": picked, "name": "Mars", "kind": "planet"},
    )
    assert refused.status_code == 400
    page = chosen.get("/g", params={"folder": "lib"}).text
    assert "data-bulk-place-name" in page
