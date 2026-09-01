"""The canonical store, held against the invariant it exists for.

GOVERNING INVARIANT: if it is output by any producer, store it as whatever it
actually is, without semantic narrowing, and prove that using the stored result
is the same as doing it live, without cheating.

THE CACHE MANDATE is user law and it is one branch in `producers.resolve`: a
HIT returns the stored bytes and the producer does not run. `_Counted` below is
the instrument -- every test that claims "nothing reran" reads a call count off
the producer itself rather than inferring it from a timing or a row.
"""

from __future__ import annotations

import sqlite3

import pytest

from db import producers
from tests.staging import NOW, fresh_schema

CONTRACT = "insightface/antelopev2"
CODEC = "sgface2"


@pytest.fixture
def db():
    conn = fresh_schema()
    producers.declare_determinism(conn, CONTRACT, NOW, kind="bitwise")
    return conn


class _Counted:
    """A producer that says how many times it actually ran.

    The count is the evidence. A test asserting "the producer did not run"
    against anything else -- elapsed time, a row that did not appear -- is
    asserting a symptom, and a resolver that ran the producer and threw the
    answer away would satisfy every one of those.
    """

    def __init__(self, value: object = b"x"):
        self.value = value
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return self.value


def _leaf(digest: str = "a" * 64, slot: str = "image") -> producers.InputRef:
    return producers.InputRef(slot=slot, kind="content", content_sha256=digest)


def _keep(conn, produce, *, preimage=None, inputs=None, runtime=None, contract=CONTRACT):
    return producers.resolve(
        conn,
        NOW,
        contract=contract,
        preimage=preimage if preimage is not None else {"impl": "v1"},
        runtime_observed=runtime if runtime is not None else {},
        inputs=inputs if inputs is not None else [_leaf()],
        codec_version=CODEC,
        execute=produce,
        capture=lambda value: value if isinstance(value, bytes) else repr(value).encode(),
    )


# --- the cache mandate ------------------------------------------------------


def test_a_miss_runs_the_producer_exactly_once(db):
    produce = _Counted(b"one")
    held = _keep(db, produce)
    assert produce.calls == 1
    assert held.was_hit is False
    assert held.envelope == b"one"


def test_a_hit_does_not_run_the_producer_at_all(db):
    """The mandate. Not 'runs it less', not 'runs it faster' -- zero."""
    first = _Counted(b"one")
    _keep(db, first)
    assert first.calls == 1

    second = _Counted(b"one")
    held = _keep(db, second)
    assert second.calls == 0, "a stored identity re-ran the producer"
    assert held.was_hit is True
    assert held.envelope == b"one"


def test_the_hit_returns_the_stored_bytes_not_the_live_ones(db):
    """A hit is served from the store even when the live producer would now
    answer differently -- which is what makes it a cache and not a coincidence."""
    _keep(db, _Counted(b"first answer"))
    later = _Counted(b"a DIFFERENT answer")
    held = _keep(db, later)
    assert later.calls == 0
    assert held.envelope == b"first answer"


# --- what the identity is made of -------------------------------------------


@pytest.mark.parametrize(
    "preimage",
    [
        {"impl": "v2"},
        {"impl": "v1", "weights": "deadbeef"},
        {"impl": "v1", "det_thresh": 0.3},
        {"impl": "v1", "provider": "CUDAExecutionProvider"},
        {"impl": "v1", "codec": "sgface3"},
    ],
    ids=["implementation", "weights", "configuration", "provider", "codec"],
)
def test_a_changed_preimage_field_is_a_different_identity(db, preimage):
    """Every field the old freshness keys never carried. `face_items` keyed on
    (file_id, content_sha256) alone -- not even the model -- so a backend swap
    was invisible to it."""
    base = producers.identity_of({"impl": "v1"}, [_leaf()])
    assert producers.identity_of(preimage, [_leaf()]) != base


def test_the_same_inputs_in_another_order_are_another_identity(db):
    """Argument order is identity-bearing: the same producer over the same two
    images the other way round is a different call with a different answer."""
    one, two = _leaf("a" * 64, "left"), _leaf("b" * 64, "right")
    assert producers.identity_of({}, [one, two]) != producers.identity_of({}, [two, one])


def test_nothing_that_cannot_move_the_bytes_reaches_the_identity(db):
    """The negative half, and the one that keeps the cache warm. A key that
    carries the clock or a row id never hits twice."""
    produce = _Counted(b"one")
    _keep(db, produce)
    # a second resolve at a different wall clock, from a different job, must hit
    again = _Counted(b"one")
    held = producers.resolve(
        db,
        NOW + 86_400,
        contract=CONTRACT,
        preimage={"impl": "v1"},
        runtime_observed={},
        inputs=[_leaf()],
        codec_version=CODEC,
        execute=again,
        capture=lambda value: value,
    )
    assert again.calls == 0, "the clock reached the identity"
    assert held.was_hit is True


def test_an_upstream_result_is_an_edge_not_a_digest(db):
    """The DAG. A derived input names the upstream RESULT's identity, so
    invalidating the parent invalidates everything computed from it by
    construction -- with no invalidation code to write and none to forget."""
    parent = _keep(db, _Counted(b"parent"))
    child = _keep(
        db,
        _Counted(b"child"),
        preimage={"impl": "crop"},
        inputs=[producers.InputRef(slot="detection", kind="result", upstream_identity=parent.identity)],
    )
    edge = db.execute(
        "SELECT kind, upstream_result_id FROM producer_input WHERE invocation_id = ?", (child.invocation_id,)
    ).fetchone()
    assert edge[0] == "result"
    assert edge[1] == parent.result_id


def test_an_upstream_the_store_does_not_hold_is_refused_by_name(db):
    with pytest.raises(LookupError, match="does not hold"):
        _keep(
            db,
            _Counted(b"child"),
            inputs=[producers.InputRef(slot="detection", kind="result", upstream_identity="f" * 64)],
        )


# --- contradiction ----------------------------------------------------------


def test_a_second_answer_under_one_identity_is_recorded_never_overwritten(db):
    """B24. `record_faces` did DELETE-then-INSERT, so a re-run under the same
    model name destroyed a disagreeing earlier answer with no trace."""
    _keep(db, _Counted(b"first"))
    with pytest.raises(producers.Contradicted) as raised:
        producers.remember(
            db,
            NOW,
            contract=CONTRACT,
            preimage={"impl": "v1"},
            runtime_observed={},
            inputs=[_leaf()],
            envelope=b"second",
            codec_version=CODEC,
        )
    row = db.execute(
        "SELECT offered_envelope FROM producer_contradiction WHERE id = ?", (raised.value.contradiction_id,)
    ).fetchone()
    assert row[0] == b"second", "the disagreement IS the evidence and must be kept"
    held = producers.stored(db, raised.value.identity)
    assert held is not None, "the held result vanished"
    assert held.envelope == b"first", "the held answer was overwritten"


def test_the_same_bytes_twice_is_not_a_contradiction(db):
    """Idempotent on agreement, which is what makes a retried item cheap."""
    first = _keep(db, _Counted(b"one"))
    again = producers.remember(
        db,
        NOW,
        contract=CONTRACT,
        preimage={"impl": "v1"},
        runtime_observed={},
        inputs=[_leaf()],
        envelope=b"one",
        codec_version=CODEC,
    )
    assert again.result_id == first.result_id
    assert db.execute("SELECT count(*) FROM producer_contradiction").fetchone()[0] == 0


def test_a_stored_result_cannot_be_edited_or_deleted(db):
    held = _keep(db, _Counted(b"one"))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("UPDATE producer_result SET envelope = ? WHERE id = ?", (b"other", held.result_id))
    with pytest.raises(sqlite3.IntegrityError, match="not deletable"):
        db.execute("DELETE FROM producer_result WHERE id = ?", (held.result_id,))


def test_a_contradiction_cannot_be_deleted_to_clean_the_board(db):
    """The asymmetry that would otherwise defeat the whole audit: nobody needs
    to edit a tolerance to clear a backlog, they delete the evidence, and the
    re-blessing join then truthfully reports nothing was re-blessed."""
    _keep(db, _Counted(b"first"))
    with pytest.raises(producers.Contradicted) as raised:
        producers.remember(
            db,
            NOW,
            contract=CONTRACT,
            preimage={"impl": "v1"},
            runtime_observed={},
            inputs=[_leaf()],
            envelope=b"second",
            codec_version=CODEC,
        )
    with pytest.raises(sqlite3.IntegrityError, match="never delete it"):
        db.execute("DELETE FROM producer_contradiction WHERE id = ?", (raised.value.contradiction_id,))


# --- declaration, never a default -------------------------------------------


def test_a_contract_with_no_determinism_declaration_is_refused_by_name(db):
    """A permissive default nobody chose is how a tolerance stops being a
    decision and starts being an inheritance -- `Case`'s rtol=1e-3 is the
    precedent this refusal exists to avoid repeating."""
    with pytest.raises(producers.Undeclared, match="no determinism declaration"):
        _keep(db, _Counted(b"one"), contract="nobody/declared-me")


def test_a_loosened_tolerance_is_a_new_row_not_an_edit(db):
    first = producers.determinism_for(db, CONTRACT)
    second = producers.declare_determinism(db, CONTRACT, NOW + 1, kind="approx", rtol=1e-3, atol=1e-7)
    assert second != first
    assert producers.determinism_for(db, CONTRACT) == second
    with pytest.raises(sqlite3.IntegrityError, match="new declaration"):
        db.execute("UPDATE producer_determinism SET rtol = 1.0 WHERE id = ?", (second,))


# --- re-verification candidates ---------------------------------------------


def test_a_changed_neutral_fact_raises_a_candidate_on_the_hit_path(db):
    """Half (b) of the contradiction fix: the population that can expose an
    under-declared contract is not the library, it is the set of results being
    served under conditions that changed. Costs a dict comparison, no producer."""
    _keep(db, _Counted(b"one"), runtime={"provider": "CPUExecutionProvider"})
    again = _Counted(b"one")
    _keep(db, again, runtime={"provider": "CUDAExecutionProvider"})
    assert again.calls == 0
    rows = db.execute("SELECT field, stored, observed FROM producer_reverify_candidate").fetchall()
    assert rows == [("provider", "CPUExecutionProvider", "CUDAExecutionProvider")]


def test_an_unchanged_runtime_raises_no_candidate(db):
    """The half that keeps the channel worth reading: if every hit raised a
    candidate, the queue would be the library and nobody would work it."""
    _keep(db, _Counted(b"one"), runtime={"provider": "CPUExecutionProvider"})
    _keep(db, _Counted(b"one"), runtime={"provider": "CPUExecutionProvider"})
    assert db.execute("SELECT count(*) FROM producer_reverify_candidate").fetchone()[0] == 0


# --- waivers ----------------------------------------------------------------


def _a_contradiction(db) -> int:
    _keep(db, _Counted(b"first"))
    with pytest.raises(producers.Contradicted) as raised:
        producers.remember(
            db,
            NOW,
            contract=CONTRACT,
            preimage={"impl": "v1"},
            runtime_observed={},
            inputs=[_leaf()],
            envelope=b"second",
            codec_version=CODEC,
        )
    return raised.value.contradiction_id


def test_a_waiver_cannot_pre_authorize_a_later_loosening(db):
    """The load-bearing property is the COMPOSITE KEY, not the triggers: a
    waiver names one contradiction under ONE declaration, so a later loosening
    mints a determinism_id it does not cover and which needs its own human act.
    A blanket waiver is unrepresentable -- the difference between a signature
    and a mute button."""
    contradiction = _a_contradiction(db)
    first = producers.determinism_for(db, CONTRACT)
    assert first is not None
    producers.waive(db, contradiction, first, "will", "measured, within the batch-width band", NOW)

    loosened = producers.declare_determinism(db, CONTRACT, NOW + 1, kind="approx", rtol=1e-3, atol=1e-7)
    covered = db.execute(
        "SELECT count(*) FROM producer_contradiction_waiver WHERE contradiction_id = ? AND determinism_id = ?",
        (contradiction, loosened),
    ).fetchone()[0]
    assert covered == 0, "an old waiver covered a declaration made after it"


def test_a_waiver_needs_an_author_and_a_reason(db):
    contradiction = _a_contradiction(db)
    declaration = producers.determinism_for(db, CONTRACT)
    assert declaration is not None
    with pytest.raises(ValueError, match="signed act"):
        producers.waive(db, contradiction, declaration, "   ", "because", NOW)
    with pytest.raises(ValueError, match="signed act"):
        producers.waive(db, contradiction, declaration, "will", "  ", NOW)


def test_re_blessing_is_a_row_rather_than_an_absence(db):
    """A contradiction cleared under a LATER declaration than the one it was
    raised under is precisely a re-blessing, and the gate reads this."""
    contradiction = _a_contradiction(db)
    assert producers.re_blessed(db) == []
    loosened = producers.declare_determinism(db, CONTRACT, NOW + 1, kind="approx", rtol=1e-3, atol=1e-7)
    producers.judge(db, contradiction, loosened, "within-tolerance", NOW + 2)
    found = producers.re_blessed(db)
    assert len(found) == 1
    assert found[0]["contradiction_id"] == contradiction
    assert found[0]["cleared_under"] == loosened


def test_a_judgment_that_lets_the_contradiction_stand_is_not_a_re_blessing(db):
    """The half that keeps the gate readable: judging a contradiction and
    upholding it must not look like clearing it."""
    contradiction = _a_contradiction(db)
    loosened = producers.declare_determinism(db, CONTRACT, NOW + 1, kind="approx", rtol=1e-3, atol=1e-7)
    producers.judge(db, contradiction, loosened, "stands", NOW + 2)
    assert producers.re_blessed(db) == []
