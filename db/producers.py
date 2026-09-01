"""The canonical store: what a producer emitted, kept whole and keyed by its inputs.

GOVERNING INVARIANT: if it is output by any producer, store it as whatever it
actually is, without semantic narrowing, and prove that using the stored result
is the same as doing it live.

This module owns the storage boundary for that. It knows nothing about faces,
captions or vectors -- it takes an envelope (`vision/facestore.py`) and the
ORDERED preimage the envelope was computed from, and it is the only thing that
writes the `producer_*` tables.

The identity is the point. `identity_of` hashes the complete ordered preimage:
the contract and its revision, the implementation and adapter digests, the
weight FILE digests, the invocation configuration, the bit-affecting runtime
facts, the capture codec version, and the ordered inputs -- each input either a
leaf's content digest or an upstream RESULT's identity, which is what makes the
whole thing a DAG rather than a flat key. Invalidating a detection invalidates
everything computed from it by construction, with no invalidation code to write
and none to forget.

Provenance record and cache key are deliberately the same object. Two of them
drift, and the drift is invisible until a hit serves bytes that were computed
from something else.

What is NOT in the identity, and why:

  - Wall clock, job id, worker, row ids, absolute paths. They cannot change the
    bytes, and a key that includes them never hits.
  - Runtime facts the contract declared neutral. Hashing a machine's identity
    into the key means moving machines invalidates the library. `remember`
    records them instead, and `stored` compares them on a HIT so a
    declared-neutral fact that has in fact changed raises a re-verification
    candidate -- which is what makes the narrow key auditable rather than
    merely convenient.
  - The determinism tolerance. It changes neither what the producer computes
    nor what a hit returns; it only decides whether a recomputation counts as a
    contradiction. In the key it would force a full library recompute over a
    judgment rule that moved no bytes. It is versioned instead, so a loosening
    is a new row and "was this contradiction re-blessed?" is a join rather than
    an absence.

A second answer under one identity is never an overwrite. `remember` raises
`Contradicted` and records the disagreement -- both envelopes kept, because the
disagreement IS the evidence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "Contradicted",
    "InputRef",
    "Stored",
    "Undeclared",
    "declare_determinism",
    "determinism_for",
    "identity_of",
    "judge",
    "remember",
    "resolve",
    "stored",
    "waive",
]


class Contradicted(Exception):
    """One identity, two different answers.

    Raised instead of overwriting. Carries the contradiction row's id so a
    caller can judge or waive it by name rather than searching for it.
    """

    def __init__(self, message: str, *, contradiction_id: int, identity: str):
        super().__init__(message)
        self.contradiction_id = contradiction_id
        self.identity = identity


class Undeclared(LookupError):
    """A contract stored a result without declaring how its output compares
    with itself.

    Refused rather than defaulted. A permissive default nobody chose is how a
    tolerance stops being a decision and starts being an inheritance.
    """


@dataclass(frozen=True)
class InputRef:
    """One input edge. Exactly one of the two digests, matching `kind`."""

    slot: str
    kind: str  # 'content' | 'result'
    content_sha256: str | None = None
    upstream_identity: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "content":
            if not self.content_sha256 or self.upstream_identity:
                raise ValueError(f"input {self.slot!r} is a leaf and must carry exactly a content digest")
        elif self.kind == "result":
            if not self.upstream_identity or self.content_sha256:
                raise ValueError(f"input {self.slot!r} is derived and must carry exactly an upstream identity")
        else:
            raise ValueError(f"input {self.slot!r} has kind {self.kind!r}, which is neither 'content' nor 'result'")

    def preimage(self) -> list[str]:
        """The three fields that reach the digest, in order."""
        return [self.slot, self.kind, self.content_sha256 or self.upstream_identity or ""]


@dataclass(frozen=True)
class Stored:
    """One canonical result as the store holds it."""

    identity: str
    result_id: int
    invocation_id: int
    contract: str
    codec_version: str
    envelope: bytes
    was_hit: bool


def identity_of(preimage: Mapping[str, Any], inputs: Sequence[InputRef]) -> str:
    """sha256 over the ORDERED complete preimage.

    `preimage` holds the non-edge fields; `inputs` holds the edges, and their
    ORDER is identity-bearing -- the same producer over the same two images in
    the other order is a different call with a different answer, so the inputs
    are hashed as a list and never as a set.

    `sort_keys` on the mapping and insertion order on the list together mean
    the digest depends on the VALUES and not on how a caller happened to build
    the dict.
    """
    body = json.dumps(
        {"preimage": preimage, "inputs": [one.preimage() for one in inputs]},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def declare_determinism(
    conn: sqlite3.Connection,
    contract: str,
    now: float,
    *,
    kind: str,
    rtol: float | None = None,
    atol: float | None = None,
) -> int:
    """Declare how this contract's output compares with itself, and return the
    declaration's id.

    Append-only: a loosened tolerance is a NEW row. The declaration in force
    when a contradiction was raised is recorded on it, so a judgment recorded
    under a LATER declaration is a re-blessing and reads as one.
    """
    if kind not in ("bitwise", "approx"):
        raise ValueError(f"a determinism class is 'bitwise' or 'approx', not {kind!r}")
    cursor = conn.execute(
        "INSERT INTO producer_determinism(contract_name, kind, rtol, atol, declared_at) VALUES(?, ?, ?, ?, ?)",
        (contract, kind, rtol, atol, now),
    )
    return cursor.lastrowid or 0


def determinism_for(conn: sqlite3.Connection, contract: str) -> int | None:
    """The id of the declaration currently in force for this contract, or None
    when it has never declared one."""
    row = conn.execute(
        "SELECT id FROM producer_determinism WHERE contract_name = ? ORDER BY declared_at DESC, id DESC LIMIT 1",
        (contract,),
    ).fetchone()
    return None if row is None else int(row[0])


def _held(conn: sqlite3.Connection, identity: str) -> tuple[int, int, str, str, bytes] | None:
    row = conn.execute(
        "SELECT r.id, i.id, i.contract_name, r.codec_version, r.envelope"
        "  FROM producer_invocation i JOIN producer_result r ON r.invocation_id = i.id"
        " WHERE i.identity = ?",
        (identity,),
    ).fetchone()
    return None if row is None else (int(row[0]), int(row[1]), str(row[2]), str(row[3]), bytes(row[4]))


def stored(conn: sqlite3.Connection, identity: str, *, observed: Mapping[str, str] | None = None, now: float = 0.0):
    """The result held under this identity, or None.

    THE CACHE MANDATE lives here: a hit returns the stored bytes and the
    producer does not run. Nothing in this function executes anything.

    When `observed` is supplied it is compared against the runtime facts
    recorded at capture, and a declared-neutral fact that has in fact changed
    raises a `producer_reverify_candidate` row. That costs a dict comparison
    and no producer time, and it is what makes re-verification affordable: the
    population that can expose an under-declared contract is this table, so the
    re-verification budget is bounded by the number of distinct
    (contract, changed-fact) pairs and NOT by the size of the library.
    """
    found = _held(conn, identity)
    if found is None:
        return None
    result_id, invocation_id, contract, codec_version, envelope = found
    if observed is not None:
        _note_divergence(conn, invocation_id, result_id, observed, now)
    return Stored(
        identity=identity,
        result_id=result_id,
        invocation_id=invocation_id,
        contract=contract,
        codec_version=codec_version,
        envelope=envelope,
        was_hit=True,
    )


def resolve(
    conn: sqlite3.Connection,
    now: float,
    *,
    contract: str,
    preimage: Mapping[str, Any],
    runtime_observed: Mapping[str, str],
    inputs: Sequence[InputRef],
    codec_version: str,
    execute: Callable[[], Any],
    capture: Callable[[Any], bytes],
) -> Stored:
    """The single doorway to producer execution.

    THE CACHE MANDATE, and it is one branch: on a HIT `execute` is never
    called. Nothing reruns when the compute inputs are unchanged, and the
    identity is a strict superset of every freshness key it replaces -- same
    content digest, plus the weights, adapter, configuration, codec and
    bit-affecting runtime those keys never carried.

    On a MISS the producer runs once, its return is frozen, and the LIVE
    OBJECT IS DROPPED: what comes back is decoded from bytes that are already
    stored, so a projection built off this result cannot be reading something
    the envelope failed to preserve. A value the envelope cannot carry raises
    HERE -- at the pass that could still rerun -- rather than at a replay years
    later discovering a hole.

    `capture` is injected rather than imported so this module keeps knowing
    nothing about what a producer emits.
    """
    identity = identity_of(preimage, inputs)
    held = stored(conn, identity, observed=runtime_observed, now=now)
    if held is not None:
        return held
    return remember(
        conn,
        now,
        contract=contract,
        preimage=preimage,
        runtime_observed=runtime_observed,
        inputs=inputs,
        envelope=capture(execute()),
        codec_version=codec_version,
    )


def _note_divergence(
    conn: sqlite3.Connection, invocation_id: int, result_id: int, observed: Mapping[str, str], now: float
) -> None:
    """Record every declared-neutral fact that differs from the one in force at
    capture. First sighting wins: the row says when the divergence STARTED, and
    re-serving the same result under the same changed condition is not news."""
    row = conn.execute("SELECT runtime_observed FROM producer_invocation WHERE id = ?", (invocation_id,)).fetchone()
    # Narrowed to str on the way out of json: the stored map is whatever was
    # written, and comparing a parsed value against an observed one only means
    # something once both are the same shape.
    held: dict[str, str] = {
        str(key): str(value) for key, value in (json.loads(row[0]) if row and row[0] else {}).items()
    }
    for field in sorted(set(held) | set(observed)):
        before, after = held.get(field, ""), observed.get(field, "")
        if before == after:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO producer_reverify_candidate(result_id, field, stored, observed, first_seen)"
            " VALUES(?, ?, ?, ?, ?)",
            (result_id, field, before, after, now),
        )


def remember(
    conn: sqlite3.Connection,
    now: float,
    *,
    contract: str,
    preimage: Mapping[str, Any],
    runtime_observed: Mapping[str, str],
    inputs: Sequence[InputRef],
    envelope: bytes,
    codec_version: str,
) -> Stored:
    """Store one producer's complete output, or return what is already held.

    Idempotent on agreement: the same identity offering the same bytes returns
    the held row and writes nothing, which is what makes a retried item cheap.

    On DISAGREEMENT it raises `Contradicted` and records the disagreement --
    the held result untouched, the offered envelope kept beside it. An
    overwrite here would destroy the evidence that the two runs disagreed,
    which is the one fact worth having.

    A contract with no determinism declaration is refused by name
    (`Undeclared`) rather than given a default. The declaration is what a
    contradiction is judged against, and one nobody chose is one nobody can
    defend.
    """
    if not envelope:
        raise ValueError(f"{contract} offered an empty envelope; there is nothing to store")
    declaration = determinism_for(conn, contract)
    if declaration is None:
        raise Undeclared(
            f"{contract} has no determinism declaration: declare_determinism(conn, {contract!r}, ...) first. "
            f"A tolerance nobody chose is the defaulting this store exists to refuse."
        )
    identity = identity_of(preimage, inputs)
    digest = hashlib.sha256(envelope).hexdigest()

    found = _held(conn, identity)
    if found is not None:
        result_id, invocation_id, held_contract, held_codec, held_envelope = found
        if hashlib.sha256(held_envelope).hexdigest() == digest:
            return Stored(
                identity=identity,
                result_id=result_id,
                invocation_id=invocation_id,
                contract=held_contract,
                codec_version=held_codec,
                envelope=held_envelope,
                was_hit=True,
            )
        cursor = conn.execute(
            "INSERT OR IGNORE INTO producer_contradiction(identity, held_result_id, offered_sha256,"
            " offered_envelope, determinism_id, observed_at) VALUES(?, ?, ?, ?, ?, ?)",
            (identity, result_id, digest, envelope, declaration, now),
        )
        contradiction = cursor.lastrowid or 0
        if not contradiction:  # already recorded; name the existing row rather than 0
            contradiction = int(
                conn.execute(
                    "SELECT id FROM producer_contradiction WHERE identity = ? AND offered_sha256 = ?",
                    (identity, digest),
                ).fetchone()[0]
            )
        raise Contradicted(
            f"{contract} produced different bytes under an identity already held "
            f"({identity[:12]}...): recorded as contradiction {contradiction}, neither answer overwritten",
            contradiction_id=contradiction,
            identity=identity,
        )

    cursor = conn.execute(
        "INSERT INTO producer_invocation(identity, contract_name, preimage_json, runtime_observed, invoked_at)"
        " VALUES(?, ?, ?, ?, ?)",
        (
            identity,
            contract,
            json.dumps(preimage, sort_keys=True, separators=(",", ":"), default=str),
            json.dumps(dict(runtime_observed), sort_keys=True, separators=(",", ":")),
            now,
        ),
    )
    invocation_id = cursor.lastrowid or 0
    for ordinal, one in enumerate(inputs):
        upstream = None
        if one.kind == "result":
            parent = _held(conn, one.upstream_identity or "")
            if parent is None:
                raise LookupError(
                    f"input {one.slot!r} names upstream result {one.upstream_identity}, which this store does not hold"
                )
            upstream = parent[0]
        conn.execute(
            "INSERT INTO producer_input(invocation_id, ordinal, slot, kind, content_sha256, upstream_result_id)"
            " VALUES(?, ?, ?, ?, ?, ?)",
            (invocation_id, ordinal, one.slot, one.kind, one.content_sha256, upstream),
        )
    cursor = conn.execute(
        "INSERT INTO producer_result(invocation_id, codec_version, envelope, envelope_sha256, byte_len, captured_at)"
        " VALUES(?, ?, ?, ?, ?, ?)",
        (invocation_id, codec_version, envelope, digest, len(envelope), now),
    )
    return Stored(
        identity=identity,
        result_id=cursor.lastrowid or 0,
        invocation_id=invocation_id,
        contract=contract,
        codec_version=codec_version,
        envelope=envelope,
        was_hit=False,
    )


def record_variance(
    conn: sqlite3.Connection,
    result_id: int,
    now: float,
    *,
    population: str,
    max_abs: float,
    max_rel: float,
    determinism_id: int,
) -> None:
    """One re-verification's observed divergence, kept EVEN WHEN IN TOLERANCE.

    Keeping the in-tolerance observations is the whole point: a contract
    declaring rtol=1.0 on day one passes a two-sided attack trivially and
    silences everything forever. That attack tests whether a tolerance is
    ENFORCED, never whether it is JUSTIFIED, and only the recorded distribution
    can answer the second question.

    `population` separates the two samples because the resolver targets
    re-verification where declared-neutral facts CHANGED, which is exactly
    where variance is largest -- justifying a tolerance against that
    distribution would let the mechanism built to catch under-declaration
    inflate the baseline that excuses it.
    """
    if population not in ("same-runtime", "changed-runtime"):
        raise ValueError(f"a variance population is 'same-runtime' or 'changed-runtime', not {population!r}")
    conn.execute(
        "INSERT INTO producer_variance(result_id, observed_at, population, max_abs, max_rel, determinism_id)"
        " VALUES(?, ?, ?, ?, ?, ?)",
        (result_id, now, population, max_abs, max_rel, determinism_id),
    )


def judge(conn: sqlite3.Connection, contradiction_id: int, determinism_id: int, verdict: str, now: float) -> None:
    """Re-judge a contradiction under a declaration. Append-only.

    This exists so DELETE is not the only route. A tolerance gets loosened
    because somebody wants a backlog cleared; if re-judging is not a SUPPORTED
    operation people reach for the forbidden one regardless of intent, and a
    guarantee that depends on nobody wanting the forbidden thing is not a
    guarantee.
    """
    if verdict not in ("stands", "within-tolerance"):
        raise ValueError(f"a verdict is 'stands' or 'within-tolerance', not {verdict!r}")
    conn.execute(
        "INSERT INTO producer_contradiction_judgment(contradiction_id, determinism_id, verdict, judged_at)"
        " VALUES(?, ?, ?, ?)",
        (contradiction_id, determinism_id, verdict, now),
    )


def waive(
    conn: sqlite3.Connection, contradiction_id: int, determinism_id: int, waived_by: str, reason: str, now: float
) -> None:
    """A human accepting ONE contradiction under ONE declaration.

    The composite key is the load-bearing part, not the triggers: a waiver
    names an existing contradiction under an existing declaration, so it cannot
    pre-authorize. A later loosening mints a determinism_id this waiver does
    not cover and which therefore needs its own human act. A blanket waiver is
    unrepresentable, which is the difference between a signature and a mute
    button.
    """
    if not waived_by.strip() or not reason.strip():
        raise ValueError("a waiver is a signed act: it needs an author and a reason")
    conn.execute(
        "INSERT INTO producer_contradiction_waiver(contradiction_id, determinism_id, waived_by, reason, waived_at)"
        " VALUES(?, ?, ?, ?, ?)",
        (contradiction_id, determinism_id, waived_by, reason, now),
    )


def identity_disagreements(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Stored identities that do NOT match the preimage they name.

    The acceptance rule's last clause: identical result identities cannot
    silently disagree. Provenance record and cache key are meant to be one
    object, and until the invocation row was made immutable a single UPDATE to
    preimage_json separated them -- the identity kept indexing immutable bytes
    while naming a preimage that no longer produces it, so a hit served real
    bytes under fabricated provenance.

    The triggers prevent that; this reads it. Prevention nobody can observe is
    indistinguishable from prevention that quietly stopped working, and a
    relation no gate runs is a record produced and never read.
    """
    found: list[dict[str, Any]] = []
    for invocation_id, identity, preimage_json in conn.execute(
        "SELECT id, identity, preimage_json FROM producer_invocation ORDER BY id"
    ).fetchall():
        inputs = [
            InputRef(
                slot=str(slot),
                kind=str(kind),
                content_sha256=str(content) if kind == "content" else None,
                upstream_identity=str(upstream) if kind == "result" else None,
            )
            for slot, kind, content, upstream in conn.execute(
                "SELECT p.slot, p.kind, p.content_sha256, up.identity"
                "  FROM producer_input p"
                "  LEFT JOIN producer_result r ON r.id = p.upstream_result_id"
                "  LEFT JOIN producer_invocation up ON up.id = r.invocation_id"
                " WHERE p.invocation_id = ? ORDER BY p.ordinal",
                (invocation_id,),
            ).fetchall()
        ]
        recomputed = identity_of(json.loads(preimage_json), inputs)
        if recomputed != identity:
            found.append({"invocation_id": int(invocation_id), "stored": str(identity), "recomputed": recomputed})
    return found


def re_blessed(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Contradictions whose most recent judgment cleared them under a
    declaration LATER than the one in force when they were raised.

    A queryable relation no gate runs is a record produced and never read, so
    this exists to be a closure condition rather than to be available. Each row
    is a case where loosening a tolerance retroactively cleared a disagreement,
    which is exactly the move the versioning was introduced to make visible.
    """
    rows = conn.execute(
        "SELECT c.id, c.identity, c.determinism_id, j.determinism_id, j.verdict, j.judged_at"
        "  FROM producer_contradiction c"
        "  JOIN producer_contradiction_judgment j ON j.contradiction_id = c.id"
        " WHERE j.verdict = 'within-tolerance' AND j.determinism_id > c.determinism_id"
        " ORDER BY c.id, j.judged_at"
    ).fetchall()
    return [
        {
            "contradiction_id": int(row[0]),
            "identity": str(row[1]),
            "raised_under": int(row[2]),
            "cleared_under": int(row[3]),
            "verdict": str(row[4]),
            "judged_at": float(row[5]),
        }
        for row in rows
    ]
