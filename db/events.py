"""Events: grouping hypotheses over media contexts.

A trip, a working session, a burst is a HYPOTHESIS over a set of files
-- never a property stamped onto them. This module owns the grouping
Seam: a Grouper consumes the Metadata package's MediaContext rows
(never source tables -- one definition of time and origin for every
algorithm) and proposes kind/interval/ordered-membership; persistence
turns proposals into rebuildable `derived_event_run` rows whose events
carry a member_hash over the ordered file uuids, so a changed
membership is VISIBLY a different event.

TIME HAS A DOMAIN here. A grouper clusters within one domain at a
time: media with knowable instants cluster on instants, media with
only a wall clock cluster on wall clocks, and the two are NEVER
subtracted from each other -- an unzoned afternoon and a UTC instant
are not seconds apart, they are incomparable. Each persisted event
carries the interval pair(s) of the domain(s) it actually knows.

CURRENTNESS IS PROVEN, not assumed. Every run names the context
generation and policy it was computed over; proposals are computed
outside the writer lane, the lane is claimed, the generation is
revalidated with one cheap read, and only then does the run become
durable -- a commit in the handoff triggers one recompute, a second
race refuses. A run whose generation is no longer current is a stale
hypothesis, whoever its members are; the timeline reads only current
runs.

Two adapters prove the Seam. GenerationSessionGrouper splits ONLY on
temporal separation: prompt and workflow changes are the history INSIDE
a session and become phase boundaries in a later slice (parent_id
already waits); splitting on them would summarize every changed prompt
as its own event. CaptureSessionGrouper clusters camera media into
temporal moments over a wider gap. Participation is the explicit
has_generation / has_capture fact -- a mixed file belongs to both
stories. A calendar day is deliberately NOT a grouper.

Regrouping keeps only each grouper's LATEST run: events are rebuildable
interpretations, not history.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing

from . import context

#: How coarse each precision is, in seconds. A member may enter gap
#: arithmetic only when its granule fits inside the gap: a
#: day-resolution date subtracted from anything yields
#: confident-looking nonsense, and 'insufficient temporal precision'
#: is an answer, never a defect.
_GRANULE = {"day": 86_400.0, "hour": 3_600.0, "minute": 60.0, "second": 1.0, "subsecond": 0.001}


@dataclasses.dataclass(frozen=True)
class GroupProposal:
    """What a Grouper claims: these files, in this order, are one event
    of this kind -- over a wall-clock interval, an instant interval, or
    both when every member makes both knowable."""

    kind: str
    file_ids: tuple[int, ...]
    uuids: tuple[str, ...]
    local_start: float | None = None
    local_end: float | None = None
    instant_start: float | None = None
    instant_end: float | None = None
    confidence: float | None = None
    place_id: int | None = None


def _split(members: list, key, gap: float) -> list[list]:
    made: list[list] = []
    current: list = []
    for one in members:
        if current and key(one) - key(current[-1]) > gap:
            if len(current) >= 2:
                made.append(current)
            current = []
        current.append(one)
    if len(current) >= 2:
        made.append(current)
    return made


def _proposed_instant(kind: str, members: list[context.MediaContext]) -> GroupProposal:
    instants = [one.instant_at for one in members]
    locals_known = [one.local_at for one in members]
    both = all(one is not None for one in locals_known)
    return GroupProposal(
        kind=kind,
        file_ids=tuple(one.file_id for one in members),
        uuids=tuple(one.uuid for one in members),
        instant_start=min(instants),
        instant_end=max(instants),
        # the wall interval rides along only when EVERY member knows it
        local_start=min(locals_known) if both else None,
        local_end=max(locals_known) if both else None,
    )


def _proposed_local(kind: str, members: list[context.MediaContext]) -> GroupProposal:
    walls = [one.local_at for one in members]
    return GroupProposal(
        kind=kind,
        file_ids=tuple(one.file_id for one in members),
        uuids=tuple(one.uuid for one in members),
        local_start=min(walls),
        local_end=max(walls),
    )


def _gapped(held: list[context.MediaContext], kind: str, takes_part, gap: float) -> list[GroupProposal]:
    """The shared temporal-clustering implementation, per DOMAIN: media
    with knowable instants cluster on instants; media with only a wall
    clock cluster among themselves on wall clocks. Unlike domains are
    never subtracted from each other, a claim too coarse for the gap
    never enters the arithmetic, and a singleton is not a session."""
    eligible = [
        one
        for one in held
        if takes_part(one) and one.time_precision is not None and _GRANULE[one.time_precision] <= gap
    ]
    made: list[GroupProposal] = []
    instants = sorted(
        (one for one in eligible if one.instant_at is not None), key=lambda one: (one.instant_at, one.file_id)
    )
    made.extend(_proposed_instant(kind, members) for members in _split(instants, lambda one: one.instant_at, gap))
    walls = sorted(
        (one for one in eligible if one.instant_at is None and one.local_at is not None),
        key=lambda one: (one.local_at, one.file_id),
    )
    made.extend(_proposed_local(kind, members) for members in _split(walls, lambda one: one.local_at, gap))
    return made


class GenerationSessionGrouper:
    """Generated media clustered into working sessions by time alone:
    the prompt evolution INSIDE the interval is the session's story,
    never its boundary. Participation is the has_generation FACT, so a
    photograph that was also generated belongs to this story too."""

    name = "generation_session"
    version = "3"
    settings: typing.ClassVar[dict] = {"gap_minutes": 30}

    def groups(self, held: list[context.MediaContext]) -> list[GroupProposal]:
        return _gapped(held, "generation_session", lambda one: one.has_generation, self.settings["gap_minutes"] * 60.0)


class CaptureSessionGrouper:
    """Camera media clustered into temporal moments -- an afternoon at
    the beach, a dinner -- over a wider gap, because humans put the
    camera down between pictures. Participation is the has_capture
    FACT."""

    name = "capture_session"
    version = "2"
    settings: typing.ClassVar[dict] = {"gap_minutes": 180}

    def groups(self, held: list[context.MediaContext]) -> list[GroupProposal]:
        return _gapped(held, "capture_session", lambda one: one.has_capture, self.settings["gap_minutes"] * 60.0)


#: The grouping adapters this build runs, in one place. The events job
#: is one item per entry, so a smarter grouper failing never costs the
#: others their run.
GROUPERS = (GenerationSessionGrouper(), CaptureSessionGrouper())


def settings_hash(grouper) -> str:
    return hashlib.sha256(json.dumps(grouper.settings, sort_keys=True).encode()).hexdigest()[:16]


def member_hash(uuids) -> str:
    """The membership's identity: ordered file uuids, hashed -- change
    one member or their order and every consumer sees a different
    event."""
    return hashlib.sha256(",".join(uuids).encode()).hexdigest()[:16]


def _persist(conn, grouper, proposals, generation: int, policy: int, now: float) -> int:
    run_id = int(
        conn.execute(
            "INSERT INTO derived_event_run(grouper, grouper_version, settings_hash,"
            " context_generation, context_policy_version, created_at) VALUES(?, ?, ?, ?, ?, ?)",
            (grouper.name, grouper.version, settings_hash(grouper), generation, policy, now),
        ).lastrowid
        or 0
    )
    for proposal in proposals:
        event_id = int(
            conn.execute(
                "INSERT INTO derived_event(run_id, parent_id, kind, local_start, local_end,"
                " instant_start, instant_end, place_id, confidence, member_hash)"
                " VALUES(?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    proposal.kind,
                    proposal.local_start,
                    proposal.local_end,
                    proposal.instant_start,
                    proposal.instant_end,
                    proposal.place_id,
                    proposal.confidence,
                    member_hash(proposal.uuids),
                ),
            ).lastrowid
            or 0
        )
        conn.executemany(
            "INSERT INTO derived_event_file(event_id, file_id, ordinal, score) VALUES(?, ?, ?, NULL)",
            [(event_id, file_id, ordinal) for ordinal, file_id in enumerate(proposal.file_ids)],
        )
    conn.execute("DELETE FROM derived_event_run WHERE grouper = ? AND id <> ?", (grouper.name, run_id))
    return run_id


def regroup_one(conn, grouper, now: float) -> int:
    """One grouper re-proposes over the current contexts, PROVING the
    contexts held still: proposals are computed outside the writer
    lane; the lane is claimed; the generation is revalidated with one
    cheap read; only then does the run become durable, tagged with the
    generation it proves. A commit in the handoff triggers one
    recompute outside the lane; a second race refuses with nothing
    written. The transaction is left open for the caller (the job
    runner) to commit with its own bookkeeping."""
    for last in (False, True):
        held_state = context.state(conn)
        if held_state is None:
            raise ValueError("no interpretation exists yet; run the context job first")
        generation, policy = held_state
        if policy != context.POLICY_VERSION:
            raise ValueError("the interpretation is from an older policy; run the context job first")
        proposals = grouper.groups(context.contexts(conn))
        conn.execute("BEGIN IMMEDIATE")
        if context.state(conn) == (generation, policy):
            return _persist(conn, grouper, proposals, generation, policy, now)
        conn.execute("ROLLBACK")
        if last:
            raise ValueError("the contexts kept moving during grouping; nothing was persisted")
    raise AssertionError("unreachable")


def regroup(conn, now: float, groupers=GROUPERS) -> dict[str, int]:
    runs = {}
    for grouper in groupers:
        runs[grouper.name] = regroup_one(conn, grouper, now)
        conn.commit()
    return runs
