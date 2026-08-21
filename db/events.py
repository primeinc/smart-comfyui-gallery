"""Events: grouping hypotheses over media contexts.

A trip, a working session, a burst is a HYPOTHESIS over a set of files
-- never a property stamped onto them. This module owns the grouping
Seam: a Grouper consumes the Metadata package's MediaContext rows
(never source tables -- one definition of time and origin for every
algorithm) and proposes kind/interval/ordered-membership; persistence
turns proposals into rebuildable `derived_event_run` rows whose events
carry a member_hash over the ordered file uuids, so a changed
membership is VISIBLY a different event and anything built on one -- a
future story, a cached summary -- invalidates by hash instead of by
bespoke bookkeeping.

Two adapters prove the Seam. GenerationSessionGrouper splits ONLY on
temporal separation: prompt and workflow changes are the history INSIDE
a session -- the exploration, the revision, the LoRA experiment -- and
become phase boundaries in a later slice (parent_id already waits for
them); splitting on them would summarize every changed prompt as its
own event, a triumph of technically correct uselessness.
CaptureSessionGrouper clusters camera media into temporal moments the
same way, over a wider gap. A calendar day is deliberately NOT a
grouper: days are presentation, read straight off the contexts.

Regrouping keeps only each grouper's LATEST run: events are rebuildable
interpretations, not history, and the consumers' question is "what does
the library look like now".
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing

from . import context


@dataclasses.dataclass(frozen=True)
class GroupProposal:
    """What a Grouper claims: these files, in this order, are one event
    of this kind over this interval."""

    kind: str
    start_at: float
    end_at: float
    file_ids: tuple[int, ...]
    uuids: tuple[str, ...]
    confidence: float | None = None
    place_id: int | None = None


def _gapped(held: list[context.MediaContext], kind: str, origin: str, gap: float) -> list[GroupProposal]:
    """The shared temporal-clustering implementation: media of one
    origin, chronological, split where the maker walked away. A
    singleton is not a session."""
    made: list[GroupProposal] = []
    current: list[context.MediaContext] = []

    def close() -> None:
        if len(current) >= 2:
            made.append(_proposed(kind, current))

    for one in held:
        if one.origin != origin or one.moment is None:
            continue
        if current and one.moment - (current[-1].moment or 0.0) > gap:
            close()
            current = []
        current.append(one)
    close()
    return made


class GenerationSessionGrouper:
    """Generated media clustered into working sessions by time alone:
    the prompt evolution INSIDE the interval is the session's story,
    never its boundary."""

    name = "generation_session"
    version = "2"
    settings: typing.ClassVar[dict] = {"gap_minutes": 30}

    def groups(self, held: list[context.MediaContext]) -> list[GroupProposal]:
        return _gapped(held, "generation_session", "generated", self.settings["gap_minutes"] * 60.0)


class CaptureSessionGrouper:
    """Camera media clustered into temporal moments -- an afternoon at
    the beach, a dinner -- over a wider gap than a generation session,
    because humans put the camera down between pictures."""

    name = "capture_session"
    version = "1"
    settings: typing.ClassVar[dict] = {"gap_minutes": 180}

    def groups(self, held: list[context.MediaContext]) -> list[GroupProposal]:
        return _gapped(held, "capture_session", "captured", self.settings["gap_minutes"] * 60.0)


def _proposed(kind: str, members: list[context.MediaContext]) -> GroupProposal:
    moments = [one.moment for one in members if one.moment is not None]
    return GroupProposal(
        kind=kind,
        start_at=min(moments),
        end_at=max(moments),
        file_ids=tuple(one.file_id for one in members),
        uuids=tuple(one.uuid for one in members),
    )


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


def regroup_one(conn, grouper, now: float) -> int:
    """One grouper re-proposes over the current contexts and keeps only
    its latest run. Returns the run id."""
    held = context.contexts(conn)
    run_id = int(
        conn.execute(
            "INSERT INTO derived_event_run(grouper, grouper_version, settings_hash, created_at) VALUES(?, ?, ?, ?)",
            (grouper.name, grouper.version, settings_hash(grouper), now),
        ).lastrowid
        or 0
    )
    for proposal in grouper.groups(held):
        event_id = int(
            conn.execute(
                "INSERT INTO derived_event(run_id, parent_id, kind, start_at, end_at,"
                " place_id, confidence, member_hash)"
                " VALUES(?, NULL, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    proposal.kind,
                    proposal.start_at,
                    proposal.end_at,
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


def regroup(conn, now: float, groupers=GROUPERS) -> dict[str, int]:
    return {grouper.name: regroup_one(conn, grouper, now) for grouper in groupers}
