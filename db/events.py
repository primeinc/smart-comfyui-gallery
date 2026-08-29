"""Events: grouping hypotheses over media contexts.

A trip, a working session, a burst is a HYPOTHESIS over a set of files
-- never a property stamped onto them. This module owns the grouping
Seam: a Grouper consumes the Metadata package's Occurrence rows
(never source tables -- one definition of time and claim for every
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

CURRENTNESS IS PROVEN, not assumed -- and STABLE IS NOT COMPLETE.
Every run names the context generation and policy it was computed
over, and grouping first proves the interpretation COVERS the library:
every present file holds a current-policy context, or the hypothesis
would be a confident statement about media nobody interpreted.
Proposals are computed outside the writer lane, the lane is claimed,
generation and coverage are revalidated cheaply, and only then does
the run become durable -- a commit in the handoff triggers one
recompute, a second race refuses. A run whose generation is no longer
current is a stale hypothesis, whoever its members are; the timeline
reads only current runs.

Two adapters prove the Seam. Each grouper names the CLAIM it consumes
and reads that claim's own occurrence rows: a mixed file tells the
capture story at the camera's time and the generation story at the
generator's claimed time -- one media identity, two historical acts,
never one timestamp impersonating both. GenerationSessionGrouper
splits ONLY on temporal separation: prompt and workflow changes are
the history INSIDE a session and become phase boundaries in a later
slice (parent_id already waits). CaptureSessionGrouper clusters camera
media over a wider gap. A calendar day is deliberately NOT a grouper.

PRECISION GATES THE GAP. A member enters gap arithmetic at the finest
CONSISTENT reading it has -- its refined moment, the estimate inside its
claim, when one exists, else the claim itself -- and only when that
granule fits inside the gap. A bare day-fine date with nothing to refine
it is not minutes from anything, and 'insufficient temporal precision' is
the answer.

Regrouping keeps only each grouper's LATEST run: events are rebuildable
interpretations, not history.
"""

from __future__ import annotations

import dataclasses
import json
import typing

from vision.decode import RAW_SUFFIXES

from . import context, naming, when

#: How coarse each precision is, in seconds: `db/when.py` SPAN itself and not a
#: copy, so how wide a claim is cannot be two numbers at once. The module
#: docstring states how a granule enters gap arithmetic.
_GRANULE = when.SPAN


def _moment(one: context.Occurrence) -> float | None:
    """The finest consistent wall-clock reading: refined when the
    estimate lands inside the claim, the claim otherwise."""
    return one.refined_at if one.refined_at is not None else one.local_at


def _granule(one: context.Occurrence) -> float:
    return 1.0 if one.refined_at is not None else _GRANULE[one.time_precision]


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


def _proposed_instant(kind: str, members: list[context.Occurrence]) -> GroupProposal:
    instants = [moment for one in members if (moment := one.instant_at) is not None]
    locals_known = [moment for one in members if (moment := one.local_at) is not None]
    # the wall interval rides along only when EVERY member knows it
    both = len(locals_known) == len(members)
    return GroupProposal(
        kind=kind,
        file_ids=tuple(one.file_id for one in members),
        uuids=tuple(one.uuid for one in members),
        instant_start=min(instants),
        instant_end=max(instants),
        local_start=min(locals_known) if both else None,
        local_end=max(locals_known) if both else None,
    )


def _proposed_local(kind: str, members: list[context.Occurrence]) -> GroupProposal:
    walls = [moment for one in members if (moment := _moment(one)) is not None]
    return GroupProposal(
        kind=kind,
        file_ids=tuple(one.file_id for one in members),
        uuids=tuple(one.uuid for one in members),
        local_start=min(walls),
        local_end=max(walls),
    )


def _rendition_rank(one: context.Occurrence) -> tuple[int, str, int]:
    """Inside one act the RAW file leads -- it is what the camera
    recorded -- then the rest by name: a stable, explainable order."""
    suffix = ("." + one.name.rsplit(".", 1)[-1].lower()) if "." in one.name else ""
    return (0 if suffix in RAW_SUFFIXES else 1, one.name.lower(), one.file_id)


def _acts(eligible: list[context.Occurrence]) -> list[list[context.Occurrence]]:
    """The occurrences as ACTS: files sharing an act key are one act
    (a RAW and its JPEG are one shutter press), ranked inside it; a
    file with no act key is an act of its own. Only acts enter the gap
    arithmetic and only acts count toward a session."""
    by_key: dict[str, list[context.Occurrence]] = {}
    made: list[list[context.Occurrence]] = []
    for one in eligible:
        if one.act_key is None:
            made.append([one])
        else:
            by_key.setdefault(one.act_key, []).append(one)
    made.extend(sorted(files, key=_rendition_rank) for files in by_key.values())
    return made


def _flat(acts: list[list[context.Occurrence]]) -> list[context.Occurrence]:
    return [one for act in acts for one in act]


def _gapped(held: list[context.Occurrence], kind: str, gap: float) -> list[GroupProposal]:
    """The shared temporal-clustering implementation, per DOMAIN: media
    with knowable instants cluster on instants; media with only a wall
    clock cluster among themselves on wall clocks. Unlike domains are
    never subtracted from each other, a claim too coarse for the gap
    never enters the arithmetic, and a singleton ACT is not a session
    -- two renditions of one shutter press are one act, not two."""
    # a claim the generator disputes with itself is recorded, not
    # sequenced: the judge said it is unfit for chronology
    eligible = [one for one in held if _granule(one) <= gap and one.usable]
    made: list[GroupProposal] = []

    def order(one):
        return one.source_order if one.source_order is not None else 0

    acts = _acts(eligible)
    instants = sorted(
        (act for act in acts if act[0].instant_at is not None),
        key=lambda act: (act[0].instant_at, order(act[0]), act[0].file_id),
    )
    made.extend(
        _proposed_instant(kind, _flat(members)) for members in _split(instants, lambda act: act[0].instant_at, gap)
    )
    # On the wall clock every act is placed at its finest consistent reading,
    # then the generator's own counter, and file ids break only what both left
    # tied. Gaps and intervals use that same reading.
    walls = sorted(
        (act for act in acts if act[0].instant_at is None and act[0].local_at is not None),
        key=lambda act: (_moment(act[0]), order(act[0]), act[0].file_id),
    )
    made.extend(_proposed_local(kind, _flat(members)) for members in _split(walls, lambda act: _moment(act[0]), gap))
    return made


class GenerationSessionGrouper:
    """Generated media clustered into working sessions by time alone:
    the prompt evolution INSIDE the interval is the session's story,
    never its boundary. It consumes the GENERATION occurrence -- a
    photograph edited by a generator years after capture joins this
    story at the generator's claimed time, never the camera's."""

    name = "generation_session"
    #: v5: gaps and bounds on the refined second inside a claimed minute
    #: (db/events.py _moment); v4 gapped on the claim alone
    version = "5"
    claim = "generation"
    settings: typing.ClassVar[dict] = {"gap_minutes": 30}

    def groups(self, held: list[context.Occurrence]) -> list[GroupProposal]:
        return _gapped(held, "generation_session", self.settings["gap_minutes"] * when.MINUTE)


class CaptureSessionGrouper:
    """Camera media clustered into temporal moments -- an afternoon at
    the beach, a dinner -- over a wider gap, because humans put the
    camera down between pictures. It consumes the CAPTURE occurrence:
    the same mixed file tells this story at the camera's time."""

    name = "capture_session"
    #: v5: the same refined-time gap rule as every wall-clock grouper
    version = "5"
    claim = "capture"
    settings: typing.ClassVar[dict] = {"gap_minutes": 180}

    def groups(self, held: list[context.Occurrence]) -> list[GroupProposal]:
        return _gapped(held, "capture_session", self.settings["gap_minutes"] * when.MINUTE)


class FileSessionGrouper:
    """Media with neither a camera's nor a generator's claim -- a
    screenshot, a download, a scan -- clustered by the FILE's own claim
    (db/when.py judge_file): a stamped name or a dated folder on the
    wall clock, else the earliest the bytes are known to exist on the
    instant axis. Other implementations fall back to "file modified"
    alone; this one takes whatever the file itself says first and
    keeps the filesystem as evidence beside it."""

    name = "file_session"
    #: v2: the same refined-time gap rule as every wall-clock grouper
    version = "2"
    claim = "file"
    settings: typing.ClassVar[dict] = {"gap_minutes": 180}

    def groups(self, held: list[context.Occurrence]) -> list[GroupProposal]:
        return _gapped(held, "file_session", self.settings["gap_minutes"] * when.MINUTE)


#: The grouping adapters this build runs, in one place. The events job
#: is one item per entry, so a smarter grouper failing never costs the
#: others their run.
GROUPERS = (GenerationSessionGrouper(), CaptureSessionGrouper(), FileSessionGrouper())


def settings_hash(grouper) -> str:
    return naming.short_hash(json.dumps(grouper.settings, sort_keys=True))


def member_hash(uuids) -> str:
    """The membership's identity: ordered file uuids, hashed -- change
    one member or their order and every consumer sees a different
    event."""
    return naming.short_hash(",".join(uuids))


def _shared_place(conn, file_ids, policy: int) -> int | None:
    """The one place the members that have a place agree on, or None:
    a session happened somewhere when nobody who said where disagrees.
    Members nobody placed do not veto; two places do."""
    ids = list(file_ids)
    if not ids:
        return None
    rows = conn.execute(
        "SELECT DISTINCT place_id FROM derived_media_context WHERE policy_version = ? AND place_id IS NOT NULL"
        " AND file_id IN (" + ",".join("?" for _ in ids) + ")",
        [policy, *ids],
    ).fetchall()
    return int(rows[0][0]) if len(rows) == 1 else None


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
                    proposal.place_id
                    if proposal.place_id is not None
                    else _shared_place(conn, proposal.file_ids, policy),
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
        # STABLE is not COMPLETE: a paused context job holds the generation
        # still over a mostly-uninterpreted library, so coverage is part of
        # what the run proves.
        have, present = context.coverage(conn)
        if have != present:
            raise ValueError(
                f"the interpretation is incomplete: {have} of {present} present files"
                " have a current context; run the context job first"
            )
        proposals = grouper.groups(context.occurrences(conn, grouper.claim))
        conn.execute("BEGIN IMMEDIATE")
        if context.state(conn) == (generation, policy) and context.coverage(conn) == (have, present):
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
