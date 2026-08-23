"""StorySnapshot: an immutable, self-contained record of exactly what
the application knew about ONE current event at ONE instant.

It freezes EVIDENCE, not prose. Everything downstream of it -- a
planner that decides what is interesting, a writer that decides how a
human is told -- consumes this one frozen document and is forbidden
from reaching back into today's database for a more convenient fact.
That is the Seam between mutable library truth and storytelling: no
LLM ever decides whether July 18 happened in August because some
timestamps looked convenient.

What a snapshot freezes, by VALUE and never by foreign key: the event's
provenance (grouper, version, settings, kind, member hash, the context
generation and policy it was proven over, its interval in the domain
it actually knows); every member's file uuid AND the content hash the
library held when the snapshot was taken -- an entity survives an
in-place replacement of its bytes, so a story about yesterday's pixels
must never silently render against today's; the OCCURRENCE that placed
each member in this event (a photograph captured in 2023 and edited in
2026 tells two stories at two times, and each snapshot says which act
it froze); and the source facts the evidence Adapters collect. It is
deliberately not linked to derived_event or derived_event_run: those
are rebuildable hypotheses replaced on every regroup.

Identity is the canonical document's SHA-256: the same evidence is one
row, so retries are idempotent, and any change in the library, the
policy or a producer yields a visibly different snapshot. Rows are
insert-only -- the schema's trigger refuses UPDATE -- because the point
of a snapshot is that yesterday's prose stays explainable forever.

Freezing proves currentness twice, the shape regroup_one earned: the
event must be proven over the CURRENT generation and policy with
COMPLETE coverage; the evidence is collected; the lane is claimed; all
of it is revalidated with cheap reads; only then does the row land. A
commit in the handoff triggers one recollect; a second race refuses.
Snapshotting is database work measured in milliseconds and is done
synchronously -- "freeze the event I am looking at right now" cannot be
queued, because the event could change before a worker reached it.
Model work happens AFTER, on the frozen input, as durable jobs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing

from . import context, prompt_sections
from . import prompts as prompts_module

#: The document shape. Bump when a field's MEANING changes; a consumer
#: that reads a document checks this before trusting any key.
FORMAT_VERSION = 1

#: The session kinds a snapshot may be of -- the schema's own CHECK
#: (story_snapshot.event_kind), spelled once for every caller that
#: filters by it.
EVENT_KINDS = ("generation_session", "capture_session", "file_session")

_CANONICAL = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": False}


def canonical(document: dict) -> str:
    """One spelling of one document: sorted keys, no whitespace -- the
    bytes the identity is hashed over."""
    return json.dumps(document, **_CANONICAL)


def digest(spelled: str) -> str:
    return hashlib.sha256(spelled.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class SnapshotRef:
    id: int
    sha256: str
    reused: bool


# --- the evidence Seam -------------------------------------------------------
#
# Each Adapter answers one question about a set of member files and
# returns {file_id: bundle}. The snapshot Module chooses them and owns
# assembly; adding OCR later, or swapping a caption producer, changes
# one Adapter and nothing downstream learns where OCR happens.


class _Evidence(typing.Protocol):
    name: str

    def collect(self, conn, file_ids: list[int]) -> dict[int, typing.Any]: ...


def _marks(file_ids: list[int]) -> str:
    return ",".join("?" * len(file_ids))


class GenerationEvidence:
    """Prompts BY ROLE (effective, original, negative -- each with its
    stable uuid, exact text and text hash, so a plan can prove which
    bytes it compared), the workflow, the sampling parameters, every
    artifact the generation named, and the generator's own parameter
    bag (wildcards, date, timings -- whatever it recorded). `prompt`
    and `negative_prompt` are the effective and negative texts, spelled
    again for readers of the roles' two commonest entries."""

    name = "generation"

    def collect(self, conn, file_ids):
        if not file_ids:
            return {}
        held: dict[int, dict] = {}
        for row in conn.execute(
            "SELECT g.file_id, g.tool, g.detection, g.seed, g.steps, g.cfg, g.denoise, g.clip_skip,"
            " g.sampler, g.scheduler, g.width, g.height, we.uuid"
            " FROM generation g"
            " LEFT JOIN entity we ON we.id = g.workflow_id"
            f" WHERE g.file_id IN ({_marks(file_ids)})",
            file_ids,
        ):
            held[row[0]] = {
                "tool": row[1],
                "detection": row[2],
                "seed": row[3],
                "steps": row[4],
                "cfg": row[5],
                "denoise": row[6],
                "clip_skip": row[7],
                "sampler": row[8],
                "scheduler": row[9],
                "width": row[10],
                "height": row[11],
                "prompt": None,
                "negative_prompt": None,
                "prompts": [],
                "workflow_uuid": row[12].hex() if row[12] else None,
                "artifacts": [],
                "params": {},
            }
        if not held:
            return {}
        members = list(held)
        for row in conn.execute(
            "SELECT gp.file_id, gp.role, e.uuid, p.text, p.text_hash FROM generation_prompt gp"
            " JOIN prompt p ON p.id = gp.prompt_id JOIN entity e ON e.id = p.id"
            f" WHERE gp.file_id IN ({_marks(members)}) ORDER BY gp.file_id, gp.role",
            members,
        ):
            one = held[row[0]]
            # the MAIN section as read by the parser of the moment, frozen
            # with the text: a later grammar re-reads the library, never
            # what a plan or a view was made from
            grammar = prompts_module.grammar_for(one["tool"])
            main = prompt_sections.main(row[3], grammar)
            one["prompts"].append(
                {
                    "role": row[1],
                    "uuid": row[2].hex(),
                    "text": row[3],
                    "text_hash": row[4],
                    "main": main,
                    "main_hash": prompts_module.text_hash(main),
                    "grammar": grammar,
                    "parser": prompt_sections.VERSION,
                }
            )
            if row[1] == "effective":
                one["prompt"] = row[3]
            elif row[1] == "negative":
                one["negative_prompt"] = row[3]
        for row in conn.execute(
            "SELECT fa.file_id, fa.ordinal, fa.role, e.uuid, a.kind, a.name, fa.model_weight, fa.clip_weight"
            " FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id JOIN entity e ON e.id = a.id"
            f" WHERE fa.file_id IN ({_marks(members)})"
            "   AND fa.role NOT IN ('captured_with', 'mounted_lens')"
            " ORDER BY fa.file_id, fa.role, fa.ordinal",
            members,
        ):
            held[row[0]]["artifacts"].append(
                {
                    "ordinal": row[1],
                    "role": row[2],
                    "uuid": row[3].hex(),
                    "kind": row[4],
                    "name": row[5],
                    "model_weight": row[6],
                    "clip_weight": row[7],
                }
            )
        for row in conn.execute(
            "SELECT file_id, key, value_text FROM file_param"
            f" WHERE source = 'generation' AND file_id IN ({_marks(members)})",
            members,
        ):
            held[row[0]]["params"][row[1]] = row[2]
        return held


class CaptureEvidence:
    """The camera's claims: exposure, optics, GPS, and the camera/lens
    artifacts the file was captured with."""

    name = "capture"

    def collect(self, conn, file_ids):
        if not file_ids:
            return {}
        held: dict[int, dict] = {}
        for row in conn.execute(
            "SELECT file_id, captured_at, tz_offset_min, iso, f_number, exposure_time, focal_length,"
            " focal_35mm, orientation, gps_lat, gps_lon, gps_alt, subsec_ms, body_serial, maker_tz_offset_min"
            f" FROM capture WHERE file_id IN ({_marks(file_ids)})",
            file_ids,
        ):
            held[row[0]] = {
                "captured_at": row[1],
                "tz_offset_min": row[2],
                "iso": row[3],
                "f_number": row[4],
                "exposure_time": row[5],
                "focal_length": row[6],
                "focal_35mm": row[7],
                "orientation": row[8],
                "gps": {"lat": row[9], "lon": row[10], "alt": row[11]} if row[9] is not None else None,
                "subsec_ms": row[12],
                "body_serial": row[13],
                "maker_tz_offset_min": row[14],
                "equipment": [],
            }
        if not held:
            return {}
        members = list(held)
        for row in conn.execute(
            "SELECT fa.file_id, fa.role, e.uuid, a.kind, a.name"
            " FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id JOIN entity e ON e.id = a.id"
            f" WHERE fa.file_id IN ({_marks(members)}) AND fa.role IN ('captured_with', 'mounted_lens')"
            " ORDER BY fa.file_id, fa.role, fa.ordinal",
            members,
        ):
            held[row[0]]["equipment"].append({"role": row[1], "uuid": row[2].hex(), "kind": row[3], "name": row[4]})
        return held


class PeopleEvidence:
    """Who is in the picture -- what a human ASSERTED kept apart from
    what a model INFERRED, each with the person's uuid and the name as
    it stood at the instant of the snapshot."""

    name = "people"

    def collect(self, conn, file_ids):
        if not file_ids:
            return {}
        held: dict[int, list] = {}
        for row in conn.execute(
            "SELECT pa.file_id, e.uuid, p.name FROM person_assertion pa"
            " JOIN person p ON p.id = pa.person_id JOIN entity e ON e.id = p.id"
            f" WHERE pa.file_id IN ({_marks(file_ids)}) ORDER BY pa.file_id, e.uuid",
            file_ids,
        ):
            held.setdefault(row[0], []).append({"uuid": row[1].hex(), "name": row[2], "basis": "asserted"})
        for row in conn.execute(
            "SELECT DISTINCT fp.file_id, e.uuid, p.name, fp.model_id, fp.model_version FROM derived_file_person fp"
            " JOIN person p ON p.id = fp.person_id JOIN entity e ON e.id = p.id"
            f" WHERE fp.file_id IN ({_marks(file_ids)}) ORDER BY fp.file_id, e.uuid",
            file_ids,
        ):
            held.setdefault(row[0], []).append(
                {"uuid": row[1].hex(), "name": row[2], "basis": "inferred", "model": [row[3], row[4]]}
            )
        return held


class PlaceEvidence:
    """A place by stable uuid plus its human-readable hierarchy, frozen
    as labelled at the instant -- a renamed region later must not
    rewrite where yesterday's story happened."""

    name = "place"

    def collect(self, conn, file_ids):
        if not file_ids:
            return {}
        held: dict[int, dict | None] = {}
        for row in conn.execute(
            "SELECT mc.file_id, mc.place_id FROM derived_media_context mc"
            f" WHERE mc.place_id IS NOT NULL AND mc.policy_version = ? AND mc.file_id IN ({_marks(file_ids)})",
            (context.POLICY_VERSION, *file_ids),
        ):
            held[row[0]] = hierarchy(conn, row[1])
        return held


def hierarchy(conn, place_id: int | None) -> dict | None:
    """One place and every ancestor, leaf first, by uuid and label."""
    if place_id is None:
        return None
    from . import places

    chain = []
    for one in places.chain(conn, place_id):
        uuid = conn.execute("SELECT uuid FROM entity WHERE id = ?", (one["id"],)).fetchone()[0]
        chain.append({"uuid": uuid.hex(), "kind": one["kind"], "name": one["name"]})
    return {"uuid": chain[0]["uuid"], "chain": chain} if chain else None


class LineageEvidence:
    """What each member came from and what came from it -- by uuid, so a
    parent outside the event is still nameable later."""

    name = "lineage"

    def collect(self, conn, file_ids):
        if not file_ids:
            return {}
        held: dict[int, dict] = {}
        for row in conn.execute(
            "SELECT d.child_id, d.kind, e.uuid FROM file_derivation d JOIN entity e ON e.id = d.parent_id"
            f" WHERE d.child_id IN ({_marks(file_ids)}) ORDER BY d.child_id, d.kind, e.uuid",
            file_ids,
        ):
            held.setdefault(row[0], {"parents": [], "children": []})["parents"].append(
                {"kind": row[1], "uuid": row[2].hex()}
            )
        for row in conn.execute(
            "SELECT d.parent_id, d.kind, e.uuid FROM file_derivation d JOIN entity e ON e.id = d.child_id"
            f" WHERE d.parent_id IN ({_marks(file_ids)}) ORDER BY d.parent_id, d.kind, e.uuid",
            file_ids,
        ):
            held.setdefault(row[0], {"parents": [], "children": []})["children"].append(
                {"kind": row[1], "uuid": row[2].hex()}
            )
        return held


class AnnotationEvidence:
    """Captions, tags, OCR, titles -- with the producer and version that
    said so, because a model's sentence is inference, and a better
    model later means a NEW snapshot, not a rewritten one."""

    name = "annotations"

    def collect(self, conn, file_ids):
        if not file_ids:
            return {}
        held: dict[int, list] = {}
        for row in conn.execute(
            "SELECT file_id, kind, text, confidence, model_id, model_version FROM derived_annotation"
            f" WHERE file_id IN ({_marks(file_ids)}) ORDER BY file_id, kind, model_id, model_version, id",
            file_ids,
        ):
            held.setdefault(row[0], []).append(
                {"kind": row[1], "text": row[2], "confidence": row[3], "model": [row[4], row[5]]}
            )
        return held


#: The Adapters this build freezes, in one place. Order is irrelevant to
#: the document (keys are sorted) and to identity (the hash is over the
#: canonical spelling).
EVIDENCE: tuple[_Evidence, ...] = (
    GenerationEvidence(),
    CaptureEvidence(),
    PeopleEvidence(),
    PlaceEvidence(),
    LineageEvidence(),
    AnnotationEvidence(),
)


# --- the subject: a CURRENT event ----------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Subject:
    event_id: int
    run_id: int
    kind: str
    grouper: str
    grouper_version: str
    settings_hash: str
    claim: str
    member_hash: str
    generation: int
    policy: int
    local_start: float | None
    local_end: float | None
    instant_start: float | None
    instant_end: float | None
    place_id: int | None
    confidence: float | None


_CLAIM_OF = {"generation_session": "generation", "capture_session": "capture", "file_session": "file"}


def _current_subject(conn, event_id: int) -> _Subject:
    """The event, PROVEN current: its run names the generation and
    policy the interpretation is at, and the interpretation covers the
    library. Anything less is a stale hypothesis, not a subject."""
    row = conn.execute(
        "SELECT e.id, e.run_id, e.kind, r.grouper, r.grouper_version, r.settings_hash, e.member_hash,"
        " r.context_generation, r.context_policy_version, e.local_start, e.local_end,"
        " e.instant_start, e.instant_end, e.place_id, e.confidence"
        " FROM derived_event e JOIN derived_event_run r ON r.id = e.run_id WHERE e.id = ?",
        (event_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"no event {event_id}; hypotheses are rebuilt by the events job and ids do not survive it")
    state = context.state(conn)
    if state is None or (row[7], row[8]) != state or row[8] != context.POLICY_VERSION:
        raise ValueError("the event is not current: its interpretation moved on; run the context and events jobs")
    have, present = context.coverage(conn)
    if have != present:
        raise ValueError(f"the interpretation is incomplete: {have} of {present} present files; run the context job")
    return _Subject(
        event_id=row[0],
        run_id=row[1],
        kind=row[2],
        grouper=row[3],
        grouper_version=row[4],
        settings_hash=row[5],
        claim=_CLAIM_OF[row[2]],
        member_hash=row[6],
        generation=row[7],
        policy=row[8],
        local_start=row[9],
        local_end=row[10],
        instant_start=row[11],
        instant_end=row[12],
        place_id=row[13],
        confidence=row[14],
    )


def _members(conn, subject: _Subject) -> list[dict]:
    """Every member in its event order, with the identity and the bytes
    the library holds RIGHT NOW, and the occurrence of the subject's
    claim that placed it here."""
    rows = conn.execute(
        "SELECT ef.ordinal, f.id, e.uuid, f.content_sha256, f.kind, f.name,"
        " o.basis, o.local_at, o.instant_at, o.tz_offset_min, o.time_precision, o.certainty,"
        " o.supports, o.conflicts, o.finished_at, o.estimated_at, o.source_order, o.act_key, f.duration"
        " FROM derived_event_file ef"
        " JOIN file f ON f.id = ef.file_id"
        " JOIN entity e ON e.id = f.id"
        " LEFT JOIN derived_media_occurrence o"
        "   ON o.file_id = f.id AND o.kind = ? AND o.policy_version = ?"
        " WHERE ef.event_id = ? ORDER BY ef.ordinal",
        (subject.claim, context.POLICY_VERSION, subject.event_id),
    ).fetchall()
    return [
        {
            "ordinal": row[0],
            "_file_id": row[1],
            "file_uuid": row[2].hex(),
            "content_sha256": row[3],
            "media_kind": row[4],
            "name": row[5],
            "duration": row[18],
            "occurrence": {
                "kind": subject.claim,
                "basis": row[6],
                "local_at": row[7],
                "instant_at": row[8],
                "tz_offset_min": row[9],
                "precision": row[10],
                "certainty": row[11],
                "supports": json.loads(row[12]) if row[12] else [],
                "conflicts": json.loads(row[13]) if row[13] else [],
                "finished_at": row[14],
                "estimated_at": row[15],
                "source_order": row[16],
                "act_key": row[17],
            }
            if row[6] is not None
            else None,
        }
        for row in rows
    ]


def _document(conn, subject: _Subject, now: float) -> dict:
    members = _members(conn, subject)
    file_ids = [one["_file_id"] for one in members]
    gathered = {adapter.name: adapter.collect(conn, file_ids) for adapter in EVIDENCE}
    for one in members:
        file_id = one.pop("_file_id")
        for name, bundles in gathered.items():
            one[name] = bundles.get(file_id)
    return {
        "v": FORMAT_VERSION,
        "frozen_at": now,
        "subject": {
            "kind": "event",
            "event_kind": subject.kind,
            "grouper": subject.grouper,
            "grouper_version": subject.grouper_version,
            "settings_hash": subject.settings_hash,
            "claim": subject.claim,
            "member_hash": subject.member_hash,
            "context_generation": subject.generation,
            "context_policy_version": subject.policy,
            "time": {
                "local": [subject.local_start, subject.local_end] if subject.local_start is not None else None,
                "instant": [subject.instant_start, subject.instant_end] if subject.instant_start is not None else None,
            },
            "place": hierarchy(conn, subject.place_id),
            "confidence": subject.confidence,
            "observed_event_id": subject.event_id,
        },
        "members": members,
    }


def _identity(document: dict) -> tuple[str, str]:
    """The hash is over the EVIDENCE: `frozen_at` is provenance of the
    act of freezing, not of what was frozen, so identical evidence
    frozen twice is one snapshot."""
    evidence = {key: value for key, value in document.items() if key != "frozen_at"}
    spelled = canonical(evidence)
    return spelled, digest(spelled)


def snapshot_event(conn, event_id: int, now: float) -> SnapshotRef:
    """Freeze one current event. Proves currentness, collects outside
    the writer lane, claims the lane, revalidates, inserts -- or returns
    the existing row when the evidence is byte-identical. The
    transaction is left open for the caller to commit."""
    for last in (False, True):
        subject = _current_subject(conn, event_id)
        document = _document(conn, subject, now)
        _spelled, sha = _identity(document)
        conn.execute("BEGIN IMMEDIATE")
        still = context.state(conn)
        event_row = conn.execute(
            "SELECT e.member_hash, e.run_id FROM derived_event e WHERE e.id = ?", (event_id,)
        ).fetchone()
        unmoved = (
            still == (subject.generation, subject.policy)
            and context.coverage(conn)[0] == context.coverage(conn)[1]
            and event_row is not None
            and (event_row[0], event_row[1]) == (subject.member_hash, subject.run_id)
        )
        if unmoved:
            held = conn.execute("SELECT id FROM story_snapshot WHERE document_sha256 = ?", (sha,)).fetchone()
            if held:
                return SnapshotRef(int(held[0]), sha, True)
            snapshot_id = int(
                conn.execute(
                    "INSERT INTO story_snapshot(format_version, source_kind, event_kind, grouper,"
                    " context_generation, context_policy_version, member_hash, document_json,"
                    " document_sha256, created_at) VALUES(?, 'event', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        FORMAT_VERSION,
                        subject.kind,
                        subject.grouper,
                        subject.generation,
                        subject.policy,
                        subject.member_hash,
                        canonical(document),
                        sha,
                        now,
                    ),
                ).lastrowid
                or 0
            )
            return SnapshotRef(snapshot_id, sha, False)
        conn.execute("ROLLBACK")
        if last:
            raise ValueError("the library kept moving while freezing; nothing was persisted")
    raise AssertionError("unreachable")


def load_snapshot(conn, snapshot_id: int) -> dict:
    """The frozen document, as frozen -- and PROVEN so on every read: a
    row whose bytes no longer hash to its identity is refused, never
    served. A GET of history: reads nothing live, consults no current
    table."""
    row = conn.execute(
        "SELECT document_json, document_sha256 FROM story_snapshot WHERE id = ?", (snapshot_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no story snapshot {snapshot_id}")
    document = parsed(row[0], f"story snapshot {snapshot_id}")
    if not verify(document, row[1]):
        raise Corrupt(f"story snapshot {snapshot_id} no longer hashes to its identity; refusing to serve it")
    return document


class Corrupt(ValueError):
    """A stored row that is not the document its table promises."""


def parsed(raw: str, what: str) -> dict:
    """The stored JSON as a document, or a controlled refusal: a row
    holding `[]` or `null` parses fine and is still not a document, and
    hashing it would raise something a page turns into a 500."""
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise Corrupt(f"{what} is not a document; refusing to serve it")
    return document


def verify(document: dict, sha: str) -> bool:
    """Does this document's evidence still hash to the identity it was
    stored under? The consumer's own check before trusting a row."""
    return _identity(document)[1] == sha
