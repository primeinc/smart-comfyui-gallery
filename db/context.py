"""One interpretation of when, where and how each media item happened.

The MediaContext half of the Metadata package. Raw evidence
(blob/file_blob) is what the media actually said and is never
normalized away; source facts (capture, generation, file, file_param)
are per-source CLAIMS -- EXIF's DateTimeOriginal and the filesystem's
mtime are independent observations, never two spellings of one value.
`derived_media_context` is the application's best CURRENT understanding
built from those claims: derived, rebuildable, always carrying its
BASIS so no date is ever unexplained, and stamped with the
POLICY_VERSION that produced it -- a better ladder tomorrow visibly
obsoletes today's rows instead of impersonating them.

TWO time concepts, held apart on purpose and never handed out fused.
`local_at` is what the human clock said -- the wall time a camera or a
generator claimed, the thing "Saturday in Hawaii" is made of.
`instant_at` is the actual UTC instant, present ONLY when knowable --
and every time carries its PRECISION, orthogonal to certainty: a
day-resolution generator date is excellent evidence for WHAT DAY and
no evidence at all for WHAT MINUTE, so the highest-ranked claim FIT
FOR THE QUESTION wins, never merely the highest-ranked claim. A
camera claim with an offset yields both at full certainty; without the
offset the wall clock STANDS and the instant stays honestly absent.
After the camera speaks the GENERATOR -- and not alone: db/when.py
judges the generator's day, SwarmUI's request minute in the file name,
the file's mtime and btime and the generation time TOGETHER, refining
the day to the second when the finer sources agree, recording every
source that does not. Only claimless media fall to mtime then btime,
instants with no local story. A claim that does not parse is no claim.
Model-derived annotations are inference, not evidence: nothing they say
may enter this ladder. There is deliberately NO fused
"moment" accessor: a consumer chooses a domain, because a convenience
value that is secretly sometimes each is how unlike times get
subtracted from each other.

Coexistence is fact, never precedence: `has_capture` and
`has_generation` are recorded as they stand and `origin` is DETERMINED
from them by CHECK -- a photograph that was also run through a
generator is `mixed`, with neither claim erased.

`derived_context_state` is the interpretation's identity: `generation`
advances on EVERY context add, change or delete, so anything computed
over the contexts (an event run, someday a story) can prove it was
computed over THESE contexts. Invalidation lives HERE, called from the
source-fact writer seams (db/ingest.py after a parse, db/scan.py when
a file's times change) -- deliberately not as schema triggers, because
a source-table trigger referencing a derived table breaks the
drop-derived-and-reindex contract. A stale context is DELETED, never
silently served; the explicit context job makes it current again.
Nothing expensive starts by itself.
"""

from __future__ import annotations

import dataclasses

ORIGINS = ("captured", "generated", "mixed", "imported")
TIME_BASES = ("capture", "embedded", "filename", "btime", "mtime", "first_seen")
LOCATION_BASES = ("gps", "sidecar", "inferred", "authored")

#: WHICH MEANING of the ladder is current. Bump when the interpretation
#: itself changes meaning -- v2 added the embedded generator-date rung,
#: v3 the precision dimension and the coexistence facts, v4 the
#: per-claim occurrence rows, v5 the generation judge, v6 the capture
#: judge and the act key. Every reader binds THIS constant, never
#: the version a database happens to remember: after an upgrade the old
#: rows are honestly invisible until the context job re-interprets.
POLICY_VERSION = 6

#: The human timeline's one axis, defined ONCE: the wall clock when one
#: was claimed, the knowable instant otherwise. The day facet and the
#: timeline shelves compose around this same fragment, so the door and
#: the shelf cannot drift apart.
HUMAN_MOMENT = "COALESCE(mc.local_at, mc.instant_at)"

#: The whole ladder, one statement, applied to whichever files the
#: caller names. Basis and certainty recorded beside every value they
#: explain; place_id stays NULL until an explicit resolver or an
#: authored assertion mints real place identity -- GPS alone never does.
#: Each temporal CLAIM as its own row: the capture act at capture time,
#: the generation act at the time the judge (db/when.py) settles from
#: every claim the file carries. The context keeps the ONE primary
#: human-timeline interpretation; these exist so a grouper reads the
#: time of ITS OWN claim -- a photograph edited by a generator years
#: later is 2023 in the capture story and 2026 in the generation story.
_OCCUR_CAPTURE = """
INSERT INTO derived_media_occurrence(file_id, kind, local_at, instant_at,
  tz_offset_min, basis, certainty, supports, conflicts, finished_at, act_key,
  time_precision, policy_version)
VALUES(?, 'capture', ?, ?, ?, 'capture', ?, ?, ?, ?, ?, ?, ?)
"""

_SOURCES = """
SELECT f.id, f.name, f.mtime, f.btime, f.folder_id,
  c.captured_at, c.tz_offset_min, c.gps_lat, c.gps_lon,
  g.tool,
  (SELECT value_text FROM file_param d WHERE d.file_id = f.id AND d.source = 'generation' AND d.key = 'date'),
  (SELECT value_text FROM file_param d WHERE d.file_id = f.id AND d.source = 'generation'
     AND d.key = 'generation_time'),
  c.subsec_ms, c.body_serial, c.maker_tz_offset_min, f.duration
FROM file f
LEFT JOIN capture c ON c.file_id = f.id
LEFT JOIN generation g ON g.file_id = f.id
"""

_CONTEXT = """
INSERT INTO derived_media_context(file_id, has_capture, has_generation, origin,
  local_at, instant_at, tz_offset_min, time_basis, time_certainty, time_supports,
  time_conflicts, time_precision, gps_lat, gps_lon, place_id, location_basis,
  location_certainty, policy_version, rebuilt_at)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
"""

_OCCUR_GENERATION = """
INSERT INTO derived_media_occurrence(file_id, kind, local_at, instant_at,
  tz_offset_min, basis, certainty, supports, conflicts, finished_at, estimated_at,
  source_order, time_precision, policy_version)
VALUES(?, 'generation', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _json(items) -> str | None:
    import json

    return json.dumps(list(items)) if items else None


def _seconds(text) -> float | None:
    """A duration as a generator spells it -- SwarmUI writes
    `generation_time` as "64.33 sec" or "2.13 min" (T2IEngine.cs:221)
    -- as seconds, or None when it is not one."""
    import re

    if text is None:
        return None
    match = re.match(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?)?\s*$", str(text), re.IGNORECASE
    )
    if not match:
        return None
    unit = (match.group(2) or "s").lower()
    return float(match.group(1)) * (60.0 if unit.startswith("m") else 1.0)


def act_key(body_serial: str | None, local_at: float, name: str) -> str:
    """One shutter press across its renditions: the body, the capture
    clock to the millisecond, and the camera's own frame name (the stem
    -- `666A0200` for both `666A0200.CR2` and `666A0200.JPG`). Two files
    that agree on all three are one act wherever they were copied; two
    frames of a burst differ in the clock, two bodies in the serial."""
    import hashlib

    stem = name.rsplit(".", 1)[0].lower() if "." in name else name.lower()
    spelled = f"{body_serial or ''}|{local_at:.3f}|{stem}"
    return hashlib.sha256(spelled.encode()).hexdigest()[:16]


def _interpret(conn, now: float, file_id: int | None = None) -> int:
    """The primary interpretation and the occurrences for every file (or
    one): the camera's act as the judge settles it from DateTimeOriginal,
    its subsecond, the zone the camera knew, mtime and btime; the
    generation act as the judge settles it from the generator's day,
    SwarmUI's request minute, the file's mtime and btime, and the
    generation time -- supports and conflicts named beside the value;
    only a claimless file falls to the filesystem, an instant with no
    local story. Returns the number of contexts written."""
    from . import when

    where = " WHERE f.id = ?" if file_id is not None else ""
    rows = conn.execute(_SOURCES + where, (file_id,) if file_id is not None else ()).fetchall()
    made = 0
    for (
        fid,
        name,
        mtime,
        btime,
        _folder,
        captured_at,
        tz,
        lat,
        lon,
        tool,
        date_text,
        gen_time,
        subsec_ms,
        body_serial,
        maker_tz,
        duration,
    ) in rows:
        has_capture, has_generation = int(captured_at is not None), int(tool is not None)
        origin = {(1, 1): "mixed", (1, 0): "captured", (0, 1): "generated"}.get(
            (has_capture, has_generation), "imported"
        )
        generation = when.judge_generation(
            date_text=date_text, name=name, tool=tool, mtime=mtime, btime=btime, generation_time=_seconds(gen_time)
        )
        capture = when.judge_capture(
            captured_at=captured_at,
            subsec_ms=subsec_ms,
            tz_offset_min=tz,
            maker_tz_offset_min=maker_tz,
            mtime=mtime,
            btime=btime,
            duration=duration,
        )
        if capture is not None:
            time = (
                capture.local_at,
                capture.instant_at,
                capture.tz_offset_min,
                "capture",
                capture.certainty,
                _json(capture.supports),
                _json(capture.conflicts),
                capture.precision,
            )
        elif generation is not None:
            time = (
                generation.local_at,
                generation.instant_at,
                generation.tz_offset_min,
                generation.basis,
                generation.certainty,
                _json(generation.supports),
                _json(generation.conflicts),
                generation.precision,
            )
        else:
            fallback = when.judge_filesystem(mtime, btime)
            time = (
                (
                    None,
                    fallback.instant_at,
                    None,
                    fallback.basis,
                    fallback.certainty,
                    _json(fallback.supports),
                    None,
                    fallback.precision,
                )
                if fallback
                else (None, None, None, None, None, None, None, None)
            )
        conn.execute(
            _CONTEXT,
            (
                fid,
                has_capture,
                has_generation,
                origin,
                *time,
                lat,
                lon,
                "gps" if lat is not None else None,
                1.0 if lat is not None else None,
                POLICY_VERSION,
                now,
            ),
        )
        made += 1
        if capture is not None:
            conn.execute(
                _OCCUR_CAPTURE,
                (
                    fid,
                    capture.local_at,
                    capture.instant_at,
                    capture.tz_offset_min,
                    capture.certainty,
                    _json(capture.supports),
                    _json(capture.conflicts),
                    capture.finished_at,
                    act_key(body_serial, capture.local_at or 0.0, name),
                    capture.precision,
                    POLICY_VERSION,
                ),
            )
        if generation is not None:
            conn.execute(
                _OCCUR_GENERATION,
                (
                    fid,
                    generation.local_at,
                    generation.instant_at,
                    generation.tz_offset_min,
                    generation.basis,
                    generation.certainty,
                    _json(generation.supports),
                    _json(generation.conflicts),
                    generation.finished_at,
                    generation.estimated_at,
                    generation.source_order,
                    generation.precision,
                    POLICY_VERSION,
                ),
            )
    return made


def _advance(conn, *, create: bool = True) -> None:
    """Every mutation of the interpretation is a new generation --
    including deletions, so a hypothesis over contexts that no longer
    exist can never prove itself current. Invalidation alone does not
    MINT the identity (create=False): staling a never-interpreted
    library leaves "no interpretation exists" true."""
    if create:
        conn.execute(
            "INSERT INTO derived_context_state(id, generation, policy_version) VALUES(1, 1, ?)"
            " ON CONFLICT(id) DO UPDATE SET generation = generation + 1,"
            " policy_version = excluded.policy_version",
            (POLICY_VERSION,),
        )
    else:
        conn.execute("UPDATE derived_context_state SET generation = generation + 1 WHERE id = 1")


def repopulated(conn) -> None:
    """The PRESENT-FILE POPULATION changed: a new file was minted, or a
    row was marked missing or restored. Events prove themselves over
    interpretation AND population, so either moving obsoletes every
    published hypothesis -- a session that was complete yesterday is
    not current over today's library. Advances the generation without
    minting state on a never-interpreted library."""
    _advance(conn, create=False)


def state(conn) -> tuple[int, int] | None:
    """`(generation, policy_version)` of the current interpretation, or
    None when nothing has ever been interpreted."""
    row = conn.execute("SELECT generation, policy_version FROM derived_context_state WHERE id = 1").fetchone()
    return (row[0], row[1]) if row else None


def coverage(conn) -> tuple[int, int]:
    """(present files holding a current-policy context, present files).
    A stable interpretation is not necessarily a COMPLETE one: a paused
    context job leaves the generation still while most of the library
    is uninterpreted, and a hypothesis proven over that silence would
    stay current indefinitely. Equality of these two numbers is the
    completeness proof the event seam demands."""
    row = conn.execute(
        "SELECT count(mc.file_id), count(*) FROM file f"
        " LEFT JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = ?"
        " WHERE f.missing_since IS NULL",
        (POLICY_VERSION,),
    ).fetchone()
    return (row[0], row[1])


def rebuild(conn, now: float) -> int:
    """The whole projection, replaced -- never merged, because a merge
    is where a stale interpretation survives its sources."""
    conn.execute("DELETE FROM derived_media_context")
    conn.execute("DELETE FROM derived_media_occurrence")
    made = _interpret(conn, now)
    _advance(conn)
    return made


def rebuild_one(conn, file_id: int, now: float) -> None:
    """One file's interpretation, refreshed -- the context job's item
    grain, so cancellation and resume land at file boundaries."""
    conn.execute("DELETE FROM derived_media_context WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM derived_media_occurrence WHERE file_id = ?", (file_id,))
    _interpret(conn, now, file_id)
    _advance(conn)


def stale(conn, file_id: int) -> None:
    """A source claim about this file changed: its interpretation is no
    longer knowably true -- DELETED now, rebuilt by the explicit jobs
    -- and the generation advances, so EVERY event hypothesis stops
    being current, including ones this file was never a member of: a
    changed outsider may belong beside an existing event's members, and
    that event's absence of it is now itself stale."""
    conn.execute("DELETE FROM derived_media_context WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM derived_media_occurrence WHERE file_id = ?", (file_id,))
    conn.execute(
        "DELETE FROM derived_event_run WHERE id IN ("
        " SELECT e.run_id FROM derived_event e"
        " JOIN derived_event_file ef ON ef.event_id = e.id WHERE ef.file_id = ?)",
        (file_id,),
    )
    _advance(conn, create=False)


@dataclasses.dataclass(frozen=True)
class Occurrence:
    """One temporal claim of one KIND about one media item -- the
    grouping input. A grouper consumes the occurrences of its OWN claim,
    so each story is told at that claim's time."""

    file_id: int
    uuid: str  # hex; membership hashes are built over these
    kind: str
    local_at: float | None
    instant_at: float | None
    time_precision: str
    #: the generator's own order inside its claimed bucket, or None
    source_order: int | None = None
    #: fit for chronology -- the judge's answer (db/when.py Verdict.usable),
    #: not the grouper's reinterpretation of its supports
    usable: bool = True
    #: one act across its renditions (capture only); None elsewhere
    act_key: str | None = None
    #: the file's name, for a grouper to rank renditions of one act
    name: str = ""


_OCCURRENCES = """
SELECT o.file_id, e.uuid, o.kind, o.local_at, o.instant_at, o.time_precision, o.source_order, o.conflicts,
       o.act_key, f.name
  FROM derived_media_occurrence o
  JOIN file f ON f.id = o.file_id AND f.missing_since IS NULL
  JOIN entity e ON e.id = o.file_id
 WHERE o.kind = ? AND o.policy_version = ?
 ORDER BY o.file_id
"""


def occurrences(conn, kind: str) -> list[Occurrence]:
    """Every present file's occurrence of one claim, in stable id order
    -- current policy only, so an upgraded ladder blinds the groupers
    exactly as it blinds every other reader."""
    import json

    from . import when

    return [
        Occurrence(
            row[0],
            row[1].hex(),
            kind,
            row[3],
            row[4],
            row[5],
            row[6],
            not any(one.startswith(when.GENERATOR) for one in (json.loads(row[7]) if row[7] else [])),
            row[8],
            row[9],
        )
        for row in conn.execute(_OCCURRENCES, (kind, POLICY_VERSION))
    ]
