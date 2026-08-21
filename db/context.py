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
After the camera speaks the GENERATOR: an embedded `date` claim
(file_param generation/date) is a wall claim about when the media
HAPPENED and outranks every filesystem time -- btime records when
bytes landed on this disk. Only claimless media fall to btime then
mtime, instants with no local story. A claim that does not parse is no
claim. Model-derived annotations are inference, not evidence: nothing
they say may enter this ladder. There is deliberately NO fused
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
TIME_BASES = ("capture", "embedded", "btime", "mtime", "first_seen")
LOCATION_BASES = ("gps", "sidecar", "inferred", "authored")

#: WHICH MEANING of the ladder is current. Bump when the interpretation
#: itself changes meaning -- v2 added the embedded generator-date rung,
#: v3 the precision dimension and the coexistence facts.
POLICY_VERSION = 3

#: The human timeline's one axis, defined ONCE: the wall clock when one
#: was claimed, the knowable instant otherwise. The day facet and the
#: timeline shelves compose around this same fragment, so the door and
#: the shelf cannot drift apart.
HUMAN_MOMENT = "COALESCE(mc.local_at, mc.instant_at)"

#: The whole ladder, one statement, applied to whichever files the
#: caller names. Basis and certainty recorded beside every value they
#: explain; place_id stays NULL until an explicit resolver or an
#: authored assertion mints real place identity -- GPS alone never does.
_INTERPRET = """
INSERT INTO derived_media_context(file_id, has_capture, has_generation, origin,
  local_at, instant_at, tz_offset_min, time_basis, time_certainty, time_precision,
  gps_lat, gps_lon, place_id, location_basis, location_certainty,
  policy_version, rebuilt_at)
SELECT f.id,
  CASE WHEN c.file_id IS NOT NULL THEN 1 ELSE 0 END,
  CASE WHEN g.file_id IS NOT NULL THEN 1 ELSE 0 END,
  CASE WHEN g.file_id IS NOT NULL AND c.file_id IS NOT NULL THEN 'mixed'
       WHEN g.file_id IS NOT NULL THEN 'generated'
       WHEN c.file_id IS NOT NULL THEN 'captured'
       ELSE 'imported' END,
  CASE WHEN c.captured_at IS NOT NULL THEN c.captured_at
       WHEN strftime('%s', d.value_text) IS NOT NULL
         THEN CAST(strftime('%s', d.value_text) AS REAL) END,
  CASE WHEN c.captured_at IS NOT NULL AND c.tz_offset_min IS NOT NULL
         THEN c.captured_at - c.tz_offset_min * 60
       WHEN c.captured_at IS NOT NULL THEN NULL
       WHEN strftime('%s', d.value_text) IS NOT NULL THEN NULL
       WHEN f.btime IS NOT NULL THEN f.btime
       ELSE f.mtime END,
  CASE WHEN c.captured_at IS NOT NULL THEN c.tz_offset_min END,
  CASE WHEN c.captured_at IS NOT NULL THEN 'capture'
       WHEN strftime('%s', d.value_text) IS NOT NULL THEN 'embedded'
       WHEN f.btime IS NOT NULL THEN 'btime'
       ELSE 'mtime' END,
  CASE WHEN c.captured_at IS NOT NULL AND c.tz_offset_min IS NOT NULL THEN 1.0
       WHEN c.captured_at IS NOT NULL THEN 0.8
       WHEN strftime('%s', d.value_text) IS NOT NULL THEN 0.6
       WHEN f.btime IS NOT NULL THEN 0.5
       ELSE 0.3 END,
  CASE WHEN c.captured_at IS NOT NULL THEN 'second'
       WHEN strftime('%s', d.value_text) IS NOT NULL THEN
         CASE WHEN length(trim(d.value_text)) <= 10 THEN 'day' ELSE 'second' END
       ELSE 'subsecond' END,
  c.gps_lat, c.gps_lon,
  NULL,
  CASE WHEN c.gps_lat IS NOT NULL THEN 'gps' END,
  CASE WHEN c.gps_lat IS NOT NULL THEN 1.0 END,
  ?, ?
FROM file f
LEFT JOIN capture c ON c.file_id = f.id
LEFT JOIN generation g ON g.file_id = f.id
LEFT JOIN file_param d ON d.file_id = f.id AND d.source = 'generation' AND d.key = 'date'
"""


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


def state(conn) -> tuple[int, int] | None:
    """`(generation, policy_version)` of the current interpretation, or
    None when nothing has ever been interpreted."""
    row = conn.execute("SELECT generation, policy_version FROM derived_context_state WHERE id = 1").fetchone()
    return (row[0], row[1]) if row else None


def rebuild(conn, now: float) -> int:
    """The whole projection, replaced -- never merged, because a merge
    is where a stale interpretation survives its sources."""
    conn.execute("DELETE FROM derived_media_context")
    made = conn.execute(_INTERPRET, (POLICY_VERSION, now)).rowcount
    _advance(conn)
    return made


def rebuild_one(conn, file_id: int, now: float) -> None:
    """One file's interpretation, refreshed -- the context job's item
    grain, so cancellation and resume land at file boundaries."""
    conn.execute("DELETE FROM derived_media_context WHERE file_id = ?", (file_id,))
    conn.execute(_INTERPRET + " WHERE f.id = ?", (POLICY_VERSION, now, file_id))
    _advance(conn)


def stale(conn, file_id: int) -> None:
    """A source claim about this file changed: its interpretation is no
    longer knowably true -- DELETED now, rebuilt by the explicit jobs
    -- and the generation advances, so EVERY event hypothesis stops
    being current, including ones this file was never a member of: a
    changed outsider may belong beside an existing event's members, and
    that event's absence of it is now itself stale."""
    conn.execute("DELETE FROM derived_media_context WHERE file_id = ?", (file_id,))
    conn.execute(
        "DELETE FROM derived_event_run WHERE id IN ("
        " SELECT e.run_id FROM derived_event e"
        " JOIN derived_event_file ef ON ef.event_id = e.id WHERE ef.file_id = ?)",
        (file_id,),
    )
    _advance(conn, create=False)


@dataclasses.dataclass(frozen=True)
class MediaContext:
    """What a grouper is allowed to know about one media item: the ONE
    interpretation, plus the generation identities that make session
    phases decidable later. Groupers consume this interface, never the
    source tables. There is NO fused moment here: a consumer names the
    domain it works in, and unlike domains are never compared."""

    file_id: int
    uuid: str  # hex; membership hashes are built over these
    origin: str
    has_capture: bool
    has_generation: bool
    local_at: float | None
    instant_at: float | None
    tz_offset_min: int | None
    time_precision: str | None
    prompt_id: int | None
    workflow_id: int | None


_GROUPING = """
SELECT mc.file_id, e.uuid, mc.origin, mc.has_capture, mc.has_generation,
       mc.local_at, mc.instant_at, mc.tz_offset_min, mc.time_precision,
       g.prompt_id, g.workflow_id
  FROM derived_media_context mc
  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL
  JOIN entity e ON e.id = mc.file_id
  LEFT JOIN generation g ON g.file_id = mc.file_id
 ORDER BY mc.file_id
"""


def contexts(conn) -> list[MediaContext]:
    """Every present file's context, in stable id order -- the grouping
    input, read once per regroup. Chronology is the consumer's to
    establish, in the domain it chose."""
    return [
        MediaContext(
            row[0], row[1].hex(), row[2], bool(row[3]), bool(row[4]), row[5], row[6], row[7], row[8], row[9], row[10]
        )
        for row in conn.execute(_GROUPING)
    ]
