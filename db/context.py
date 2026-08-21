"""One interpretation of when, where and how each media item happened.

The MediaContext half of the Metadata package. Raw evidence
(blob/file_blob) is what the media actually said and is never
normalized away; source facts (capture, generation, file, file_param)
are per-source CLAIMS -- EXIF's DateTimeOriginal and the filesystem's
mtime are independent observations, never two spellings of one value.
`derived_media_context` is the application's best CURRENT understanding
built from those claims: derived, rebuildable, and always carrying its
BASIS, so no date is ever unexplained and a better ladder tomorrow is a
rebuild, not a migration.

TWO time concepts, held apart on purpose. `local_at` is what the human
clock said -- the wall time a camera claimed, the thing "Saturday in
Hawaii" is made of. `instant_at` is the actual UTC instant, present
ONLY when knowable. A camera claim with an offset yields both at full
certainty; without the offset the wall clock STANDS and the instant
stays honestly absent -- a known human clock is never replaced by a
filesystem time to make a column easier to sort. After the camera
speaks the GENERATOR: an embedded `date` claim the tool wrote into the
file (file_param generation/date) is a wall claim about when the media
HAPPENED, and it outranks every filesystem time -- btime records when
bytes landed on this disk, which for a copied-in batch is the wrong
day entirely. Only media with no capture and no embedded claim fall to
btime, then mtime, instants with no local story. A claim that does not
parse as a date is no claim: the ladder falls through rather than
inventing chronology. Model-derived annotations are inference, not
evidence: nothing they say may enter this ladder.

Invalidation lives HERE, called from the source-fact writer seams
(db/ingest.py after a parse, db/scan.py when a file's times change) --
deliberately not as schema triggers, because a trigger on a source
table that references a derived table breaks the drop-derived-and-
reindex contract the moment the namespace is dropped. A stale context
is DELETED, never silently served; the explicit context job is what
makes it current again. Nothing expensive starts by itself.
"""

from __future__ import annotations

import dataclasses

ORIGINS = ("captured", "generated", "imported", "unknown")
TIME_BASES = ("capture", "embedded", "btime", "mtime", "first_seen")
LOCATION_BASES = ("gps", "sidecar", "inferred", "authored")

#: The whole ladder, one statement, applied to whichever files the
#: caller names. Basis and certainty recorded beside every value they
#: explain; place_id stays NULL until an explicit resolver or an
#: authored assertion mints real place identity -- GPS alone never does.
_INTERPRET = """
INSERT INTO derived_media_context(file_id, origin, local_at, instant_at, tz_offset_min,
  time_basis, time_certainty, gps_lat, gps_lon, place_id, location_basis,
  location_certainty, rebuilt_at)
SELECT f.id,
  CASE WHEN g.file_id IS NOT NULL THEN 'generated'
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
  c.gps_lat, c.gps_lon,
  NULL,
  CASE WHEN c.gps_lat IS NOT NULL THEN 'gps' END,
  CASE WHEN c.gps_lat IS NOT NULL THEN 1.0 END,
  ?
FROM file f
LEFT JOIN capture c ON c.file_id = f.id
LEFT JOIN generation g ON g.file_id = f.id
LEFT JOIN file_param d ON d.file_id = f.id AND d.source = 'generation' AND d.key = 'date'
"""


def rebuild(conn, now: float) -> int:
    """The whole projection, replaced -- never merged, because a merge
    is where a stale interpretation survives its sources."""
    conn.execute("DELETE FROM derived_media_context")
    return conn.execute(_INTERPRET, (now,)).rowcount


def rebuild_one(conn, file_id: int, now: float) -> None:
    """One file's interpretation, refreshed -- the context job's item
    grain, so cancellation and resume land at file boundaries."""
    conn.execute("DELETE FROM derived_media_context WHERE file_id = ?", (file_id,))
    conn.execute(_INTERPRET + " WHERE f.id = ?", (now, file_id))


def stale(conn, file_id: int) -> None:
    """A source claim about this file changed: its interpretation, and
    every event hypothesis built over it, is no longer knowably true --
    DELETED now, rebuilt by the explicit jobs. Stale derived truth never
    masquerades as current truth, and nothing expensive runs here."""
    conn.execute("DELETE FROM derived_media_context WHERE file_id = ?", (file_id,))
    conn.execute(
        "DELETE FROM derived_event_run WHERE id IN ("
        " SELECT e.run_id FROM derived_event e"
        " JOIN derived_event_file ef ON ef.event_id = e.id WHERE ef.file_id = ?)",
        (file_id,),
    )


@dataclasses.dataclass(frozen=True)
class MediaContext:
    """What a grouper is allowed to know about one media item: the ONE
    interpretation, plus the generation identities that make session
    phases decidable later. Groupers consume this interface, never the
    source tables -- one definition of time and origin for every
    grouping algorithm."""

    file_id: int
    uuid: str  # hex; membership hashes are built over these
    origin: str
    local_at: float | None
    instant_at: float | None
    tz_offset_min: int | None
    prompt_id: int | None
    workflow_id: int | None

    @property
    def moment(self) -> float | None:
        """One axis for ordering: the instant when knowable, the wall
        clock otherwise -- the same coalesce the timeline reads, named
        once."""
        return self.instant_at if self.instant_at is not None else self.local_at


_GROUPING = """
SELECT mc.file_id, e.uuid, mc.origin, mc.local_at, mc.instant_at, mc.tz_offset_min,
       g.prompt_id, g.workflow_id
  FROM derived_media_context mc
  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL
  JOIN entity e ON e.id = mc.file_id
  LEFT JOIN generation g ON g.file_id = mc.file_id
 ORDER BY COALESCE(mc.instant_at, mc.local_at), mc.file_id
"""


def contexts(conn) -> list[MediaContext]:
    """Every present file's context, chronological -- the grouping
    input, read once per regroup."""
    return [
        MediaContext(row[0], row[1].hex(), row[2], row[3], row[4], row[5], row[6], row[7])
        for row in conn.execute(_GROUPING)
    ]
