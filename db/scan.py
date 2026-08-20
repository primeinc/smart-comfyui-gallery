"""Deciding what a file found on disk refers to.

This is the scanner's identity matcher. It lives here rather than in the test
file because an algorithm defined inside its own tests tests only itself: the
tests pass regardless of what the real scanner does, and whoever writes the
scanner writes a second implementation with nothing binding it to this one.

The obvious implementation is wrong in a way that silently corrupts identity.
Treating a `(folder_id, name)` hit as a match means that when two files
exchange names, both lookups succeed and every rating stays with the *path*
instead of following the bytes -- the exact defect the schema exists to remove,
reintroduced one layer up.

So a path hit is provisional. It is confirmed only when the content also
matches; everything else joins a changed set reconciled by content first. The
candidate pool includes rows still sitting at a path that now holds different
bytes, not only rows that vanished -- without that, a swap looks like two
unrelated edits.
"""

from __future__ import annotations

import enum
from typing import NamedTuple


class Outcome(enum.Enum):
    """What a scanned path turned out to be.

    AMBIGUOUS exists because sha256 proves byte equality, not object
    continuity: with two identical copies a delete/add pair cannot be
    attributed, and guessing moves one file's ratings onto another.
    """

    UNIQUE_MATCH = "unique_match"
    AMBIGUOUS = "ambiguous"
    NEW = "new"


class Resolution(NamedTuple):
    outcome: Outcome
    file_id: int | None


def resolve_scan(conn, observed: dict[tuple[int, str], str | None]):
    """Map each observed ``(folder_id, name) -> content_sha256`` to a decision.

    Returns ``(resolutions, missing)`` where *resolutions* is keyed by the same
    tuples and *missing* lists file ids whose content was found nowhere. A
    missing file is never deleted here; the caller records ``missing_since``,
    because unreachable and deleted are different things.
    """
    rows = {
        (r[1], r[2]): (r[0], r[3])
        for r in conn.execute("SELECT id, folder_id, name, content_sha256 FROM file")
    }
    result: dict[tuple[int, str], Resolution] = {}
    settled: set[int] = set()

    # Pass 1 -- same place, same bytes. Nothing to reconcile, no hashing beyond
    # what the caller already had to do to fill `observed`.
    for key, sha in observed.items():
        row = rows.get(key)
        if row and sha is not None and row[1] == sha:
            result[key] = Resolution(Outcome.UNIQUE_MATCH, row[0])
            settled.add(row[0])

    # Pass 2 -- reconcile by content across everything still unsettled.
    candidates: dict[str, list[int]] = {}
    for file_id, sha in rows.values():
        if file_id not in settled and sha is not None:
            candidates.setdefault(sha, []).append(file_id)

    for key, sha in observed.items():
        if key in result:
            continue
        pool = [f for f in candidates.get(sha, []) if f not in settled] if sha else []
        if len(pool) == 1:
            result[key] = Resolution(Outcome.UNIQUE_MATCH, pool[0])
            settled.add(pool[0])
        elif len(pool) > 1:
            result[key] = Resolution(Outcome.AMBIGUOUS, None)
        else:
            result[key] = Resolution(Outcome.NEW, None)

    missing = [file_id for file_id, _ in rows.values() if file_id not in settled]
    return result, missing
