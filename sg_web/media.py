"""Serving bytes: range arithmetic, disk streaming, and the avatar's face.

Litestar's File response streams whole files but implements no Range
handling -- the only 206 in the package is the constant in
litestar/status_codes.py (litestar-org/litestar@64cd7da) -- and a video
element that cannot seek is a slideshow. So the range grammar lives here,
per RFC 9110 section 14: one bytes-range, first-byte/last-byte inclusive,
suffix form for "the last N bytes". Syntactically malformed or multipart
ranges are ignored (the whole body is the permitted response); a
syntactically valid range that selects nothing is 416, said with the
`bytes */size` form so the client learns the size it misjudged.
"""

from __future__ import annotations

#: Bytes per chunk when streaming from disk.
CHUNK = 64 * 1024


class Unsatisfiable(Exception):
    """A well-formed range that selects no bytes of this file."""


def _digits(text: str) -> bool:
    """RFC 9110's DIGIT: %x30-39 and nothing else. `str.isdigit()` admits
    Unicode numerics `int()` rejects ('²' -- a single latin-1 octet
    on the wire), and `str.isdecimal()` admits ones `int()` accepts
    (the Arabic-Indic digits, U+0660..U+0669), so a header field defined
    as ASCII must be checked as ASCII."""
    return text.isascii() and text.isdigit()


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """`(first, last)` inclusive for one satisfiable bytes-range, or None
    to serve the whole file. Raises Unsatisfiable for the 416 case, and
    nothing else, ever: a Range header is attacker-supplied text on an
    unauthenticated route, and anything unparseable is "no opinion", not
    an error."""
    if not header or size == 0:
        return None
    unit, _, spec = header.partition("=")
    if unit.strip() != "bytes" or "," in spec:
        return None
    start_text, dash, end_text = spec.strip().partition("-")
    if not dash:
        return None
    if not start_text:
        # suffix form: the last N bytes
        if not _digits(end_text):
            return None
        last_n = int(end_text)
        if last_n == 0:
            raise Unsatisfiable
        return max(0, size - last_n), size - 1
    if not _digits(start_text) or (end_text and not _digits(end_text)):
        return None
    first = int(start_text)
    if first >= size:
        raise Unsatisfiable
    last = min(int(end_text), size - 1) if end_text else size - 1
    if last < first:
        return None
    return first, last


def chunks(path: str, first: int, length: int):
    """Yield `length` bytes of `path` from offset `first`, chunked."""
    remaining = length
    with open(path, "rb") as handle:
        handle.seek(first)
        while remaining > 0:
            piece = handle.read(min(CHUNK, remaining))
            if not piece:
                return
            remaining -= len(piece)
            yield piece


def exemplar_face(conn, person_id: int):
    """The face that stands for a person: their highest-confidence
    detection in the primary clustering run.

    `(face_id, file_id, sample_id, x, y, w, h)` or None when no primary
    run holds a cluster attributed to them -- an unclustered person has no
    face to show, and that is a valid result, not an error.
    """
    return conn.execute(
        "SELECT fi.id, fi.file_id, fi.sample_id, r.x, r.y, r.w, r.h"
        " FROM derived_face_cluster c"
        " JOIN derived_face_membership m ON m.cluster_id = c.id"
        " JOIN derived_face_instance fi ON fi.id = m.face_id"
        " JOIN region r ON r.id = fi.region_id"
        " JOIN derived_face_run run ON run.id = c.run_id AND run.is_primary = 1"
        " WHERE c.person_id = ?"
        " ORDER BY fi.det_score DESC, fi.id LIMIT 1",
        (person_id,),
    ).fetchone()
