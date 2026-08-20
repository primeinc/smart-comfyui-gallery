"""Choosing the moments of a video that anything else will look at.

A face on a video is a face on a frame. `derived_media_sample` exists so a
claim can say which moment it was looking at -- without one, "Ilse is in this
video" cannot be checked, cropped or corrected, and re-running the detector
cannot tell it has already done this part. The table had a producer and no
caller: nothing had ever chosen a moment.

Choosing is separate from decoding on purpose. A sample row says which
moment matters and how it was picked; `vision.decode.frames_at` fetches the
pixels when a job wants them. Keeping them apart is what lets the sampling
be re-run, resumed and reasoned about without touching a decoder.

`policy` is the token that makes a re-run recognisable. Two runs at the same
cadence produce the same rows and `add_sample` returns the existing ones; a
different cadence is a different set, side by side, because a job that
sampled every two seconds and a job that sampled every ten did not look at
the same video.

Pages are the same shape one medium over: `probe.pages_of` writes one
'page' sample per PDF page, so a caption or a piece of OCR on a
document has a moment to point at exactly as a face on a video does.
"""

from __future__ import annotations

from . import derived, probe

#: Two seconds. Close enough that a person entering and leaving a shot is
#: caught, far enough apart that an hour of video is 1,800 samples and not
#: 108,000 -- the difference between a job somebody waits for and a job
#: nobody finishes.
EVERY_MS = 2_000

#: A sampler that reads the whole file would make the cost of deciding the
#: same as the cost of doing. This many is where a long video stops being
#: sampled evenly and starts being sampled across its length.
MOST = 5_000


def cadence(every_ms: int = EVERY_MS) -> str:
    """The policy token for a fixed interval.

    A token, not a sentence: the schema refuses spaces and capitals for it,
    because the same policy spelled three ways cannot be grouped and a re-run
    spelled differently cannot tell it already did this.
    """
    if every_ms % 1000 == 0:
        return f"every-{every_ms // 1000}s"
    return f"every-{every_ms}ms"


def moments(duration: float | None, *, every_ms: int = EVERY_MS, most: int = MOST):
    """The offsets to sample, in milliseconds.

    Widened rather than truncated when a video is long enough that the
    cadence would produce more than `most`. Truncating would sample the first
    hour of a three-hour film and call the rest unexamined; widening samples
    all of it, less finely, and says so in the policy token it is stored
    under.
    """
    if not duration or duration <= 0:
        return [], every_ms
    span = int(duration * 1000)
    if span // every_ms + 1 > most:
        every_ms = max(1, -(-span // max(1, most - 1)))
    return list(range(0, span, every_ms)), every_ms


def frames(conn, file_id: int, path, *, every_ms: int = EVERY_MS, most: int = MOST):
    """Pick the moments of one video. Returns the sample ids, in order.

    Idempotent: asking twice for the same cadence returns the same rows, so
    an interrupted sampling job resumes instead of raising on the first frame
    it had already taken.
    """
    found = probe.read(path)
    if found.duration is None:
        return []
    offsets, spacing = moments(found.duration, every_ms=every_ms, most=most)
    policy = cadence(spacing)
    return [derived.add_sample(conn, file_id, "frame", policy, offset_ms=offset) for offset in offsets]


def taken(conn, file_id: int, policy: str | None = None):
    """The moments already chosen for this file, oldest first."""
    if policy is None:
        return conn.execute(
            "SELECT id, offset_ms, policy FROM derived_media_sample"
            " WHERE file_id = ? AND kind = 'frame' ORDER BY offset_ms",
            (file_id,),
        ).fetchall()
    return conn.execute(
        "SELECT id, offset_ms, policy FROM derived_media_sample"
        " WHERE file_id = ? AND kind = 'frame' AND policy = ? ORDER BY offset_ms",
        (file_id, policy),
    ).fetchall()
