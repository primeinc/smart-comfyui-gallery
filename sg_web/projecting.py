"""Time to x, with empty time collapsed.

A timeline that spends its pixels in proportion to ELAPSED time spends
them on nothing. A library with a picture in July and a week's work in
August draws one bar and three weeks of blank; a library with a scanned
photograph from 2002 draws twenty-four years of blank to reach this
month. Both are the same defect at different zooms, and neither is
communicated by the blank: empty pixels are ambiguous between "no
pictures", "no data" and "the render broke". A band that says
`3 weeks · nothing` says it exactly, and costs a fortieth of the space.

So this module is the axis itself, as an object rather than as
`((t - lo) / span) * W` written out at eleven call sites. Every mark on
the surface -- bars, ticks, session cards, faces, the scrubber handle,
the year labels -- goes through one projection, so collapsing is a
property of the axis and every mark follows it without being told.

**The client shares it.** The browser turns a click, a drag and a pan
back into a time (frontend/src/timeline.ts), so a piecewise axis the
server keeps to itself would put every gesture in the wrong year. The
segments ride in the payload and the browser inverts through the same
piecewise function.

**A gap qualifies by SHARE, not by duration.** Anything else needs a
different threshold per zoom, and the complaint this module answers is
that the defect is the same at two weeks and at twenty years. A run of
empty time is collapsed when drawing it honestly would cost more than
`SHARE` of the axis -- which is scale-free, so one rule holds at every
width.

**Inside kept time the axis is still linear.** A gap between two days of
one week is real information at that zoom; what is not information is
three weeks of identical blank. Collapsing only removes the runs that
were paying no rent.
"""

from __future__ import annotations

import dataclasses

#: The axis, in the same arbitrary units the surface already draws in
#: (sg_web/timeline_view.py `_W`). Stated here because a projection is
#: only meaningful against a width.
WIDTH = 1000.0

#: A run of empty time is collapsed when drawing it to scale would cost
#: more than this share of the axis. Scale-free on purpose: the same
#: number governs a fortnight and a quarter-century.
SHARE = 0.04

#: The least of the axis the pictures keep, however many runs were
#: worth collapsing. It is what bounds the NUMBER of bands: past this
#: the marks would want more room than the axis has and would run off
#: the end of it.
LEAST_DRAWN = 0.5

#: What a collapsed run is drawn as instead. Wide enough to carry a
#: label and to be clicked, narrow enough that a dozen of them still
#: leave the pictures the axis.
COLLAPSED = WIDTH * 0.022


@dataclasses.dataclass(frozen=True)
class Segment:
    """One run of the axis: `[t0, t1)` drawn from `x0` to `x1`.

    `skipped` marks a run that holds nothing and was collapsed, which is
    the only thing a reader of this needs to tell apart -- it is what
    gets a band and a label rather than bars.
    """

    t0: float
    t1: float
    x0: float
    x1: float
    skipped: bool = False

    @property
    def seconds(self) -> float:
        return max(0.0, self.t1 - self.t0)


def merged(spans, lo: float, hi: float) -> list[tuple[float, float]]:
    """The occupied time in `[lo, hi)`, clipped and coalesced.

    Adjacent and overlapping runs become one: two bins that touch are
    not a gap, and a session lying across several bins must not be cut
    into pieces by the bins it happens to cross.
    """
    held = sorted(
        (max(lo, float(a)), min(hi, float(b)))
        for a, b in spans
        if float(b) > lo and float(a) < hi and float(b) > float(a)
    )
    if not held:
        return []
    out = [held[0]]
    for start, end in held[1:]:
        last_start, last_end = out[-1]
        if start <= last_end:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


class Projection:
    """A piecewise-linear map from time to x, and back.

    Linear when nothing was collapsed, which is the ordinary case and
    has to stay byte-identical to the arithmetic it replaced -- a dense
    window must not move by a pixel because this module now exists.
    """

    __slots__ = ("hi", "lo", "segments", "width")

    def __init__(self, segments: tuple[Segment, ...], lo: float, hi: float, width: float):
        self.segments = segments
        self.lo = lo
        self.hi = hi
        self.width = width

    @property
    def collapsed(self) -> tuple[Segment, ...]:
        return tuple(one for one in self.segments if one.skipped)

    def x(self, t: float) -> float:
        """Where a moment lands. Clamped at both ends, because a session
        that starts before the window still has to draw its left edge
        somewhere and that somewhere is the edge."""
        if t <= self.lo:
            return 0.0
        if t >= self.hi:
            return self.width
        for one in self.segments:
            if t < one.t1:
                if one.seconds <= 0:
                    return one.x0
                return one.x0 + ((t - one.t0) / one.seconds) * (one.x1 - one.x0)
        return self.width

    def t(self, x: float) -> float:
        """The moment at an x -- what a click means. Inside a collapsed
        run this is linear over the run's real time, so the answer is
        honest about a place the axis is not drawing to scale."""
        if x <= 0:
            return self.lo
        if x >= self.width:
            return self.hi
        for one in self.segments:
            if x < one.x1:
                drawn = one.x1 - one.x0
                if drawn <= 0:
                    return one.t0
                return one.t0 + ((x - one.x0) / drawn) * one.seconds
        return self.hi

    def told(self) -> list[dict]:
        """The segments as the browser reads them."""
        return [
            {"t0": one.t0, "t1": one.t1, "x0": round(one.x0, 3), "x1": round(one.x1, 3), "skipped": one.skipped}
            for one in self.segments
        ]


def linear(lo: float, hi: float, width: float = WIDTH) -> Projection:
    """The axis that collapses nothing -- what every surface drew before
    this module, and what a window with no worthwhile gap still draws."""
    span = max(1.0, hi - lo)
    return Projection((Segment(lo, lo + span, 0.0, width),), lo, lo + span, width)


def projected(lo: float, hi: float, occupied, *, width: float = WIDTH, share: float = SHARE) -> Projection:
    """The axis for a window, with worthless runs of empty time taken out.

    `occupied` is every `(start, end)` that holds something -- the bins
    that came back non-empty, and the spans too coarse to have landed in
    one. It is merged here rather than by the caller, because whether
    two bins touch is a fact about the axis and not about the query.
    """
    span = max(1.0, hi - lo)
    hi = lo + span
    held = merged(occupied, lo, hi)
    if not held:
        return linear(lo, hi, width)

    # The gaps are the complement, including before the first run and
    # after the last: a leading quarter-century of nothing is the exact
    # case the scrubber suffers from, and it is a gap like any other.
    gaps: list[tuple[float, float]] = []
    edge = lo
    for start, end in held:
        if start > edge:
            gaps.append((edge, start))
        edge = end
    if edge < hi:
        gaps.append((edge, hi))

    worth = [one for one in gaps if (one[1] - one[0]) / span > share]
    # Bounded, and the bound is load-bearing rather than tidy: a window
    # riddled with gaps would otherwise want more band than there is
    # axis -- fifty of them is 1100 units of a 1000-unit width -- and the
    # marks would run off the end. The LONGEST ones are kept, because
    # they are the ones whose collapse buys the most; the rest stay drawn
    # to scale, which is also the honest answer for a short gap.
    most = int((width * (1.0 - LEAST_DRAWN)) // COLLAPSED)
    if len(worth) > most:
        worth = sorted(sorted(worth, key=lambda one: one[0] - one[1])[:most])
    if not worth:
        return linear(lo, hi, width)

    # What is left after every collapsed run has taken its fixed band,
    # shared out over the time that is still drawn to scale. Never below
    # LEAST_DRAWN of the axis: collapsing must not make the pictures
    # harder to see than the nothing did.
    taken = COLLAPSED * len(worth)
    drawn_seconds = span - sum(one[1] - one[0] for one in worth)
    usable = max(width * LEAST_DRAWN, width - taken)
    if drawn_seconds <= 0:
        return linear(lo, hi, width)

    skipped = set(worth)
    runs: list[tuple[float, float, bool]] = []
    edge = lo
    for start, end in held:
        if start > edge:
            runs.append((edge, start, (edge, start) in skipped))
        runs.append((start, end, False))
        edge = end
    if edge < hi:
        runs.append((edge, hi, (edge, hi) in skipped))

    segments: list[Segment] = []
    at = 0.0
    for t0, t1, is_gap in runs:
        step = COLLAPSED if is_gap else ((t1 - t0) / drawn_seconds) * usable
        segments.append(Segment(t0, t1, at, at + step, is_gap))
        at += step
    # The last segment ends ON the edge whatever the rounding did, so
    # `x(hi)` and the axis width are the same number.
    if segments:
        last = segments[-1]
        segments[-1] = Segment(last.t0, last.t1, last.x0, width, last.skipped)
    return Projection(tuple(segments), lo, hi, width)
