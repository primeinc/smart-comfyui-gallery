"""Find the smallest retained extent that still reproduces the baseline.

An ablation asks "is this needed", which is the right question for a primitive
that is either present or absent. A patch of pixels is not that: it has four
edges and a size, and asking "is the margin needed" answers yes or no about
whichever constant somebody typed. That is folklore with a threshold attached
-- it passes while the constant is too generous, and says nothing about where
the real edge is.

So this searches instead. Each side is moved inward independently by binary
search until the replay stops reproducing, which gives the furthest that side
can travel on its own. The four results are then applied TOGETHER and probed,
because sides interact: a warp reading a diagonal band can tolerate losing the
left column or the top row but not both. Where the combination fails, it is
walked back one side at a time rather than reported as if it held.

The search is over integer pixel offsets and every probe is the real replay
against the real baseline, so the number that comes out is a measurement of
the implementation, not of a model of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: Left, top, right, bottom, as signed inward moves on (x0, y0, x1, y1).
#: Named rather than indexed because an off-by-one in a side index produces a
#: plausible smaller rectangle and a silently wrong measurement.
SIDES: tuple[str, ...] = ("left", "top", "right", "bottom")


@dataclass(frozen=True)
class Rect:
    """A half-open source rectangle, in whole source pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)

    def inset(self, side: str, amount: int) -> Rect:
        """This rectangle with one side moved inward by `amount` pixels."""
        if side == "left":
            return Rect(self.x0 + amount, self.y0, self.x1, self.y1)
        if side == "top":
            return Rect(self.x0, self.y0 + amount, self.x1, self.y1)
        if side == "right":
            return Rect(self.x0, self.y0, self.x1 - amount, self.y1)
        if side == "bottom":
            return Rect(self.x0, self.y0, self.x1, self.y1 - amount)
        raise ValueError(f"{side!r} is not one of {SIDES}")

    def insets(self, amounts: dict[str, int]) -> Rect:
        out = self
        for side in SIDES:
            out = out.inset(side, amounts.get(side, 0))
        return out

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


#: Answers one question: does a replay restricted to this rectangle still
#: reproduce the baseline? Anything that raises counts as "no" at the call
#: site, never here -- a probe that swallows its own exception cannot be told
#: apart from one that answered.
Probe = Callable[[Rect], bool]


@dataclass(frozen=True)
class SideResult:
    """How far one side travelled before the replay stopped reproducing."""

    side: str
    max_inset: int
    probes: int
    ceiling_hit: bool
    """True when the search ran out of room rather than out of margin: the
    side reached the limit and still reproduced, so the real maximum is at
    least this and the number is a lower bound, not the edge."""


@dataclass(frozen=True)
class Minimum:
    """The measured smallest extent, and everything needed to re-derive it."""

    start: Rect
    per_side: tuple[SideResult, ...]
    combined: Rect
    combined_holds: bool
    walked_back: tuple[str, ...]
    probes: int

    @property
    def saved_pixels(self) -> int:
        return self.start.area - self.combined.area

    @property
    def saved_fraction(self) -> float:
        return (self.saved_pixels / self.start.area) if self.start.area else 0.0

    def margin_on(self, side: str) -> int:
        for one in self.per_side:
            if one.side == side:
                return one.max_inset
        raise KeyError(side)


def _safe(probe: Probe, rect: Rect) -> bool:
    """A probe result, with a degenerate rectangle answering False.

    A rectangle that has collapsed cannot reproduce anything, and letting the
    replay raise on it would make "too small" indistinguishable from "the
    runner is broken".
    """
    if rect.width <= 0 or rect.height <= 0:
        return False
    return probe(rect)


def largest_inset(probe: Probe, start: Rect, side: str, limit: int) -> SideResult:
    """The furthest one side can move inward and still reproduce.

    Binary search on the inset, not a linear walk: the property is monotone in
    the amount removed -- pixels the warp reads do not come back once they are
    gone -- so the first failure bounds every larger inset too.

    Monotonicity is the assumption this rests on, and it is checked rather
    than trusted: `limit` is probed first, and a `limit` that reproduces makes
    the answer a lower bound (`ceiling_hit`) instead of an edge.
    """
    probes = 0

    if limit <= 0:
        return SideResult(side=side, max_inset=0, probes=0, ceiling_hit=False)

    probes += 1
    if _safe(probe, start.inset(side, limit)):
        return SideResult(side=side, max_inset=limit, probes=probes, ceiling_hit=True)

    low, high = 0, limit  # low reproduces, high does not
    while high - low > 1:
        middle = (low + high) // 2
        probes += 1
        if _safe(probe, start.inset(side, middle)):
            low = middle
        else:
            high = middle
    return SideResult(side=side, max_inset=low, probes=probes, ceiling_hit=False)


def minimum_extent(probe: Probe, start: Rect, *, limit: int | None = None) -> Minimum:
    """The smallest rectangle measured to still reproduce the baseline.

    Four independent searches, then the combination, then a walk-back. The
    combination is probed rather than assumed because the per-side maxima were
    each measured with the other three sides intact; applying all four at once
    removes strictly more than any one search saw.

    `start` must itself reproduce. If it does not, the runner or the derived
    footprint is wrong and there is nothing here to measure -- that is a
    failed case, not a minimum of zero.
    """
    probes = 1
    if not _safe(probe, start):
        raise ValueError(
            f"the starting extent {start.as_tuple()} does not reproduce the baseline, so there is no minimum to "
            f"measure: fix the footprint or the runner before searching inside it"
        )

    room = limit if limit is not None else max(start.width, start.height)
    per_side: list[SideResult] = []
    for side in SIDES:
        span = start.width if side in {"left", "right"} else start.height
        found = largest_inset(probe, start, side, min(room, max(span - 1, 0)))
        per_side.append(found)
        probes += found.probes

    amounts: dict[str, int] = {one.side: one.max_inset for one in per_side}
    combined = start.insets(amounts)
    probes += 1
    holds = _safe(probe, combined)

    # Sides interact. Where the combination fails, give pixels back one side
    # at a time -- largest claimed inset first, since that side took the most
    # and is likeliest to be the one over-reaching -- rather than reporting a
    # rectangle that was never observed to work.
    walked_back: list[str] = []
    if not holds:
        order = sorted(SIDES, key=lambda one: amounts[one], reverse=True)
        for side in order:
            while amounts[side] > 0:
                amounts[side] -= 1
                probes += 1
                if _safe(probe, start.insets(amounts)):
                    break
            if side not in walked_back and amounts[side] != next(one.max_inset for one in per_side if one.side == side):
                walked_back.append(side)
            combined = start.insets(amounts)
            if _safe(probe, combined):
                probes += 1
                holds = True
                break
            probes += 1

    return Minimum(
        start=start,
        per_side=tuple(per_side),
        combined=combined,
        combined_holds=holds,
        walked_back=tuple(walked_back),
        probes=probes,
    )
