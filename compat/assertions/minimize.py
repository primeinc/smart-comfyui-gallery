from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SIDES: tuple[str, ...] = ("left", "top", "right", "bottom")


@dataclass(frozen=True)
class Rect:
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


Probe = Callable[[Rect], bool]


@dataclass(frozen=True)
class SideResult:
    side: str
    max_inset: int
    probes: int
    ceiling_hit: bool


@dataclass(frozen=True)
class Minimum:
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
    if rect.width <= 0 or rect.height <= 0:
        return False
    return probe(rect)


def largest_inset(probe: Probe, start: Rect, side: str, limit: int) -> SideResult:
    probes = 0

    if limit <= 0:
        return SideResult(side=side, max_inset=0, probes=0, ceiling_hit=False)

    probes += 1
    if _safe(probe, start.inset(side, limit)):
        return SideResult(side=side, max_inset=limit, probes=probes, ceiling_hit=True)

    low, high = 0, limit
    while high - low > 1:
        middle = (low + high) // 2
        probes += 1
        if _safe(probe, start.inset(side, middle)):
            low = middle
        else:
            high = middle
    return SideResult(side=side, max_inset=low, probes=probes, ceiling_hit=False)


def minimum_extent(probe: Probe, start: Rect, *, limit: int | None = None) -> Minimum:
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
