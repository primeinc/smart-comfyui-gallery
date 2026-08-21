"""How a value is spelled for a human -- decided once, here, so five
templates never each choose between "July 18", "18 July" and "07/18".

Pure functions over frozen values. A wall-clock epoch (the snapshot's
`local_at` spelling of what a clock on the wall read) is rendered as
that wall clock, never shifted: the seconds are already the human's
own day. Locale-ready by construction: every spelling passes through
one place.
"""

from __future__ import annotations

import datetime

#: The range dash, U+2013: a typographic fact decided once.
EN_DASH = "\u2013"

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def count(n: int, singular: str, plural: str | None = None) -> str:
    """`1 image`, `55 images`."""
    word = singular if n == 1 else (plural or singular + "s")
    return f"{n} {word}"


def day_label(epoch: float | None, *, utc: bool = False) -> str | None:
    """`July 18, 2026` from an epoch that spells a wall clock -- the
    human's own calendar day; `July 18, 2026 UTC` from a true instant
    whose local day nobody knows. None when nothing claimed a day."""
    if epoch is None:
        return None
    moment = datetime.datetime.fromtimestamp(epoch, datetime.UTC)
    told = f"{_MONTHS[moment.month - 1]} {moment.day}, {moment.year}"
    return f"{told} UTC" if utc else told


def day_range(start: float | None, end: float | None, *, utc: bool = False) -> str | None:
    """The days an interval touches, as a human reads them: `July 18,
    2026` when both ends fall on one day; `July 18` EN DASH `19, 2026`
    across days of one month; `July 31` EN DASH `August 1, 2026` across
    months; both full dates across years. A session that crosses
    midnight is not "generated on" either day."""
    if start is None:
        return None
    if end is None:
        return day_label(start, utc=utc)
    first = datetime.datetime.fromtimestamp(start, datetime.UTC)
    last = datetime.datetime.fromtimestamp(end, datetime.UTC)
    suffix = " UTC" if utc else ""
    if (first.year, first.month, first.day) == (last.year, last.month, last.day):
        return day_label(start, utc=utc)
    if (first.year, first.month) == (last.year, last.month):
        return f"{_MONTHS[first.month - 1]} {first.day}{EN_DASH}{last.day}, {first.year}{suffix}"
    if first.year == last.year:
        opening, closing = f"{_MONTHS[first.month - 1]} {first.day}", f"{_MONTHS[last.month - 1]} {last.day}"
        return f"{opening} {EN_DASH} {closing}, {first.year}{suffix}"
    return f"{day_label(start)} {EN_DASH} {day_label(end)}{suffix}"


def percent(fraction: float) -> str:
    """`92%` from 0.92314 -- whole percents; a story is not a spreadsheet."""
    return f"{round(max(0.0, min(1.0, fraction)) * 100)}%"


def join_names(names: list[str]) -> str:
    """`foo`, `foo and bar`, `foo, bar and baz`."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def duration(seconds: float) -> str:
    """`1.5 s`, `4 min 12 s`, `2 h 5 min` -- the camera's own units."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s" if seconds != int(seconds) else f"{int(seconds)} s"
    minutes, rest = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {rest} s" if rest else f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def shutter(seconds: float) -> str:
    """`1/250 s` below a second, `2 s` at or above."""
    if seconds >= 1:
        return f"{seconds:g} s"
    return f"1/{round(1 / seconds):d} s" if seconds > 0 else "0 s"


def span(pair: list, spell) -> str:
    """`f/4` when both ends agree, `f/4` + EN DASH + `f/8` otherwise."""
    low, high = pair
    return spell(low) if low == high else f"{spell(low)}{EN_DASH}{spell(high)}"


def plural_verb(n: int, singular: str, plural: str) -> str:
    """`appears` for one, `appear` for many."""
    return singular if n == 1 else plural
