"""When did this happen -- judged from EVERY claim, not the first one.

The old interpretation was a ladder: the most trusted source won and
the rest were never consulted. A ladder cannot say "these agree", so a
generator's day-precision date outranked a minute-precision request
time sitting in the file name and a finish time the filesystem kept,
and whole libraries stayed at day precision -- too coarse to ever form
a session.

This is a corroboration stack. Every source makes a CLAIM with its own
precision; the judge settles the OCCURRENCE from the generator's own
claims, then lets every other source SUPPORT it, satisfy a CONSTRAINT,
or CONFLICT -- named, persisted, never compressed into a score. The
sources all descend from the same request, so agreement is consistency,
not independent probability: `quality` is an ORDINAL -- corroborated >
claimed > contested -- and `certainty` is that ordinal's fixed
spelling, nothing finer.

Two times come out, kept apart. The CLAIMED occurrence (`local_at`,
`precision`) is what the generator itself said -- the stamp, the
request minute, or the day -- and is what groupers sequence by. The
ESTIMATE (`estimated_at`, `finished_at`) is what the filesystem adds:
mtime is the finish, and `mtime - generation_time` is the request to
the second -- an inference consistent with the claim, shown with its
own basis and never written over the claim. A consumer wanting seconds
reads the estimate and says so.

Sources, for a generated file:

    gen_day      the generator's `date` parameter         day    (SwarmUI writes
                                                                 yyyy-MM-dd)
    gen_minute   SwarmUI's file name `[hour][minute]`     minute -- documented as a
                 `[request_time_inc]` counter orders      unique linear id prefix
                 requests inside the minute               (docs/User Settings.md)
    gen_stamp    the opt-in stamped name                  second -- the day and the
                 `[year][month][day]T[hour][minute]       second in the name; the T
                 [second][request_time_inc]-...`          is the marker no default
                                                          name carries
    gen_second   a full generator stamp in `date`         second (A1111-family)
    gen_time     `generation_time`, a duration            request -> finish
    mtime        filesystem modified                      the FINISH instant (the
                                                          one NTFS time a copy
                                                          preserves)
    btime        filesystem created                       a constraint: on NTFS
                                                          a copy is born at copy
                                                          time; never an instant

Rules, all evaluated, none short-circuits:

 1. The occurrence is the generator's finest claim: the stamp (second),
    else the request minute (minute), else the day. A stamp or a full
    `date` outside the claimed day conflicts and the day stands.
 2. mtime is the finish. With a duration, `mtime - generation_time`
    inside the claimed window is `mtime_finish_consistent` and becomes
    `estimated_at`; without one, an mtime inside the window is
    `mtime_consistent`. Outside: a conflict that says how far off, and
    an mtime BEFORE the request is the loud case. The claim is never
    replaced: "the request was at 09:47" stays exactly that, with "and
    it finished at 09:48:32, so it started at 09:47:28" beside it.
 3. btime at or after the occurrence satisfies `btime_after_generation`;
    before it -- bytes born before they were generated -- conflicts.
    btime never supplies an instant while any claim exists.
 4. No generator claim: the filesystem's own instants, mtime first, as
    the context's fallback only -- never an occurrence.

Quality: contested when anything conflicts; corroborated when nothing
conflicts and at least one support was found; claimed otherwise.

Nothing here reads a database: claims in, a verdict out, so the table
of cases in the tests IS the policy. One file's verdict depends on that
file alone -- sibling profiles (a folder copied in one go, every btime
alike) are a later, folder-level derived interpretation with its own
invalidation, not a per-file rule.
"""

from __future__ import annotations

import dataclasses
import datetime
import re

#: Seconds of slack around a finish: queue wait, encoding, disk.
SLACK = 90.0

#: The ordinal, and its fixed spelling in the certainty column.
CERTAINTY = {"corroborated": 0.9, "claimed": 0.6, "contested": 0.4}

#: SwarmUI's default output name: `[hour][minute][request_time_inc]-...`
#: (refs/mcmonkeyprojects/SwarmUI src/Core/Settings.cs:399).
_SWARM_NAME = re.compile(r"^(\d{2})(\d{2})(\d{3})-")
#: The opt-in STAMPED name this gallery recommends for Swarm's
#: OutpathBuilder: `[year][month][day]T[hour][minute][second]
#: [request_time_inc]-[prompthash]-[model]`. The `T` is the marker: no
#: default Swarm name has one, so the two grammars never collide.
_SWARM_STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(\d{3})?-")


@dataclasses.dataclass(frozen=True)
class Verdict:
    local_at: float | None
    instant_at: float | None
    tz_offset_min: int | None
    precision: str
    basis: str
    supports: tuple[str, ...]
    conflicts: tuple[str, ...]
    #: the filesystem's finish instant, when consistent with the claim
    finished_at: float | None = None
    #: the request to the second, inferred as finish - generation time,
    #: a wall-clock reading beside the claim -- never in its place
    estimated_at: float | None = None

    @property
    def quality(self) -> str:
        if self.conflicts:
            return "contested"
        return "corroborated" if self.supports else "claimed"

    @property
    def certainty(self) -> float:
        return CERTAINTY[self.quality]


def _wall(text: str) -> tuple[float, str] | None:
    """A generator's date text as (wall-clock epoch, precision): a bare
    day, or a full stamp. Anything else is no claim."""
    held = text.strip()
    for pattern, precision in (("%Y-%m-%d %H:%M:%S", "second"), ("%Y-%m-%dT%H:%M:%S", "second"), ("%Y-%m-%d", "day")):
        try:
            moment = datetime.datetime.strptime(held, pattern).replace(tzinfo=datetime.UTC)
        except ValueError:
            continue
        return moment.timestamp(), precision
    return None


def swarm_minute(name: str) -> tuple[int, int, int] | None:
    """(hour, minute, request counter) from a SwarmUI-format file name,
    or None when the name is not one."""
    match = _SWARM_NAME.match(name)
    if not match:
        return None
    hour, minute, counter = (int(match.group(i)) for i in (1, 2, 3))
    if hour > 23 or minute > 59:
        return None
    return hour, minute, counter


def swarm_stamp(name: str) -> tuple[float, int] | None:
    """(wall-clock epoch, request counter) from a STAMPED Swarm name, or
    None when the name is not one."""
    match = _SWARM_STAMP.match(name)
    if not match:
        return None
    try:
        moment = datetime.datetime(*(int(match.group(i)) for i in range(1, 7)), tzinfo=datetime.UTC)
    except ValueError:
        return None
    return moment.timestamp(), int(match.group(7) or 0)


def _wall_of(instant: float) -> float:
    """An instant's reading on the HOST's wall clock -- the clock a
    local generator wrote its day and minute from."""
    local = datetime.datetime.fromtimestamp(instant).astimezone()
    return local.replace(tzinfo=datetime.UTC).timestamp()


def judge_generation(
    *,
    date_text: str | None,
    name: str,
    tool: str | None,
    mtime: float | None,
    btime: float | None,
    generation_time: float | None,
) -> Verdict | None:
    """The generation act's time, from every claim the file carries."""
    claimed = _wall(date_text) if date_text else None
    if claimed is None:
        return None
    day_start, precision = claimed
    conflicts: list[str] = []
    supports: list[str] = []
    if precision == "second":
        at, basis, window = day_start, "embedded", (day_start, day_start + 1.0)
    else:
        at, basis, window = day_start, "embedded", (day_start, day_start + 86400.0)
        precision = "day"
        swarm = (tool or "").lower().startswith("swarm")
        stamp = swarm_stamp(name) if swarm else None
        minute = swarm_minute(name) if swarm and stamp is None else None
        if stamp is not None:
            stamped, _counter = stamp
            if day_start <= stamped < day_start + 86400.0:
                at, precision, basis = stamped, "second", "filename"
                supports.append("embedded_day")
                window = (stamped, stamped + 1.0)
            else:
                conflicts.append(
                    f"the stamped name {_spell(stamped)} is not inside the claimed day {_spell(day_start)}"
                )
        elif minute is not None:
            hour, mins, _counter = minute
            at, precision, basis = day_start + hour * 3600.0 + mins * 60.0, "minute", "filename"
            supports.append("embedded_day")
            window = (at, at + 60.0)
    finished_at = estimated_at = None
    # rule 2: mtime is the finish, evidence beside the claim
    if mtime is not None:
        finish = _wall_of(mtime)
        if generation_time is not None:
            started = finish - generation_time
            if window[0] - SLACK <= started < window[1] + SLACK:
                supports.append("mtime_finish_consistent")
                finished_at, estimated_at = mtime, started
            elif finish < window[0]:
                conflicts.append(f"mtime {_spell(finish)} is before the claimed {_spell(window[0])}")
            else:
                conflicts.append(f"mtime {_spell(finish)} is {_hours(started - window[1])} after the claimed window")
        elif window[0] <= finish < window[1] + SLACK:
            supports.append("mtime_consistent")
            finished_at = mtime
        elif finish < window[0]:
            conflicts.append(f"mtime {_spell(finish)} is before the claimed {_spell(window[0])}")
        else:
            conflicts.append(f"mtime {_spell(finish)} is {_hours(finish - window[1])} after the claimed window")
    # rule 3: btime is a constraint
    if btime is not None:
        born = _wall_of(btime)
        if born >= at - 1.0:
            supports.append("btime_after_generation")
        else:
            conflicts.append(f"btime {_spell(born)} is before the claimed {_spell(at)}: bytes born before generated")
    return Verdict(at, None, None, precision, basis, tuple(supports), tuple(conflicts), finished_at, estimated_at)


def judge_filesystem(mtime: float | None, btime: float | None) -> Verdict | None:
    """Rule 4: a file with no claim of its own. The filesystem is not an
    act, so this is the context's fallback only -- never an occurrence:
    mtime first (a copy keeps it), btime only when mtime is absent, a
    btime within a day of the mtime noted as consistent."""
    if mtime is None and btime is None:
        return None
    if mtime is None:
        return Verdict(None, btime, None, "subsecond", "btime", (), ())
    supports = ("btime_consistent",) if btime is not None and abs(btime - mtime) <= 86400.0 else ()
    return Verdict(None, mtime, None, "subsecond", "mtime", supports, ())


def _spell(wall: float) -> str:
    return datetime.datetime.fromtimestamp(wall, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


def _hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"
