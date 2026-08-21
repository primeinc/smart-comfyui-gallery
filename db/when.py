"""When did this happen -- judged from EVERY claim, not the first one.

A ladder lets the most trusted source win and never consults the rest,
so it cannot say "these agree": a generator's day-precision date
outranked a minute-precision request time sitting in the file name and
whole libraries stayed at day precision -- too coarse to form a session.

This is a corroboration stack. Every source makes a CLAIM with its own
precision; the judge settles the OCCURRENCE from the generator's own
claims, then lets every other source SUPPORT it, satisfy a CONSTRAINT,
or CONFLICT -- named, persisted, never compressed into a score. The
sources all descend from the same request, so agreement is consistency,
not independent probability: `quality` is an ORDINAL -- corroborated >
claimed > contested -- and `certainty` is that ordinal's fixed spelling.

The FILESYSTEM is read in the host's zone: mtime and btime are UTC
instants, the generator's claims are unzoned wall clocks, and the only
bridge is the zone this machine runs in today. A library generated in
one zone and opened in another makes that bridge wrong by whole hours,
so every filesystem support or conflict carries the assumption
(`host_zone_assumed`) and a filesystem conflict never decides whether
a claim is fit for chronology -- only the generator's own claims
disagreeing with each other does (`usable`).

Two times come out, kept apart. The CLAIMED occurrence (`local_at`,
`precision`) is what the generator itself said -- the stamp, the
request minute, or the day -- and is what groupers sequence by, with
`source_order` (SwarmUI's request counter) breaking ties inside the
minute. The ESTIMATE (`estimated_at`, `finished_at`) is what the
filesystem adds: mtime is the finish, `mtime - generation_time` is the
request to the second -- an inference consistent with the claim, shown
with its own basis and never written over the claim.

Sources, for a generated file:

    gen_day      the generator's `date` parameter         day    (SwarmUI writes
                                                                 yyyy-MM-dd)
    gen_minute   SwarmUI's file name `[hour][minute]`     minute -- documented as a
                 `[request_time_inc]` counter orders      unique linear id prefix
                 requests inside the minute               (docs/User Settings.md)
    gen_stamp    a stamped name, either grammar:          second (.millisecond)
                 `[year][month][day]T[hour][minute]       -- every clock field
                 [second][request_time_inc]-...`          is Swarm's RequestTime;
                 `[year][month][day]_[hour]h[minute]m     the `T` or the
                 [second]s[millisecond]ms_...`            `h..m..s..ms` is the
                                                          marker no default name
                                                          carries
    gen_second   a full generator stamp in `date`         second (A1111-family)
    gen_time     `generation_time`, a duration            request -> finish
    mtime        filesystem modified                      the FINISH instant (the
                                                          one NTFS time a copy
                                                          preserves)
    btime        filesystem created                       a constraint: on NTFS
                                                          a copy is born at copy
                                                          time; never an instant

Rules, all evaluated, none short-circuits:

 1. The occurrence is the generator's finest claim: the stamped name
    (second, standing on its own -- a name is not optional metadata),
    else the request minute placed in the embedded day (minute), else
    the day. An embedded day that contradicts the stamp is a generator
    conflict and the stamp stands; a full `date` stamp is a claim of
    its own. The request counter is ORDER inside the minute, persisted
    as `source_order` and never turned into seconds.
 2. mtime is the finish. With a duration, `mtime - generation_time`
    inside the claimed window is `mtime_finish_consistent` and becomes
    `estimated_at`; without one, an mtime inside the window is
    `mtime_consistent`. Outside: a conflict that says how far off, and
    an mtime BEFORE the request is the loud case. The claim is never
    replaced.
 3. btime at or after the occurrence satisfies `btime_after_generation`;
    before it -- bytes born before they were generated -- conflicts.
    btime never supplies an instant while any claim exists.
 4. No generator claim: the filesystem's own instants, mtime first, as
    the context's fallback only -- never an occurrence.

Quality: contested when anything conflicts; corroborated when nothing
conflicts and at least one support was found; claimed otherwise.

Nothing here reads a database: claims in, a verdict out, so the table
of cases in the tests IS the policy. One file's verdict depends on that
file alone; sibling profiles (a folder copied in one go, every btime
alike) are a later, folder-level interpretation with its own
invalidation, not a per-file rule.
"""

from __future__ import annotations

import dataclasses
import datetime
import re
import typing

#: Seconds of slack around a finish: queue wait, encoding, disk.
SLACK = 90.0

#: The ordinal, and its fixed spelling in the certainty column.
CERTAINTY = {"corroborated": 0.9, "claimed": 0.6, "contested": 0.4}

#: A conflict's spelling says who disagreed: the generator with itself,
#: or the filesystem (read in the host's zone) with the generator.
GENERATOR = "generator: "
FILESYSTEM = "filesystem: "

#: SwarmUI's default output name: `[hour][minute][request_time_inc]-...`
#: (refs/mcmonkeyprojects/SwarmUI src/Core/Settings.cs:399).
_SWARM_NAME = re.compile(r"^(\d{2})(\d{2})(\d{3})-")
#: Stamped names. Every clock field is Swarm's RequestTime, so the name
#: is the generator's own second. The `T` and the `h..m..s..ms` are the
#: markers: no default Swarm name has either, so the grammars never
#: collide.
_SWARM_STAMPS = (
    re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?P<order>\d{3})?-"),
    re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})h(\d{2})m(\d{2})s(?P<ms>\d{3})ms_"),
)


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
    #: the generator's own order inside the claimed bucket (SwarmUI's
    #: request counter): ordering evidence, never seconds
    source_order: int | None = None

    @property
    def quality(self) -> str:
        if self.conflicts:
            return "contested"
        return "corroborated" if self.supports else "claimed"

    @property
    def certainty(self) -> float:
        return CERTAINTY[self.quality]

    @property
    def usable(self) -> bool:
        """Fit for chronology: the generator does not disagree with
        itself. A filesystem conflict is recorded but, read through the
        host's zone, cannot demote the generator's own claim."""
        return not any(one.startswith(GENERATOR) for one in self.conflicts)


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


def swarm_stamp(name: str) -> tuple[float, int | None] | None:
    """(wall-clock epoch, request counter) from a STAMPED Swarm name --
    either grammar -- or None when the name is not one. The counter is
    None for a grammar that does not carry one."""
    for pattern in _SWARM_STAMPS:
        match = pattern.match(name)
        if not match:
            continue
        try:
            moment = datetime.datetime(*(int(match.group(i)) for i in range(1, 7)), tzinfo=datetime.UTC)
        except ValueError:
            return None
        at = moment.timestamp()
        fields = match.groupdict()
        if fields.get("ms") is not None:
            at += int(fields["ms"]) / 1000.0
        order = fields.get("order")
        return at, (int(order) if order is not None else None)
    return None


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
    swarm = (tool or "").lower().startswith("swarm")
    stamp = swarm_stamp(name) if swarm else None
    claimed = _wall(date_text) if date_text else None
    if claimed is None and stamp is None:
        return None
    conflicts: list[str] = []
    supports: list[str] = []
    order: int | None = None
    if stamp is not None:
        # rule 1: the stamped name stands on its own; the embedded day
        # corroborates it or disagrees with it
        at, order = stamp
        precision, basis, window = "second", "filename", (at, at + 1.0)
        if claimed is not None:
            day_start, claimed_precision = claimed
            inside = day_start <= at < day_start + 86400.0 if claimed_precision == "day" else abs(at - day_start) < 1.0
            if inside:
                supports.append("embedded_day" if claimed_precision == "day" else "embedded_stamp")
            else:
                conflicts.append(
                    f"{GENERATOR}the stamped name {_spell(at)} is not inside the embedded {claimed_precision}"
                    f" {_spell(day_start)}"
                )
    else:
        day_start, precision = typing.cast("tuple[float, str]", claimed)
        if precision == "second":
            at, basis, window = day_start, "embedded", (day_start, day_start + 1.0)
        else:
            at, basis, window = day_start, "embedded", (day_start, day_start + 86400.0)
            precision = "day"
            minute = swarm_minute(name) if swarm else None
            if minute is not None:
                hour, mins, order = minute
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
                conflicts.append(f"{FILESYSTEM}mtime {_spell(finish)} is before the claimed {_spell(window[0])}")
            else:
                conflicts.append(
                    f"{FILESYSTEM}mtime {_spell(finish)} is {_hours(started - window[1])} after the claimed window"
                )
        elif window[0] <= finish < window[1] + SLACK:
            supports.append("mtime_consistent")
            finished_at = mtime
        elif finish < window[0]:
            conflicts.append(f"{FILESYSTEM}mtime {_spell(finish)} is before the claimed {_spell(window[0])}")
        else:
            conflicts.append(
                f"{FILESYSTEM}mtime {_spell(finish)} is {_hours(finish - window[1])} after the claimed window"
            )
    # rule 3: btime is a constraint
    if btime is not None:
        born = _wall_of(btime)
        if born >= at - 1.0:
            supports.append("btime_after_generation")
        else:
            conflicts.append(
                f"{FILESYSTEM}btime {_spell(born)} is before the claimed {_spell(at)}: bytes born before generated"
            )
    if mtime is not None or btime is not None:
        supports.append("host_zone_assumed")
    return Verdict(
        at, None, None, precision, basis, tuple(supports), tuple(conflicts), finished_at, estimated_at, order
    )


#: A capture conflict between the camera's own zone claims.
CAMERA = "camera: "


def judge_capture(
    *,
    captured_at: float | None,
    subsec_ms: int | None,
    tz_offset_min: int | None,
    maker_tz_offset_min: int | None,
    mtime: float | None,
    btime: float | None,
    duration: float | None = None,
) -> Verdict | None:
    """The capture act's time, from every claim the file carries.

    The camera's DateTimeOriginal is the claim; SubSecTimeOriginal makes
    it subsecond-fine (`exif_subsecond`). The zone comes from
    OffsetTimeOriginal when written (`exif_offset`), else from the maker note's
    clock zone (`maker_timezone`), which turns the wall clock into a
    knowable instant; with both, they must agree or the camera disagrees
    with itself (`camera: `). A camera closes the file as the shutter
    closes (a clip: as recording ends), so mtime is the WRITE and is
    compared on instants when the zone is known -- no host zone in the
    way -- and on the host's wall clock otherwise (`host_zone_assumed`).
    btime is the constraint it always is.
    """
    if captured_at is None:
        return None
    supports: list[str] = []
    conflicts: list[str] = []
    precision = "second"
    fine = (subsec_ms or 0) / 1000.0
    if subsec_ms is not None:
        precision = "subsecond"
        supports.append("exif_subsecond")
    if tz_offset_min is not None:
        local = captured_at + fine
        instant = local - tz_offset_min * 60.0
        zone = tz_offset_min
        supports.append("exif_offset")
        if maker_tz_offset_min is not None:
            if maker_tz_offset_min == tz_offset_min:
                supports.append("maker_timezone")
            else:
                conflicts.append(
                    f"{CAMERA}OffsetTimeOriginal says {_zone(tz_offset_min)} but the maker note's clock zone"
                    f" says {_zone(maker_tz_offset_min)}"
                )
    elif maker_tz_offset_min is not None:
        local = captured_at + fine
        zone = maker_tz_offset_min
        instant = local - zone * 60.0
        supports.append("maker_timezone")
    else:
        local, instant, zone = captured_at + fine, None, None
    finished_at = None
    window_end = local + (duration or 0.0)
    if mtime is not None:
        if instant is not None:
            wrote = mtime
            start, end = instant, instant + (duration or 0.0)
            tag = "mtime_write_consistent"
        else:
            wrote = _wall_of(mtime)
            start, end = local, window_end
            tag = "mtime_write_consistent"
            supports.append("host_zone_assumed")
        if start - 2.0 <= wrote <= end + SLACK:
            supports.append(tag)
            finished_at = mtime
        elif wrote < start:
            conflicts.append(
                f"{FILESYSTEM}mtime {_spell(wrote)} is {_hours(start - wrote)} before the capture {_spell(start)}"
            )
        else:
            conflicts.append(f"{FILESYSTEM}mtime {_spell(wrote)} is {_hours(wrote - end)} after the capture")
    if btime is not None:
        born = btime if instant is not None else _wall_of(btime)
        floor = instant if instant is not None else local
        if instant is None and "host_zone_assumed" not in supports:
            supports.append("host_zone_assumed")
        if born >= floor - 1.0:
            supports.append("btime_after_capture")
        else:
            conflicts.append(
                f"{FILESYSTEM}btime {_spell(born)} is before the capture {_spell(floor)}: bytes born before taken"
            )
    return Verdict(
        local, instant, zone, precision, "capture", tuple(supports), tuple(conflicts), finished_at, None, None
    )


def _zone(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    minutes = abs(minutes)
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


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
