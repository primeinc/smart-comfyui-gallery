"""One searchable list of every fact this library can be asked about.

The filter drawer offers a curated vocabulary (db/vocabulary.py: 41
dimensions this application understands well enough to name) and, behind
an "advanced" heading, a text box whose placeholder is `key=value`. That
box is the proof this module is needed: it can only be used by somebody
who already knows the internal spelling, which is the one thing the
application is supposed to remember for them.

So: one list, over both registries, that a person searches by typing
what they half-remember.

    Add filter…    [ edit                          ]
    ─────────────────────────────────────────────
    Used local editor                     SwarmUI
    Edit reference megapixels          Generation

The curated/discovered split stays real INSIDE -- one has semantics we
understand, the other is a recorded fact about what some tool wrote --
and stops being visible to the person. A curated dimension carries its
own operators and value list; a discovered key is asked through
`param.is`, and the catalog says which is which so the surface can build
the right control without the person ever learning the difference.

Three things make this harder than a list, all measured against a real
3,748-file library with 108 distinct `file_param` keys.

**Indexed families are one concept wearing seven names.** `_param()`
flattens lists positionally, so `used_wildcards.0` through
`used_wildcards.6` are seven keys, 55 files each, and "did this use a
wildcard" is currently seven separate questions. A family collapses to
one field here whose repeats OR -- which is the `multi="any"` machinery
the vocabulary already has.

**Rank by what DISCRIMINATES, not by what is common.** About 40 of those
108 keys are EXIF plumbing -- `StripOffsets`, `YCbCrPositioning`,
`FocalPlaneXResolution` -- and a picker built straight off `param_key`
is a haystack. A key every file carries with one value between them
separates nothing; a key whose every file has its own value (a seed, a
timestamp) is not a filter either. Usefulness lives in between, and is
counted WITHIN the answer being looked at rather than over the library,
because "what could I ask next about these" is the question a filter
surface is answering.

**The observed type is already recorded.** `param_key.value_kind` is
maintained by a trigger over a three-state lattice that only widens
(db/schema.sql `param_key_learn`), so `automaticvae` -- the string
'True' with a NULL `value_num` on all 155 files -- is honestly `text`
and gets a value list rather than a number control. Nothing here has to
infer a type from the values.
"""

from __future__ import annotations

import dataclasses
import re

from . import resultset, vocabulary

#: How many fields one search answers with. A person scanning a list
#: stops well before this; what is past it is reachable by typing more.
MOST = 30

#: The `source` values that are a camera's own plumbing rather than anything
#: somebody asks about; by source, not by key, so a new camera's new tag needs
#: no new spelling here. Ranked down, never hidden.
PLUMBING = ("exif", "iptc", "container", "filesystem")

#: A key ending `.<digits>` is one member of a positional family.
_INDEXED = re.compile(r"^(?P<stem>.+)\.(?P<at>\d+)$")

#: The share of a full list reserved for keys nothing here named, so the curated
#: dimensions cannot fill the list and leave the long tail unreachable. The tail
#: holds this share whenever it has anything to put in it, and gives back the rest.
SHARE = 0.4

#: Where a field's usefulness stops growing with its value count. Two to
#: fifty distinct values is a list somebody can pick from; past that it
#: still filters, and is worth less as something to OFFER.
PICKABLE = 50


@dataclasses.dataclass(frozen=True)
class Field:
    """One filterable fact, as a person would look for it."""

    #: What the URL carries. A curated dimension's own key, or
    #: `param.is` for a discovered one -- the surface never has to know
    #: which registry it came from to build the address.
    key: str
    #: For a discovered field, the raw metadata key its clause names.
    #: None for a curated dimension, whose key IS the address.
    param: str | None
    #: What a person reads.
    label: str
    #: The section it belongs to, or the source that wrote it.
    group: str
    #: Everything this field also answers to, lower case: the raw key,
    #: the family stem, the source. So `edit`, `editor` and the ugly
    #: serialised spelling all arrive at the same fact.
    aliases: tuple[str, ...]
    #: text | number | mixed for a discovered key; the vocabulary's own
    #: value_kind for a curated one.
    value_kind: str
    #: The operators this field allows.
    ops: tuple[str, ...]
    #: How several values read: "", "any" or "both" (db/vocabulary.py).
    multi: str
    #: A sentence for the surface where the label alone misleads.
    note: str
    #: Files in the answer carrying this fact. 0 for a curated
    #: dimension, which is not counted here -- opening it counts it.
    covered: int
    #: Distinct values it takes across those files.
    values: int
    #: How many positional members collapsed into this one.
    #: `used_wildcards.0..6` is 7; everything else is 1.
    repeats: int

    @property
    def curated(self) -> bool:
        """Whether this application understands the fact or merely
        recorded that some tool wrote it."""
        return self.param is None


def _stem(key: str) -> tuple[str, bool]:
    """A positional family's shared name, and whether it was one."""
    found = _INDEXED.match(key)
    return (found.group("stem"), True) if found else (key, False)


def _readable(key: str) -> str:
    """A metadata key as a sentence, best effort.

    `used_wildcards` -> "used wildcards"; `FocalPlaneXResolution` ->
    "Focal Plane X Resolution". Best effort and nothing more: this is a
    string some tool chose, and the honest thing is to make it legible
    rather than to pretend we named it. The raw spelling stays in
    `aliases`, so somebody who knows it still finds it by typing it.
    """
    spaced = re.sub(r"[_.\-]+", " ", key)
    # Two rules, because camel case has two seams: `focalPlane` breaks after the
    # lower-case run, `XResolution` breaks inside an upper-case run before the
    # last capital that starts a word.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return spaced.strip() or key


def usefulness(covered: int, values: int, total: int) -> float:
    """How much a field would DISCRIMINATE within this answer, 0..1.

    Two ways to be useless and they are opposite ends of one axis. A
    field every file carries with one value between them cuts nothing --
    every camera writes `YCbCrPositioning` and it is always 1. A field
    whose every file has its own value cuts to one file at a time, which
    is a lookup rather than a filter -- a seed, a capture timestamp.

    So: coverage says how much of the answer it can speak about, and the
    value count is worth most in the band a person can pick from.
    Deliberately not entropy. Entropy needs the whole value distribution
    -- a second aggregate over the long tail's widest table -- to rank a
    list somebody is going to read the top ten of.
    """
    if total <= 0 or covered <= 0 or values <= 1:
        return 0.0
    reach = min(1.0, covered / total)
    band = 1.0 if values <= PICKABLE else PICKABLE / values
    return reach * band


def _matches(field: Field, wanted: str) -> int | None:
    """How well a field answers to what somebody typed, or None.

    Lower is better, so the caller sorts ascending and never has to
    remember which way round the score runs:

        0  the label starts with it        "edit" -> "Edit reference…"
        1  a word inside the label does    "ref"  -> "Edit reference…"
        2  the label contains it anywhere
        3  an alias does -- the raw key, the family stem, the source

    A field is never dropped for matching only by its ugly spelling:
    that IS the door for somebody who knows it, and it is the door this
    module exists to stop being the ONLY one.
    """
    if not wanted:
        return 2
    label = field.label.casefold()
    if label.startswith(wanted):
        return 0
    if any(word.startswith(wanted) for word in re.split(r"\W+", label) if word):
        return 1
    if wanted in label:
        return 2
    if any(wanted in one for one in field.aliases):
        return 3
    return None


def _curated(kind: str | None) -> list[Field]:
    """The vocabulary, as catalog entries.

    Offered dimensions only, and only those that are facts about the
    kind being asked about -- a LoRA is not a fact about an audio file,
    and the drawer already refuses to list one there.
    """
    made = []
    for one in vocabulary.offered():
        if not vocabulary.applies_to(one, kind):
            continue
        made.append(
            Field(
                key=one.key,
                param=None,
                label=one.label,
                group=one.group,
                aliases=(one.key.casefold(), one.group.casefold()),
                value_kind=one.value_kind,
                ops=one.ops,
                multi=one.multi,
                note=one.note,
                covered=0,
                values=0,
                repeats=1,
            )
        )
    return made


#: Every discovered key in the answer, with what it would cut it by: one scan of
#: the long tail's table for how many files carry each key and how many values
#: they hold. The family collapse over a trailing dot and digits happens in Python.
_DISCOVERED = (
    "SELECT fp.source, fp.key, COUNT(DISTINCT f.id), COUNT(DISTINCT fp.value_text) FROM file f"
    " JOIN file_param fp ON fp.file_id = f.id"
    " WHERE f.missing_since IS NULL {scope}"
    " GROUP BY fp.source, fp.key"
)

#: How many media the question answers with, over the same conjunct the
#: aggregate above uses -- so "carried by 40% of this answer" is a
#: fraction of the same denominator, taken on the same connection.
_COUNTED = "SELECT COUNT(*) FROM file f WHERE f.missing_since IS NULL {scope}"

#: The observed type of every key, from the registry the trigger keeps. Read
#: whole, one row per (source, key), rather than joined into the aggregate
#: above, which would make a scan of the widest table carry a lookup.
_KINDS = "SELECT source, key, value_kind FROM param_key"


def discovered(
    conn,
    query: resultset.GalleryQuery,
    *,
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
) -> tuple[list[Field], int]:
    """Every metadata key the answer carries, families collapsed, and
    how many media the answer holds.

    Counted WITH the whole question, not without one dimension: this
    list is "what else is true about what I am looking at", and a key no
    file here carries is not something to offer.

    The size comes back with the fields because it is the denominator
    they are ranked by, and taking it here means one `scope_of` and one
    connection rather than a second read that could straddle a commit.

    A field is numeric when its kind is `number` exactly. `param.num` reads
    `fp.value_num`, the column the schema fills whenever a value parses as one,
    so it can offer the comparisons a number has; `param.is` reads `value_text`
    and can only mean equals, because over text 9 is more than 30. `mixed` stays
    on `param.is`: a family where some values parsed as numbers and some did not
    cannot be compared as numbers without dropping the rest. schema.sql CHECKs
    `value_kind IN ('text','number','mixed')`, so a longer list of numeric kinds
    would be guessing at kinds the column cannot hold.
    """
    conjunct, args, _ = resultset.scope_of(conn, query, actor_id, models_dir=models_dir, now=now)
    kinds = {(source, key): kind for source, key, kind in conn.execute(_KINDS)}
    # The answer's size from the SAME conjunct, so the ratios below are a
    # fraction of what they were counted against. Not `resultset.describe`,
    # which builds the whole ordered answer to hand back a length.
    total = int(conn.execute(_COUNTED.replace("{scope}", conjunct), args).fetchone()[0])

    held: dict[tuple[str, str], dict] = {}
    for source, key, covered, values in conn.execute(_DISCOVERED.replace("{scope}", conjunct), args):
        stem, indexed = _stem(str(key))
        at = held.setdefault(
            (str(source), stem),
            {"covered": 0, "values": 0, "repeats": 0, "kinds": set(), "raw": set()},
        )
        # A family's coverage is the WIDEST member, never the sum: the
        # same file carries `used_wildcards.0` and `.1`, and adding them
        # would report more files than the answer holds.
        at["covered"] = max(at["covered"], int(covered))
        at["values"] += int(values)
        at["repeats"] += 1 if indexed else 0
        at["kinds"].add(kinds.get((str(source), str(key)), "text"))
        at["raw"].add(str(key).casefold())

    made = []
    for (source, stem), one in held.items():
        # A family whose members disagree about their type is mixed,
        # which is exactly what the lattice means.
        kind = "mixed" if len(one["kinds"]) > 1 else next(iter(one["kinds"]))
        # `number` exactly; the docstring names the column and the CHECK.
        numeric = kind == "number"
        made.append(
            Field(
                key="param.num" if numeric else "param.is",
                param=stem,
                label=_readable(stem),
                group=source,
                aliases=tuple(sorted({*one["raw"], stem.casefold(), source.casefold()})),
                value_kind=kind,
                ops=("eq", "gte", "lte") if numeric else ("eq", "any"),
                multi="any",
                note="",
                covered=one["covered"],
                values=one["values"],
                repeats=max(1, one["repeats"]),
            )
        )
    return made, total


def catalog(
    conn,
    query: resultset.GalleryQuery,
    *,
    search: str = "",
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    most: int = MOST,
) -> tuple[tuple[Field, ...], int]:
    """The fields worth offering from here, best first, and how many more.

    Curated before discovered at equal match quality, because a fact
    this application named is one it can offer values and operators for,
    and a raw key is one it can only compare. Within the discovered, by
    what would cut the answer most usefully, with a camera's own
    plumbing ranked down rather than hidden.

    And then a SHARE of the list is kept for discovered fields, which is
    not a tweak -- it is the defect this module exists to fix, found
    reproducing itself inside the fix. There are forty-one curated
    dimensions and thirty rows: ranked purely by match quality, any
    broad search fills the list with names before a single raw key
    appears, and the long tail is unreachable again for exactly the
    person who does not know what it is called. So the tail keeps
    `SHARE` of the rows whenever it has anything to put in them, and
    gives back whatever it does not use.
    """
    wanted = search.strip().casefold()
    found, total = discovered(conn, query, actor_id=actor_id, models_dir=models_dir, now=now)
    fields = [*_curated(query.kind), *found]

    ranked = []
    for one in fields:
        score = _matches(one, wanted)
        if score is None:
            continue
        ranked.append(
            (
                score,
                0 if one.curated else 1,
                1 if one.group in PLUMBING else 0,
                -usefulness(one.covered, one.values, total),
                one.label.casefold(),
                one,
            )
        )
    ranked.sort(key=lambda held: held[:-1])
    order = [held[-1] for held in ranked]

    kept = max(0, most - int(most * SHARE))
    named = [one for one in order if one.curated][:kept]
    tail = [one for one in order if not one.curated][: most - len(named)]
    # Back-fill from whichever side had less to give, so a library with
    # no metadata keys at all still answers with a full list of names.
    if len(named) + len(tail) < most:
        room = most - len(named) - len(tail)
        held = {id(one) for one in (*named, *tail)}
        named += [one for one in order if one.curated and id(one) not in held][:room]
    # Back in the ranked order they were chosen from: the reservation
    # decides WHICH rows are shown, never which is best.
    chosen = {id(one) for one in (*named, *tail)}
    shown = tuple(one for one in order if id(one) in chosen)
    return shown, max(0, len(order) - len(shown))


#: What ONE discovered key holds across this answer, most used first. `LIKE` and
#: not `=` because a positional family is one field, so the stem matches exactly or
#: followed by a dot and digits; the escape is explicit because a key may hold `%` or `_` (`Exif_Offset`).
_VALUES = (
    "SELECT fp.value_text, COUNT(DISTINCT f.id) FROM file f"
    " JOIN file_param fp ON fp.file_id = f.id"
    " WHERE f.missing_since IS NULL {scope}"
    r"   AND (fp.key = ? OR fp.key LIKE ? ESCAPE '\')"
    "   AND fp.value_text IS NOT NULL"
    " GROUP BY fp.value_text"
    " ORDER BY COUNT(DISTINCT f.id) DESC, fp.value_text COLLATE NOCASE"
)


def values(
    conn,
    query: resultset.GalleryQuery,
    param: str,
    *,
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    search: str = "",
    most: int = MOST,
) -> tuple[tuple[tuple[str, int], ...], int]:
    """What one discovered key holds here: (value, files), and how many
    more.

    The half of the catalog that turns "type sniffed format equals…"
    into picking `png` off a list. The curated dimensions have carried
    their own value lists since the drawer was built (db/vocabulary.py
    `discover`); the long tail never could, because there is no
    statement per key to write -- there is one statement for every key,
    and this is it.

    Counted WITH the whole question, like `discovered`: these are the
    values present in what is being looked at, so choosing one always
    leaves something. A value the library holds and this answer does not
    is not an option here.
    """
    conjunct, args, _ = resultset.scope_of(conn, query, actor_id, models_dir=models_dir, now=now)
    # The backslash first, or escaping the wildcards would then escape
    # the escapes. `Exif_Offset` is a real key and its underscore is a
    # LIKE wildcard, so this is not hypothetical tidiness.
    stem = param.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = list(conn.execute(_VALUES.replace("{scope}", conjunct), [*args, param, f"{stem}.%"]))
    wanted = search.strip().casefold()
    made = [(str(value), int(count)) for value, count in rows if not wanted or wanted in str(value).casefold()]
    return tuple(made[:most]), max(0, len(made) - most)
