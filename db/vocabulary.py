"""What a filter is CALLED, where it belongs, and how its values are found.

The knowledge this module holds was spread across five places, and the
spread was visible in the product: `db/facets.py` knew a key's type,
operators and SQL; `sg_web/gallery.py` independently knew its human
label, in a private dict, for chips; `gallery.html` independently
hard-coded kind, rating and favorite as three `<select>` elements; and
nothing at all knew what values a key could take, so no surface could
offer them. Adding one filter meant coordinated edits in Python
predicate code, chip-label code, Jinja and TypeScript -- which is why
the browser could ask for four things out of a vocabulary of twenty.

So one Dimension record per filter, here. A dimension says:

    what it is called          a person's word, never the key
    where it belongs           the section of the filter surface
    how the URL carries it     a scope parameter, or `f=key:op:value`
    what its values are        the kind, the operators, and the SQL
                               that DISCOVERS the candidates
    which media have it        a LoRA is not a fact about an audio file

Two rules keep it honest.

**The vocabulary describes; it never decides membership.** The SQL here
finds candidate VALUES to offer. Which media match is
db/resultset.py's, through `scope_of`, and no surface built on this
module may answer that question a second way -- one definition of "these
videos" or eventually one of them says 412 and the other 407.

**A dimension is curated, not discovered.** The schema deliberately
records arbitrary `file_param` keys, because tools emit arbitrary
metadata. Those are not dimensions. A dimension is a fact whose meaning
this application understands well enough to name in a person's words.
The long tail belongs behind an explicit "advanced" door, asked by key.
"""

from __future__ import annotations

import dataclasses

from . import facets as facets_module

#: The sections of the filter surface, in the order a person looks for
#: them. Ordering lives here rather than in a template so a new
#: dimension lands in the right place by naming its group.
GROUPS: tuple[tuple[str, str], ...] = (
    ("mine", "my stuff"),
    ("media", "media"),
    ("people", "people"),
    ("places", "places"),
    ("time", "time"),
    ("generation", "generation"),
    ("camera", "camera"),
    #: LAST, and deliberately last: the long tail nothing here
    #: curated. A dimension above is a fact this application
    #: understands well enough to name in a person's words; a
    #: `param_key` row is a string some tool wrote. Four hundred
    #: discovered keys in the drawer would bury the twenty that mean
    #: something, so they live behind their own heading.
    ("advanced", "advanced metadata"),
)

#: Every file kind -- `db/facets.py KINDS`, not a copy of it (this was
#: the third Python copy of the tuple). A dimension that names no kinds
#: applies to all of them, which is the default because most questions
#: are cross-media: favorite, rating, folder, album, people, place and
#: date mean the same thing about a photograph, a video and a document.
EVERY_KIND = tuple(facets_module.KINDS)

#: The kinds that have pixels -- width, height and aspect are facts
#: about these and about nothing else. `vision/thumbs.py PICTURED`
#: states the same three and cannot import them: vision must not import
#: db (vision/semantic/__init__.py states the rule), so the pair is held
#: by test_the_kinds_with_pixels_are_one_set instead.
VISUAL = ("image", "animated_image", "video")
#: The kinds that have a length.
TIMED = ("animated_image", "video", "audio")

#: VISUAL as the SQL conjunct every pixels-only sweep appends, over the
#: file alias `f`. BUILT from the tuple: six hand-written copies of this
#: clause existed across db/inspecting.py and db/runner.py, and
#: db/inspecting.py's own comment records what happens when the count
#: and the queue disagree about what a picture is.
PICTURE_SQL = " AND f.kind IN (" + ", ".join(f"'{one}'" for one in VISUAL) + ")"


@dataclasses.dataclass(frozen=True)
class Dimension:
    """One filterable fact, described once."""

    #: The URL spelling: a GalleryQuery field name when `carried` is
    #: "scope", a db/facets.py registry key when it is "facet".
    key: str
    #: What a person calls it. Lower case: the surface decides casing.
    label: str
    #: Which section of the filter surface it appears in.
    group: str
    #: "scope" -- its own URL parameter, one value, established
    #: semantics and smart-collection compatibility. "facet" -- the
    #: repeatable `f=key:op:value` spelling. The interface hides the
    #: distinction; the implementation keeps whichever it already has,
    #: because rewriting `folder=` into a facet for tidiness would break
    #: every bookmark and every saved collection to no one's benefit.
    carried: str
    #: What the values are: int | num | text | date | bool | slug | id.
    #: "id" is an entity id whose chip must be resolved to a name.
    value_kind: str
    #: The operators this dimension allows.
    ops: tuple[str, ...] = ("eq",)
    #: The kinds that carry this fact. Empty means every kind.
    kinds: tuple[str, ...] = ()
    #: A short sentence for the surface, where the label alone misleads.
    note: str = ""
    #: How choosing SEVERAL values reads.
    #:
    #:   ""      one at a time; choosing another replaces it
    #:   "any"   several, OR'd -- the only sane reading for a dimension a
    #:           file has exactly ONE of. "image or video". AND there
    #:           would ask for a file that is two things at once and
    #:           always answer nothing.
    #:   "both"  several, and BOTH readings are real, so the surface
    #:           offers the choice. A picture carries several LoRAs and
    #:           several people at once, so "any of these" and "all of
    #:           these" are questions somebody genuinely asks separately.
    #:
    #: Which is right is a fact about the dimension, not a preference, so
    #: it is stated once here rather than guessed at by each surface.
    multi: str = ""
    #: Whether the filter surface LISTS this dimension.
    #:
    #: False for the links another surface makes rather than a person:
    #: a timeline bar opens `context.moment` between two epoch seconds
    #: at a `context.granule` fine enough for its width, and a session
    #: opens `event.id`. Nobody picks those off a list -- there is no
    #: list, the values are a machine's arithmetic about one bar. They
    #: are still described here, because when one arrives in the URL its
    #: chip has to read as words, and "the thing that labels chips" and
    #: "the thing that lists filters" being two registries is how the
    #: labels went stale in the first place.
    offered: bool = True
    #: SELECT value, label, COUNT(*) -- the candidate values reachable
    #: from a question, counted. `{scope}` is where the ResultSet's
    #: membership conjunct goes, over the file alias `f`. None means the
    #: values are not enumerable (a free number, a date) and the surface
    #: offers a control rather than a list.
    discover: str | None = None
    #: For `id` kinds: SELECT id, name over a `json_each` array of ids.
    #: A chip that said "#412" would be the database's answer to a
    #: question a person asked in words.
    names: str | None = None


ARTIFACT_NAMES = "SELECT a.id, a.name FROM artifact a JOIN json_each(?) ids ON ids.value = a.id"

PLACE_NAMES = "SELECT p.id, p.name FROM place p JOIN json_each(?) ids ON ids.value = p.id"


def _artifact_values(role: str) -> str:
    """Candidate artifacts in one role, by entity id, most used first."""
    return (
        "SELECT a.id, a.name, COUNT(DISTINCT f.id) FROM file f"
        " JOIN file_artifact fa ON fa.file_id = f.id AND fa.role = '" + role + "'"
        " JOIN artifact a ON a.id = fa.artifact_id"
        " WHERE f.missing_since IS NULL {scope}"
        " GROUP BY a.id ORDER BY COUNT(DISTINCT f.id) DESC, a.name COLLATE NOCASE"
    )


def _generation_values(column: str) -> str:
    """Candidate values of one recipe column, most used first."""
    return (
        "SELECT gen." + column + ", gen." + column + ", COUNT(*) FROM file f"
        " JOIN generation gen ON gen.file_id = f.id"
        " WHERE f.missing_since IS NULL {scope} AND gen." + column + " IS NOT NULL"
        " GROUP BY gen." + column + " ORDER BY COUNT(*) DESC, gen." + column + " COLLATE NOCASE"
    )


#: A yes/no dimension's two values, offered as a list like any other so
#: the surface needs no special case for them.
_YES_NO = (
    "SELECT v.value, v.label, ("
    "  SELECT COUNT(*) FROM file f WHERE f.missing_since IS NULL {scope} AND ({fact}) = v.value"
    ") FROM (SELECT 1 AS value, 'yes' AS label UNION ALL SELECT 0, 'no') v"
)


def _yes_no(fact: str) -> str:
    return _YES_NO.replace("{fact}", fact)


#: The vocabulary. Order within a group is the order the surface shows.
DIMENSIONS: tuple[Dimension, ...] = (
    # --- my stuff ------------------------------------------------------
    Dimension(
        key="favorite",
        label="favorite",
        group="mine",
        carried="scope",
        value_kind="bool",
        note="your own mark, never anyone else's",
    ),
    Dimension(
        key="rating_min",
        label="rating",
        group="mine",
        carried="scope",
        value_kind="int",
        ops=("gte",),
        note="at least this many stars",
    ),
    #: The keyword. Last of the "mine" group to be built and the first
    #: thing most people reach for -- a word you typed because it is the
    #: word you will look for.
    #:
    #: `multi="both"` and it is the dimension that makes the distinction
    #: worth offering: "beach AND sunset" and "beach OR sunset" are both
    #: ordinary questions about keywords, where for `kind` only one of
    #: them means anything.
    Dimension(
        key="tag",
        label="keyword",
        group="mine",
        carried="facet",
        value_kind="text",
        ops=("eq", "any"),
        multi="both",
        note="a word you wrote on a picture",
        discover=(
            "SELECT t.tag, t.label, COUNT(DISTINCT f.id) FROM file f"
            " JOIN file_tag ft ON ft.file_id = f.id"
            " JOIN tag t ON t.id = ft.tag_id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY t.id ORDER BY COUNT(DISTINCT f.id) DESC, t.label COLLATE NOCASE"
        ),
    ),
    #: A scope, and now a discoverable one. It held a slug and offered no
    #: list, so the drawer showed a heading with nothing under it -- a
    #: filter you can only use if you already know the answer.
    Dimension(
        key="album",
        label="album",
        group="mine",
        carried="scope",
        value_kind="slug",
        discover=(
            "SELECT e.slug, c.name, COUNT(DISTINCT f.id) FROM file f"
            " JOIN collection_file cf ON cf.file_id = f.id"
            " JOIN collection c ON c.id = cf.collection_id AND c.archived_at IS NULL"
            " JOIN entity e ON e.id = c.id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY c.id ORDER BY COUNT(DISTINCT f.id) DESC, c.name COLLATE NOCASE"
        ),
    ),
    Dimension(
        key="folder",
        label="folder",
        group="mine",
        carried="scope",
        value_kind="slug",
        discover=(
            "SELECT e.slug, fo.name, COUNT(*) FROM file f"
            " JOIN folder fo ON fo.id = f.folder_id"
            " JOIN entity e ON e.id = fo.id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY fo.id ORDER BY COUNT(*) DESC, fo.name COLLATE NOCASE"
        ),
    ),
    # --- media ---------------------------------------------------------
    #: Two spellings of one question, on purpose.
    #:
    #: `kind=` is a GalleryQuery scope and every bookmark, saved
    #: collection and cross-surface link uses it -- so it stays, and
    #: stays described, or those chips would print a raw key. But a scope
    #: holds ONE value, and "image or video" is the most ordinary
    #: multi-select there is. The surface writes `media.kind`, which
    #: can be OR'd. Both compose; asking one thing twice is redundant
    #: rather than wrong.
    Dimension(
        key="kind",
        label="kind",
        group="media",
        carried="scope",
        value_kind="text",
        offered=False,
    ),
    Dimension(
        key="media.kind",
        label="kind",
        group="media",
        carried="facet",
        value_kind="text",
        ops=("any", "eq"),
        multi="any",
        discover=(
            "SELECT f.kind, f.kind, COUNT(*) FROM file f"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY f.kind ORDER BY COUNT(*) DESC, f.kind"
        ),
    ),
    Dimension(
        key="media.width",
        label="width",
        group="media",
        carried="facet",
        value_kind="int",
        ops=("eq", "gte", "lte"),
        kinds=VISUAL,
        note="the pixels on disk, not what a recipe asked for",
    ),
    Dimension(
        key="media.height",
        label="height",
        group="media",
        carried="facet",
        value_kind="int",
        ops=("eq", "gte", "lte"),
        kinds=VISUAL,
    ),
    Dimension(
        key="media.duration",
        label="length",
        group="media",
        carried="facet",
        value_kind="num",
        ops=("gte", "lte"),
        kinds=TIMED,
        note="seconds",
    ),
    # --- people --------------------------------------------------------
    #: As with `kind`: the scope is what a person's own page links with,
    #: the facet is what the drawer writes -- because "Hannah or Lelly"
    #: and "Hannah AND Lelly" are both real questions about a
    #: photograph, and a scope can express neither.
    Dimension(key="person", label="person", group="people", carried="scope", value_kind="slug", offered=False),
    Dimension(
        key="people.person",
        label="person",
        group="people",
        carried="facet",
        value_kind="id",
        ops=("any", "eq"),
        multi="both",
        note="any of them, or all of them in one picture",
        names="SELECT p.id, p.name FROM person p JOIN json_each(?) ids ON ids.value = p.id",
        discover=(
            "SELECT p.id, COALESCE(p.name, e.slug), COUNT(DISTINCT f.id) FROM file f"
            " JOIN derived_file_person fp ON fp.file_id = f.id"
            " JOIN derived_face_run fr ON fr.id = fp.run_id AND fr.is_primary = 1"
            " JOIN person p ON p.id = fp.person_id"
            " JOIN entity e ON e.id = p.id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY p.id ORDER BY COUNT(DISTINCT f.id) DESC, COALESCE(p.name, e.slug) COLLATE NOCASE"
        ),
    ),
    Dimension(
        key="has.people",
        label="anybody in it",
        group="people",
        carried="facet",
        value_kind="bool",
        discover=_yes_no(
            "EXISTS (SELECT 1 FROM derived_file_person fp"
            " JOIN derived_face_run fr ON fr.id = fp.run_id AND fr.is_primary = 1"
            " WHERE fp.file_id = f.id)"
        ),
    ),
    # --- places --------------------------------------------------------
    Dimension(
        key="place.id",
        label="place",
        group="places",
        carried="facet",
        value_kind="id",
        discover=(
            "SELECT p.id, p.name, COUNT(*) FROM file f"
            " JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = {policy}"
            " JOIN place p ON p.id = mc.place_id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY p.id ORDER BY COUNT(*) DESC, p.name COLLATE NOCASE"
        ),
        names=PLACE_NAMES,
        ops=("any", "eq"),
        multi="both",
    ),
    Dimension(
        key="has.place",
        label="anywhere said",
        group="places",
        carried="facet",
        value_kind="bool",
        discover=_yes_no(
            "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
            " AND mc.policy_version = {policy} AND mc.place_id IS NOT NULL)"
        ),
    ),
    # --- time ----------------------------------------------------------
    Dimension(
        key="context.local_day",
        label="day",
        group="time",
        carried="facet",
        value_kind="date",
        ops=("eq", "gte", "lte"),
        note="the local calendar day, the way the timeline counts one",
    ),
    Dimension(
        key="context.origin",
        label="origin",
        group="time",
        carried="facet",
        value_kind="text",
        note="what the evidence adds up to; `mixed` carries both kinds",
        discover=(
            "SELECT mc.origin, mc.origin, COUNT(*) FROM file f"
            " JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = {policy}"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY mc.origin ORDER BY COUNT(*) DESC, mc.origin"
        ),
        ops=("any", "eq"),
        multi="any",
    ),
    Dimension(
        key="context.disputed",
        label="date disputed",
        group="time",
        carried="facet",
        value_kind="bool",
        discover=_yes_no(
            "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
            " AND mc.policy_version = {policy} AND mc.time_conflicts IS NOT NULL)"
        ),
    ),
    #: The three another surface links WITH, never a person from a list.
    #: A timeline bar opens a half-open window of the human moment at a
    #: granule fine enough for its width; a session opens its own
    #: members. Described so their chips read as words; not offered,
    #: because there is no list of epoch seconds anybody wants.
    Dimension(
        key="context.moment",
        label="moment",
        group="time",
        carried="facet",
        value_kind="int",
        ops=("gte", "lt", "lte"),
        offered=False,
    ),
    Dimension(
        key="context.granule",
        label="claimed within",
        group="time",
        carried="facet",
        value_kind="int",
        ops=("lte",),
        offered=False,
    ),
    Dimension(
        key="event.id",
        label="session",
        group="time",
        carried="facet",
        value_kind="id",
        offered=False,
    ),
    # --- generation ----------------------------------------------------
    Dimension(
        key="has.generation",
        label="AI generated",
        group="generation",
        carried="facet",
        value_kind="bool",
        note="has a recorded recipe -- which `mixed` files do too",
        discover=_yes_no("EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id)"),
    ),
    Dimension(
        key="generation.checkpoint",
        label="checkpoint",
        group="generation",
        carried="facet",
        value_kind="id",
        discover=_artifact_values("checkpoint"),
        names=ARTIFACT_NAMES,
        ops=("any", "eq"),
        multi="both",
    ),
    Dimension(
        key="generation.lora",
        label="LoRA",
        group="generation",
        carried="facet",
        value_kind="id",
        note="choosing two means both were applied",
        discover=_artifact_values("lora"),
        names=ARTIFACT_NAMES,
        ops=("any", "eq"),
        multi="both",
    ),
    Dimension(
        key="generation.workflow",
        label="workflow",
        group="generation",
        carried="facet",
        value_kind="id",
        discover=(
            "SELECT a.id, a.name, COUNT(*) FROM file f"
            " JOIN generation gen ON gen.file_id = f.id"
            " JOIN artifact a ON a.id = gen.workflow_id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY a.id ORDER BY COUNT(*) DESC, a.name COLLATE NOCASE"
        ),
        names=ARTIFACT_NAMES,
        ops=("any", "eq"),
        multi="both",
    ),
    Dimension(
        key="generation.tool",
        label="tool",
        group="generation",
        carried="facet",
        value_kind="text",
        discover=_generation_values("tool"),
        ops=("any", "eq"),
        multi="any",
    ),
    Dimension(
        key="generation.sampler",
        label="sampler",
        group="generation",
        carried="facet",
        value_kind="text",
        discover=_generation_values("sampler"),
        ops=("any", "eq"),
        multi="any",
    ),
    Dimension(
        key="generation.scheduler",
        label="scheduler",
        group="generation",
        carried="facet",
        value_kind="text",
        discover=_generation_values("scheduler"),
        ops=("any", "eq"),
        multi="any",
    ),
    Dimension(
        key="generation.steps",
        label="steps",
        group="generation",
        carried="facet",
        value_kind="int",
        ops=("eq", "gte", "lte"),
        discover=_generation_values("steps"),
    ),
    Dimension(
        key="generation.cfg",
        label="CFG",
        group="generation",
        carried="facet",
        value_kind="num",
        ops=("eq", "gte", "lte"),
        discover=_generation_values("cfg"),
    ),
    Dimension(
        key="generation.denoise",
        label="denoise",
        group="generation",
        carried="facet",
        value_kind="num",
        ops=("eq", "gte", "lte"),
        discover=_generation_values("denoise"),
    ),
    Dimension(
        key="generation.clip_skip",
        label="clip skip",
        group="generation",
        carried="facet",
        value_kind="int",
        ops=("eq", "gte", "lte"),
        discover=_generation_values("clip_skip"),
    ),
    Dimension(
        key="generation.seed",
        label="seed",
        group="generation",
        carried="facet",
        value_kind="int",
    ),
    # --- camera --------------------------------------------------------
    Dimension(
        key="has.capture",
        label="from a camera",
        group="camera",
        carried="facet",
        value_kind="bool",
        discover=_yes_no("EXISTS (SELECT 1 FROM capture cap WHERE cap.file_id = f.id)"),
    ),
    Dimension(
        key="capture.camera",
        label="camera",
        group="camera",
        carried="facet",
        value_kind="id",
        discover=_artifact_values("captured_with"),
        names=ARTIFACT_NAMES,
        ops=("any", "eq"),
        multi="both",
    ),
    Dimension(
        key="capture.lens",
        label="lens",
        group="camera",
        carried="facet",
        value_kind="id",
        discover=_artifact_values("mounted_lens"),
        names=ARTIFACT_NAMES,
        ops=("any", "eq"),
        multi="both",
    ),
    Dimension(
        key="capture.iso",
        label="ISO",
        group="camera",
        carried="facet",
        value_kind="int",
        ops=("eq", "gte", "lte"),
        discover=(
            "SELECT cap.iso, cap.iso, COUNT(*) FROM file f JOIN capture cap ON cap.file_id = f.id"
            " WHERE f.missing_since IS NULL {scope} AND cap.iso IS NOT NULL"
            " GROUP BY cap.iso ORDER BY COUNT(*) DESC, cap.iso"
        ),
    ),
    Dimension(
        key="capture.f_number",
        label="aperture",
        group="camera",
        carried="facet",
        value_kind="num",
        ops=("eq", "gte", "lte"),
    ),
    Dimension(
        key="capture.focal_length",
        label="focal length",
        group="camera",
        carried="facet",
        value_kind="num",
        ops=("eq", "gte", "lte"),
        note="mm",
    ),
    Dimension(
        key="capture.exposure_time",
        label="exposure",
        group="camera",
        carried="facet",
        value_kind="num",
        ops=("eq", "gte", "lte"),
        note="seconds",
    ),
    # --- the long tail ---------------------------------------------------
    Dimension(
        key="param.has",
        label="carries the field",
        group="advanced",
        carried="facet",
        value_kind="text",
        ops=("any", "eq"),
        multi="any",
        note="every metadata key any tool wrote, as the registry recorded it",
        discover=(
            "SELECT fp.key, fp.key, COUNT(DISTINCT f.id) FROM file f"
            " JOIN file_param fp ON fp.file_id = f.id"
            " WHERE f.missing_since IS NULL {scope}"
            " GROUP BY fp.key ORDER BY COUNT(DISTINCT f.id) DESC, fp.key COLLATE NOCASE"
        ),
    ),
    Dimension(
        key="param.is",
        label="field is",
        group="advanced",
        carried="facet",
        value_kind="pair",
        ops=("eq", "any"),
        multi="any",
        note="written key=value, for a field this application has no name of its own for",
    ),
    Dimension(
        key="param.num",
        label="field above or below",
        group="advanced",
        carried="facet",
        value_kind="numpair",
        ops=("eq", "gte", "lte"),
        multi="any",
        note=(
            "written key=number, for a field whose values parsed as numbers"
            " -- `param.is` compares them as text, where 9 is more than 30"
        ),
    ),
)

BY_KEY: dict[str, Dimension] = {one.key: one for one in DIMENSIONS}


def dimension(key: str) -> Dimension | None:
    return BY_KEY.get(key)


def offered() -> tuple[Dimension, ...]:
    """The dimensions a person picks from, in the surface's order."""
    return tuple(one for one in DIMENSIONS if one.offered)


def grouped(kind: str | None = None) -> list[tuple[str, str, list[Dimension]]]:
    """The vocabulary by section, in the surface's order.

    A group with no dimensions is not returned, and neither is one whose
    every dimension is inapplicable to the kind being asked about: an
    empty heading is a promise the application cannot keep.
    """
    made = []
    for name, label in GROUPS:
        held = [one for one in offered() if one.group == name and applies_to(one, kind)]
        if held:
            made.append((name, label, held))
    return made


def applies_to(one: Dimension, kind: str | None) -> bool:
    """Whether a dimension is a fact about this kind of media.

    An audio file has no LoRA and no aperture. Offering those under a
    `kind=audio` question is offering a filter whose every answer is
    empty, which reads as a broken library rather than an inapplicable
    question. `kind=None` is "no kind chosen", where everything applies
    because the answer may hold anything.
    """
    if not one.kinds or kind is None:
        return True
    return kind in one.kinds


# --- what a chip says -------------------------------------------------------


#: How an operator reads in a chip. `eq` says nothing: "sampler Euler"
#: is how a person says it, and "sampler = Euler" is how a database
#: does.
OP_WORDS = {"eq": "", "gte": "from ", "lte": "to ", "lt": "under "}


def spelled_value(one: Dimension, value, named: dict | None = None) -> str:
    """One value, as a person reads it.

    `named` resolves the entity-id kinds -- an artifact or a place is
    stored by id because names move, and a chip that said "#412" would
    be the database's answer to a question a person asked in words.
    """
    if one.value_kind == "bool":
        held = value if isinstance(value, bool) else str(value) not in ("0", "False")
        return "yes" if held else "no"
    if one.value_kind == "id":
        found = (named or {}).get(int(value))
        return str(found) if found else f"#{value} (gone)"
    return str(value)


def chip(one: Dimension, op: str, value, named: dict | None = None) -> str:
    """A whole clause, as a person reads it: `LoRA detail-tweaker`,
    `rating from 4`, `AI generated yes`."""
    return f"{one.label} {OP_WORDS.get(op, op + ' ')}{spelled_value(one, value, named)}".strip()


def chip_any(one: Dimension, values, named: dict | None = None) -> str:
    """Several values of one dimension, OR'd, as ONE clause.

    An OR group is one thing the question says, so it is one chip that
    removes as one. Three chips reading `kind image`, `kind video`,
    `kind audio` would look exactly like three ANDed clauses -- which is
    the opposite question, and one that answers nothing.
    """
    spelled = [spelled_value(one, value, named) for value in values]
    if len(spelled) == 1:
        return f"{one.label} {spelled[0]}"
    return f"{one.label} {', '.join(spelled[:-1])} or {spelled[-1]}"


def chip_all(one: Dimension, values, named: dict | None = None) -> str:
    """Several values of one dimension, ANDed, as ONE chip.

    Said in words, because `LoRA A` beside `LoRA B` is ambiguous to
    read: it is exactly what an OR would look like. "all of" is the
    difference.
    """
    spelled = [spelled_value(one, value, named) for value in values]
    if len(spelled) == 1:
        return f"{one.label} {spelled[0]}"
    return f"{one.label} all of {', '.join(spelled)}"


def unknown_facets() -> tuple[str, ...]:
    """Registered facet keys with no dimension describing them.

    A key the ResultSet can answer and no surface can offer is not a
    secret feature, it is an invisible one. Named here so a test can
    fail on it rather than a person never finding it.
    """
    return tuple(sorted(set(facets_module.REGISTRY) - set(BY_KEY)))
