"""The typed facet vocabulary: every metadata key the gallery can
filter by, registered once.

The Facet half of the Metadata package. Each key declares its value
type, its legal operators, its parser and its predicate implementation
HERE, so the ResultSet consumes one closed vocabulary instead of
growing a field per key -- and a key added here is automatically a
timed AND a semantic (pre-RRF) constraint, because both ride the one
eligibility construction. Deliberately not a predicate AST: the
vocabulary is registered, closed, and loud about what it does not know.

The URL spelling is `f=key:op:value`, split on exactly two colons so a
value may carry colons of its own; facets normalize -- deduplicated and
canonically ordered -- before they enter a GalleryQuery, so ?f=A&f=B
and ?f=B&f=A are one question with one fingerprint.
"""

from __future__ import annotations

import dataclasses
import re

from . import context
from .context import HUMAN_MOMENT, ORIGINS


@dataclasses.dataclass(frozen=True)
class Facet:
    """One registered metadata predicate: a key from the vocabulary, an
    operator that key allows, and a value of that key's type. Frozen and
    hashable, so it rides the GalleryQuery and its fingerprint."""

    key: str
    op: str
    value: int | float | str


@dataclasses.dataclass(frozen=True)
class _Spec:
    value_kind: str  # 'int' | 'num' | 'text' | 'date'
    ops: tuple[str, ...]
    #: The predicate over the ResultSet's file alias `f`, with {op}
    #: substituted from _OP_SQL -- structure from this closed registry,
    #: the VALUE always bound.
    template: str
    choices: tuple[str, ...] | None = None


_OP_SQL = {"eq": "=", "gte": ">=", "lt": "<", "lte": "<="}

#: The vocabulary, closed: each key names where the fact lives and how
#: it may be asked. Adding a key here is the WHOLE work of adding a
#: gallery filter -- URL spelling, canonical qs, fingerprint, timed
#: walk and pre-RRF eligibility all follow.
REGISTRY: dict[str, _Spec] = {
    "capture.iso": _Spec(
        "int",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM capture cap WHERE cap.file_id = f.id AND cap.iso {op} ?)",
    ),
    "generation.sampler": _Spec(
        "text",
        ("eq",),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.sampler {op} ?)",
    ),
    "generation.seed": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.seed {op} ?)",
    ),
    "context.origin": _Spec(
        "text",
        ("eq",),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND mc.origin {op} ?)",
        choices=ORIGINS,
    ),
    #: The timeline's own link into the gallery: one LOCAL calendar day,
    #: by the same coalesce the timeline aggregates -- the wall clock
    #: when one was claimed, the instant otherwise. /timeline links here
    #: instead of ever becoming a second media membership engine.
    "context.local_day": _Spec(
        "date",
        ("eq", "gte", "lte"),
        # composed around context.HUMAN_MOMENT -- the link and the
        # timeline shelf share one definition of the human day
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy}"
        " AND strftime('%Y-%m-%d', " + HUMAN_MOMENT + ", 'unixepoch') {op} ?)",
    ),
    #: The surface's link: a bin of the human moment, as epoch seconds
    #: on the SAME axis the density is counted on -- so a bar of 14
    #: pictures opens a gallery of exactly those 14. The axis is REAL
    #: (a claimless file's moment is its fractional mtime), so a link is
    #: the half-open [at, at+width) the count uses, never [at, at+width-1].
    "context.moment": _Spec(
        "int",
        ("gte", "lt", "lte"),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND " + HUMAN_MOMENT + " {op} ?)",
    ),
    #: The contested: pictures whose sources disagreed about when (the
    #: context carries named conflicts). 1 asks for the disputed, 0 for
    #: the undisputed -- the timeline's "N contested" opens here.
    "context.disputed": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND (mc.time_conflicts IS NOT NULL) {op} ?)",
    ),
    #: How fine a claim is, in seconds of granule: a day-precision claim
    #: is 86400, a subsecond one 0. `lte:<bin width>` is exactly "fine
    #: enough for this bin" (db/pages.py _FINE_ENOUGH), so a bar's link
    #: opens the pictures the bar counted and no coarser claim that
    #: happens to sit inside its window.
    "context.granule": _Spec(
        "int",
        ("lte",),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND CASE mc.time_precision"
        " WHEN 'subsecond' THEN 0 WHEN 'second' THEN 1 WHEN 'minute' THEN 60 WHEN 'hour' THEN 3600"
        " WHEN 'day' THEN 86400 ELSE 2147483647 END {op} ?)",
    ),
    #: Where it happened, by place entity id: the link a place name opens.
    "place.id": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND mc.place_id {op} ?)",
    ),
    #: A session's link: the members of one CURRENT event -- a run proven
    #: over this interpretation, so a stale hypothesis answers nothing.
    #: The timeline links here instead of growing a membership engine.
    "event.id": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM derived_event_file ef JOIN derived_event ev ON ev.id = ef.event_id"
        " JOIN derived_event_run r ON r.id = ev.run_id WHERE ef.file_id = f.id AND ef.event_id {op} ?"
        " AND r.context_generation = (SELECT generation FROM derived_context_state)"
        " AND r.context_policy_version = {policy})",
    ),
    #: WAS THIS MADE BY A MODEL. Deliberately not `context.origin` --
    #: origin is the interpretation's verdict and has a fourth value,
    #: `mixed`, for a file carrying BOTH capture and generation
    #: evidence. Asking "AI generated" and getting `origin=generated`
    #: silently drops every mixed file, and because repeated facets are
    #: ANDed there is no way to spell "generated OR mixed" either. This
    #: asks the fact instead: is there a generation row. Origin stays
    #: for the forensic question of what the evidence adds up to.
    #:
    #: It also answers before the context job has run, which origin
    #: cannot: the generation row is written by ingest.
    "has.generation": _Spec(
        "int",
        ("eq",),
        "(EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id)) {op} ?",
    ),
    #: The same question for a camera: is there EXIF from a capture.
    "has.capture": _Spec(
        "int",
        ("eq",),
        "(EXISTS (SELECT 1 FROM capture cap WHERE cap.file_id = f.id)) {op} ?",
    ),
    #: Whether anybody is attributed in it, under the PRIMARY clustering
    #: -- the same run the `person` scope means, so "has people" and
    #: "has Hannah" cannot disagree about which answer they are reading.
    "has.people": _Spec(
        "int",
        ("eq",),
        "(EXISTS (SELECT 1 FROM derived_file_person fp"
        " JOIN derived_face_run fr ON fr.id = fp.run_id AND fr.is_primary = 1"
        " WHERE fp.file_id = f.id)) {op} ?",
    ),
    #: Whether the interpretation put it anywhere.
    "has.place": _Spec(
        "int",
        ("eq",),
        "(EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND mc.place_id IS NOT NULL)) {op} ?",
    ),
    # --- the resources, by role ----------------------------------------
    #: An artifact BY ROLE, as a repeatable facet, because the `artifact`
    #: scope holds exactly one and "this checkpoint with that LoRA" is
    #: the ordinary question. The value is the artifact's entity id: the
    #: stable identity, since renaming a model is a thing people do and
    #: a bookmark must survive it. The chip says the name
    #: (db/vocabulary.py), never the number.
    #:
    #: Repeating one key ANDs, so two LoRAs mean "both were applied".
    "generation.checkpoint": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'checkpoint' AND fa.artifact_id {op} ?)",
    ),
    "generation.lora": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'lora' AND fa.artifact_id {op} ?)",
    ),
    #: A workflow hangs off the generation row rather than file_artifact
    #: -- the one place an artifact's own kind changes its relation.
    "generation.workflow": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.workflow_id {op} ?)",
    ),
    "capture.camera": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'captured_with' AND fa.artifact_id {op} ?)",
    ),
    "capture.lens": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'mounted_lens' AND fa.artifact_id {op} ?)",
    ),
    # --- the recipe, by its numbers ------------------------------------
    "generation.tool": _Spec(
        "text",
        ("eq",),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.tool {op} ?)",
    ),
    "generation.scheduler": _Spec(
        "text",
        ("eq",),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.scheduler {op} ?)",
    ),
    "generation.steps": _Spec(
        "int",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.steps {op} ?)",
    ),
    "generation.cfg": _Spec(
        "num",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.cfg {op} ?)",
    ),
    "generation.denoise": _Spec(
        "num",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.denoise {op} ?)",
    ),
    "generation.clip_skip": _Spec(
        "int",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.clip_skip {op} ?)",
    ),
    # --- the camera, by its numbers ------------------------------------
    "capture.f_number": _Spec(
        "num",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM capture cap WHERE cap.file_id = f.id AND cap.f_number {op} ?)",
    ),
    "capture.focal_length": _Spec(
        "num",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM capture cap WHERE cap.file_id = f.id AND cap.focal_length {op} ?)",
    ),
    "capture.exposure_time": _Spec(
        "num",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM capture cap WHERE cap.file_id = f.id AND cap.exposure_time {op} ?)",
    ),
    # --- the bytes, which every medium has -----------------------------
    #: `file.width`/`file.height` are the PIXELS ON DISK, never what a
    #: recipe asked for (that is `generation.width`, and the two
    #: differing is the interesting part -- db/schema.sql says so at the
    #: column).
    "media.width": _Spec(
        "int",
        ("eq", "gte", "lte"),
        "f.width {op} ?",
    ),
    "media.height": _Spec(
        "int",
        ("eq", "gte", "lte"),
        "f.height {op} ?",
    ),
    #: Seconds. A still picture has none and is not a member of any
    #: duration question, which is the honest answer rather than zero.
    "media.duration": _Spec(
        "num",
        ("gte", "lte"),
        "f.duration {op} ?",
    ),
}

_INT = re.compile(r"-?\d+")
#: A real number, written plainly. CFG is 7 and it is also 7.5; a
#: vocabulary that took only integers would refuse half the library's
#: own recipes.
_NUM = re.compile(r"-?\d+(\.\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def facet(key: str, op: str, raw: str) -> Facet:
    """A validated Facet from request-shaped parts. Refusals name the
    rule: an unregistered key, a disallowed operator or a value of the
    wrong shape must fail where it is asked, never become an empty page
    wearing an answer's clothes."""
    spec = REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"there is no filter named {key!r}; the filters are {', '.join(sorted(REGISTRY))}")
    if op not in spec.ops:
        raise ValueError(f"{key} allows {', '.join(spec.ops)}, not {op!r}")
    if type(raw) is not str:
        raise ValueError(f"{key} takes a text value, not {raw!r}")
    if spec.value_kind == "int":
        if _INT.fullmatch(raw) is None:
            raise ValueError(f"{key} takes an integer, not {raw!r}")
        return Facet(key, op, int(raw))
    if spec.value_kind == "num":
        if _NUM.fullmatch(raw) is None:
            raise ValueError(f"{key} takes a number, not {raw!r}")
        # int where it is one, so `cfg:eq:7` and `cfg:eq:7.0` are not two
        # spellings of one question with two fingerprints
        made = float(raw)
        return Facet(key, op, int(made) if made.is_integer() else made)
    if spec.value_kind == "date":
        if _DATE.fullmatch(raw) is None:
            raise ValueError(f"{key} takes a date written YYYY-MM-DD, not {raw!r}")
        return Facet(key, op, raw)
    value = raw.strip()
    if not value:
        raise ValueError(f"{key} takes a non-empty value")
    if spec.choices is not None and value not in spec.choices:
        raise ValueError(f"{key} is one of {', '.join(spec.choices)}, not {value!r}")
    return Facet(key, op, value)


def parse_spelling(spelled: str) -> Facet:
    """The URL spelling: `key:op:value`, exactly two colons of
    structure; the value keeps any colons of its own."""
    parts = spelled.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"a filter is written key:op:value, not {spelled!r}")
    return facet(parts[0], parts[1], parts[2])


def normalized(spellings) -> tuple[Facet, ...]:
    """Request-shaped facet spellings into ONE canonical tuple:
    validated, deduplicated, ordered -- so two orders of the same
    conjunction are one question with one fingerprint."""
    held = [spellings] if isinstance(spellings, str) else list(spellings or ())
    return tuple(sorted({parse_spelling(one) for one in held if one and one.strip()}, key=spell))


def spell(held: Facet) -> str:
    return f"{held.key}:{held.op}:{held.value}"


#: No facets: no conjunct, no values -- what every scoped reader takes
#: by default.
UNSCOPED: tuple[str, list] = ("", [])


def conjunction(held) -> tuple[str, list]:
    """Every facet as one SQL conjunct over the ResultSet's file alias
    `f`, values bound: `(" AND p1 AND p2", [v1, v2])`, UNSCOPED when
    there are none. What the timeline appends to its own statements so
    a scoped surface counts exactly what the gallery's link would open."""
    parts: list[str] = []
    values: list = []
    for one in held:
        sql, value = predicate(one)
        parts.append(sql)
        values.append(value)
    return ("".join(" AND " + part for part in parts), values)


def predicate(held: Facet) -> tuple[str, int | float | str]:
    """The registered SQL for one facet -- structure from the closed
    registry, the value bound."""
    spec = REGISTRY[held.key]
    # {policy} is the RUNNING interpretation policy, read at call time:
    # after a software upgrade the old rows are honestly invisible here
    # exactly as they are on the timeline, until the context job runs.
    # An int constant from code, never request data -- still structure.
    return spec.template.format(op=_OP_SQL[held.op], policy=int(context.POLICY_VERSION)), held.value
