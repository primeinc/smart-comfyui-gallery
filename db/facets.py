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
    value: int | str


@dataclasses.dataclass(frozen=True)
class _Spec:
    value_kind: str  # 'int' | 'text' | 'date'
    ops: tuple[str, ...]
    #: The predicate over the ResultSet's file alias `f`, with {op}
    #: substituted from _OP_SQL -- structure from this closed registry,
    #: the VALUE always bound.
    template: str
    choices: tuple[str, ...] | None = None


_OP_SQL = {"eq": "=", "gte": ">=", "lte": "<="}

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
    #: The timeline's own door into the gallery: one LOCAL calendar day,
    #: by the same coalesce the timeline aggregates -- the wall clock
    #: when one was claimed, the instant otherwise. /timeline links here
    #: instead of ever becoming a second media membership engine.
    "context.local_day": _Spec(
        "date",
        ("eq", "gte", "lte"),
        # composed around context.HUMAN_MOMENT -- the door and the
        # timeline shelf share one definition of the human day
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy}"
        " AND strftime('%Y-%m-%d', " + HUMAN_MOMENT + ", 'unixepoch') {op} ?)",
    ),
    #: The surface's door: a bin of the human moment, as epoch seconds
    #: on the SAME axis the density is counted on -- so a bar of 14
    #: pictures opens a gallery of exactly those 14.
    "context.moment": _Spec(
        "int",
        ("gte", "lte"),
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
    #: Where it happened, by place entity id: the door a place name opens.
    "place.id": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND mc.place_id {op} ?)",
    ),
    #: A session's door: the members of one CURRENT event -- a run proven
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
}

_INT = re.compile(r"-?\d+")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def facet(key: str, op: str, raw: str) -> Facet:
    """A validated Facet from request-shaped parts. Refusals name the
    rule: an unregistered key, a disallowed operator or a value of the
    wrong shape must fail where it is asked, never become an empty page
    wearing an answer's clothes."""
    spec = REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"nothing is registered to filter by {key!r}; the vocabulary is {', '.join(sorted(REGISTRY))}")
    if op not in spec.ops:
        raise ValueError(f"{key} allows {', '.join(spec.ops)}, not {op!r}")
    if type(raw) is not str:
        raise ValueError(f"{key} takes a spelled value, not {raw!r}")
    if spec.value_kind == "int":
        if _INT.fullmatch(raw) is None:
            raise ValueError(f"{key} takes an integer, not {raw!r}")
        return Facet(key, op, int(raw))
    if spec.value_kind == "date":
        if _DATE.fullmatch(raw) is None:
            raise ValueError(f"{key} takes a day spelled YYYY-MM-DD, not {raw!r}")
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
        raise ValueError(f"a facet is spelled key:op:value, not {spelled!r}")
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
    a scoped surface counts exactly what the gallery's door would open."""
    parts: list[str] = []
    values: list = []
    for one in held:
        sql, value = predicate(one)
        parts.append(sql)
        values.append(value)
    return ("".join(" AND " + part for part in parts), values)


def predicate(held: Facet) -> tuple[str, int | str]:
    """The registered SQL for one facet -- structure from the closed
    registry, the value bound."""
    spec = REGISTRY[held.key]
    # {policy} is the RUNNING interpretation policy, read at call time:
    # after a software upgrade the old rows are honestly invisible here
    # exactly as they are on the timeline, until the context job runs.
    # An int constant from code, never request data -- still structure.
    return spec.template.format(op=_OP_SQL[held.op], policy=int(context.POLICY_VERSION)), held.value
