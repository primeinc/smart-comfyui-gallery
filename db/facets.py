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

from . import context, when
from .context import HUMAN_MOMENT, ORIGINS

#: `CASE mc.time_precision WHEN ... END`, BUILT from `db/when.py SPAN`
#: rather than restated. It was restated: a hand-written CASE listed five
#: precisions and sent everything else to 2147483647, so the day a file
#: could be dated to its month, every folder-dated scan compared as
#: infinitely coarse and was `lte`-excluded from every bin link it
#: belonged in. The vocabulary is `db/schema.sql`'s and the widths are
#: `when.SPAN`'s; neither is this module's to copy.
#:
#: Interpolated, never bound: these are float literals derived from a
#: frozen dict at import, not input, and a CASE cannot take parameters
#: for its branches.
_GRANULE_CASE = (
    "CASE mc.time_precision" + "".join(f" WHEN '{name}' THEN {width!r}" for name, width in when.SPAN.items()) + " END"
)

#: The file kinds, stated here rather than imported from
#: db/resultset.py, which imports this module. Held against
#: `resultset.KINDS` by a test, so the two cannot drift.
KINDS = ("image", "animated_image", "video", "audio", "document")


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
    #: Whether a text value has an identity narrower than the bytes
    #: typed. `tag` does: "Sunset" and "sunset" are one keyword, so both
    #: spellings must fold to one Facet or they are two questions with
    #: two fingerprints, two cached answers and one of them empty.
    folded: bool = False


#: `any` compares like `eq`. What makes it different is not the
#: comparison but the ASSEMBLY: several `any` clauses on one key become
#: one OR'd group (see `clauses`), where several `eq` clauses stay
#: separate conjuncts.
_OP_SQL = {"eq": "=", "any": "=", "gte": ">=", "lt": "<", "lte": "<="}

#: The operator whose repeats mean "or", not "and".
ANY = "any"

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
    #: A word somebody typed on a picture. A FACET rather than a scope
    #: because a picture wears several at once, which makes both
    #: readings real: repeated `eq` asks for all of them, repeated `any`
    #: for any of them, and both are questions people actually ask.
    #: Stored normalised, so the value here is folded to match.
    "tag": _Spec(
        "text",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_tag ft JOIN tag t ON t.id = ft.tag_id WHERE ft.file_id = f.id AND t.tag {op} ?)",
        folded=True,
    ),
    "generation.sampler": _Spec(
        "text",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.sampler {op} ?)",
    ),
    "generation.seed": _Spec(
        "int",
        ("eq",),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.seed {op} ?)",
    ),
    "context.origin": _Spec(
        "text",
        ("eq", "any"),
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
    #: is 86400, a subsecond one a millisecond. `lte:<bin width>` is
    #: exactly "fine enough for this bin" (db/pages.py _FINE_ENOUGH), so a
    #: bar's link opens the pictures the bar counted and no coarser claim
    #: that happens to sit inside its window.
    "context.granule": _Spec(
        "int",
        ("lte",),
        "EXISTS (SELECT 1 FROM derived_media_context mc WHERE mc.file_id = f.id"
        " AND mc.policy_version = {policy} AND " + _GRANULE_CASE + " {op} ?)",
    ),
    #: Where it happened, by place entity id: the link a place name opens.
    "place.id": _Spec(
        "int",
        ("eq", "any"),
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
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'checkpoint' AND fa.artifact_id {op} ?)",
    ),
    "generation.lora": _Spec(
        "int",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'lora' AND fa.artifact_id {op} ?)",
    ),
    #: A workflow hangs off the generation row rather than file_artifact
    #: -- the one place an artifact's own kind changes its relation.
    "generation.workflow": _Spec(
        "int",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.workflow_id {op} ?)",
    ),
    "capture.camera": _Spec(
        "int",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'captured_with' AND fa.artifact_id {op} ?)",
    ),
    "capture.lens": _Spec(
        "int",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id"
        " AND fa.role = 'mounted_lens' AND fa.artifact_id {op} ?)",
    ),
    # --- the recipe, by its numbers ------------------------------------
    "generation.tool": _Spec(
        "text",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM generation gen WHERE gen.file_id = f.id AND gen.tool {op} ?)",
    ),
    "generation.scheduler": _Spec(
        "text",
        ("eq", "any"),
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
    #: WHICH MEDIUM, as a facet, so it can be OR'd.
    #:
    #: `kind=` is also a GalleryQuery scope, and stays one: every
    #: bookmark, every smart collection and every other surface's link
    #: spells it that way, and rewriting them would break all of it to no
    #: one's benefit. But a scope holds exactly ONE value, so "image or
    #: video" was unaskable -- and it is the most ordinary multi-select
    #: there is. The filter surface writes this; the scope keeps working;
    #: both compose, and asking one thing twice is merely redundant.
    "media.kind": _Spec(
        "text",
        ("eq", "any"),
        "f.kind {op} ?",
        choices=KINDS,
    ),
    #: WHO IS IN IT, as a facet, so several people can be asked for at
    #: once -- either "any of these" or "all of these", which are both
    #: real questions about a photograph and mean opposite things.
    #:
    #: Bound to the PRIMARY clustering, exactly as the `person` scope is
    #: (db/resultset.py bind), so the two cannot disagree about which
    #: answer they are reading.
    "people.person": _Spec(
        "int",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM derived_file_person fp"
        " JOIN derived_face_run fr ON fr.id = fp.run_id AND fr.is_primary = 1"
        " WHERE fp.file_id = f.id AND fp.person_id {op} ?)",
    ),
    #: THE LONG TAIL, asked by name.
    #:
    #: The schema records every key any tool emitted into `file_param`
    #: and registers it in `param_key` -- whose own comment says the
    #: registry "is what the facet UI is generated from". Nothing
    #: generated one, so a library full of ComfyUI's own parameters had
    #: no way to ask about any of them.
    #:
    #: These stay OUT of the curated sections on purpose. A dimension is
    #: a fact this application understands well enough to name in a
    #: person's words; a `param_key` row is a string some tool wrote. The
    #: two do not belong in one list, and dumping four hundred discovered
    #: keys into the drawer would bury the twenty that mean something.
    "param.has": _Spec(
        "text",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_param fp WHERE fp.file_id = f.id AND fp.key {op} ?)",
    ),
    #: One key AND what it holds, spelled `key=value`. Two binds, which
    #: is why `predicate` returns a list: the tail is rows, not columns,
    #: so naming a field costs a value of its own.
    "param.is": _Spec(
        "pair",
        ("eq", "any"),
        "EXISTS (SELECT 1 FROM file_param fp WHERE fp.file_id = f.id AND fp.key = ? AND fp.value_text {op} ?)",
    ),
    #: The same long tail, asked as a NUMBER: `steps=30` with `gte`.
    #:
    #: A second spec beside `param.is` rather than widening it, because
    #: they compare different columns. `param.is` reads `fp.value_text`
    #: and can only ever mean equals -- "steps at least 30" over text is
    #: string comparison, where 9 is more than 30. This reads
    #: `fp.value_num`, which the schema fills whenever a value parses as
    #: a number and indexes for exactly this
    #: (`file_param_key_num ON file_param(key, value_num)
    #: WHERE value_num IS NOT NULL`).
    #:
    #: ONE key with three operators rather than the `param.atleast` /
    #: `param.atmost` pair this was first sketched as: the operator is
    #: already the middle of the URL spelling, so a key named "atleast"
    #: carrying `gte` says it twice, and asking "at most" through it
    #: would be unspellable.
    #:
    #: A key whose value never parsed as a number matches nothing here,
    #: which is honest: `value_num` is NULL for those rows and NULL is
    #: not greater than anything. The catalog is what stops it being
    #: OFFERED -- `param_key.value_kind` says which keys are numbers.
    "param.num": _Spec(
        "numpair",
        ("eq", "gte", "lte"),
        "EXISTS (SELECT 1 FROM file_param fp WHERE fp.file_id = f.id AND fp.key = ? AND fp.value_num {op} ?)",
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
    if spec.value_kind == "pair":
        name, sign, held = raw.partition("=")
        if not sign or not name.strip() or not held.strip():
            raise ValueError(f"{key} is written key=value, not {raw!r}")
        return Facet(key, op, f"{name.strip()}={held.strip()}")
    if spec.value_kind == "numpair":
        name, sign, held = raw.partition("=")
        if not sign or not name.strip() or not held.strip():
            raise ValueError(f"{key} is written key=number, not {raw!r}")
        if _NUM.fullmatch(held.strip()) is None:
            raise ValueError(f"{key} compares a number, and {held.strip()!r} is not one")
        # Normalised the way every other number here is, so `steps=30`
        # and `steps=30.0` are not two spellings of one question with two
        # fingerprints.
        made = float(held.strip())
        return Facet(key, op, f"{name.strip()}={int(made) if made.is_integer() else made}")
    value = raw.strip()
    if not value:
        raise ValueError(f"{key} takes a non-empty value")
    if spec.choices is not None and value not in spec.choices:
        raise ValueError(f"{key} is one of {', '.join(spec.choices)}, not {value!r}")
    if spec.folded:
        # Locally imported: a keyword's identity rule belongs with the
        # keyword, and db/authored.py reaches naming and scan, which this
        # module sits underneath.
        from .authored import normalised

        return Facet(key, op, normalised(value))
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


def said(key: str, value, op: str = "eq", named: dict | None = None) -> str:
    """One clause as a person reads it: `kind image`, `favorite yes`,
    `rating from 4`, `keyword beach`, `place Lisbon`.

    `spell` above is the URL form and this is the sentence; they are not
    interchangeable and a surface printing the first is printing a query
    string at a reader. A key with no dimension keeps `key=value`, because
    inventing a label for it here is the drift db/vocabulary.py opens by
    describing.

    `named` resolves the id-valued dimensions. Without it an artifact or a
    place reads `#11 (gone)`, which says the row was deleted about one that
    is sitting there -- worse than the address it replaced. `chips` below
    is how a caller with a connection gets it right.

    It lives here rather than in the surface: an adapter may speak to
    facets, and what a clause is CALLED belongs to the vocabulary, so the
    two are joined on this side of the boundary (sglint SG402).
    """
    from . import vocabulary

    dimension = vocabulary.dimension(key)
    if dimension is None:
        return f"{key}={value}"
    return vocabulary.chip(dimension, op, value, named)


def chips(conn, question, scopes: dict[str, str] | None = None) -> list[str]:
    """Every clause a question carries, as sentences, names resolved.

    One place, so a surface never writes its own `key=value` fallback --
    which is how a page ends up printing `favorite=False` at a person.

    `scopes` names the non-equality comparisons: `rating_min` states a
    bound rather than a value, and reads `rating from 4`.
    """
    from . import discovery, vocabulary

    scopes = scopes or {}
    held = discovery.labels(conn, question)
    said_for = []
    for key, value in (
        ("folder", question.folder),
        ("album", question.album),
        ("person", question.person),
        ("artifact", question.artifact),
        ("kind", question.kind),
        ("favorite", question.favorite),
        ("rating_min", question.rating_min),
    ):
        if value is None:
            continue
        one = vocabulary.dimension(key)
        said_for.append(said(key, value, scopes.get(key, "eq"), held.get(key) if one else None))
    said_for += [said(one.key, one.value, one.op, held.get(one.key)) for one in question.facets]
    return said_for


#: No facets: no conjunct, no values -- what every scoped reader takes
#: by default.
UNSCOPED: tuple[str, list] = ("", [])


def clauses(held) -> list[tuple[str, list]]:
    """Facets as SQL clauses, with `any` REPEATS OR'D TOGETHER.

    Repeating a key is how a question says more than one thing about one
    dimension, and there are two meanings for it:

        eq   repeated -> AND. "this checkpoint AND that LoRA", and
             "both these LoRAs were applied" -- which is the only
             reading that makes sense for a dimension a file can hold
             several of at once.
        any  repeated -> OR. "image or video", "generated or mixed" --
             which is the only reading that makes sense for a dimension
             a file has exactly one of, where AND would ask for a file
             that is two things and always answer nothing.

    One vocabulary cannot pick for both: which is right is a fact about
    the DIMENSION, so it is spelled in the operator and the surface
    chooses it (db/vocabulary.py `multi`).

    Ordering is by first appearance, so the SQL a question produces is
    stable and its plan is comparable between runs.
    """
    order: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[tuple[str, list]]] = {}
    for one in held:
        # `eq` clauses never share a group, so each gets its own key --
        # which is what keeps them separate conjuncts.
        at = (one.key, ANY) if one.op == ANY else (one.key, f"eq:{len(order)}")
        if at not in grouped:
            grouped[at] = []
            order.append(at)
        grouped[at].append(predicate(one))
    made: list[tuple[str, list]] = []
    for at in order:
        parts = grouped[at]
        if len(parts) == 1:
            made.append((parts[0][0], list(parts[0][1])))
            continue
        values: list = []
        for _, bound in parts:
            values.extend(bound)
        made.append(("(" + " OR ".join(sql for sql, _ in parts) + ")", values))
    return made


def conjunction(held) -> tuple[str, list]:
    """Every facet as one SQL conjunct over the ResultSet's file alias
    `f`, values bound: `(" AND p1 AND p2", [v1, v2])`, UNSCOPED when
    there are none. What the timeline appends to its own statements so
    a scoped surface counts exactly what the gallery's link would open."""
    parts: list[str] = []
    values: list = []
    for sql, bound in clauses(held):
        parts.append(sql)
        values.extend(bound)
    return ("".join(" AND " + part for part in parts), values)


def bound_values(held: Facet) -> list:
    """The values one facet binds, in the order its template asks.

    One, for every key whose value is one thing. TWO for a `pair`, whose
    value is `key=value` and whose template asks about a registered
    metadata key AND what it holds -- the long tail is stored as rows,
    not columns, so naming a field there takes a value of its own.
    """
    spec = REGISTRY[held.key]
    if spec.value_kind == "pair":
        name, _, value = str(held.value).partition("=")
        return [name, value]
    if spec.value_kind == "numpair":
        name, _, value = str(held.value).partition("=")
        # Bound as a NUMBER: `value_num` is REAL, and a string bound
        # against it compares by SQLite's type order rather than by
        # magnitude -- every row would be on the wrong side.
        made = float(value)
        return [name, int(made) if made.is_integer() else made]
    return [held.value]


def predicate(held: Facet) -> tuple[str, list]:
    """The registered SQL for one facet -- structure from the closed
    registry, the value or values bound."""
    spec = REGISTRY[held.key]
    # {policy} is the RUNNING interpretation policy, read at call time:
    # after a software upgrade the old rows are honestly invisible here
    # exactly as they are on the timeline, until the context job runs.
    # An int constant from code, never request data -- still structure.
    return spec.template.format(op=_OP_SQL[held.op], policy=int(context.POLICY_VERSION)), bound_values(held)
