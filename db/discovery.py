"""What useful refinements are available from here.

The ResultSet answers "which media match?". This answers the other
question a filter surface has to ask: "given what is already being
asked, what could be asked next, and how many would each leave?" They
are different questions and only one of them was implemented, which is
why the gallery's filters were four `<select>` elements -- a control can
only offer what something has counted.

Two rules.

**Membership comes from db/resultset.py and nowhere else.** Every count
here is taken through `resultset.scope_of`, the same conjunct the
gallery walks and the timeline appends. A second definition of "these
videos" is how one surface says 412 and another says 407, and no
amount of care keeps two hand-written WHERE clauses agreeing across a
schema change.

**A dimension's own options are counted with that dimension REMOVED.**
This is disjunctive faceting and it is not a detail. Counted against
the whole question, choosing `Hannah` collapses every other person's
count to "pictures that already contain Hannah", so the list a person
opens to broaden their question can only ever narrow it. Removing the
dimension first means the People counts always mean "people among the
rest of this question", which is what the number is for.
"""

from __future__ import annotations

import dataclasses

from . import context, resultset, vocabulary

#: How many candidate values one dimension offers at once. A checkpoint
#: list is 900 rows in a real library and no one reads 900 rows; the
#: surface searches within a dimension instead, and says what it cut.
MOST = 40


@dataclasses.dataclass(frozen=True)
class Option:
    """One value a dimension could take, and what it would leave."""

    #: The value as the URL spells it.
    value: str
    #: The value as a person reads it.
    label: str
    #: How many media it would leave, from the rest of this question.
    count: int
    #: Whether the question already carries it.
    chosen: bool


@dataclasses.dataclass(frozen=True)
class Options:
    """Everything a surface needs to draw one dimension's list."""

    key: str
    label: str
    options: tuple[Option, ...]
    #: How many values exist beyond the ones returned. Never silently
    #: zero: a truncated list that does not say so reads as a complete
    #: one, and then a model that is in the library looks absent.
    more: int


def without(query: resultset.GalleryQuery, key: str) -> resultset.GalleryQuery:
    """The question with one dimension taken out of it.

    The base a dimension's own options are counted against, and also
    exactly what a chip's remove link needs -- so the two cannot come to
    disagree about what "without this filter" means.
    """
    one = vocabulary.dimension(key)
    if one is None:
        raise ValueError(f"there is no filter named {key!r}")
    if one.carried == "scope":
        return resultset.with_scope(query, key, None)
    return dataclasses.replace(query, facets=tuple(held for held in query.facets if held.key != key))


def chosen_values(query: resultset.GalleryQuery, key: str) -> tuple[str, ...]:
    """What this question already asks of one dimension, as URL values."""
    one = vocabulary.dimension(key)
    if one is None:
        return ()
    if one.carried == "scope":
        held = getattr(query, key, None)
        if held is None:
            return ()
        if isinstance(held, bool):
            return ("1" if held else "0",)
        return (str(held),)
    return tuple(str(held.value) for held in query.facets if held.key == key)


def options(
    conn,
    query: resultset.GalleryQuery,
    key: str,
    *,
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    search: str | None = None,
    most: int = MOST,
) -> Options:
    """The values one dimension could take, counted from here.

    Counted against the question WITHOUT this dimension, so opening a
    list can broaden as well as narrow. A dimension with nothing to
    enumerate -- a free number, a date -- answers with no options, and
    the surface offers a control instead of a list.
    """
    one = vocabulary.dimension(key)
    if one is None:
        raise ValueError(f"there is no filter named {key!r}")
    if one.discover is None:
        return Options(key=key, label=one.label, options=(), more=0)

    base = without(query, key)
    conjunct, args, _ = resultset.scope_of(conn, base, actor_id, models_dir=models_dir, now=now)
    # `.replace`, not `.format`: these statements carry SQL of their own
    # and a stray brace in it would be read as a field.
    sql = one.discover.replace("{scope}", conjunct).replace("{policy}", str(int(context.POLICY_VERSION)))
    # One row past the limit, so "there are more" is observed rather
    # than assumed from a full page.
    rows = list(conn.execute(sql + " LIMIT ?", [*args, most + 1]))

    held = set(chosen_values(query, key))
    made = []
    for value, label, count in rows:
        spelled = str(value)
        shown = str(label)
        if search and search.strip().casefold() not in shown.casefold():
            continue
        made.append(Option(value=spelled, label=shown, count=int(count), chosen=spelled in held))
    over = max(0, len(made) - most)
    return Options(key=key, label=one.label, options=tuple(made[:most]), more=over)


def breakdown(
    conn,
    query: resultset.GalleryQuery,
    key: str,
    *,
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    most: int = MOST,
) -> Options:
    """What one dimension holds ACROSS THIS ANSWER.

    The same statement `options` runs, scoped differently, because they
    answer different questions and the difference is the whole reason
    both exist:

        options    counted WITHOUT this dimension -- "what could I ask
                   next", so the list can widen the question
        breakdown  counted WITH the whole question -- "what is in what I
                   am looking at", which is what an analysis says

    Scoped through `resultset.scope_of` either way, so an analysis and
    the grid beneath it are describing the same media.
    """
    one = vocabulary.dimension(key)
    if one is None:
        raise ValueError(f"there is no filter named {key!r}")
    if one.discover is None:
        return Options(key=key, label=one.label, options=(), more=0)

    conjunct, args, _ = resultset.scope_of(conn, query, actor_id, models_dir=models_dir, now=now)
    sql = one.discover.replace("{scope}", conjunct).replace("{policy}", str(int(context.POLICY_VERSION)))
    rows = list(conn.execute(sql + " LIMIT ?", [*args, most + 1]))
    held = set(chosen_values(query, key))
    made = [
        Option(value=str(value), label=str(label), count=int(count), chosen=str(value) in held)
        for value, label, count in rows
        # a value nothing in this answer carries is not part of what this
        # answer IS; the options list is where "it exists and gives none"
        # belongs
        if int(count) > 0
    ]
    return Options(key=key, label=one.label, options=tuple(made[:most]), more=max(0, len(made) - most))


def labels(conn, query: resultset.GalleryQuery) -> dict[str, dict[int, str]]:
    """The names behind every id-valued clause the question carries.

    An artifact and a place ride the URL as entity ids, because renaming
    a model is a thing people do and a bookmark has to survive it. A
    chip that printed the id would be the database's answer to a
    question a person asked in words. One statement per dimension, over
    the ids that question actually holds -- never a table read.
    """
    import json

    made: dict[str, dict[int, str]] = {}
    for one in vocabulary.DIMENSIONS:
        if one.value_kind != "id" or one.names is None:
            continue
        held = [int(value) for value in chosen_values(query, one.key) if value.lstrip("-").isdigit()]
        if not held:
            continue
        made[one.key] = {int(row[0]): row[1] for row in conn.execute(one.names, (json.dumps(held),))}
    return made


def asked_kind(query: resultset.GalleryQuery) -> str | None:
    """The one medium this question is about, or None.

    Which dimensions apply is decided by the medium being asked about --
    a sound has no aperture -- and the question can now say which medium
    in two places: the `kind=` scope that every bookmark carries, and the
    `media.kind` facet the drawer writes so kinds can be OR'd.

    Exactly ONE kind, from either. Two OR'd kinds mean the answer holds
    both, so a dimension either of them carries still applies; and no
    kind at all means the answer may hold anything.
    """
    if query.kind is not None:
        return query.kind
    held = {str(one.value) for one in query.facets if one.key == "media.kind"}
    return next(iter(held)) if len(held) == 1 else None


def counts(query: resultset.GalleryQuery) -> dict[str, int]:
    """How many clauses the question carries per dimension.

    What the `Filters 6` badge counts and what puts a number beside a
    section heading. Cheap: it reads the question, never the database.
    """
    made: dict[str, int] = {}
    for one in vocabulary.DIMENSIONS:
        held = len(chosen_values(query, one.key))
        if held:
            made[one.key] = held
    return made


#: Whether the phrase somebody typed names a person this library knows, named
#: people only. An unnamed cluster's slug is `person-<short-id>`, which nobody
#: types.
_NAMED_PEOPLE = (
    "SELECT p.name, e.slug FROM person p JOIN entity e ON e.id = p.id WHERE p.name IS NOT NULL AND trim(p.name) <> ''"
)


def person_in(conn, phrase: str) -> tuple[str, str, str] | None:
    """`(name, slug, what is left of the phrase)`, or None.

    "Sarah at the beach" is the thing somebody types, and it fails
    today: the ranking is over image embeddings and the text encoder has
    never heard of Sarah and never will. No amount of captioning fixes
    that -- the answer is the question splitting into a person FILTER,
    which the vocabulary already has, plus the phrase that is left.

    Whole words only, and case-insensitively. A person called `Ana`
    must not match `banana`, and matching a fragment would be a
    suggestion nobody can explain.

    The LONGEST name wins. With both `Ana` and `Ana Torres` in the
    library, "ana torres at the beach" means the second, and offering
    the first would leave "torres at the beach" as a phrase -- a worse
    answer arrived at more confidently.

    Nothing is applied here. This only says a split is available; the
    surface offers it and somebody chooses, because rewriting a typed
    question silently is how a person stops trusting what the box does.
    """
    said = " ".join(phrase.split())
    if not said:
        return None
    folded = said.casefold()
    best: tuple[str, str, str] | None = None
    for name, slug in conn.execute(_NAMED_PEOPLE):
        held = " ".join(str(name).split()).casefold()
        if not held:
            continue
        at = _word_at(folded, held)
        if at is None:
            continue
        rest = " ".join((said[:at] + " " + said[at + len(held) :]).split())
        if best is None or len(held) > len(best[0]):
            best = (str(name), str(slug), rest)
    return best


def _word_at(haystack: str, needle: str) -> int | None:
    """Where `needle` sits in `haystack` as whole words, or None.

    Written out rather than a regex because a person's name is somebody
    else's text: it can hold a bracket, a dot or a plus, and escaping it
    into a pattern is one forgotten call away from a name that matches
    everything or raises.
    """
    at = haystack.find(needle)
    while at != -1:
        before = at == 0 or not haystack[at - 1].isalnum()
        after = at + len(needle) == len(haystack) or not haystack[at + len(needle)].isalnum()
        if before and after:
            return at
        at = haystack.find(needle, at + 1)
    return None
