"""What is IN this answer.

The gallery could produce a result set and then offered exactly one
thing to do with it: look at thumbnails. "Show me every video that was
generated, and tell me which prompts and which LoRAs made them" is the
obvious next question and there was nowhere to ask it.

So: a question has ONE membership and several presentations.

    GalleryQuery
        |
        +---> ResultSet ---> ordered members
                |
                +--> Gallery    the pictures
                +--> Analyze    what the pictures are made of

Two rules, and the first is the one that matters.

**Analysis reads the same membership as the grid.** Every count here is
taken through `resultset.scope_of`. Nothing in this module writes a
WHERE clause about which media are included, because the moment two
places decide that, one of them says 412 videos and the other says 407
and no one can tell which is lying.

**Every number is a question you can ask.** A count that cannot be
clicked back into the query is a dashboard, and a dashboard is where
data goes to be looked at instead of used. Each row carries the
dimension key and value that would add it to the question, so the
surface refines rather than reports.

What is deliberately NOT here: word frequencies over prompt text. Exact
prompt identity is a fact -- `prompt.text_hash` is a real column and two
files carrying one prompt genuinely share one -- and counting the word
"cinematic" is a different claim wearing the same clothes. The second is
worth building; conflating them is not.
"""

from __future__ import annotations

import dataclasses
import re

from . import discovery, resultset, vocabulary

#: How many rows one breakdown carries. Past this a chart is decoration.
MOST = 12

#: How many distinct prompts a panel lists at once.
PROMPTS_MOST = 40

#: The dimensions worth breaking an answer down by, in the order they
#: answer "what am I looking at". Curated rather than "every dimension",
#: because thirty charts is not an analysis, it is a wall.
BREAKDOWNS: tuple[str, ...] = (
    # `media.kind`, not the `kind` scope: the scope holds one value and
    # has no value list, so breaking an answer down by it counted
    # nothing at all.
    "media.kind",
    "has.generation",
    "people.person",
    "generation.checkpoint",
    # NOT `generation.lora`: LoRAs have their own panel, with the weight
    # each was applied at, which is a strictly better telling of the same
    # fact. Two panels counting one thing side by side is not twice the
    # information, it is a reader wondering which one is wrong.
    "generation.tool",
    "generation.sampler",
    "generation.scheduler",
    "generation.steps",
    "generation.cfg",
    "capture.camera",
    "capture.lens",
    "capture.iso",
    "place.id",
    "context.origin",
)


@dataclasses.dataclass(frozen=True)
class Share:
    """One row of a breakdown, and the question it would make."""

    #: what the URL would carry to narrow to this
    value: str
    label: str
    count: int
    #: of the answer, as a percentage 0..100 -- computed here so a bar
    #: and its number cannot disagree
    share: float
    #: whether the question already holds it
    chosen: bool


@dataclasses.dataclass(frozen=True)
class Breakdown:
    """One dimension, across the answer."""

    key: str
    label: str
    #: how many members carry this dimension at all -- NOT the answer's
    #: total. "18 of 684 have a camera" and "18 of 18 cameras are a
    #: Canon" are different sentences and a surface needs both.
    covered: int
    rows: tuple[Share, ...]
    more: int


@dataclasses.dataclass(frozen=True)
class PromptUse:
    """One exact prompt, and how much of this answer used it."""

    id: int
    text: str
    role: str
    uses: int


@dataclasses.dataclass(frozen=True)
class TermUse:
    """One recurring term, and how many files asked for it.

    `files`, never `uses`: a term written three times in one prompt is
    one file that wanted it.
    """

    term: str
    files: int


@dataclasses.dataclass(frozen=True)
class Weighted:
    """One LoRA, how often, and at what strength.

    The weight is why this is not a list of names: a LoRA without its
    number does not reproduce the picture. Median rather than mean --
    one picture at 1.5 should not move what "usually" means.
    """

    id: int
    name: str
    uses: int
    typical: float | None
    lowest: float | None
    highest: float | None


@dataclasses.dataclass(frozen=True)
class Analysis:
    """Everything said about one answer."""

    total: int
    breakdowns: tuple[Breakdown, ...]
    prompts: tuple[PromptUse, ...]
    #: The terms that recur across those prompts. A separate field, not
    #: folded into `prompts`, because it is a separate CLAIM: an exact
    #: prompt count is a fact and a term count is a reading of a comma
    #: convention, and mixing them would let the reading borrow the
    #: fact's certainty.
    terms: tuple[TermUse, ...]
    loras: tuple[Weighted, ...]
    #: How many distinct prompts the answer holds, beyond those listed.
    more_prompts: int
    #: And how many more terms.
    more_terms: int


_PROMPTS = (
    "SELECT p.id, p.text, gp.role, COUNT(*) FROM file f"
    " JOIN generation_prompt gp ON gp.file_id = f.id"
    " JOIN prompt p ON p.id = gp.prompt_id"
    " WHERE f.missing_since IS NULL {scope} AND gp.role = ?"
    " GROUP BY p.id, gp.role ORDER BY COUNT(*) DESC, p.id LIMIT ?"
)

#: The prompt TEXTS this answer used, with how many files each. The
#: terms are counted in Python because splitting one is a reading of a
#: convention rather than a fact the schema holds -- see `terms`.
_PROMPT_TEXTS = (
    "SELECT p.text, COUNT(*) FROM file f"
    " JOIN generation_prompt gp ON gp.file_id = f.id"
    " JOIN prompt p ON p.id = gp.prompt_id"
    " WHERE f.missing_since IS NULL {scope} AND gp.role = ?"
    " GROUP BY p.id"
)

#: Every LoRA in the answer, with the strengths it was applied AT.
#:
#: The weights come back as one column and the middle one is taken here.
#: SQLite has no percentile function, and the obvious spelling -- a
#: correlated subquery ordering one artifact's weights and taking the
#: middle by OFFSET -- needs to reach the outer alias from two levels
#: down, which SQLite refuses ("no such column: a.id"). One column and
#: one line of Python is the honest version of the same answer.
_LORAS = (
    "SELECT a.id, a.name, COUNT(DISTINCT f.id),"
    " group_concat(fa.model_weight), MIN(fa.model_weight), MAX(fa.model_weight)"
    " FROM file f JOIN file_artifact fa ON fa.file_id = f.id AND fa.role = 'lora'"
    " JOIN artifact a ON a.id = fa.artifact_id"
    " WHERE f.missing_since IS NULL {scope}"
    " GROUP BY a.id ORDER BY COUNT(DISTINCT f.id) DESC, a.name COLLATE NOCASE LIMIT ?"
)


def _median(spelled: str | None) -> float | None:
    """The middle weight, or nothing when none was recorded.

    Median rather than mean: one picture at 1.5 must not move what
    "usually" means for a LoRA applied at 0.8 forty times.
    """
    if not spelled:
        return None
    held = sorted(float(one) for one in spelled.split(",") if one not in ("", "None"))
    if not held:
        return None
    middle = len(held) // 2
    if len(held) % 2:
        return held[middle]
    return (held[middle - 1] + held[middle]) / 2


def _scoped(conn, query, actor_id, models_dir, now) -> tuple[str, list]:
    conjunct, args, _ = resultset.scope_of(conn, query, actor_id, models_dir=models_dir, now=now)
    return conjunct, args


def prompts(
    conn,
    query: resultset.GalleryQuery,
    *,
    role: str = "effective",
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    most: int = PROMPTS_MOST,
) -> tuple[tuple[PromptUse, ...], int]:
    """The exact prompts this answer used, most-used first.

    EXACT, not similar. `prompt.text_hash` is a real identity and two
    files carrying one prompt share one row, so this is a count and not
    an estimate. Grouping by meaning is a different feature with a
    different error mode and it does not get to borrow this one's
    certainty.
    """
    conjunct, args = _scoped(conn, query, actor_id, models_dir, now)
    rows = list(conn.execute(_PROMPTS.replace("{scope}", conjunct), [*args, role, most + 1]))
    made = tuple(PromptUse(id=int(one[0]), text=str(one[1]), role=str(one[2]), uses=int(one[3])) for one in rows[:most])
    return made, max(0, len(rows) - most)


#: Weight syntax, stripped so `(rim light:1.3)` and `rim light` are one
#: term. The number is a strength, not part of what was asked for.
_WEIGHTED = re.compile(r"^[(\[{]+\s*(.*?)\s*(?::\s*-?\d+(?:\.\d+)?\s*)?[)\]}]+$")

#: A LoRA or embedding reference, ANYWHERE in a fragment. Not a term: it
#: is an ARTIFACT, counted by `loras` with the strengths it was used at,
#: and letting it in here would report the same fact twice under two
#: different kinds of claim.
#:
#: Removed rather than rejected, because it does not arrive on its own.
#: A1111 writes `a castle on a hill <lora:filmGrain:0.35>` -- one
#: comma-free fragment with the reference glued to the end of it -- so a
#: rule that only refused a fragment that was ENTIRELY a reference left
#: every one of them inside a term.
_ARTIFACT = re.compile(r"<[^>]*>")


def term_of(raw: str) -> str | None:
    """One prompt fragment as the term it names, or None for no term.

    This is the interpretation `prompts` deliberately refuses to make,
    kept in one place so it can be read and disagreed with:

    - **Commas separate terms.** The convention every diffusion UI's
      prompt box follows, and it is a convention rather than a grammar:
      a sentence prompt with commas in it splits into clauses, which is
      wrong, and there is no way to tell the two apart from the text.
    - **Weights are not part of the term.** `(rim light:1.3)` is the
      same thing asked for as `rim light`, more loudly.
    - **An `<...>` reference is not a term, wherever it sits.** It names
      an artifact, and `loras` already counts those WITH their strengths
      -- counting it here too would report one fact twice under two
      kinds of claim. It is REMOVED rather than refused, because it does
      not arrive alone: A1111 writes it glued to the end of the prose.
    - **Case is not meaning.** Folded, so `Rim Light` and `rim light`
      are one term.
    """
    held = _ARTIFACT.sub(" ", raw).strip()
    for _ in range(4):
        # Nested wrapping: `((rim light:1.3))`. Bounded rather than
        # `while`, because a prompt is somebody's text and an unbalanced
        # bracket must not become a loop.
        found = _WEIGHTED.match(held)
        if found is None:
            break
        held = found.group(1).strip()
    if not held:
        return None
    # The reference may have left two spaces where it was.
    return " ".join(held.split()).casefold()


def terms(
    conn,
    query: resultset.GalleryQuery,
    *,
    role: str = "effective",
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    most: int = MOST,
) -> tuple[tuple[TermUse, ...], int]:
    """The terms that RECUR across this answer, most-used first.

    A different claim from `prompts`, with a different error mode, and
    the reason the two are separate panels rather than one.

    `prompts` is a count: `prompt.text_hash` is a real identity, two
    files carrying one prompt share one row, and "twelve files used this
    prompt" is a fact. This is a READING: it assumes commas separate
    terms, which is a convention every prompt box follows and no grammar
    enforces. A prompt written as a sentence splits into clauses here and
    is counted as terms nobody asked for.

    So it says `files`, not `uses`: a term appearing three times in one
    prompt is one file that wanted it, and counting the repeats would
    make a habit of typing look like a habit of generating.
    """
    conjunct, args = _scoped(conn, query, actor_id, models_dir, now)
    counted: dict[str, int] = {}
    for text, files in conn.execute(_PROMPT_TEXTS.replace("{scope}", conjunct), [*args, role]):
        # Per PROMPT, so one prompt saying "sunset" twice is one wanting.
        seen = {held for held in (term_of(one) for one in str(text).split(",")) if held}
        for one in seen:
            counted[one] = counted.get(one, 0) + int(files)
    ranked = sorted(counted.items(), key=lambda one: (-one[1], one[0]))
    made = tuple(TermUse(term=term, files=files) for term, files in ranked[:most])
    return made, max(0, len(ranked) - most)


def loras(
    conn,
    query: resultset.GalleryQuery,
    *,
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    most: int = MOST,
) -> tuple[Weighted, ...]:
    """Every LoRA this answer used, with the strengths it was used at."""
    conjunct, args = _scoped(conn, query, actor_id, models_dir, now)
    rows = conn.execute(_LORAS.replace("{scope}", conjunct), [*args, most])
    return tuple(
        Weighted(
            id=int(one[0]),
            name=str(one[1]),
            uses=int(one[2]),
            typical=_median(one[3]),
            lowest=None if one[4] is None else float(one[4]),
            highest=None if one[5] is None else float(one[5]),
        )
        for one in rows
    )


def analyze(
    conn,
    query: resultset.GalleryQuery,
    total: int,
    *,
    actor_id: int | None = None,
    models_dir: str | None = None,
    now: float | None = None,
    keys: tuple[str, ...] = BREAKDOWNS,
) -> Analysis:
    """What this answer is made of.

    `total` is the ResultSet's own count, passed in rather than counted
    again here: the share a bar shows is a fraction of THE ANSWER, and
    an analysis that counted its own denominator could show 108%.
    """
    made: list[Breakdown] = []
    for key in keys:
        one = vocabulary.dimension(key)
        if one is None:
            continue
        held = discovery.breakdown(conn, query, key, actor_id=actor_id, models_dir=models_dir, now=now, most=MOST)
        if not held.options:
            continue
        # A breakdown with ONE value covering the whole answer says
        # nothing: asking `has.generation=1` and being shown "AI
        # generated: yes, 100%" is the question read back as an answer.
        # It is dropped rather than drawn, which is also what stops a
        # filtered analysis being half bars at 100%.
        if len(held.options) == 1 and held.options[0].count >= total:
            continue
        # How many members carry this dimension AT ALL. Not the answer's
        # total: "18 of 684 have a camera" and "18 of 18 are a Canon" are
        # different sentences, and a share drawn against the wrong one is
        # a bar that lies quietly.
        covered = sum(each.count for each in held.options)
        made.append(
            Breakdown(
                key=key,
                label=one.label,
                covered=covered,
                rows=tuple(
                    Share(
                        value=each.value,
                        label=each.label,
                        count=each.count,
                        share=round(100.0 * each.count / total, 1) if total else 0.0,
                        chosen=each.chosen,
                    )
                    for each in held.options
                ),
                more=held.more,
            )
        )
    said, over = prompts(conn, query, actor_id=actor_id, models_dir=models_dir, now=now)
    recurring, over_terms = terms(conn, query, actor_id=actor_id, models_dir=models_dir, now=now)
    return Analysis(
        total=total,
        breakdowns=tuple(made),
        prompts=said,
        terms=recurring,
        loras=loras(conn, query, actor_id=actor_id, models_dir=models_dir, now=now),
        more_prompts=over,
        more_terms=over_terms,
    )
