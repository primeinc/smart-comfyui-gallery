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

from . import discovery, resultset, vocabulary

#: How many rows one breakdown carries. Past this a chart is decoration.
MOST = 12

#: How many distinct prompts a panel lists at once.
PROMPTS_MOST = 40

#: The dimensions worth breaking an answer down by, in the order they
#: answer "what am I looking at". Curated rather than "every dimension",
#: because thirty charts is not an analysis, it is a wall.
BREAKDOWNS: tuple[str, ...] = (
    "kind",
    "has.generation",
    "generation.checkpoint",
    "generation.lora",
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
    loras: tuple[Weighted, ...]
    #: How many distinct prompts the answer holds, beyond those listed.
    more_prompts: int


_PROMPTS = (
    "SELECT p.id, p.text, gp.role, COUNT(*) FROM file f"
    " JOIN generation_prompt gp ON gp.file_id = f.id"
    " JOIN prompt p ON p.id = gp.prompt_id"
    " WHERE f.missing_since IS NULL {scope} AND gp.role = ?"
    " GROUP BY p.id, gp.role ORDER BY COUNT(*) DESC, p.id LIMIT ?"
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
    return Analysis(
        total=total,
        breakdowns=tuple(made),
        prompts=said,
        loras=loras(conn, query, actor_id=actor_id, models_dir=models_dir, now=now),
        more_prompts=over,
    )
