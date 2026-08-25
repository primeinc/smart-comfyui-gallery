"""What the thumbs add up to, and what they cannot be made to say.

Verdicts are the only ground truth this application will ever have about
its own models on THIS library. A benchmark somebody else ran is about
somebody else's pictures; forty judgements over yours are worth more than
a leaderboard, and they are free -- somebody already looked.

The counting is trivial. Everything hard here is refusing to overstate
it, and each refusal is a rule the numbers below obey:

**A verdict set is a BIASED SAMPLE and always will be.** People judge
what they happen to look at, and they reach for `wrong` far more readily
than for `right` -- nobody stops to confirm a caption that is simply
correct. So a raw error rate is unpublishable: "12% wrong" is a
statement about which pictures got looked at. What survives the bias is
a COMPARISON BETWEEN PRODUCERS OVER THE SAME JUDGED FILES, where both
sides were chosen by the same distracted person on the same day.

**Say the n, and say nothing under a floor.** A model is not worse than
another on four verdicts. Below `ENOUGH` the honest answer is how many
more judgements it would take, not a percentage with a wide invisible
error bar.

**An observation is not a cause.** "Wrong more often on video" is a fact
about a correlation in a biased sample. It is worth showing and it is
worth being able to click into; it is not worth a sentence that says
`because`.
"""

from __future__ import annotations

import dataclasses

#: How many verdicts a producer needs before a rate is shown at all.
#:
#: Not a statistical threshold -- with a biased sample there is no honest
#: one -- but the point below which the number is obviously noise and
#: showing it would invite a decision it cannot support.
ENOUGH = 10


@dataclasses.dataclass(frozen=True)
class Judged:
    """What one producer was told about one kind of claim."""

    model_id: str
    model_version: str
    kind: str
    right: int
    wrong: int
    unsure: int

    @property
    def judged(self) -> int:
        return self.right + self.wrong + self.unsure

    @property
    def enough(self) -> bool:
        """Whether a rate may be shown at all."""
        return self.judged >= ENOUGH

    @property
    def wrong_share(self) -> float | None:
        """0..1, or None below the floor. None means "not enough said",
        never zero -- a zero would read as "never wrong"."""
        return None if not self.enough else self.wrong / self.judged

    @property
    def needs(self) -> int:
        """How many more judgements before a rate is shown."""
        return max(0, ENOUGH - self.judged)


_BY_PRODUCER = (
    "SELECT model_id, model_version, annotation_kind, verdict, count(*) FROM feedback"
    " WHERE target_kind = 'annotation' AND model_id IS NOT NULL AND annotation_kind IS NOT NULL"
    " GROUP BY model_id, model_version, annotation_kind, verdict"
)


def by_producer(conn) -> list[Judged]:
    """Every producer that has been judged, most-judged first.

    Read from `feedback` alone and never joined to `derived_annotation`:
    the judgement is the durable half and the annotation is the
    disposable one, so joining would make a rebuild look like people
    changed their minds.
    """
    held: dict[tuple[str, str, str], dict[str, int]] = {}
    for model_id, model_version, kind, verdict, n in conn.execute(_BY_PRODUCER):
        at = held.setdefault((str(model_id), str(model_version), str(kind)), {})
        at[str(verdict)] = at.get(str(verdict), 0) + int(n)
    made = [
        Judged(
            model_id=model_id,
            model_version=model_version,
            kind=kind,
            right=counts.get("right", 0),
            wrong=counts.get("wrong", 0),
            unsure=counts.get("unsure", 0),
        )
        for (model_id, model_version, kind), counts in held.items()
    ]
    made.sort(key=lambda one: (-one.judged, one.model_id, one.model_version))
    return made


@dataclasses.dataclass(frozen=True)
class Contest:
    """Two producers over the files where BOTH were judged.

    The only comparison a biased sample supports. Over all verdicts, a
    model judged forty times and one judged five are not comparable at
    all -- the second's rate is about five pictures somebody happened to
    open. Restricted to the files both were judged on, the person, the
    day and the pictures are shared, and what is left is the difference
    between the models.
    """

    kind: str
    #: (model_id, model_version) -> how many of the shared files it got
    #: WRONG, by the same person's judgement
    wrong: dict[tuple[str, str], int]
    #: how many files both were judged on
    shared: int

    @property
    def enough(self) -> bool:
        return self.shared >= ENOUGH


_CONTESTED = (
    "SELECT annotation_kind, model_id, model_version, file_id, verdict FROM feedback"
    " WHERE target_kind = 'annotation' AND model_id IS NOT NULL"
    "   AND annotation_kind IS NOT NULL AND file_id IS NOT NULL"
)


def contests(conn) -> list[Contest]:
    """Every pair of producers judged over some of the same files.

    In Python rather than SQL: this is a self-join over a small authored
    table whose answer is a handful of rows, and the pairing rule -- the
    intersection of two file sets -- reads as what it is here and as a
    puzzle in SQL.
    """
    seen: dict[str, dict[tuple[str, str], dict[int, str]]] = {}
    for kind, model_id, model_version, file_id, verdict in conn.execute(_CONTESTED):
        by_kind = seen.setdefault(str(kind), {})
        by_kind.setdefault((str(model_id), str(model_version)), {})[int(file_id)] = str(verdict)

    made: list[Contest] = []
    for kind, producers in seen.items():
        named = sorted(producers)
        for at, one in enumerate(named):
            for other in named[at + 1 :]:
                shared = set(producers[one]) & set(producers[other])
                if not shared:
                    continue
                made.append(
                    Contest(
                        kind=kind,
                        wrong={
                            one: sum(1 for f in shared if producers[one][f] == "wrong"),
                            other: sum(1 for f in shared if producers[other][f] == "wrong"),
                        },
                        shared=len(shared),
                    )
                )
    made.sort(key=lambda one: -one.shared)
    return made
