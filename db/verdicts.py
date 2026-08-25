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


#: What a verdict export carries WITHOUT being asked, and the reason the
#: list is this short.
#:
#: Verdicts are the cheapest valuable thing this application accumulates
#: and the easiest to share safely: "this model got these 41 wrong" is
#: an eval set, and it can leave the machine with none of the media
#: leaving with it. So the default is a producer identity, what kind of
#: claim it was, the verdict, the bytes it was about, and when.
#:
#: What is NOT here is the point. No paths, no file names, no folder, no
#: person's name, no embeddings, no note. A note is free text somebody
#: typed and can hold anything at all, so it is opt-in per field rather
#: than something an export decides on their behalf.
#:
#: The content hash IS in the default, deliberately: without something
#: joinable a row cannot be checked against a picture, and an eval set
#: nobody can verify is not one. It names the bytes and nothing else --
#: no path, and no way back to a name.
EXPORTED = ("judged", "verdict", "model_id", "model_version", "annotation_kind", "sha256", "other_sha256", "at")

#: Fields whose VALUE an export withholds until asked for by name.
#:
#: The key is always there and the shape is fixed -- a route's answer has
#: to describe itself or nothing downstream can be typed against it
#: (sglint SG413) -- so what is opt-in is the CONTENT, which is the part
#: that could carry anything. A null note says "not asked for" and
#: carries nothing either way.
BY_REQUEST = ("note",)

#: Every column the export reads, and the shape it hands back. LEFT JOIN
#: because `feedback`'s pointers are ON DELETE SET NULL on purpose: a
#: judgement outlives the derived thing it judged, and a row whose file
#: is gone still says a model got something wrong.
EXPORT = (
    "SELECT f.target_kind AS judged, f.verdict, f.model_id, f.model_version,"
    "       f.annotation_kind, one.content_sha256 AS sha256,"
    "       two.content_sha256 AS other_sha256, f.created_at AS at, f.note"
    "  FROM feedback f"
    "  LEFT JOIN file one ON one.id = f.file_id"
    "  LEFT JOIN file two ON two.id = f.other_file_id"
    " ORDER BY f.created_at, f.id"
)


def exported(conn, *, include: tuple[str, ...] = ()) -> list[dict]:
    """Every verdict, as an eval set that carries no pictures.

    Every row is the same shape; `include` decides whether the fields in
    `BY_REQUEST` carry their VALUE or a null. Anything else is refused
    loudly, because an export that quietly ignored a field somebody
    asked for would hand them a file they believe holds something it
    does not.
    """
    unknown = [one for one in include if one not in BY_REQUEST]
    if unknown:
        raise ValueError(
            f"an export adds {', '.join(BY_REQUEST)} by name and nothing else, not {', '.join(sorted(unknown))}"
        )
    withheld = tuple(one for one in BY_REQUEST if one not in include)
    cursor = conn.execute(EXPORT)
    columns = [c[0] for c in cursor.description]
    return [{k: (None if k in withheld else v) for k, v in zip(columns, row, strict=True)} for row in cursor]


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
class Corrected:
    """How many of a face producer's attributions people took back.

    A COUNT and never a rate, and the reason is the bias rule above
    taken to its limit. A correction is only ever recorded when
    somebody says a person is NOT in a picture, so this producer's
    verdicts are 100% `wrong` by construction -- nobody stops to click
    "yes, that is her" on a face that is simply right. There is no
    denominator, so there is no share to show, and putting one here
    would be the most confidently wrong number in the application.

    What it IS good for: a producer nobody has had to correct, beside
    one corrected forty times, over the same library. That is a
    comparison the bias survives, because the same person was doing the
    same looking for both.
    """

    model_id: str
    model_version: str
    #: attributions taken back by hand
    corrections: int
    #: how many distinct people those corrections were about
    people: int


_CORRECTED = (
    "SELECT model_id, model_version, count(*), count(DISTINCT person_id) FROM feedback"
    " WHERE target_kind = 'person' AND verdict = 'wrong' AND model_id IS NOT NULL"
    " GROUP BY model_id, model_version"
)


def corrections(conn) -> list[Corrected]:
    """Every face producer somebody has corrected, most-corrected first.

    Written by `authored.deny_person`, and only when the denial actually
    withdrew an attribution a run had made: denying a person no run put
    there judges nothing, and counting it would put a correction against
    a producer that never spoke.
    """
    made = [
        Corrected(
            model_id=str(model_id),
            model_version=str(model_version),
            corrections=int(n),
            people=int(distinct),
        )
        for model_id, model_version, n, distinct in conn.execute(_CORRECTED)
    ]
    made.sort(key=lambda one: (-one.corrections, one.model_id, one.model_version))
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
