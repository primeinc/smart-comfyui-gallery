"""What a person wrote down, and the rules that keep it.

Ratings, comments, favourites, albums, named people and verdicts on what a
model said cannot be reproduced from the files. `derived_*` rows can be
recomputed; these cannot, at any cost. The operations here are narrow and
every deletion is one a person asked for by name.

`person` sits here rather than with the face pipeline. A cluster can be
recomputed; the name a person gave it cannot, so the name lives on the
person and `person_assertion` records the claim against the file.
Recomputing the derived tables leaves both standing.

The export queries under "taking it with you" are keyed by `content_sha256` and
never by a row id, which is the whole difference between an export and a dump:
ids belong to one database file and mean nothing in the next one, while a hash
names the same photograph in any library that holds it, so what comes back reads
against a rebuilt library, a moved one, or somebody else's copy of the same
pictures. `SAID` lists only the files carrying something authored, because a
library is mostly pictures nobody has said anything about and a page of
`rating: null` buries the rows that are the point. `APPEARS` carries `stance`,
because a negative is a claim here rather than the absence of one: "not her" has
to survive a rebuild and constrain the next clustering exactly as a positive
does, and an export that dropped the negatives hands back a library that makes
the same mistake again. `KEYWORDED` is its own query rather than a
`group_concat` beside the collections, because a keyword holds SPACES -- "new
york" is one word here -- while the collections ride a space-joined list of slugs
that splits back apart; it carries both `tag`, the identity a filter is built
from, and `label`, what somebody typed, so a reader never has to re-fold one into
the other and possibly fold it differently.
"""

from __future__ import annotations

import dataclasses

from .naming import rename
from .scan import mint

# --- who is using it -------------------------------------------------------


def add_user(conn, username: str, password_hash: str, role: str, now: float) -> int:
    conn.execute(
        "INSERT INTO user(username, password_hash, role, created_at) VALUES(?, ?, ?, ?)",
        (username, password_hash, role, now),
    )
    return conn.execute("SELECT id FROM user WHERE username = ?", (username,)).fetchone()[0]


def local_actor(conn, now: float) -> int:
    """The one local authored identity, resolved at application start.

    Ratings and favorites are per-user by schema, and "user_id = 1"
    hard-coded at a write site is how that stops being true. This is
    the single place the local-first deployment answers "who is
    writing": the first registered user, created if the library has
    none. When real sessions arrive, the request's actor replaces this
    resolution while every authored signature stays as it is. The
    password hash is '!' -- not a hash of anything, so nothing can log
    in as the implicit identity."""
    row = conn.execute("SELECT id FROM user ORDER BY id LIMIT 1").fetchone()
    if row:
        return row[0]
    return add_user(conn, "local", "!", "ADMIN", now)


# --- judgements about one picture -----------------------------------------


#: One statement per fact, shared by the one-item and many-item shapes --
#: two spellings of an upsert is where their semantics quietly fork.
_RATE = (
    "INSERT INTO rating(file_id, user_id, rating, created_at) VALUES(?, ?, ?, ?)"
    " ON CONFLICT(file_id, user_id) DO UPDATE SET rating = excluded.rating"
)
_UNRATE = "DELETE FROM rating WHERE file_id = ? AND user_id = ?"
_FAVOURITE = "INSERT OR IGNORE INTO favorite(file_id, user_id, created_at) VALUES(?, ?, ?)"
_UNFAVOURITE = "DELETE FROM favorite WHERE file_id = ? AND user_id = ?"


def rate(conn, file_id: int, user_id: int, stars: int, now: float) -> None:
    """One rating per person per picture; rating again replaces their own."""
    conn.execute(_RATE, (file_id, user_id, stars, now))


def unrate(conn, file_id: int, user_id: int) -> None:
    conn.execute(_UNRATE, (file_id, user_id))


def favourite(conn, file_id: int, user_id: int, now: float) -> None:
    conn.execute(_FAVOURITE, (file_id, user_id, now))


def unfavourite(conn, file_id: int, user_id: int) -> None:
    conn.execute(_UNFAVOURITE, (file_id, user_id))


# --- one picture's authored state, as desired facts ------------------------

# The write interface is DESIRED STATE, never a toggle, and every operation is
# idempotent by the primitives beneath it: a retried "favorite = true" lands
# where it already was, where a retried toggle lands on the opposite.


@dataclasses.dataclass(frozen=True)
class MediaAuthoredState:
    """What this actor has written down about one file."""

    favorite: bool
    rating: int | None
    collections: tuple[dict, ...]  # {"slug", "name"}, static memberships by name
    tags: tuple[dict, ...]  # {"tag", "label"}, shared keywords, not this actor's


def media_state(conn, file_id: int, user_id: int) -> MediaAuthoredState:
    return MediaAuthoredState(
        favorite=conn.execute("SELECT 1 FROM favorite WHERE file_id = ? AND user_id = ?", (file_id, user_id)).fetchone()
        is not None,
        rating=(
            row[0]
            if (
                row := conn.execute(
                    "SELECT rating FROM rating WHERE file_id = ? AND user_id = ?", (file_id, user_id)
                ).fetchone()
            )
            else None
        ),
        collections=tuple(
            {"slug": slug, "name": name}
            for slug, name in conn.execute(
                "SELECT e.slug, c.name FROM collection_file cf"
                "  JOIN collection c ON c.id = cf.collection_id"
                "  JOIN entity e ON e.id = c.id"
                " WHERE cf.file_id = ? ORDER BY c.name COLLATE NOCASE",
                (file_id,),
            )
        ),
        tags=tags_of(conn, file_id),
    )


def _rating(value) -> int | None:
    """Exact-integer rating semantics, once: `None` clears, 1..5 sets,
    and Python's bool-IS-an-int coercion never turns JSON true into one
    star (the same trap the rule validator closes)."""
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError(f"a rating is 1..5 stars or null to clear, not {value!r}")
    return value


def set_favorite_many(conn, file_ids, user_id: int, value: bool, now: float) -> None:
    """The desired fact over MANY files, one statement -- the single-item
    interface delegates here, so there is exactly one implementation for
    two adapters to share."""
    if value:
        conn.executemany(_FAVOURITE, [(file_id, user_id, now) for file_id in file_ids])
    else:
        conn.executemany(_UNFAVOURITE, [(file_id, user_id) for file_id in file_ids])


def set_rating_many(conn, file_ids, user_id: int, value: int | None, now: float) -> None:
    held = _rating(value)
    if held is None:
        conn.executemany(_UNRATE, [(file_id, user_id) for file_id in file_ids])
    else:
        conn.executemany(_RATE, [(file_id, user_id, held, now) for file_id in file_ids])


def set_favorite(conn, file_id: int, user_id: int, value: bool, now: float) -> None:
    set_favorite_many(conn, (file_id,), user_id, value, now)


def set_place_many(conn, file_ids, user_id: int, place_id: int | None, now: float) -> None:
    """Where these pictures happened, as desired state: one place per
    file, replaced on a change of mind, None to withdraw. The context
    is re-interpreted by the caller (db/context.py rebuild_many) so the
    rows read through at once."""
    ids = [(int(one),) for one in file_ids]
    if place_id is None:
        conn.executemany("DELETE FROM file_place WHERE file_id = ?", ids)
        return
    conn.executemany(
        "INSERT INTO file_place(file_id, place_id, user_id, asserted_at) VALUES(?, ?, ?, ?)"
        " ON CONFLICT(file_id) DO UPDATE SET place_id = excluded.place_id, user_id = excluded.user_id,"
        " asserted_at = excluded.asserted_at",
        [(file_id, place_id, user_id, now) for (file_id,) in ids],
    )


def set_place(conn, file_id: int, user_id: int, place_id: int | None, now: float) -> None:
    set_place_many(conn, [file_id], user_id, place_id, now)


def set_rating(conn, file_id: int, user_id: int, value: int | None, now: float) -> None:
    """`None` clears; 1..5 sets. Validated in `_rating` so every caller
    gets the same refusal instead of a CHECK-constraint traceback."""
    set_rating_many(conn, (file_id,), user_id, value, now)


#: A pasted paragraph is not a keyword. The cap is on the NORMALISED
#: form, so it counts characters somebody meant rather than the spaces
#: between them.
LONGEST_TAG = 100


def normalised(name: str) -> str:
    """The identity of a keyword: whitespace collapsed, case folded.

    Case is not identity -- "New York" and "new york" are one keyword,
    and a library that let them split would be answering two different
    questions with the same word. Folded here rather than by COLLATE
    NOCASE, which folds ASCII only: SQLite would keep CAFE and cafe
    together and let CAFE and cafE apart, which is worse than either
    rule applied consistently.
    """
    held = " ".join(name.split())
    if not held:
        raise ValueError("a keyword needs a word in it")
    folded = held.casefold()
    if len(folded) > LONGEST_TAG:
        raise ValueError(f"a keyword is at most {LONGEST_TAG} characters, not {len(folded)}")
    return folded


def tag(conn, name: str, now: float) -> int:
    """The keyword by that name, minted if this library has never held
    it. The spelling somebody typed becomes the label the FIRST time and
    is left alone after: whoever wrote it down first named it, and a
    later `set_tag` should not silently restyle a word on every page
    that shows it."""
    held = normalised(name)
    found = conn.execute("SELECT id FROM tag WHERE tag = ?", (held,)).fetchone()
    if found:
        return int(found[0])
    return int(
        conn.execute(
            "INSERT INTO tag(tag, label, created_at) VALUES(?, ?, ?) RETURNING id",
            (held, " ".join(name.split()), now),
        ).fetchone()[0]
    )


def set_tag_many(conn, file_ids, user_id: int, name: str, value: bool, now: float) -> None:
    """This keyword on these pictures, or off them, as desired state.

    Removing the last picture from a keyword removes the keyword. A word
    with nothing under it is not a vocabulary somebody built, it is the
    typo they just corrected -- and left standing it would haunt the
    filter menu forever, where the cost of being wrong is a list nobody
    trusts. Typing it again is what recreates it.
    """
    ids = [(int(one),) for one in file_ids]
    if value:
        tag_id = tag(conn, name, now)
        conn.executemany(
            "INSERT OR IGNORE INTO file_tag(file_id, tag_id, user_id, created_at) VALUES(?, ?, ?, ?)",
            [(file_id, tag_id, user_id, now) for (file_id,) in ids],
        )
        return
    held = normalised(name)
    found = conn.execute("SELECT id FROM tag WHERE tag = ?", (held,)).fetchone()
    if not found:
        return
    tag_id = int(found[0])
    conn.executemany("DELETE FROM file_tag WHERE file_id = ? AND tag_id = ?", [(f, tag_id) for (f,) in ids])
    conn.execute(
        "DELETE FROM tag WHERE id = ? AND NOT EXISTS (SELECT 1 FROM file_tag WHERE tag_id = ?)", (tag_id, tag_id)
    )


def set_tag(conn, file_id: int, user_id: int, name: str, value: bool, now: float) -> None:
    set_tag_many(conn, (file_id,), user_id, name, value, now)


def rename_tag(conn, name: str, to: str, now: float) -> int:
    """Rename a keyword, folding it into the one already there if that
    name is taken.

    The whole reason a tag is not an entity: renaming is this, rather
    than a retired address somebody may still hold a bookmark to. A fold
    is the ordinary case -- somebody typed "beach" and "Beaches" over a
    year and is now saying they were always the same word -- so a
    collision merges rather than refusing, and the pictures under both
    end up under one.
    """
    held, want = normalised(name), normalised(to)
    found = conn.execute("SELECT id FROM tag WHERE tag = ?", (held,)).fetchone()
    if not found:
        raise ValueError(f"no keyword {name!r} to rename")
    from_id = int(found[0])
    onto = conn.execute("SELECT id FROM tag WHERE tag = ?", (want,)).fetchone()
    if onto is None or int(onto[0]) == from_id:
        conn.execute("UPDATE tag SET tag = ?, label = ? WHERE id = ?", (want, " ".join(to.split()), from_id))
        return from_id
    onto_id = int(onto[0])
    # OR IGNORE, not a plain UPDATE: a picture already under both words
    # would collide on (file_id, tag_id), and the honest result of "these
    # were always one word" is one row rather than a refusal.
    conn.execute("UPDATE OR IGNORE file_tag SET tag_id = ? WHERE tag_id = ?", (onto_id, from_id))
    conn.execute("DELETE FROM tag WHERE id = ?", (from_id,))
    return onto_id


def forget_tag(conn, name: str, *, expecting: int | None = None) -> int:
    """Take a keyword off every picture and delete it. Returns how many
    pictures wore it.

    The one gesture in the keyword vocabulary that destroys authored
    work: retyping a word on two hundred pictures is not a recovery, so
    this follows the same doctrine as removing a root rather than
    inventing a lighter one. `expecting` is the count the caller was
    LOOKING AT. If the number moved between the page being drawn and the
    button being pressed -- another window, a bulk write -- the refusal
    says so instead of taking more than somebody meant.

    Passing None skips the check, which is for callers that have no
    screen to have read a number off.
    """
    held = normalised(name)
    found = conn.execute("SELECT id FROM tag WHERE tag = ?", (held,)).fetchone()
    if not found:
        raise ValueError(f"there is no keyword {name!r} to forget")
    tag_id = int(found[0])
    wearing = int(conn.execute("SELECT count(*) FROM file_tag WHERE tag_id = ?", (tag_id,)).fetchone()[0])
    if expecting is not None and expecting != wearing:
        raise ValueError(
            f"{name!r} is on {wearing} picture(s) now, not the {expecting} you were shown"
            " -- look again before forgetting it"
        )
    conn.execute("DELETE FROM file_tag WHERE tag_id = ?", (tag_id,))
    conn.execute("DELETE FROM tag WHERE id = ?", (tag_id,))
    return wearing


TAGS_OF = (
    "SELECT t.tag, t.label FROM file_tag ft JOIN tag t ON t.id = ft.tag_id"
    " WHERE ft.file_id = ? ORDER BY t.label COLLATE NOCASE"
)


def tags_of(conn, file_id: int) -> tuple[dict, ...]:
    return tuple({"tag": one, "label": label} for one, label in conn.execute(TAGS_OF, (file_id,)))


def vocabulary(conn) -> list[tuple[str, str, int]]:
    """Every keyword in the library with how many pictures wear it,
    commonest first. The filter menu's whole content, and the answer to
    "what have I actually been calling things"."""
    return [
        (one, label, int(count))
        for one, label, count in conn.execute(
            "SELECT t.tag, t.label, count(ft.file_id) FROM tag t"
            "  JOIN file_tag ft ON ft.tag_id = t.id"
            " GROUP BY t.id ORDER BY count(ft.file_id) DESC, t.label COLLATE NOCASE"
        )
    ]


def comment(conn, file_id: int, user_id: int, body: str, now: float) -> int:
    cursor = conn.execute(
        "INSERT INTO comment(file_id, user_id, body, created_at) VALUES(?, ?, ?, ?)",
        (file_id, user_id, body, now),
    )
    return int(cursor.lastrowid or 0)


def edit_comment(conn, comment_id: int, body: str, now: float) -> None:
    conn.execute("UPDATE comment SET body = ?, edited_at = ? WHERE id = ?", (body, now, comment_id))


# --- people ----------------------------------------------------------------


def person(conn, name: str | None, now: float) -> int:
    """Someone the library knows about, named or not.

    An unnamed person is addressable from the moment they exist, so naming
    is a product action rather than a precondition for having a page.
    """
    person_id = mint(conn, "person", name or "person")
    conn.execute("INSERT INTO person(id, name, created_at) VALUES(?, ?, ?)", (person_id, name, now))
    return person_id


def name_person(conn, person_id: int, name: str, now: float) -> str:
    conn.execute("UPDATE person SET name = ? WHERE id = ?", (name, person_id))
    return rename(conn, person_id, name, now)


def assert_named_cluster(conn, person_id: int, user_id: int | None, now: float) -> int:
    """The durable form of naming a group: one assertion per file.

    Naming a person on the People page is a human confirming "this group
    is them". The cluster carrying that confirmation is derived and will
    dissolve on the next re-cluster; what re-attaches the name afterwards
    is `person_assertion`, so the confirmation is written down here, one
    row per file with the face's box -- the highest-confidence face where
    a file holds several.

    Written from ONE run: the primary if its clusters carry this person
    (that is the page the human is looking at); otherwise the EARLIEST
    run that does -- the closest thing to the run that minted them the
    rows record -- with the run id as the final tiebreak, so the pick is
    a function of the data and never of a query plan. Never the newest,
    which would write whichever model ran last into the record; never the
    union across runs, which would write every model's inference into the
    authored ground truth the run rankings are judged against. Returns
    how many files were asserted.
    """
    addressed = conn.execute(
        "SELECT c.run_id FROM derived_face_cluster c"
        "  JOIN derived_face_run r ON r.id = c.run_id"
        " WHERE c.person_id = ?"
        " ORDER BY r.is_primary DESC, r.computed_at ASC, r.id ASC LIMIT 1",
        (person_id,),
    ).fetchone()
    if addressed is None:
        return 0
    seen: set[int] = set()
    rows = conn.execute(
        "SELECT fi.file_id, fi.sample_id, fi.region_id FROM derived_face_membership m"
        "  JOIN derived_face_instance fi ON fi.id = m.face_id"
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id"
        " WHERE c.person_id = ? AND c.run_id = ?"
        " ORDER BY fi.det_score DESC",
        (person_id, addressed[0]),
    ).fetchall()
    for file_id, sample_id, region_id in rows:
        if file_id in seen:
            continue
        seen.add(file_id)
        assert_person(conn, person_id, file_id, user_id, now, sample_id=sample_id, region_id=region_id)
    return len(seen)


def assert_person(
    conn,
    person_id: int,
    file_id: int,
    user_id: int | None,
    now: float,
    *,
    sample_id=None,
    region_id=None,
    stance: str = "is",
) -> None:
    """A person states that this person appears in this file -- or, with
    `stance="is_not"`, that they do not.

    The negative is a CLAIM and not the absence of one, which is the
    whole reason it is a row. `retract_person` deletes, and deleting
    means "I take that back": the next clustering run is then free to
    decide the same thing again, because nothing recorded that it was
    wrong. A denial survives the rebuild and constrains it
    (db/derived.py `seed_clusters_from_assertions`), which is what makes
    a correction permanent rather than a chore repeated after each run.

    One row per (person, file) either way -- a person cannot both be and
    not be in one picture -- so saying the second withdraws the first.

    Kept apart from `derived_file_person`, which is what a model inferred.
    After a re-index the inference is gone and this is what re-attributes the
    clusters, so the naming survives by a record rather than by a similarity
    heuristic that usually works.

    The upsert is guarded: a record with no author (`user_id` NULL -- the
    system writing a naming down) never replaces one a person signed. A
    signed write replaces anything, including another signature. Signed
    rows are today an out-of-band affordance -- no route writes one --
    and the guard protects them for the tooling and surfaces that do.
    """
    if stance not in ("is", "is_not"):
        raise ValueError(f"a claim is 'is' or 'is_not', not {stance!r}")
    conn.execute(
        "INSERT INTO person_assertion(person_id, file_id, sample_id, region_id,"
        " user_id, created_at, stance) VALUES(?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(person_id, file_id) DO UPDATE SET"
        " sample_id = excluded.sample_id, region_id = excluded.region_id,"
        " user_id = excluded.user_id, stance = excluded.stance"
        " WHERE person_assertion.user_id IS NULL OR excluded.user_id IS NOT NULL",
        (person_id, file_id, sample_id, region_id, user_id, now, stance),
    )


def deny_person(conn, person_id: int, file_id: int, user_id: int | None, now: float, *, region_id=None) -> None:
    """This person is NOT in this file -- said out loud, and kept.

    The thing there was no way to say. Naming a region says WHICH face
    was wrong, which is the ordinary case: a picture with two people in
    it cannot express "not her" by naming the file alone.
    """
    from . import derived

    assert_person(conn, person_id, file_id, user_id, now, region_id=region_id, stance="is_not")
    # Who said it, read before it is taken away -- after the delete
    # nothing records which model's output was corrected.
    said_so = derived.attributing_producers(conn, person_id, file_id)
    # And take the name off the picture NOW: `derived_file_person` is what the
    # page reads, so leaving it shows the picture contradicting what somebody
    # just said until a re-run the claim only constrains once it happens.
    gone = derived.withdraw_attribution(conn, person_id, file_id)
    if not gone:
        # Nothing was attributed, so no model got this wrong. Denying a person
        # no run ever put here is a claim about the picture, and a correction
        # recorded here would land against a producer that never spoke.
        return
    for model_id, model_version in said_so:
        retract_person_feedback(conn, person_id, file_id, model_id, model_version, user_id)
        feedback(
            conn,
            "person",
            "wrong",
            now,
            file_id=file_id,
            person_id=person_id,
            user_id=user_id,
            model_id=model_id,
            model_version=model_version,
        )


def reject_duplicate(conn, file_id: int, other_file_id: int, user_id: int | None, now: float) -> int:
    """These two are not the same picture -- said out loud, and kept.

    The same doctrine `deny_person` follows, because it is the same
    problem. A perceptual group is a guess: pHash sees composition, and
    two photographs of one scene a second apart are close in it. Told
    otherwise, the application must both stop showing them together NOW
    and refuse to group them again -- a correction that survives only
    until the next sweep is a chore repeated for ever.

    So it writes a verdict AND drops the pair, and the sweep reads the
    verdicts back (db/runner.py `_dupes_item`). Returns how many rows it
    took out of the group.

    Recorded as a judgement of the PRODUCER too, which is what makes the
    console able to say a fingerprinting threshold is costing somebody
    time -- the same count `deny_person` contributes for faces.
    """
    low, high = (file_id, other_file_id) if file_id < other_file_id else (other_file_id, file_id)
    conn.execute(
        "DELETE FROM feedback WHERE target_kind = 'duplicate' AND file_id = ? AND other_file_id = ? AND user_id IS ?",
        (low, high, user_id),
    )
    feedback(
        conn,
        "duplicate",
        "wrong",
        now,
        file_id=low,
        other_file_id=high,
        user_id=user_id,
        model_id="perceptual",
        model_version="phash64",
    )
    # And take them apart NOW: the page reads `derived_dupe_group`, so leaving
    # them together shows two pictures somebody just called different sitting
    # in one group until the next sweep.
    return int(
        conn.execute(
            "DELETE FROM derived_dupe_group WHERE file_id = ? AND group_id IN"
            " (SELECT group_id FROM derived_dupe_group WHERE file_id = ?)",
            (high, low),
        ).rowcount
        or 0
    )


def rejected_pairs(conn) -> set[tuple[int, int]]:
    """Every pair somebody has said is not one picture, low id first.

    Read by the dupes sweep before it writes a group, which is what
    makes the correction permanent instead of a chore repeated after
    every run.
    """
    return {
        (int(low), int(high))
        for low, high in conn.execute(
            "SELECT file_id, other_file_id FROM feedback"
            " WHERE target_kind = 'duplicate' AND verdict = 'wrong'"
            "   AND file_id IS NOT NULL AND other_file_id IS NOT NULL"
        )
    }


def choose_face(conn, person_id: int, file_id: int | None) -> None:
    """Take this person's face from this picture, or go back to automatic.

    A FILE and never a face. The avatar is cropped from a
    `derived_face_instance`, and every one of those is deleted by
    `derived.drop_all` and minted afresh by the next detection -- so
    remembering the face would remember something the next re-detect
    destroys. Naming the picture is the same durability
    `person_assertion` gets by naming a file and a region rather than a
    cluster.

    `None` clears it, which is not "no avatar": it is the confident
    automatic choice they had before saying anything.
    """
    conn.execute("UPDATE person SET exemplar_file_id = ? WHERE id = ?", (file_id, person_id))


def merge_people(conn, keep_id: int, folded_id: int, user_id: int | None, now: float) -> dict:
    """Say two people are one, and keep it that way.

    The failure this answers is the ordinary one: a clustering run
    splits somebody into four, and a threshold cannot fix it without
    trading away somebody else's correct grouping. Saying it here is
    local, permanent, and survives every future run -- the assertions
    that move are re-applied by `seed_clusters_from_assertions`, which
    is the whole reason authored claims live apart from derived ones.

    What moves, and why each:

    - **Assertions**, so the correction is durable. Never over one
      already made about `keep`: a person who has said something about
      the survivor said it about the survivor, and a merge is not the
      moment to overrule them.
    - **Inferred appearances**, so the page is right NOW rather than
      after a re-run that may never come -- the same reason denying
      withdraws an attribution instead of only recording a claim.
    - **Clusters**, so the next run's groups point at the survivor
      rather than at a person who no longer exists.
    - **Verdicts**, so a correction somebody made about the folded
      person still counts against the model that earned it.

    And every ADDRESS the folded person ever had is re-pointed at the
    survivor, so a bookmark, a shared link or an exported document keeps
    working. `slug_history` already answers a retired slug with the
    entity that holds it now, and the person route already redirects on
    a hit; this makes a merge one more kind of retirement rather than a
    new sort of hole.
    """
    if keep_id == folded_id:
        raise ValueError("a person cannot be merged into themselves")
    for one in (keep_id, folded_id):
        if conn.execute("SELECT 1 FROM person WHERE id = ?", (one,)).fetchone() is None:
            raise LookupError(f"no person {one}")

    # OR IGNORE, then delete the rest: the primary keys are what refuse a
    # duplicate, so this keeps whatever the survivor already had and
    # drops the folded row rather than overwriting a person's own words.
    moved = conn.execute(
        "INSERT OR IGNORE INTO person_assertion(person_id, file_id, sample_id, region_id, user_id, created_at, stance)"
        " SELECT ?, file_id, sample_id, region_id, ?, ?, stance FROM person_assertion WHERE person_id = ?",
        (keep_id, user_id, now, folded_id),
    ).rowcount
    conn.execute(
        "INSERT OR IGNORE INTO derived_file_person(file_id, person_id, run_id, model_id, model_version, face_count)"
        " SELECT file_id, ?, run_id, model_id, model_version, face_count"
        " FROM derived_file_person WHERE person_id = ?",
        (keep_id, folded_id),
    )
    conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE person_id = ?", (keep_id, folded_id))
    conn.execute("UPDATE feedback SET person_id = ? WHERE person_id = ?", (keep_id, folded_id))

    # Every address it ever answered to, including the one it is
    # answering to now.
    conn.execute("UPDATE slug_history SET entity_id = ? WHERE entity_id = ?", (keep_id, folded_id))
    held = conn.execute("SELECT slug FROM entity WHERE id = ?", (folded_id,)).fetchone()
    if held is not None:
        conn.execute(
            "INSERT OR IGNORE INTO slug_history(kind, slug, entity_id, retired_at) VALUES('person', ?, ?, ?)",
            (held[0], keep_id, now),
        )
    # CASCADE takes the person row and anything still pointing at it.
    conn.execute("DELETE FROM entity WHERE id = ?", (folded_id,))
    return {"kept": keep_id, "folded": folded_id, "assertions": int(moved or 0)}


def retract_person(conn, person_id: int, file_id: int) -> None:
    """Withdraw a claim entirely -- neither said nor denied.

    Different from denying, and the difference is what a re-run may do
    next: a retraction leaves no record, so clustering is free to decide
    it again; a denial is a record that stops it.

    The correction goes with it. A verdict recorded because somebody
    denied a person is that denial's evidence, so taking the denial back
    and leaving the verdict standing would keep counting a mistake the
    person no longer says was one.
    """
    conn.execute(
        "DELETE FROM person_assertion WHERE person_id = ? AND file_id = ?",
        (person_id, file_id),
    )
    conn.execute(
        "DELETE FROM feedback WHERE target_kind = 'person' AND person_id = ? AND file_id = ?",
        (person_id, file_id),
    )


def retract_person_feedback(conn, person_id: int, file_id: int, model_id, model_version, user_id) -> int:
    """Take back this actor's correction of one producer; returns how many.

    Denying twice is one standing correction, not two -- the same rule
    `retract_feedback` holds for a caption, and for the same reason: a
    person has one opinion about one claim.
    """
    cursor = conn.execute(
        "DELETE FROM feedback"
        " WHERE target_kind = 'person' AND person_id = ? AND file_id = ?"
        "   AND model_id IS ? AND model_version IS ? AND user_id IS ?",
        (person_id, file_id, model_id, model_version, user_id),
    )
    return int(cursor.rowcount or 0)


def denials(conn, file_id: int | None = None) -> list[tuple[int, int, int | None]]:
    """`(person_id, file_id, region_id)` for every standing denial, or
    for one file."""
    if file_id is None:
        return conn.execute(
            "SELECT person_id, file_id, region_id FROM person_assertion WHERE stance = 'is_not'"
        ).fetchall()
    return conn.execute(
        "SELECT person_id, file_id, region_id FROM person_assertion WHERE stance = 'is_not' AND file_id = ?",
        (file_id,),
    ).fetchall()


# --- feedback on what the machine said -------------------------------------


def feedback(
    conn,
    target_kind: str,
    verdict: str,
    now: float,
    *,
    file_id=None,
    other_file_id=None,
    person_id=None,
    annotation_kind=None,
    note=None,
    user_id=None,
    model_id=None,
    model_version=None,
) -> int:
    """A person's verdict on a derived claim.

    The one authored table whose subject is derived. Its pointers are
    ON DELETE SET NULL rather than CASCADE, so recomputing the derived
    tables leaves the verdict with a nulled target instead of deleting it.

    `model_id`/`model_version` are the producer that was judged, COPIED
    rather than referenced -- for the same reason `annotation_kind` is a
    column rather than an annotation id. Without them a rebuild leaves a
    table that knows a caption was wrong and not which model wrote it,
    and "this model gets 12% of my library wrong" is the reason to
    collect verdicts at all.
    """
    cursor = conn.execute(
        "INSERT INTO feedback(target_kind, file_id, other_file_id, person_id,"
        " annotation_kind, verdict, note, user_id, model_id, model_version, created_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            target_kind,
            file_id,
            other_file_id,
            person_id,
            annotation_kind,
            verdict,
            note,
            user_id,
            model_id,
            model_version,
            now,
        ),
    )
    return int(cursor.lastrowid or 0)


#: One actor's standing verdict on one producer's claim about one file. The
#: newest wins and the older row stays: this table is a record of judgements,
#: not a settings store.
_STANDING = (
    "SELECT verdict FROM feedback"
    " WHERE target_kind = 'annotation' AND file_id = ? AND annotation_kind = ?"
    "   AND model_id IS ? AND model_version IS ?"
    "   AND user_id IS ?"
    " ORDER BY created_at DESC, id DESC LIMIT 1"
)


def standing_verdict(conn, file_id: int, annotation_kind: str, model_id, model_version, user_id) -> str | None:
    """What this actor last said about this claim, or None.

    `IS`, not `=`: a verdict from before the producer columns existed
    holds NULL there, and `= NULL` is never true -- so an old judgement
    would be invisible and the control would open blank over one that
    exists.
    """
    row = conn.execute(_STANDING, (file_id, annotation_kind, model_id, model_version, user_id)).fetchone()
    return None if row is None else str(row[0])


def retract_feedback(conn, file_id: int, annotation_kind: str, model_id, model_version, user_id) -> int:
    """Take back this actor's verdicts on one claim; returns how many.

    Clicking the thumb that is already lit means "I take that back", and
    the honest record of taking something back is that the row is gone --
    a `verdict` of "none" would be a third opinion nobody expressed.
    """
    cursor = conn.execute(
        "DELETE FROM feedback"
        " WHERE target_kind = 'annotation' AND file_id = ? AND annotation_kind = ?"
        "   AND model_id IS ? AND model_version IS ? AND user_id IS ?",
        (file_id, annotation_kind, model_id, model_version, user_id),
    )
    return int(cursor.rowcount or 0)


# --- taking it with you ------------------------------------------------------

#: What somebody SAID about their own pictures, keyed by the bytes; only the
#: files carrying something authored. The module docstring states why.
SAID = (
    "SELECT * FROM ("
    "  SELECT f.content_sha256 AS sha256, f.name,"
    "         (SELECT r.rating FROM rating r WHERE r.file_id = f.id) AS rating,"
    "         EXISTS(SELECT 1 FROM favorite v WHERE v.file_id = f.id) AS favorite,"
    "         (SELECT p.name FROM file_place fp JOIN place p ON p.id = fp.place_id"
    "           WHERE fp.file_id = f.id) AS place,"
    "         (SELECT group_concat(e.slug, ' ') FROM collection_file cf"
    "            JOIN entity e ON e.id = cf.collection_id"
    "           WHERE cf.file_id = f.id ORDER BY e.slug) AS collections,"
    "         EXISTS(SELECT 1 FROM person_assertion a WHERE a.file_id = f.id) AS appears,"
    "         EXISTS(SELECT 1 FROM file_tag ft WHERE ft.file_id = f.id) AS tagged"
    "    FROM file f WHERE f.content_sha256 IS NOT NULL)"
    " WHERE rating IS NOT NULL OR favorite OR place IS NOT NULL"
    "    OR collections IS NOT NULL OR appears OR tagged"
    " ORDER BY name"
)

#: Who is in a picture, and who is NOT; `stance` carries the negative claim.
#: The module docstring states why.
APPEARS = (
    "SELECT f.content_sha256 AS sha256, e.slug AS person, a.stance,"
    "       r.x, r.y, r.w, r.h"
    "  FROM person_assertion a"
    "  JOIN file f ON f.id = a.file_id AND f.content_sha256 IS NOT NULL"
    "  JOIN entity e ON e.id = a.person_id"
    "  LEFT JOIN region r ON r.id = a.region_id"
    " ORDER BY f.name, e.slug"
)

#: The keywords on each picture, in both spellings and in their own query.
#: The module docstring states why.
KEYWORDED = (
    "SELECT f.content_sha256 AS sha256, t.tag, t.label"
    "  FROM file_tag ft"
    "  JOIN file f ON f.id = ft.file_id AND f.content_sha256 IS NOT NULL"
    "  JOIN tag t ON t.id = ft.tag_id"
    " ORDER BY f.name, t.label COLLATE NOCASE"
)

#: The people themselves. A picture's row names a slug; this says what
#: the slug is called, which is the part somebody typed.
NAMED = (
    "SELECT e.slug, p.name FROM person p JOIN entity e ON e.id = p.id"
    " WHERE p.name IS NOT NULL ORDER BY p.name COLLATE NOCASE"
)

#: The albums, with their nesting. A collection's slug is its address in
#: the rows above; `parent` is a slug too, so a shelf rebuilds.
SHELVED = (
    "SELECT e.slug, c.name, c.kind, up.slug AS parent"
    "  FROM collection c"
    "  JOIN entity e ON e.id = c.id"
    "  LEFT JOIN entity up ON up.id = c.parent_id"
    " ORDER BY c.name COLLATE NOCASE"
)


def _rows(conn, sql: str) -> list[dict]:
    cursor = conn.execute(sql)
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def exported(conn) -> dict:
    """Everything a person told this library about their own pictures.

    The opposite shape from `verdicts.exported`, and deliberately. That
    one is for SHARING, so it carries no name and no path; this one is
    for CUSTODY -- it is yours, it is about your pictures, and an export
    that withheld the names would be withholding them from their owner.
    Deleting the application must not delete the understanding.

    Still no pixels: a picture is named by the hash of its bytes and by
    the filename it had, which is what lets somebody put this back
    against the same photographs wherever they now live.
    """
    appearances: dict[str, list[dict]] = {}
    for one in _rows(conn, APPEARS):
        box = None if one["x"] is None else {"x": one["x"], "y": one["y"], "w": one["w"], "h": one["h"]}
        appearances.setdefault(one["sha256"], []).append(
            {"person": one["person"], "stance": one["stance"], "region": box}
        )
    keywords: dict[str, list[dict]] = {}
    for one in _rows(conn, KEYWORDED):
        keywords.setdefault(one["sha256"], []).append({"tag": one["tag"], "label": one["label"]})
    pictures = [
        {
            "sha256": one["sha256"],
            "name": one["name"],
            "rating": one["rating"],
            "favorite": bool(one["favorite"]),
            "place": one["place"],
            "collections": (one["collections"] or "").split(),
            "tags": keywords.get(one["sha256"], []),
            "people": appearances.get(one["sha256"], []),
        }
        for one in _rows(conn, SAID)
    ]
    return {"people": _rows(conn, NAMED), "collections": _rows(conn, SHELVED), "pictures": pictures}
