"""What a person wrote down, and the rules that keep it.

Ratings, comments, favourites, albums, named people and verdicts on what a
model said are the one class of row nothing in this system may regenerate. Everything in
`derived_*` can be dropped and rebuilt by definition; none of this can, so
the operations here are deliberately narrow and every deletion is something
a person asked for by name.

`person` sits here rather than with the face pipeline. A cluster is evidence
and is disposable; the human's naming of it is not, which is why the name
lives on the person and `person_assertion` records the claim directly
against the file. Dropping the entire derived namespace and re-indexing must
leave both standing.
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
#
# The write interface is DESIRED STATE, never a toggle: "favorite = true"
# retried after a network hiccup lands where it already was, where a
# toggle retried lands on the opposite of what the person asked. Every
# operation is idempotent by the primitives beneath it.


@dataclasses.dataclass(frozen=True)
class MediaAuthoredState:
    """What this actor has written down about one file."""

    favorite: bool
    rating: int | None
    collections: tuple[dict, ...]  # {"slug", "name"}, static memberships by name


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


def set_collection_membership_many(conn, collection_id: int, file_ids, value: bool, now: float) -> None:
    """One membership write for every adapter. The smart refusal runs
    ONCE, before any row -- all or nothing is the transaction's job, but
    not even the first row of a doomed batch should be attempted."""
    _takes_filings(conn, collection_id, removing=not value)
    if value:
        conn.executemany(_FILE_INTO, [(collection_id, file_id, now) for file_id in file_ids])
    else:
        conn.executemany(_FILE_OUT_OF, [(collection_id, file_id) for file_id in file_ids])


def set_favorite(conn, file_id: int, user_id: int, value: bool, now: float) -> None:
    set_favorite_many(conn, (file_id,), user_id, value, now)


def set_rating(conn, file_id: int, user_id: int, value: int | None, now: float) -> None:
    """`None` clears; 1..5 sets. Validated in `_rating` so every caller
    gets the same refusal instead of a CHECK-constraint traceback."""
    set_rating_many(conn, (file_id,), user_id, value, now)


def set_collection_membership(conn, collection_id: int, file_id: int, value: bool, now: float) -> None:
    set_collection_membership_many(conn, collection_id, (file_id,), value, now)


def comment(conn, file_id: int, user_id: int, body: str, now: float) -> int:
    cursor = conn.execute(
        "INSERT INTO comment(file_id, user_id, body, created_at) VALUES(?, ?, ?, ?)",
        (file_id, user_id, body, now),
    )
    return int(cursor.lastrowid or 0)


def edit_comment(conn, comment_id: int, body: str, now: float) -> None:
    conn.execute("UPDATE comment SET body = ?, edited_at = ? WHERE id = ?", (body, now, comment_id))


# --- collections -----------------------------------------------------------


def collection(
    conn,
    name: str,
    now: float,
    *,
    kind: str = "album",
    parent_id=None,
    colour=None,
    description=None,
) -> int:
    """An album, a flag, or a saved query.

    The three are one table because they differ only in how membership is
    decided: an album is listed, a flag is a state, a smart collection is a
    query. All three are addressable, nestable and nameable, and a schema
    that split them would need three of everything to say so. A smart
    collection's rule lives in `collection_rule` (db/collection_rules.py);
    until one is saved the collection is UNEVALUATED, never empty.
    """
    collection_id = mint(conn, "collection", name)
    conn.execute(
        "INSERT INTO collection(id, parent_id, name, kind, color, description, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
        (collection_id, parent_id, name, kind, colour, description, now),
    )
    return collection_id


def rename_collection(conn, collection_id: int, name: str, now: float) -> str:
    """Rename it, and keep its old address working."""
    conn.execute("UPDATE collection SET name = ? WHERE id = ?", (name, collection_id))
    return rename(conn, collection_id, name, now)


_FILE_INTO = "INSERT OR IGNORE INTO collection_file(collection_id, file_id, added_at) VALUES(?, ?, ?)"
_FILE_OUT_OF = "DELETE FROM collection_file WHERE collection_id = ? AND file_id = ?"


def _takes_filings(conn, collection_id: int, *, removing: bool) -> None:
    """A smart collection is refused by name here, and by trigger
    beneath: its members are its rule's answer, and a stored row would
    be a second, disagreeing one -- and pretending to remove one would
    be answering under a membership model the kind does not have."""
    kind = conn.execute("SELECT kind FROM collection WHERE id = ?", (collection_id,)).fetchone()
    if kind is not None and kind[0] == "smart":
        what = "to remove" if removing else "into it"
        raise ValueError(f"a smart collection derives its members from its rule; nothing is filed {what}")


def add_to_collection(conn, collection_id: int, file_id: int, now: float) -> None:
    """File one picture into a listed collection."""
    _takes_filings(conn, collection_id, removing=False)
    conn.execute(_FILE_INTO, (collection_id, file_id, now))


def remove_from_collection(conn, collection_id: int, file_id: int) -> None:
    _takes_filings(conn, collection_id, removing=True)
    conn.execute(_FILE_OUT_OF, (collection_id, file_id))


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
) -> None:
    """A person states that this person appears in this file.

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
    conn.execute(
        "INSERT INTO person_assertion(person_id, file_id, sample_id, region_id,"
        " user_id, created_at) VALUES(?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(person_id, file_id) DO UPDATE SET"
        " sample_id = excluded.sample_id, region_id = excluded.region_id,"
        " user_id = excluded.user_id"
        " WHERE person_assertion.user_id IS NULL OR excluded.user_id IS NOT NULL",
        (person_id, file_id, sample_id, region_id, user_id, now),
    )


def retract_person(conn, person_id: int, file_id: int) -> None:
    conn.execute(
        "DELETE FROM person_assertion WHERE person_id = ? AND file_id = ?",
        (person_id, file_id),
    )


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
) -> int:
    """A person's verdict on a derived claim.

    The one authored table whose subject is disposable, so its pointers are
    ON DELETE SET NULL rather than CASCADE: dropping the derived namespace
    must leave the judgement standing with a nulled target, not delete it.
    """
    cursor = conn.execute(
        "INSERT INTO feedback(target_kind, file_id, other_file_id, person_id,"
        " annotation_kind, verdict, note, user_id, created_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            target_kind,
            file_id,
            other_file_id,
            person_id,
            annotation_kind,
            verdict,
            note,
            user_id,
            now,
        ),
    )
    return int(cursor.lastrowid or 0)
