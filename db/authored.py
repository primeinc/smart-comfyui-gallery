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

from .naming import rename
from .scan import mint

# --- who is using it -------------------------------------------------------


def add_user(conn, username: str, password_hash: str, role: str, now: float) -> int:
    conn.execute(
        "INSERT INTO user(username, password_hash, role, created_at) VALUES(?, ?, ?, ?)",
        (username, password_hash, role, now),
    )
    return conn.execute("SELECT id FROM user WHERE username = ?", (username,)).fetchone()[0]


# --- judgements about one picture -----------------------------------------


def rate(conn, file_id: int, user_id: int, stars: int, now: float) -> None:
    """One rating per person per picture; rating again replaces their own."""
    conn.execute(
        "INSERT INTO rating(file_id, user_id, rating, created_at) VALUES(?, ?, ?, ?)"
        " ON CONFLICT(file_id, user_id) DO UPDATE SET rating = excluded.rating",
        (file_id, user_id, stars, now),
    )


def unrate(conn, file_id: int, user_id: int) -> None:
    conn.execute("DELETE FROM rating WHERE file_id = ? AND user_id = ?", (file_id, user_id))


def favourite(conn, file_id: int, user_id: int, now: float) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO favorite(file_id, user_id, created_at) VALUES(?, ?, ?)",
        (file_id, user_id, now),
    )


def unfavourite(conn, file_id: int, user_id: int) -> None:
    conn.execute("DELETE FROM favorite WHERE file_id = ? AND user_id = ?", (file_id, user_id))


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
    sql_text=None,
    nl_text=None,
) -> int:
    """An album, a flag, or a saved query.

    The three are one table because they differ only in how membership is
    decided: an album is listed, a flag is a state, a smart collection is a
    query. All three are addressable, nestable and nameable, and a schema
    that split them would need three of everything to say so.
    """
    collection_id = mint(conn, "collection", name)
    conn.execute(
        "INSERT INTO collection(id, parent_id, name, kind, color, description,"
        " sql_text, nl_text, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (collection_id, parent_id, name, kind, colour, description, sql_text, nl_text, now),
    )
    return collection_id


def rename_collection(conn, collection_id: int, name: str, now: float) -> str:
    """Rename it, and keep its old address working."""
    conn.execute("UPDATE collection SET name = ? WHERE id = ?", (name, collection_id))
    return rename(conn, collection_id, name, now)


def add_to_collection(conn, collection_id: int, file_id: int, now: float) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO collection_file(collection_id, file_id, added_at) VALUES(?, ?, ?)",
        (collection_id, file_id, now),
    )


def remove_from_collection(conn, collection_id: int, file_id: int) -> None:
    conn.execute(
        "DELETE FROM collection_file WHERE collection_id = ? AND file_id = ?",
        (collection_id, file_id),
    )


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
    a file holds several. Reads every run whose clusters carry this
    person, primary first: the cluster job mints an addressable person
    per run, only one run is ever primary, and a person named from a
    non-primary run's page deserves the same durability. Returns how
    many files were asserted.
    """
    seen: set[int] = set()
    rows = conn.execute(
        "SELECT fi.file_id, fi.sample_id, fi.region_id FROM derived_face_membership m"
        "  JOIN derived_face_instance fi ON fi.id = m.face_id"
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id"
        "  JOIN derived_face_run r ON r.id = c.run_id"
        " WHERE c.person_id = ?"
        " ORDER BY r.is_primary DESC, fi.det_score DESC",
        (person_id,),
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
    """
    conn.execute(
        "INSERT INTO person_assertion(person_id, file_id, sample_id, region_id,"
        " user_id, created_at) VALUES(?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(person_id, file_id) DO UPDATE SET"
        " sample_id = excluded.sample_id, region_id = excluded.region_id,"
        " user_id = excluded.user_id",
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
