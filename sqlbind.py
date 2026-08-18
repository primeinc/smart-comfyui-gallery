"""Binding a list to a SQLite statement.

SQLite binds values, never lists, so a statement that filters on n ids
needs n placeholders written into its text. That is the one piece of SQL
this codebase assembles rather than states, and it is assembled here so
there is one thing to read: all `with_id_placeholders` can insert is
question marks, and anything else a call site wants in its SQL still has
to be written out where it can be seen.
"""

from __future__ import annotations


def with_id_placeholders(statement: str, ids) -> str:
    """`statement` with its `{ids}` slot filled: one question mark per id.

    The values themselves are still bound; only their number reaches the
    statement text.

        conn.execute(with_id_placeholders("DELETE FROM files WHERE id IN ({ids})", ids), ids)
    """
    return statement.format(ids=",".join("?" * len(ids)))
