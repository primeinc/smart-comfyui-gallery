"""SQL text may be assembled from structure, never from a value.

Every statement this program runs is parameterised: values cross as `?`
and are bound. What the code does assemble by hand is *shape* -- a column
name from a fixed registry, a table name, a WHERE clause built from
literal fragments, an ORDER BY chosen from a map, a run of placeholders
whose only content is question marks.

flake8-bandit's S608 fires on any of that, because it sees an f-string
containing SELECT and cannot tell a column name from a password. Rewriting
those statements to satisfy it means calling `.format()` on a module
constant or joining a list of parts -- both of which ruff cannot see
through at all, so a value spliced in later would go unreported forever.
That trades a rule that over-reports for one that never reports.

This check is the other way round: it knows the names. Every slot in every
SQL f-string in the shipped source has to be one this file has agreed
carries structure. A new name -- `user_id`, `search_term`, anything that
smells like a value -- fails here and has to be either bound as a
parameter or added to the list on purpose.

It is deliberately stricter than S608 in one way: S608 only looks at
strings that match its SELECT/INSERT/UPDATE/DELETE regex, and this looks
at every f-string that contains SQL keywords at all.
"""

from __future__ import annotations

import ast
import re

from source_tree import parsed, shipped

# The application, as shipped. Tests, probes and benchmarks build throwaway
# databases and are not part of what a visitor can reach.

_SQL_SHAPED = re.compile(
    r"\b(select\s+.+\s+from\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b|where\b|order\s+by\b)",
    re.IGNORECASE | re.DOTALL,
)

# Names allowed to appear in a SQL f-string, and what each carries. Adding
# to this list is a decision: it is a promise that the value is SQL text
# this codebase wrote, never anything a caller supplied. The exactness
# check below prunes it the moment a slot stops being written, so the list
# is always the tree's current truth rather than a record of what used to
# be true.
_STRUCTURE = {
    # runs of "?" -- one per bound value
    "marks": "placeholders",
    # db/stories.py _marks: one "?" per frozen member id, nothing else
    "_marks(file_ids)": "placeholders",
    "_marks(members)": "placeholders",
    "','.join('?' * len(batch))": "placeholders",
    "','.join('?' * len(kinds))": "placeholders",
    # job-claim clauses in db/jobs.py: `runnable` is a module literal, and
    # `kind_filter` is "" or an IN (...) of placeholders, params bound
    "runnable": "clause",
    "kind_filter": "clause",
    # "" or a COALESCE-over-setting guard; the key and default are bound
    "gate_filter": "clause",
    # db/resultset.py membership: a conjunction of module-literal
    # predicates (every value in them bound), and ASC/DESC chosen from
    # the validated sort vocabulary -- eligibility is an intersection,
    # constructed, so the pieces are structure by definition
    "' AND '.join(where)": "clause",
    # db/collections.py _claim_revision: "col = ?," assignments chosen
    # from the module's fixed patch vocabulary, every value bound
    "sets": "clause",
    "order": "keyword",
    # table / column names checked against sqlite_master or fixed registries
    "table": "identifier",
    "column": "identifier",
    "name": "identifier",
}


def _sql_interpolations(tree):
    """[(slot, line)] for every interpolation in a SQL-shaped f-string."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(
            part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if not _SQL_SHAPED.search(literal):
            continue
        found.extend(
            (ast.unparse(part.value), node.lineno) for part in node.values if isinstance(part, ast.FormattedValue)
        )
    return found


def test_the_sweep_finds_the_sql_that_is_there():
    """Control. Every check below is an absence, and a sweep that matched
    nothing would report the same absence."""
    slots = {}
    for source in shipped():
        for slot, _line in _sql_interpolations(parsed(source)):
            slots.setdefault(slot, 0)
            slots[slot] += 1

    assert len(slots) >= 5, f"only {len(slots)} distinct slots found: {sorted(slots)}"
    assert "marks" in slots, sorted(slots)
    assert "runnable" in slots, sorted(slots)

    # Every listed name has to still be reached by something, or the list
    # grows into a record of what used to be true.
    assert set(_STRUCTURE) == set(slots), {
        "listed but no longer written anywhere": sorted(set(_STRUCTURE) - set(slots)),
        "written but not listed": sorted(set(slots) - set(_STRUCTURE)),
    }


def test_no_sql_string_interpolates_anything_but_structure():
    """The rule S608 is really after: a value must be bound, not written
    into the statement."""
    unexpected = {}
    for source in shipped():
        for slot, line in _sql_interpolations(parsed(source)):
            if slot not in _STRUCTURE:
                unexpected[f"{source.name}:{line}"] = slot

    assert not unexpected, (
        f"{unexpected} -- written into a SQL statement. If it is a value, bind "
        f"it as ? instead. If it really is SQL structure this codebase wrote, "
        f"add the name to _STRUCTURE here and say which kind."
    )


def test_a_value_written_into_a_statement_would_be_caught():
    """Control for the check above: it has to fail for the thing it exists
    to catch, or it is passing because it understands nothing."""
    smuggled = ast.parse("q = f\"SELECT id FROM files WHERE name = '{user_input}'\"")

    slots = [slot for slot, _line in _sql_interpolations(smuggled)]

    assert slots == ["user_input"], slots
    assert "user_input" not in _STRUCTURE
