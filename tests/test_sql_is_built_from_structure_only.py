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

from source_tree import parsed, sources

# The application, as shipped. Tests, probes and benchmarks build throwaway
# databases and are not part of what a visitor can reach.
_SHIPPED = ("smartgallery.py", "sg_auth.py", "sqlbind.py", "smartgallery_ai", "omniquery", "metaparse")

_SQL_SHAPED = re.compile(
    r"\b(select\s+.+\s+from\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b|where\b|order\s+by\b)",
    re.IGNORECASE | re.DOTALL,
)

# Names allowed to appear in a SQL f-string, and what each carries. Adding
# to this list is a decision: it is a promise that the value is SQL text
# this codebase wrote, never anything a caller supplied.
_STRUCTURE = {
    # runs of "?" -- one per bound value. The {ids} slot that sqlbind fills
    # is not here: it lives in plain strings, never an f-string.
    "placeholders": "placeholders",
    "type_placeholders": "placeholders",
    # column / table names, from fixed registries
    "column": "identifier",
    "col_name": "identifier",
    "key_column": "identifier",
    "table": "identifier",
    "name": "identifier",
    "primary_hash_key": "identifier",
    "spec.table": "identifier",
    "spec.alias": "identifier",
    # clause fragments assembled from literals, with their own bound params
    "where_clause": "clause",
    "where_sql": "clause",
    "order_clause": "clause",
    "order": "clause",
    "cond": "clause",
    "q_cond": "clause",
    "w_cond": "clause",
    "hashable": "clause",
    "joins": "clause",
    "only_clause": "clause",
    "extra_cols": "clause",
    "input_key_sql": "clause",
    "inner_sql": "clause",
    "name_sql": "clause",
    "col_expr": "clause",
    "scope_sql": "clause",
    "sub_query": "clause",
    "PER_CALLER_COLUMNS": "clause",
    # operators chosen from a two-element set
    "op_in": "operator",
    # coerced to an integer before it reaches the text
    "int(coll_id)": "integer",
    # a fixed map of prompt columns, keyed by a validated field name
    "_PROMPT_SEARCH_TEXT_COLUMNS[name]": "identifier",
    # named clause fragments, each literal SQL carrying its own ? slots
    "shared_with_me": "clause",
    "shared_with_anyone": "clause",
    # clauses joined from a list whose every element was written here
    "' AND '.join(conditions)": "clause",
    "' OR '.join(stale)": "clause",
    # Not a statement: the gallery root, written into the natural-language
    # prompt the SQL model reads. Whatever the model then writes is run
    # through omniquery's read-only validator before it reaches sqlite.
    "BA_OU_PA": "prompt",
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
    for source in sources(*_SHIPPED):
        for slot, _line in _sql_interpolations(parsed(source)):
            slots.setdefault(slot, 0)
            slots[slot] += 1

    assert len(slots) >= 30, f"only {len(slots)} distinct slots found: {sorted(slots)}"
    assert "where_clause" in slots, sorted(slots)
    assert "PER_CALLER_COLUMNS" in slots, sorted(slots)

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
    for source in sources(*_SHIPPED):
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
