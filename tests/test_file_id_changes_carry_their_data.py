"""Changing a file's id must bring its ratings and comments along.

A file's id is the md5 of its path, so anything that moves or renames a
file gives it a new one. Nine tables reference `files(id)` with
`ON DELETE CASCADE` and no `ON UPDATE` clause, and six more hold a file_id
with no constraint at all -- so an id change either fails outright with a
foreign key error or silently orphans the lot.

Four places change an id: renaming a file, renaming a folder, moving files,
and the startup migration for collection notes. All four were broken in
exactly this way, and each was found separately, because the failures did
not look alike from outside: one returned a 500, one reported a "partial
success", one printed a one-line notice and carried on.

`_reassign_file_ids` is the answer to all of them, and this checks the
statements rather than the functions. A first version asked only whether
the function mentioned the helper anywhere, and a control -- deleting one
of move_batch's two calls -- passed it: the merge branch's call covered for
the standard branch's missing one. That is the exact shape of the original
bug, so the check has to see branches.
"""

from __future__ import annotations

import ast
import pathlib

_SOURCE = pathlib.Path(__file__).resolve().parent.parent / "smartgallery.py"
_MARKER = "UPDATE files SET id"
_HELPER = "_reassign_file_ids"


def _sql_of(node):
    """The SQL string a conn.execute/executemany call was given, if any."""
    if not isinstance(node, ast.Call):
        return ""
    parts = []
    for arg in node.args[:1]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                parts.append(sub.value)
    return " ".join(parts)


def _calls_helper(stmt):
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == _HELPER for n in ast.walk(stmt)
    )


def _unprotected_id_changes(tree):
    """Every statement that rewrites files.id whose own block does not also
    call the helper. Returns [(line, enclosing function)]."""

    # Which function each node belongs to, for the failure message.
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                owner.setdefault(id(node), fn.name)

    bad = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            # Does this block contain the helper anywhere in it?
            protected = any(_calls_helper(stmt) for stmt in block)
            for stmt in block:
                changes_id = any(_MARKER in _sql_of(n) for n in ast.walk(stmt))
                if changes_id and not protected:
                    bad.append((stmt.lineno, owner.get(id(stmt), "<module>")))
    return sorted(set(bad))


def test_the_check_still_sees_the_known_sites(gallery_tree):
    """Control: an empty world passes the real test for ever, so the four
    known id-changing statements are counted."""
    sites = [n for n in ast.walk(gallery_tree) if _MARKER in _sql_of(n)]

    assert len(sites) >= 4, (
        f"only {len(sites)} statements rewrite files.id; the check has "
        f"stopped finding them, which would make it pass regardless."
    )


def test_every_id_change_carries_the_files_data(gallery_tree):
    """The regression, for all four at once and for any fifth."""
    unprotected = _unprotected_id_changes(gallery_tree)

    assert unprotected == [], (
        "these rewrite files.id without their block calling "
        f"{_HELPER}: {unprotected}.\n"
        "A file's id is derived from its path, and its ratings, comments, "
        "album membership and AI rows are keyed to that id. Changing it "
        "alone either raises FOREIGN KEY constraint failed or silently "
        "orphans them, depending which table gets there first."
    )
