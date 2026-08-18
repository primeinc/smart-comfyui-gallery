"""One parsed copy of the repository, shared by the checks that sweep it.

Three test modules read every source file in this repo to hold an
invariant across it -- how SQL is assembled, how programs are started,
what is imported when a module is read. Parsing smartgallery.py alone is
17,000 lines, and each module doing its own walk paid for it again.

pytest puts a test file's own directory on sys.path under the default
prepend import mode (doc/en/explanation/pythonpath.rst), which is what
makes `from source_tree import parsed` work from a sibling test module
without making tests/ a package.
"""

from __future__ import annotations

import ast
import functools
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@functools.cache
def parsed(source: pathlib.Path) -> ast.Module:
    """The file's AST, parsed once per session."""
    return ast.parse(source.read_text(encoding="utf-8"))


@functools.cache
def sources(*entries: str) -> tuple[pathlib.Path, ...]:
    """Every .py file under the named files and directories."""
    found: list[pathlib.Path] = []
    for entry in entries:
        path = REPO_ROOT / entry
        if path.is_dir():
            found.extend(sorted(path.rglob("*.py")))
        elif path.exists():
            found.append(path)
    return tuple(found)
