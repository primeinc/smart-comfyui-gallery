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


#: Directories that are not this project's code. Everything else is,
#: automatically: the sweeps discover their scope instead of listing it,
#: because a hand-typed list is how `db/` shipped a subprocess call outside
#: every gate -- the package was born after the list was written.
_NOT_OURS = {".git", ".venv", "__pycache__", "node_modules", "vendor"}

#: Our code that does not ship to a user: tests, benchmarks, probes and
#: experiments build throwaway state on purpose.
_TOOLING = {"tests", "benchmarks", "experiments", "probes"}


@functools.cache
def every_source() -> tuple[pathlib.Path, ...]:
    """Every .py file this repository owns, discovered rather than listed."""
    import os

    found: list[pathlib.Path] = []
    for current, subdirs, names in os.walk(REPO_ROOT):
        subdirs[:] = sorted(d for d in subdirs if d not in _NOT_OURS)
        found.extend(pathlib.Path(current) / name for name in sorted(names) if name.endswith(".py"))
    return tuple(found)


@functools.cache
def shipped() -> tuple[pathlib.Path, ...]:
    """The application as a user receives it: every source outside tooling."""
    return tuple(path for path in every_source() if path.relative_to(REPO_ROOT).parts[0] not in _TOOLING)
