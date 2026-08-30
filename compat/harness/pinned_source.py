"""Execute one symbol from a pinned commit without importing its package.

Most of the frozen population cannot simply be imported. ReActor's
`reactor_utils` imports ComfyUI's `folder_paths` and `comfy.utils` at module
level; several others reach for their own runtime on the first line. Standing
all of that up would make the suite depend on twenty application environments
to answer a storage question none of them are being asked about.

The alternative is not to reimplement the function. It is to take the bytes
upstream committed, extract exactly the symbol named, and run those.

    fn, proof = load_symbol(repo, commit, path, symbol, namespace={...})

What this DOES guarantee: the code executed is byte-for-byte the source of
that symbol at that commit, and `blob_sha256` is the number
`git show <commit>:<path> | sha256sum` produces -- the same one
`provenance.py` records, so the two can be checked against each other.

What it does NOT: module-level imports and module-level state are not
executed. A symbol that depends on either gets it from `namespace`, supplied
explicitly by the caller and listed in the proof -- so a reader can see the
seam where this stops being upstream's own environment rather than having to
infer it. A symbol whose body reaches for a global nobody supplied raises
NameError when called, which is loud and correct.

This is deliberately narrower than an import. It is not a way to run a whole
consumer; it is a way to run the one function whose contract is under test.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import proc

#: Where extracted sources are materialised: written to disk rather than
#: exec'd from a string, so the loaded object has a real `__file__` and the
#: exact bytes that ran can be opened afterwards.
SCRATCH: Path = Path(tempfile.gettempdir()) / "compat_pinned_source"


@dataclass
class LoadedSymbol:
    """One symbol executed out of a pinned blob, with its provenance."""

    symbol: str
    path: str
    commit: str
    blob_sha256: str
    source_sha256: str
    lines: int
    materialised: str = ""
    supplied: list[str] = field(default_factory=list)
    """Names the caller had to provide because module scope was not executed.
    Recorded because they are the seam where this stops being upstream's own
    environment, and a reviewer needs to see the seam rather than infer it."""


def _blob(repo: Path, commit: str, path: str) -> bytes:
    argv = ["git", "-C", str(repo), "show", f"{commit}:{path}"]
    code, out, err = proc.run(argv, timeout=proc.LOCAL_SECONDS)
    if code != 0:
        raise FileNotFoundError(f"{path} is not at {commit[:12]} in {repo}: {err.decode('utf-8', 'replace')}")
    return out


def _body_of(node: ast.stmt) -> list[ast.stmt] | None:
    """The statement list a definition encloses, or None if it encloses none.

    Written as an explicit narrowing rather than an attribute access on a
    union: three node types carry `body` and every other statement does not,
    and a checker cannot know which arrived without being told here.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.body
    return None


def find_node(tree: ast.Module, symbol: str) -> ast.stmt:
    """The definition of `symbol`, walking `A.b` into the class body.

    Resolved through the AST, never by searching text. A substring match finds
    the name in a comment, an import or a call site, and would happily extract
    the wrong region -- which in this module means executing something other
    than the symbol whose provenance is about to be recorded.
    """

    def walk(body: list[ast.stmt], names: list[str]) -> ast.stmt | None:
        head, rest = names[0], names[1:]
        for node in body:
            inner = _body_of(node)
            if inner is None:
                continue
            named = getattr(node, "name", None)
            if named != head:
                continue
            return node if not rest else walk(inner, rest)
        return None

    found = walk(tree.body, symbol.split("."))
    if found is None:
        raise LookupError(f"{symbol!r} is not defined in this source")
    return found


def source_segment(text: str, node: ast.AST) -> str | None:
    """One symbol's source, decorators included.

    `ast.get_source_segment` slices from `node.lineno`, which for a decorated
    definition is the `def`/`class` line -- `decorator_list` entries carry
    their own, earlier, linenos (CPython Doc/library/ast.rst, class FunctionDef;
    confirmed by execution: decorator lineno 3 against FunctionDef lineno 4).
    Every decorator is therefore dropped, silently, and the loaded object
    behaves differently from the pinned one. `@torch.inference_mode()` was
    being removed from two of the symbols this suite executes.

    EVERY line is outdented by `col_offset`, not just the first.
    `get_source_segment` trims only the first line, which is sound when the
    `def` IS the first line: the body is then merely over-indented relative to
    a `def` at column 0, and Python accepts that. With a decorator above it,
    trimming one line puts `@decorator` at column 0 and `def` at column 4 --
    an IndentationError. Outdenting the whole block keeps the relative
    structure and produces a loadable module.
    """
    decorators = list(getattr(node, "decorator_list", ()) or ())
    if not decorators:
        return ast.get_source_segment(text, node)

    end = getattr(node, "end_lineno", None)
    col = getattr(node, "col_offset", None)
    if end is None or col is None:
        return None
    start = min(one.lineno for one in decorators)
    lines = text.splitlines(keepends=True)[start - 1 : end]
    if not lines:
        return None
    return "".join(_outdent(one, col) for one in lines)


def _outdent(line: str, col: int) -> str:
    """One line with up to `col` leading spaces removed, blanks untouched."""
    if not col or not line.strip():
        return line
    return line[col:] if line[:col].strip() == "" else line.lstrip()


def subscript_keys(repo: Path, commit: str, path: str, symbol: str, on: str) -> tuple[str, ...]:
    """Every constant string `on` is subscripted with inside `symbol`.

    `save_face_model` names the nine keys ReActor requires by writing
    `face["bbox"]`, `face["kps"]` and so on. Retyping that list into a
    constant makes the contract a thing somebody remembered rather than a
    thing upstream states: a tenth key added at a later commit would leave
    the copy stale and the case would keep passing while testing the wrong
    contract.

    So the keys are read back out of the pinned AST. Order is source order,
    which is also the order the vendor writes them into its container -- and
    for a format whose header records key order, that is part of the artifact.
    """
    source, _ = symbol_source(repo, commit, path, symbol)
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Subscript):
            continue
        value, index = node.value, node.slice
        if not isinstance(value, ast.Name) or value.id != on:
            continue
        if isinstance(index, ast.Constant) and isinstance(index.value, str) and index.value not in found:
            found.append(index.value)
    if not found:
        raise LookupError(f"{symbol!r} at {commit[:12]} subscripts {on!r} with no constant string keys")
    return tuple(found)


def symbol_source(repo: Path, commit: str, path: str, symbol: str) -> tuple[str, str]:
    """The exact source text of one symbol, and the whole blob's digest.

    The digest covers the WHOLE blob rather than the extracted segment, over
    the blob with CRLF normalised to LF -- the same normalisation
    `provenance.py:496-497` applies, so the two lanes report one number for
    one file. It is NOT what `git show <commit>:<path> | sha256sum` prints for
    a blob committed with CRLF; normalising is deliberate, so the number does
    not move with a checkout's eol settings, and saying which number it is
    matters more than it being the convenient one.

    The segment carries its own digest separately, and that one is taken over
    the text INCLUDING decorators -- see `source_segment`.
    """
    blob = _blob(repo, commit, path)
    text = blob.decode("utf-8", errors="surrogateescape")
    segment = source_segment(text, find_node(ast.parse(text), symbol))
    if segment is None:
        raise LookupError(f"{symbol!r} was located but its source segment could not be recovered")
    return segment, hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def load_symbol(repo: Path, commit: str, path: str, symbol: str, namespace: dict[str, Any]) -> tuple[Any, LoadedSymbol]:
    """Load one pinned symbol as a module, in a namespace the caller controls.

    `namespace` is bound into the module's dictionary BEFORE its body runs, so
    a symbol closing over a name upstream's module scope would have provided
    resolves to the caller's object instead. That substitution is the seam,
    and it is recorded in the returned proof.

    A real module load rather than a bare exec: the object gets a `__file__`,
    a traceback that points at readable lines, and bytes on disk that can be
    reopened and compared against the digest afterwards.
    """
    source, blob_sha256 = symbol_source(repo, commit, path, symbol)
    supplied = sorted(namespace)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    # The full path is folded in, not just its stem: two files named
    # `utils.py` in one commit would produce the same scratch name and the
    # second load would overwrite the first.
    where_from = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    stem = f"{Path(path).stem}__{symbol.replace('.', '_')}__{commit[:12]}__{where_from}"
    where = SCRATCH / f"{stem}.py"
    # newline="" so the bytes on disk are the bytes that were hashed: Windows
    # would otherwise translate every \n and the file would no longer match.
    with where.open("w", encoding="utf-8", newline="") as handle:
        handle.write(source)

    spec = importlib.util.spec_from_file_location(stem, where)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build a module spec for {where}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(namespace)
    spec.loader.exec_module(module)

    leaf = symbol.rsplit(".", 1)[-1]
    if not hasattr(module, leaf):
        raise LookupError(f"executing {symbol!r} did not define {leaf!r}")

    return getattr(module, leaf), LoadedSymbol(
        symbol=symbol,
        path=path,
        commit=commit,
        blob_sha256=blob_sha256,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        lines=source.count("\n") + 1,
        materialised=str(where),
        supplied=supplied,
    )
