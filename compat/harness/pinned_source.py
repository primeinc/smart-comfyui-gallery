from __future__ import annotations

import ast
import hashlib
import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import proc

SCRATCH: Path = Path(tempfile.gettempdir()) / "compat_pinned_source"


@dataclass
class LoadedSymbol:
    symbol: str
    path: str
    commit: str
    blob_sha256: str
    source_sha256: str
    lines: int
    materialised: str = ""
    supplied: list[str] = field(default_factory=list)


def _blob(repo: Path, commit: str, path: str) -> bytes:
    argv = ["git", "-C", str(repo), "show", f"{commit}:{path}"]
    code, out, err = proc.run(argv, timeout=proc.LOCAL_SECONDS)
    if code != 0:
        raise FileNotFoundError(f"{path} is not at {commit[:12]} in {repo}: {err.decode('utf-8', 'replace')}")
    return out


def _body_of(node: ast.stmt) -> list[ast.stmt] | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.body
    return None


def find_node(tree: ast.Module, symbol: str) -> ast.stmt:

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
    if not col or not line.strip():
        return line
    return line[col:] if line[:col].strip() == "" else line.lstrip()


def subscript_keys(repo: Path, commit: str, path: str, symbol: str, on: str) -> tuple[str, ...]:
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
    blob = _blob(repo, commit, path)
    text = blob.decode("utf-8", errors="surrogateescape")
    segment = source_segment(text, find_node(ast.parse(text), symbol))
    if segment is None:
        raise LookupError(f"{symbol!r} was located but its source segment could not be recovered")
    return segment, hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()


def load_symbol(repo: Path, commit: str, path: str, symbol: str, namespace: dict[str, Any]) -> tuple[Any, LoadedSymbol]:
    source, blob_sha256 = symbol_source(repo, commit, path, symbol)
    supplied = sorted(namespace)

    SCRATCH.mkdir(parents=True, exist_ok=True)

    where_from = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    stem = f"{Path(path).stem}__{symbol.replace('.', '_')}__{commit[:12]}__{where_from}"
    where = SCRATCH / f"{stem}.py"

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
