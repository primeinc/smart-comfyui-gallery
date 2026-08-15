"""OmniQuery v2 NL parser backends: shared types, registry, and helpers.

Every backend under this package (heuristic, needle2, fallback_qwen) speaks
the same contract:

    ParserBackend.parse(text, now_epoch) -> ParserOutcome

``ParserOutcome.ast`` is either ``None`` (the backend could not produce a
usable query) or a plain JSON-compatible dict that already round-tripped
through :func:`omniquery.ast.parse_query` and
:func:`omniquery.validation.validate` -- backends must never hand back an
AST that hasn't been validated. ``router.py`` combines backends according
to a measured policy; ``get_backend``/``make_default_router`` below are the
convenience entry points the rest of the app (and the benchmark harness)
use instead of importing individual backend modules directly.

This module also hosts ``coverage_guard``, the model-free "did the AST
actually keep every literal number/quoted string/recognized keyword in the
query" check shared by needle2.py and fallback_qwen.py -- the two backends
whose output isn't already guaranteed complete by construction.
"""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from omniquery.ast import ASTError, Query, parse_query
from omniquery.validation import AuthContext, ValidationError, validate

# A maximally-permissive AuthContext used by parsers to self-check their own
# output before returning it. Real authorization happens again, for real,
# when the engine validates the *chosen* AST against the caller's actual
# AuthContext -- this is purely "did I build something structurally and
# semantically legal", not an authorization decision.
PERMISSIVE_CTX = AuthContext(
    role="ADMIN", user_id="omniquery-parser", client_uuid="omniquery-parser", ai_enabled=True,
)


@dataclass(frozen=True)
class ParserOutcome:
    ast: Optional[dict]
    confidence: Optional[float]
    backend: str
    unsupported: bool = False
    reason: Optional[str] = None
    coverage: Optional[float] = None
    latency_ms: Optional[float] = None
    raw: Optional[dict] = None


class ParserBackend(ABC):
    """Common interface for every NL -> AST parser backend."""

    name: str = ""

    @abstractmethod
    def parse(self, text: str, now_epoch: float) -> ParserOutcome:
        """Parse `text` into a ParserOutcome. Must never raise: any internal
        failure (engine crash, invalid AST, missing runtime) is reported as
        a failed/unsupported outcome instead."""

    def available(self) -> bool:
        """True when this backend can be used right now. The default
        (zero-dependency) implementation is always available; backends with
        an optional runtime (Needle2, the grammar-constrained Qwen
        fallback) override this to check their runtime/weights."""
        return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKEND_PATHS: Dict[str, str] = {
    "heuristic": "omniquery.parsers.heuristic.HeuristicBackend",
    "needle2": "omniquery.parsers.needle2.Needle2Backend",
    "fallback_qwen": "omniquery.parsers.fallback_qwen.FallbackQwenBackend",
}


def get_backend(name: str, **kwargs: Any) -> ParserBackend:
    """Construct a fresh backend instance by name.

    Submodules are imported lazily (only on request) so asking for
    'heuristic' never pulls in `needle` or `llama_cpp`, and constructing a
    backend never itself touches an optional runtime -- only calling
    `.available()`/`.parse()` on it does.
    """
    try:
        dotted = _BACKEND_PATHS[name]
    except KeyError:
        raise KeyError(f"unknown parser backend {name!r}; known: {sorted(_BACKEND_PATHS)}") from None
    module_name, _, cls_name = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**kwargs)


def make_default_router(config: Optional[Dict[str, Any]] = None) -> "Router":  # noqa: F821
    """Convenience: build the standard heuristic/needle2/fallback Router
    with thresholds loaded from routing_defaults.json, optionally overridden
    by `config`. Backend runtimes that are unavailable are still handed to
    the Router (it checks `.available()` itself before ever calling them)."""
    from omniquery.parsers.router import Router, fallback_enabled, load_thresholds

    heuristic = get_backend("heuristic")
    primary = get_backend("needle2")
    # Benchmark-measured default: the 0.5B fallback adds no accuracy on the
    # SmartGallery corpus, so it only joins the route when explicitly enabled
    # (routing_defaults.json "fallback_enabled" or OMNIQUERY_ENABLE_FALLBACK).
    fallback = get_backend("fallback_qwen") if fallback_enabled() else None
    thresholds = load_thresholds()
    if config:
        thresholds.update(config)
    return Router(primary=primary, fallback=fallback, heuristic=heuristic, thresholds=thresholds)


# ---------------------------------------------------------------------------
# Shared validation helper
# ---------------------------------------------------------------------------

def try_validate(ast_dict: dict) -> Tuple[Optional[Query], Optional[str]]:
    """Parse + validate an AST dict against PERMISSIVE_CTX.

    Every backend must call this (directly or via a wrapper) before
    returning a successful ParserOutcome -- "the output AST must pass
    validation before returning" is enforced here once instead of being
    re-implemented per backend. Returns (query, None) on success or
    (None, reason) on failure.
    """
    try:
        query = parse_query(ast_dict)
    except ASTError as exc:
        return None, f"invalid AST structure: {exc}"
    try:
        validate(query, PERMISSIVE_CTX)
    except ValidationError as exc:
        return None, f"AST failed validation: {exc}"
    return query, None


# ---------------------------------------------------------------------------
# AST tree-walking helpers (dict-shaped ASTs, i.e. pre-parse_query() JSON)
# ---------------------------------------------------------------------------

def _walk_conds(node: Any) -> Iterator[dict]:
    """Yield every condition dict ({"field", "op", "value"}) under `node`,
    a plain dict-shaped where-tree (Group/Not/Cond JSON)."""
    if not isinstance(node, dict):
        return
    if "children" in node:
        for child in node["children"]:
            yield from _walk_conds(child)
    elif "child" in node:
        yield from _walk_conds(node["child"])
    elif "field" in node:
        yield node


def contains_not_node(node: Any) -> bool:
    """True if `node` (a dict-shaped where-tree, possibly None) contains a
    'not' node anywhere. Used by needle2.py: if the model detected negation
    but the frame-expanded AST has no 'not' node, the frame lost it."""
    if not isinstance(node, dict):
        return False
    if node.get("op") == "not":
        return True
    if "children" in node:
        return any(contains_not_node(c) for c in node["children"])
    if "child" in node:
        return contains_not_node(node["child"])
    return False


def _scalar_leaves(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _scalar_leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from _scalar_leaves(v)
    else:
        yield value


def _all_ast_values(ast_dict: dict) -> List[Any]:
    values: List[Any] = []
    for cond in _walk_conds(ast_dict.get("where")):
        values.extend(_scalar_leaves(cond.get("value")))
    if ast_dict.get("limit") is not None:
        values.append(ast_dict["limit"])
    return values


# ---------------------------------------------------------------------------
# coverage_guard: model-free "did the AST keep every literal" check, shared
# by needle2.py (checking its frame-expanded AST) and fallback_qwen.py
# (checking its grammar-constrained-but-still-hallucination-prone output).
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r"[\"']([^\"']+)[\"']")

# number optionally followed by a unit word; unit drives which scaled
# variants of the raw number are accepted as "the same value" (e.g. "100 MB"
# satisfies either a size_mb literal 100 or a size_bytes literal 104857600).
_NUMBER_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)"
    r"(?:\s*(mb|megabytes?|gb|gigabytes?|kb|kilobytes?|"
    r"seconds?|secs?|minutes?|mins?|hours?|hrs?|"
    r"stars?|days?|weeks?|months?|megapixels?|mp))?\b",
    re.I,
)

_UNIT_MULTIPLIERS: Dict[str, Tuple[float, ...]] = {
    "mb": (1.0, 1024.0 * 1024.0), "megabyte": (1.0, 1024.0 * 1024.0), "megabytes": (1.0, 1024.0 * 1024.0),
    "gb": (1024.0, 1024.0 ** 3), "gigabyte": (1024.0, 1024.0 ** 3), "gigabytes": (1024.0, 1024.0 ** 3),
    "kb": (1.0 / 1024.0, 1024.0), "kilobyte": (1.0 / 1024.0, 1024.0), "kilobytes": (1.0 / 1024.0, 1024.0),
    "second": (1.0,), "seconds": (1.0,), "sec": (1.0,), "secs": (1.0,),
    "minute": (60.0,), "minutes": (60.0,), "min": (60.0,), "mins": (60.0,),
    "hour": (3600.0,), "hours": (3600.0,), "hr": (3600.0,), "hrs": (3600.0,),
    "star": (1.0,), "stars": (1.0,),
    "day": (1.0,), "days": (1.0,),
    "week": (7.0,), "weeks": (7.0,),
    "month": (30.0,), "months": (30.0,),
    "megapixel": (1.0,), "megapixels": (1.0,), "mp": (1.0,),
}

_MEDIA_TYPE_RE = re.compile(
    r"\b(photos?|pictures?|images?|videos?|clips?|movies?|gifs?|animated[ -]images?|"
    r"sounds?|music|songs?|audio|documents?|pdfs?)\b", re.I,
)
_STATUS_WORD_RE = re.compile(
    r"\b(approved|rejected|needs?\s+review|in\s+review|review|to\s+edit|selected|select)\b", re.I,
)
_FAVORITE_WORD_RE = re.compile(r"\bfavou?rite[sd]?\b", re.I)
_COUNT_WORD_RE = re.compile(r"\bhow\s+many\b|\bcount\s+of\b|\bnumber\s+of\b", re.I)


def _number_candidates(raw: str, unit: Optional[str]) -> List[float]:
    n = float(raw)
    candidates = [n]
    if unit:
        candidates.extend(n * m for m in _UNIT_MULTIPLIERS.get(unit.lower(), ()))
    return candidates


def coverage_guard(text: str, ast_dict: dict) -> Tuple[float, List[str]]:
    """Model-free structural guard: does the AST plausibly reflect every
    literal number, every quoted string, and every recognized keyword class
    present in `text`? Returns (coverage_fraction, miss_descriptions).

    This is deliberately NOT a semantic checker (it can't tell "not
    approved" from "approved" -- that's the negation check next to it in
    needle2.py) -- it only catches the failure mode measured on this
    machine: constraints silently DROPPED from the parse.  coverage == 1.0
    when there is nothing to check (an empty/trivial query).
    """
    values = _all_ast_values(ast_dict)
    numeric_values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    string_values = [v for v in values if isinstance(v, str)]

    total = 0
    ok = 0
    missing: List[str] = []

    for m in _QUOTED_RE.finditer(text):
        literal = m.group(1)
        total += 1
        if any(literal.lower() in s.lower() for s in string_values):
            ok += 1
        else:
            missing.append(f"quoted string {literal!r} not reflected in AST")

    for m in _NUMBER_UNIT_RE.finditer(text):
        raw, unit = m.group(1), m.group(2)
        total += 1
        candidates = _number_candidates(raw, unit)
        if any(abs(v - c) < 1e-6 for v in numeric_values for c in candidates):
            ok += 1
        elif any(raw in s for s in string_values):
            ok += 1
        else:
            missing.append(f"number {raw!r} not reflected in AST")

    if _MEDIA_TYPE_RE.search(text):
        total += 1
        if any(c.get("field") == "type" for c in _walk_conds(ast_dict.get("where"))):
            ok += 1
        else:
            missing.append("media-type keyword not reflected as a 'type' condition")

    if _STATUS_WORD_RE.search(text):
        total += 1
        if any(c.get("field") == "status_flag" for c in _walk_conds(ast_dict.get("where"))):
            ok += 1
        else:
            missing.append("status keyword not reflected as a 'status_flag' condition")

    if _FAVORITE_WORD_RE.search(text):
        total += 1
        if any(c.get("field") == "is_favorite" for c in _walk_conds(ast_dict.get("where"))):
            ok += 1
        else:
            missing.append("'favorite' keyword not reflected as an 'is_favorite' condition")

    if _COUNT_WORD_RE.search(text):
        total += 1
        if ast_dict.get("result") == "count":
            ok += 1
        else:
            missing.append("'how many'/'count of' not reflected as result='count'")

    coverage = 1.0 if total == 0 else ok / total
    return coverage, missing


__all__ = [
    "PERMISSIVE_CTX",
    "ParserOutcome",
    "ParserBackend",
    "get_backend",
    "make_default_router",
    "try_validate",
    "contains_not_node",
    "coverage_guard",
]
