"""OmniQuery NL parser backends: shared types and the registry.

Every AST-emitting backend here (nlq, and the benchmark-only
nlq) speaks the same contract:

    ParserBackend.parse(text, now_epoch) -> ParserOutcome

``ParserOutcome.ast`` is either ``None`` (the backend could not produce a
usable query) or a plain JSON-compatible dict that already round-tripped
through :func:`omniquery.ast.parse_query` and
:func:`omniquery.validation.validate` -- backends must never hand back an
AST that hasn't been validated.

`nlq` (deterministic, sub-millisecond) parses EVERY query -- structural
rules consume what they recognize, every leftover term is a full-text
search, and "unsupported" does not exist. The nl2sql model lives OUTSIDE
this contract (parsers/nl2sql.SqlSearch speaks SQL through the sqlexec
sandbox, not ASTs); the app's endpoint fuses the two -- see
omniquery/__init__.py for the whole picture.

This module also hosts ``coverage_guard``, the model-free "did the AST
actually keep every literal number/quoted string/recognized keyword in the
query" check the benchmark harness applies to model-produced ASTs.
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
    """Result of one parse attempt: either a validated AST plus quality
    signals, or a diagnosis of why no usable AST exists. Backends set
    ``ast`` only to a dict that already passed :func:`try_validate`."""

    ast: Optional[dict]  # validated JSON-compatible AST; None when the parse failed
    confidence: Optional[float]  # backend's self-reported confidence in [0, 1]; None when the backend emits none; not comparable across backends
    backend: str  # registry name of the producing backend
    unsupported: bool = False  # True: no usable AST was produced; `reason` says why
    reason: Optional[str] = None  # failure diagnostics; on success, soft warnings (e.g. literals the AST missed)
    coverage: Optional[float] = None  # literal/keyword coverage fraction in [0, 1]; None when not computed
    latency_ms: Optional[float] = None  # wall-clock parse duration in milliseconds
    raw: Optional[dict] = None  # backend-specific debug payload (interpretation chips, raw generation, ...)


class ParserBackend(ABC):
    """Common interface for every NL -> AST parser backend."""

    name: str = ""  # registry identifier, reported in every ParserOutcome; each subclass overrides

    @abstractmethod
    def parse(self, text: str, now_epoch: float) -> ParserOutcome:
        """Parse `text` into a ParserOutcome. Must never raise: any internal
        failure (engine crash, invalid AST, missing runtime) is reported as
        a failed/unsupported outcome instead."""

    def available(self) -> bool:
        """True when this backend can be used right now. The default
        (zero-dependency) implementation is always available; backends with
        an optional runtime override this to check their runtime/weights."""
        return True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKEND_PATHS: Dict[str, str] = {  # backend name -> dotted class path, imported lazily by get_backend
    "nlq": "omniquery.parsers.nlq.NlqParser",
}


def get_backend(name: str, **kwargs: Any) -> ParserBackend:
    """Construct a fresh backend instance by name.

    Submodules are imported lazily (only on request) so asking for
    'nlq' never pulls in a model runtime, and constructing a backend
    never itself touches an optional runtime -- only calling
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


def make_search_parser() -> ParserBackend:
    """The deterministic search parser (nlq): sub-millisecond, always
    answers. The nl2sql model layer lives ABOVE the parser contract -- it
    speaks SQL, not ASTs (omniquery.parsers.nl2sql.SqlSearch), and the
    endpoint fuses the two: rules answer what they fully consume, the
    model answers the linguistic remainder, model failure falls back to
    the rules result."""
    return get_backend("nlq")


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
    'not' node anywhere: a model that detected negation but returned an
    AST with no 'not' node has lost it."""
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
    """Yield every non-container leaf inside `value`, recursing through
    dicts and lists (a condition value may be a scalar, an 'in'-style list,
    or a structured value like {"days_ago": 7})."""
    if isinstance(value, dict):
        for v in value.values():
            yield from _scalar_leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from _scalar_leaves(v)
    else:
        yield value


def _all_ast_values(ast_dict: dict) -> List[Any]:
    """Every scalar literal the AST commits to (all condition values,
    flattened, plus `limit` when set) -- the pool coverage_guard matches
    the query text's literals against."""
    values: List[Any] = []
    for cond in _walk_conds(ast_dict.get("where")):
        values.extend(_scalar_leaves(cond.get("value")))
    if ast_dict.get("limit") is not None:
        values.append(ast_dict["limit"])
    return values


# ---------------------------------------------------------------------------
# coverage_guard: model-free "did the AST keep every literal" check that
# gates every model-produced AST (grammar-constrained decoding guarantees
# shape, not completeness -- a model can still silently drop a constraint).
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r"[\"']([^\"']+)[\"']")  # either quote style; no escape handling inside

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

# unit word -> multipliers whose products with the raw number are each an
# accepted AST encoding of it: sizes may be stored in MB or bytes, durations
# in seconds, week/month counts in days; unitless kinds multiply by 1.
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
# Normalized (singular, space-separated, lowercase) keyword -> AST enum value.
# Kept in lockstep with the alternatives of _MEDIA_TYPE_RE above.
_MEDIA_TYPE_NORMALIZE: Dict[str, str] = {
    "photo": "image", "picture": "image", "image": "image",
    "video": "video", "clip": "video", "movie": "video",
    "gif": "animated_image", "animated image": "animated_image",
    "sound": "audio", "music": "audio", "song": "audio", "audio": "audio",
    "document": "document", "pdf": "document",
}
_STATUS_WORD_RE = re.compile(
    r"\b(approved|rejected|needs?\s+review|in\s+review|review|to\s+edit|selected|select)\b", re.I,
)
# Normalized keyword -> lowercased AST status_flag value ("To Edit" etc.).
_STATUS_NORMALIZE: Dict[str, str] = {
    "approved": "approved", "rejected": "rejected",
    "need review": "review", "needs review": "review", "in review": "review",
    "review": "review",
    "to edit": "to edit",
    "selected": "select", "select": "select",
}
_FAVORITE_WORD_RE = re.compile(r"\bfavou?rite[sd]?\b", re.I)  # both spellings, optional plural/past-tense suffix
_COUNT_WORD_RE = re.compile(r"\bhow\s+many\b|\bcount\s+of\b|\bnumber\s+of\b", re.I)  # phrasings that demand result="count"


def _number_candidates(raw: str, unit: Optional[str]) -> List[float]:
    """Every numeric value an AST may legally carry for the literal `raw`:
    the number itself plus each unit-scaled variant from _UNIT_MULTIPLIERS."""
    n = float(raw)
    candidates = [n]
    if unit:
        candidates.extend(n * m for m in _UNIT_MULTIPLIERS.get(unit.lower(), ()))
    return candidates


def _normalize_keyword(word: str, table: Dict[str, str]) -> Optional[str]:
    """Look `word` up in `table` after lowercasing, collapsing whitespace/
    hyphens, and (if needed) stripping one plural 's'."""
    w = re.sub(r"[\s-]+", " ", word.lower().strip())
    if w in table:
        return table[w]
    if w.endswith("s") and w[:-1] in table:
        return table[w[:-1]]
    return None


def _cond_values_for_field(ast_dict: dict, field: str) -> set:
    """Every scalar value (lowercased) any condition on `field` carries,
    flattening 'in'-style list values."""
    out: set = set()
    for c in _walk_conds(ast_dict.get("where")):
        if c.get("field") != field:
            continue
        v = c.get("value")
        if isinstance(v, list):
            out.update(str(x).lower() for x in v)
        else:
            out.add(str(v).lower())
    return out


def coverage_guard(text: str, ast_dict: dict) -> Tuple[float, List[str]]:
    """Model-free structural guard: does the AST plausibly reflect every
    literal number, every quoted string, and every recognized keyword class
    present in `text`? Returns (coverage_fraction, miss_descriptions).

    This is deliberately NOT a semantic checker (it can't tell "not
    approved" from "approved" -- that's the negation check next to it in
    a negation check) -- it only catches one specific failure mode: constraints
    silently DROPPED from the parse.  coverage == 1.0 when there is
    nothing to check (an empty/trivial query).
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

    # Each distinct normalized value the text names is its own coverage
    # unit: "images or videos" must find both 'image' and 'video' among
    # the AST's type-condition values, or the dropped half is a miss.
    mentioned_types = {
        t for m in _MEDIA_TYPE_RE.finditer(text)
        if (t := _normalize_keyword(m.group(1), _MEDIA_TYPE_NORMALIZE)) is not None
    }
    if mentioned_types:
        present = _cond_values_for_field(ast_dict, "type")
        for t in sorted(mentioned_types):
            total += 1
            if t in present:
                ok += 1
            else:
                missing.append(f"media-type {t!r} not reflected in any 'type' condition")

    mentioned_statuses = {
        s for m in _STATUS_WORD_RE.finditer(text)
        if (s := _normalize_keyword(m.group(1), _STATUS_NORMALIZE)) is not None
    }
    if mentioned_statuses:
        present = _cond_values_for_field(ast_dict, "status_flag")
        for s in sorted(mentioned_statuses):
            total += 1
            if s in present:
                ok += 1
            else:
                missing.append(f"status {s!r} not reflected in any 'status_flag' condition")

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
    "make_search_parser",
    "try_validate",
    "contains_not_node",
    "coverage_guard",
]
