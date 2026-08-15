"""Needle2 (cactus-needle) backend: primary NL -> AST parser.

The engine parses small tool schemas (<=6 params) reliably; a single wide
tool (13+ properties) makes it hallucinate a value for every property; 20+
properties makes it return no call at all. Two topologies are offered,
both sized to stay inside that envelope:

  'frame'  -- one tool, `query_gallery`, with exactly 8 flat parameters.
  'family' -- seven small tools (<=2 params each); tends to pick the
              right tools but DROPS constraints on multi-constraint queries.

Either way the model never emits the typed AST directly -- it emits a flat
"frame" (tool-call arguments merged into one dict, last-call-wins per key)
that `frame_to_ast` deterministically expands into the real AST. Confidence
from the engine is carried through uninterpreted (it is NOT calibrated --
see routing_defaults.json / omniquery/benchmark/harness.py for why routing
doesn't trust it alone); `coverage_guard` (parsers/__init__.py) is the
model-free signal that actually catches dropped constraints.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from omniquery import fields
from omniquery.parsers import (
    ParserBackend,
    ParserOutcome,
    contains_not_node,
    coverage_guard,
    try_validate,
)

# Enum lists sorted so the tool schemas (and thus prompts) are byte-stable.
_FILE_TYPES = sorted(fields.FILE_TYPE_VALUES)
_STATUS_FLAGS = sorted(fields.STATUS_FLAG_VALUES)
# Frame 'order_by' keyword -> (AST order field, direction).
_ORDER_MAP: Dict[str, Tuple[str, str]] = {
    "newest": ("mtime", "desc"), "oldest": ("mtime", "asc"),
    "largest": ("size_bytes", "desc"), "rating": ("rating_avg", "desc"),
}

# 'frame' topology: the entire frame vocabulary as one 8-parameter tool.
FRAME_TOOL: Dict[str, Any] = {
    "name": "query_gallery",
    "description": "Query the SmartGallery media library for files matching filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "result": {"type": "string", "enum": ["ids", "count"],
                       "description": "Return matching file ids, or just a count."},
            "media_type": {"type": "string", "enum": _FILE_TYPES,
                           "description": "Restrict to one media type."},
            "favorite": {"type": "boolean", "description": "Only favorited files."},
            "min_rating": {"type": "integer", "minimum": 1, "maximum": 5,
                           "description": "Minimum average star rating."},
            "days_ago_max": {"type": "integer", "minimum": 0,
                              "description": "Only files modified within this many days."},
            "name_or_prompt_contains": {"type": "string",
                                         "description": "Substring to match in file name or prompt."},
            "status_flag": {"type": "string", "enum": _STATUS_FLAGS,
                             "description": "Review status flag."},
            "order_by": {"type": "string", "enum": ["newest", "oldest", "largest", "rating"],
                         "description": "Sort order for results."},
        },
    },
}

# 'family' topology: the frame vocabulary split across seven tools of <=2
# parameters each; additionally exposes min_size_mb, which FRAME_TOOL lacks.
FAMILY_TOOLS: List[Dict[str, Any]] = [
    {"name": "filter_kind", "description": "Filter by media type.",
     "parameters": {"type": "object", "properties": {
         "media_type": {"type": "string", "enum": _FILE_TYPES}}}},
    {"name": "filter_rating", "description": "Filter by favorite status or star rating.",
     "parameters": {"type": "object", "properties": {
         "favorite": {"type": "boolean"},
         "min_rating": {"type": "integer", "minimum": 1, "maximum": 5}}}},
    {"name": "filter_time", "description": "Filter by recency, in days.",
     "parameters": {"type": "object", "properties": {
         "days_ago_max": {"type": "integer", "minimum": 0}}}},
    {"name": "filter_text", "description": "Filter by name or prompt text search.",
     "parameters": {"type": "object", "properties": {
         "name_or_prompt_contains": {"type": "string"}}}},
    {"name": "filter_size", "description": "Filter by minimum file size in MB.",
     "parameters": {"type": "object", "properties": {
         "min_size_mb": {"type": "number"}}}},
    {"name": "filter_status", "description": "Filter by review status flag.",
     "parameters": {"type": "object", "properties": {
         "status_flag": {"type": "string", "enum": _STATUS_FLAGS}}}},
    {"name": "present", "description": "Choose result kind and sort order.",
     "parameters": {"type": "object", "properties": {
         "result": {"type": "string", "enum": ["ids", "count"]},
         "order_by": {"type": "string", "enum": ["newest", "oldest", "largest", "rating"]}}}},
]

# Textual evidence that favorite=False in a frame is intended negation
# rather than the engine's default-argument noise (see frame_to_ast).
_FAVORITE_NEGATION_RE = re.compile(
    r"\b(?:not|except|without)\s+(?:a\s+)?favou?rite[sd]?\b|\bun-?favou?rited\b", re.I,
)


def _is_number(v: Any) -> bool:
    """True for int/float but not bool (bool is an int subclass)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def frame_to_ast(frame: Dict[str, Any], text: str) -> dict:
    """Deterministic frame -> AST expansion.

    Falsy-noise values are dropped: empty strings always; `favorite` /
    `has_workflow`-shaped booleans of False are dropped UNLESS the query
    text itself contains a negated-favorite pattern (in which case they're
    kept as an explicit is_favorite=False, since that's the one case where
    "false" is signal, not the engine's default-argument noise); 0 is
    dropped for ratings/sizes/days.
    """
    conds: List[dict] = []

    media_type = frame.get("media_type")
    if isinstance(media_type, str) and media_type:
        conds.append({"field": "type", "op": "eq", "value": media_type})

    favorite = frame.get("favorite")
    if isinstance(favorite, bool):
        if favorite:
            conds.append({"field": "is_favorite", "op": "eq", "value": True})
        elif _FAVORITE_NEGATION_RE.search(text):
            conds.append({"field": "is_favorite", "op": "eq", "value": False})

    min_rating = frame.get("min_rating")
    if _is_number(min_rating) and min_rating:
        conds.append({"field": "rating_avg", "op": "ge", "value": min_rating})

    days_ago_max = frame.get("days_ago_max")
    if _is_number(days_ago_max) and days_ago_max:
        conds.append({"field": "mtime", "op": "ge", "value": {"days_ago": days_ago_max}})

    name_or_prompt = frame.get("name_or_prompt_contains")
    if isinstance(name_or_prompt, str) and name_or_prompt:
        conds.append({"op": "or", "children": [
            {"field": "name", "op": "contains", "value": name_or_prompt},
            {"field": "workflow_prompt", "op": "contains", "value": name_or_prompt},
        ]})

    status_flag = frame.get("status_flag")
    if isinstance(status_flag, str) and status_flag in fields.STATUS_FLAG_VALUES:
        conds.append({"field": "status_flag", "op": "eq", "value": status_flag})

    min_size_mb = frame.get("min_size_mb")
    if _is_number(min_size_mb) and min_size_mb:
        conds.append({"field": "size_mb", "op": "ge", "value": min_size_mb})

    where: Optional[dict] = None
    if len(conds) == 1:
        where = conds[0]
    elif len(conds) > 1:
        where = {"op": "and", "children": conds}

    result = frame.get("result")
    if result not in ("ids", "count"):
        result = "ids"

    ast_dict: Dict[str, Any] = {"result": result}
    if where is not None:
        ast_dict["where"] = where

    order_by = frame.get("order_by")
    if order_by in _ORDER_MAP:
        field_name, direction = _ORDER_MAP[order_by]
        ast_dict["order_by"] = [{"field": field_name, "dir": direction}]

    return ast_dict


def _merge_calls(function_calls: List[dict]) -> Dict[str, Any]:
    """Merge every function call's arguments into one flat frame, last call
    wins per key (topology-agnostic: works whether all calls hit the same
    single 'frame' tool or several of the 'family' tools)."""
    frame: Dict[str, Any] = {}
    for call in function_calls:
        args = call.get("arguments")
        if isinstance(args, dict):
            frame.update(args)
    return frame


def _ms(t0: float) -> float:
    """Milliseconds elapsed since monotonic timestamp `t0`."""
    return (time.monotonic() - t0) * 1000.0


class Needle2Backend(ParserBackend):
    """Wraps the needle engine behind the ParserBackend contract: engine
    tool-calls -> merged frame -> deterministic AST, gated by validation,
    coverage_guard, and the engine's negation flag."""

    name = "needle2"

    def __init__(self, topology: str = "frame", max_new_tokens: int = 512):
        """`topology` selects FRAME_TOOL ('frame') or FAMILY_TOOLS
        ('family'); the engine runtime is not touched until
        available()/parse()."""
        if topology not in ("frame", "family"):
            raise ValueError(f"unknown topology {topology!r}; expected 'frame' or 'family'")
        self.topology = topology
        self.max_new_tokens = max_new_tokens
        self._agent: Any = None  # lazily-built needle engine; None until first use
        self._available: Optional[bool] = None  # cached probe result; None = not yet probed

    def _tools(self) -> Any:
        """Tool schema set for the configured topology."""
        return [FRAME_TOOL] if self.topology == "frame" else FAMILY_TOOLS

    def _get_agent(self) -> Any:
        """Build (once) and return the needle engine; the import is deferred
        so the optional runtime is only required when actually used."""
        if self._agent is None:
            import needle  # local import: never required unless this backend is used
            self._agent = needle.Needle(tools=self._tools())
        return self._agent

    def available(self) -> bool:
        """Probe (once) whether the needle engine can be constructed; the
        result is cached for this backend's lifetime."""
        if self._available is None:
            try:
                self._get_agent()
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def parse(self, text: str, now_epoch: float) -> ParserOutcome:  # noqa: ARG002
        """Run the engine and expand its calls into a validated AST; if the
        engine reports detecting negation but the expanded AST carries no
        'not' node, the parse is refused as unsupported. `now_epoch` is
        unused: relative dates stay symbolic as {"days_ago": N}. Never
        raises."""
        t0 = time.monotonic()
        try:
            agent = self._get_agent()
        except Exception as exc:
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=f"engine error: {exc}", latency_ms=_ms(t0))

        try:
            agent.reset()
            resp = agent.complete(text, max_new_tokens=self.max_new_tokens)
        except Exception as exc:
            # The C engine can truncate output at the token budget, and the
            # Python wrapper then raises json.JSONDecodeError (or similar)
            # trying to parse it. Never let that escape.
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=f"engine error: {exc}", latency_ms=_ms(t0))

        latency_ms = _ms(t0)
        confidence = resp.get("confidence") if isinstance(resp, dict) else None

        if not isinstance(resp, dict) or not resp.get("success") or not resp.get("function_calls"):
            error = resp.get("error") if isinstance(resp, dict) else "malformed response"
            return ParserOutcome(ast=None, confidence=confidence, backend=self.name, unsupported=True,
                                  reason=f"no function call: {error}", latency_ms=latency_ms, raw=resp)

        frame = _merge_calls(resp["function_calls"])
        ast_dict = frame_to_ast(frame, text)
        coverage, missing = coverage_guard(text, ast_dict)

        validation_info = resp.get("validation") or {}
        if validation_info.get("negation") and not contains_not_node(ast_dict.get("where")):
            return ParserOutcome(ast=None, confidence=confidence, backend=self.name, unsupported=True,
                                  reason="negation not expressible in frame", coverage=coverage,
                                  latency_ms=latency_ms, raw=resp)

        query, err = try_validate(ast_dict)
        if err is not None:
            return ParserOutcome(ast=None, confidence=confidence, backend=self.name, unsupported=True,
                                  reason=f"invalid AST: {err}", coverage=coverage,
                                  latency_ms=latency_ms, raw=resp)

        return ParserOutcome(ast=query.to_dict(), confidence=confidence, backend=self.name,
                              unsupported=False, reason=("; ".join(missing) or None),
                              coverage=coverage, latency_ms=latency_ms, raw=resp)
