"""Routing policy across the OmniQuery v2 parser backends.

Policy, in order:

  1. Run the heuristic parser. If it produced an AST with coverage >= 1.0,
     accept it immediately -- a full deterministic parse beats any model's
     guess, no matter what the model would have said.
  2. Otherwise, if the primary backend (Needle2) is available, run it and
     accept its AST iff it is valid AND its coverage >=
     thresholds['needle2_min_coverage'] AND its (uncalibrated!) confidence
     >= thresholds['needle2_min_confidence'].
  3. Otherwise, if a fallback backend (grammar-constrained Qwen) is
     available, run it and accept its AST iff valid AND coverage >=
     thresholds['fallback_min_coverage']. This backend reports no
     confidence signal at all, so coverage is the only gate.
  4. Otherwise, if the heuristic produced a *partial* AST (coverage < 1.0
     but it did parse) with coverage >= thresholds['heuristic_partial_floor'],
     accept it as a best-effort partial parse -- flagged via `reason`/`raw`
     rather than silently returned as if it were a full parse.
  5. Otherwise return an aggregated unsupported outcome collecting every
     backend's reason.

thresholds default to routing_defaults.json's PROVISIONAL values (see that
file's `_note`): they encode qualitative findings from manual probing, not
a fit to measured data. `omniquery/benchmark/harness.py` measures each
backend's actual accuracy/false-confident-rate/escalation-rate against the
SmartGallery corpus and reports recommended thresholds that should
supersede these before the router is trusted in production.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omniquery.parsers import ParserBackend, ParserOutcome

_DEFAULTS_PATH = Path(__file__).with_name("routing_defaults.json")

_THRESHOLD_KEYS = (
    "needle2_min_confidence", "needle2_min_coverage",
    "fallback_min_coverage", "heuristic_partial_floor",
)


def load_thresholds(path: Optional[str] = None) -> Dict[str, float]:
    """Load the four routing thresholds from routing_defaults.json (or an
    override path). Leading-underscore keys (documentation, not thresholds)
    are dropped."""
    p = Path(path) if path else _DEFAULTS_PATH
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {k: float(data[k]) for k in _THRESHOLD_KEYS if k in data}


def fallback_enabled(path: Optional[str] = None) -> bool:
    """Whether the constrained fallback model participates in routing.

    Benchmark-measured default is off (see routing_defaults.json's note);
    the OMNIQUERY_ENABLE_FALLBACK env var overrides in either direction.
    """
    env = os.environ.get("OMNIQUERY_ENABLE_FALLBACK")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    p = Path(path) if path else _DEFAULTS_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("fallback_enabled", False))
    except (OSError, ValueError):
        return False


def _accepts(o: ParserOutcome) -> bool:
    return o.ast is not None and not o.unsupported


class Router:
    def __init__(self, primary: Optional[ParserBackend], fallback: Optional[ParserBackend],
                 heuristic: ParserBackend, thresholds: Dict[str, float]):
        self.primary = primary
        self.fallback = fallback
        self.heuristic = heuristic
        self.thresholds = thresholds

    def route(self, text: str, now_epoch: float) -> Tuple[ParserOutcome, List[ParserOutcome]]:
        trace: List[ParserOutcome] = []

        heuristic_out = self.heuristic.parse(text, now_epoch)
        trace.append(heuristic_out)
        if _accepts(heuristic_out) and (heuristic_out.coverage or 0.0) >= 1.0:
            return heuristic_out, trace

        if self.primary is not None and self.primary.available():
            primary_out = self.primary.parse(text, now_epoch)
            trace.append(primary_out)
            if (_accepts(primary_out)
                    and (primary_out.coverage or 0.0) >= self.thresholds["needle2_min_coverage"]
                    and (primary_out.confidence or 0.0) >= self.thresholds["needle2_min_confidence"]):
                return primary_out, trace

        if self.fallback is not None and self.fallback.available():
            fallback_out = self.fallback.parse(text, now_epoch)
            trace.append(fallback_out)
            if (_accepts(fallback_out)
                    and (fallback_out.coverage or 0.0) >= self.thresholds["fallback_min_coverage"]):
                return fallback_out, trace

        if (_accepts(heuristic_out)
                and (heuristic_out.coverage or 0.0) >= self.thresholds["heuristic_partial_floor"]):
            partial_reason = "accepted as a partial heuristic parse (below full coverage)"
            if heuristic_out.reason:
                partial_reason = f"{partial_reason}: {heuristic_out.reason}"
            warned = replace(heuristic_out, reason=partial_reason,
                              raw={**(heuristic_out.raw or {}), "warning": "partial_heuristic_accept"})
            return warned, trace

        reasons = "; ".join(f"{o.backend}: {o.reason}" for o in trace if o.reason)
        aggregate = ParserOutcome(
            ast=None, confidence=None, backend="router", unsupported=True,
            reason=reasons or "no backend produced an acceptable parse",
        )
        return aggregate, trace
