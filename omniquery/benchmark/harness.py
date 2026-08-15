"""OmniQuery v2 NL-parser accuracy/latency benchmark harness.

Runs one or more parser backends (plus, optionally, the full `router`
policy) against `corpus.jsonl` and reports, per backend:

  - ast_exact_rate: canonicalized-AST equality against the corpus's
    expected AST (over entries expected to be supported).
  - execution_match_rate: same result set (ids) / same count when the
    produced AST and the expected AST are both run against the
    deterministic fixture DB (omniquery.benchmark.fixtures), through
    omniquery.engine with stub AI resolvers that return fixed ids.
  - invalid_rate: fraction of "I have an answer" outcomes whose AST the
    harness itself (not just the backend's own internal check) fails to
    validate -- a regression guard, expected to be ~0 for a correctly
    implemented backend.
  - unsupported_correct / unsupported_incorrect: for corpus entries marked
    `{"unsupported": true}` (ambiguous/adversarial), whether the backend
    also declined vs. confidently produced an AST anyway.
  - false_confident_rate(theta): swept over theta in 0.0..0.9 step 0.1 --
    of the entries where the backend both produced an AST AND reported
    confidence >= theta, what fraction were execution-mismatched (includes
    every unsupported_incorrect case, since "confidently wrong" there too).
  - latency p50/p95/mean (wall-clock around `.parse()`/`.route()`).
  - peak_rss_kb: process-wide peak resident set size in KB, sampled after
    the backend's entries all ran (getrusage on POSIX, psapi on Windows).

`router` additionally reports escalation stats: which backend ultimately
answered each entry, and what fraction of entries needed to escalate past
the heuristic's outright-accept rule.

CLI: `python -m omniquery.benchmark.harness --backends heuristic,needle2,fallback_qwen,router --out report.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time

try:
    import resource  # POSIX-only; absent on Windows
except ImportError:
    resource = None
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omniquery.ast import canonicalize
from omniquery.benchmark.fixtures import ANCHOR_EPOCH, FIXTURE_BASE_PATH, build_fixture_db
from omniquery.engine import OmniQueryEngine
from omniquery.parsers import ParserBackend, ParserOutcome, get_backend, try_validate
from omniquery.parsers.router import Router, load_thresholds
from omniquery.validation import AuthContext

_DEFAULT_CORPUS_PATH = Path(__file__).with_name("corpus.jsonl")

# Maximally privileged context, so no corpus entry is rejected on
# authorization grounds -- the benchmark measures parsing, not access control.
BENCH_CTX = AuthContext(role="ADMIN", user_id="bench", client_uuid="bench", ai_enabled=True)

# Fixed-id stand-ins for the model-backed similarity searches, keeping
# execution comparisons deterministic and model-free.
_STUB_AI_RESOLVERS = {
    "similar_to_semantic": lambda v: ["f001", "f002"],
    "similar_to_visual": lambda v: ["f001"],
    "near_dup_of": lambda v: ["f003"],
}

_THETA_STEPS = [round(i * 0.1, 1) for i in range(10)]  # 0.0 .. 0.9


# ---------------------------------------------------------------------------
# Corpus / fixture plumbing
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: Path) -> List[dict]:
    """Parse a JSONL corpus: one entry per non-blank line, in file order."""
    entries: List[dict] = []
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _date_placeholder_map(now_epoch: float) -> Dict[str, str]:
    """Calendar anchors for corpus placeholders, derived from the SAME
    injected clock the parsers receive. The corpus can't hardcode dates for
    calendar vocabulary ("yesterday" has no fixed value), so it carries
    tokens and the harness resolves them with the calendar-boundary
    semantics the parsers are contractually required to implement:
    local-timezone days, ISO weeks starting Monday, calendar months."""
    local_today = date.fromtimestamp(now_epoch)
    return {
        "<today>": local_today.isoformat(),
        "<yesterday>": (local_today - timedelta(days=1)).isoformat(),
        "<week_start>": (local_today - timedelta(days=local_today.weekday())).isoformat(),
        "<month_start>": local_today.replace(day=1).isoformat(),
    }


def _resolve_date_placeholders(obj: Any, mapping: Dict[str, str]) -> Any:
    """Deep copy of `obj` with every string equal to a placeholder token
    replaced by its resolved date; everything else passes through unchanged."""
    if isinstance(obj, dict):
        return {k: _resolve_date_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_date_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str) and obj in mapping:
        return mapping[obj]
    return obj


def _build_fixture_engine() -> OmniQueryEngine:
    """Engine over a freshly built fixture database (default seed) in a
    temp file, with the stub AI resolvers wired in. mkstemp reserves the
    file atomically, so concurrent runs can never share a database."""
    import os
    import tempfile
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="omniquery_bench_")
    os.close(fd)
    build_fixture_db(db_path, seed=42)
    return OmniQueryEngine(db_path=db_path, base_path=FIXTURE_BASE_PATH,
                            ai_resolvers=_STUB_AI_RESOLVERS)


def _exec_result(engine: OmniQueryEngine, query: Any, now_epoch: float) -> Optional[Tuple[str, Any]]:
    """Comparable execution outcome: ("count", n) or ("ids", frozenset).
    None means execution failed, and callers treat None as matching nothing,
    so a failed run can never count as an execution match."""
    out = engine.run(query, BENCH_CTX, now_epoch=now_epoch)
    if not out.ok:
        return None
    if out.kind == "count":
        return ("count", out.count)
    return ("ids", frozenset(out.ids or ()))


# ---------------------------------------------------------------------------
# Per-entry scoring
# ---------------------------------------------------------------------------

@dataclass
class _EntryRecord:
    """Scoring flags for one corpus entry under one backend."""
    expected_unsupported: bool  # corpus marks the entry {"unsupported": true}
    produced_ast: bool  # backend answered with an AST rather than declining
    valid: bool  # harness-side validation of the produced AST passed
    ast_exact: bool  # canonicalized produced AST equals the expected AST
    exec_match: bool  # produced and expected ASTs yield the same fixture-DB result
    confidence: Optional[float]  # backend's self-reported confidence, if any


def _score_entry(expected: dict, outcome: ParserOutcome, engine: OmniQueryEngine,
                  now_epoch: float) -> _EntryRecord:
    """Score one backend outcome against its corpus entry. ast_exact and
    exec_match stay False for entries expected to be unsupported -- there
    is no expected AST to compare against."""
    expected_unsupported = bool(expected.get("unsupported"))
    produced_ast_dict = outcome.ast if (outcome.ast is not None and not outcome.unsupported) else None

    valid = False
    ast_exact = False
    exec_match = False
    if produced_ast_dict is not None:
        query, err = try_validate(produced_ast_dict)
        if err is None:
            valid = True
            if not expected_unsupported:
                expected_query, _ = try_validate(expected["ast"])  # corpus is pre-verified valid
                if canonicalize(query) == canonicalize(expected_query):
                    ast_exact = True
                got = _exec_result(engine, query, now_epoch)
                want = _exec_result(engine, expected_query, now_epoch)
                if got is not None and got == want:
                    exec_match = True

    return _EntryRecord(expected_unsupported=expected_unsupported, produced_ast=produced_ast_dict is not None,
                         valid=valid, ast_exact=ast_exact, exec_match=exec_match, confidence=outcome.confidence)


def _latency_stats(latencies_ms: List[float]) -> Dict[str, Optional[float]]:
    """p50/p95/mean over millisecond latencies; all None when empty."""
    if not latencies_ms:
        return {"p50": None, "p95": None, "mean": None}
    s = sorted(latencies_ms)

    def _pct(p: float) -> float:
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    return {"p50": _pct(0.50), "p95": _pct(0.95), "mean": sum(s) / len(s)}


def _false_confident_sweep(records: List[_EntryRecord]) -> Dict[str, Optional[float]]:
    """Per theta in _THETA_STEPS (keyed by its string form): among entries
    where the backend produced an AST and reported confidence >= theta, the
    fraction that are execution-mismatched; None where nothing clears theta."""
    scored = [(r.confidence, not r.exec_match) for r in records if r.produced_ast and r.confidence is not None]
    result: Dict[str, Optional[float]] = {}
    for theta in _THETA_STEPS:
        confident = [mismatched for conf, mismatched in scored if conf >= theta]
        result[str(theta)] = (sum(confident) / len(confident)) if confident else None
    return result


def _aggregate(records: List[_EntryRecord], latencies_ms: List[float]) -> Dict[str, Any]:
    """Fold per-entry records and latencies into one backend's metrics
    block; the module docstring defines each metric."""
    n_total = len(records)
    n_supported = sum(1 for r in records if not r.expected_unsupported)
    n_unsupported = n_total - n_supported
    n_with_ast = sum(1 for r in records if r.produced_ast)

    return {
        "n_entries": n_total,
        "n_supported_expected": n_supported,
        "n_unsupported_expected": n_unsupported,
        "n_with_ast": n_with_ast,
        "ast_exact_rate": sum(1 for r in records if r.ast_exact) / max(n_supported, 1),
        "execution_match_rate": sum(1 for r in records if r.exec_match) / max(n_supported, 1),
        "invalid_rate": sum(1 for r in records if r.produced_ast and not r.valid) / max(n_with_ast, 1),
        "unsupported_correct": sum(1 for r in records if r.expected_unsupported and not r.produced_ast),
        "unsupported_incorrect": sum(1 for r in records if r.expected_unsupported and r.produced_ast),
        "unsupported_correct_rate": (
            sum(1 for r in records if r.expected_unsupported and not r.produced_ast) / max(n_unsupported, 1)
        ),
        "false_confident_rate": _false_confident_sweep(records),
        "latency_ms": _latency_stats(latencies_ms),
        "peak_rss_kb": _peak_rss_kb(),
    }


def _peak_rss_kb() -> int:
    """Process-wide peak resident set size in KB.

    POSIX: getrusage ru_maxrss (KB on Linux, bytes on macOS). Windows has no
    `resource` module; psapi GetProcessMemoryInfo's PeakWorkingSetSize (bytes)
    is the equivalent high-water mark.
    """
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak // 1024 if sys.platform == 'darwin' else peak

    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # Typed signatures are load-bearing: the GetCurrentProcess pseudo-handle
    # ((HANDLE)-1) truncates to a 32-bit int under ctypes' default
    # conversions on 64-bit Windows and yields ERROR_INVALID_HANDLE.
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
    ]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = kernel32.K32GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) // 1024 if ok else 0


# ---------------------------------------------------------------------------
# Backend / router runners
# ---------------------------------------------------------------------------

def _run_backend(backend: ParserBackend, entries: List[dict], engine: OmniQueryEngine,
                  now_epoch: float) -> Dict[str, Any]:
    """Run one backend over every entry -- timing only the .parse() call --
    and aggregate the scores."""
    records: List[_EntryRecord] = []
    latencies: List[float] = []
    for entry in entries:
        t0 = time.monotonic()
        outcome = backend.parse(entry["nl"], now_epoch)
        latencies.append((time.monotonic() - t0) * 1000.0)
        records.append(_score_entry(entry["expected"], outcome, engine, now_epoch))
    return _aggregate(records, latencies)


def _run_router(router: Router, entries: List[dict], engine: OmniQueryEngine,
                 now_epoch: float) -> Dict[str, Any]:
    """Like _run_backend but through router.route(), adding escalation
    stats: which backend ultimately answered each entry, and how often the
    heuristic's outright-accept rule did not settle it (trace longer than
    one hop)."""
    records: List[_EntryRecord] = []
    latencies: List[float] = []
    winner_counts: Counter = Counter()
    escalated_past_heuristic = 0
    for entry in entries:
        t0 = time.monotonic()
        outcome, trace = router.route(entry["nl"], now_epoch)
        latencies.append((time.monotonic() - t0) * 1000.0)
        records.append(_score_entry(entry["expected"], outcome, engine, now_epoch))
        winner_counts[outcome.backend] += 1
        if len(trace) > 1:
            escalated_past_heuristic += 1
    metrics = _aggregate(records, latencies)
    metrics["escalation"] = {
        "winner_counts": dict(winner_counts),
        "escalated_past_heuristic_rate": escalated_past_heuristic / max(len(entries), 1),
    }
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_benchmark(backend_names: List[str], corpus_path: Any = _DEFAULT_CORPUS_PATH,
                   out_path: Optional[str] = None, now_epoch: float = ANCHOR_EPOCH) -> Dict[str, Any]:
    """Run each of `backend_names` (any of 'heuristic', 'needle2',
    'fallback_qwen', 'router') against the corpus at `corpus_path`. Writes a
    JSON report to `out_path` if given, and always returns the report dict.
    """
    entries = load_corpus(Path(corpus_path))
    entries = _resolve_date_placeholders(entries, _date_placeholder_map(now_epoch))
    engine = _build_fixture_engine()

    instances: Dict[str, ParserBackend] = {
        name: get_backend(name) for name in backend_names if name != "router"
    }

    report: Dict[str, Any] = {
        "now_epoch": now_epoch,
        "corpus_path": str(corpus_path),
        "corpus_size": len(entries),
        "backends": {},
    }

    for name in backend_names:
        if name == "router":
            router = Router(
                primary=instances.get("needle2") or get_backend("needle2"),
                fallback=instances.get("fallback_qwen") or get_backend("fallback_qwen"),
                heuristic=instances.get("heuristic") or get_backend("heuristic"),
                thresholds=load_thresholds(),
            )
            report["backends"]["router"] = _run_router(router, entries, engine, now_epoch)
        else:
            report["backends"][name] = _run_backend(instances[name], entries, engine, now_epoch)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    return report


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI argument parsing; `argv=None` reads sys.argv."""
    parser = argparse.ArgumentParser(description="OmniQuery v2 NL-parser benchmark")
    parser.add_argument(
        "--backends", type=str, default="heuristic",
        help="comma-separated backend names, e.g. heuristic,needle2,fallback_qwen,router",
    )
    parser.add_argument("--corpus", type=str, default=str(_DEFAULT_CORPUS_PATH))
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point: run the benchmark, then print the report to stdout
    (or just the destination path when --out is given)."""
    args = _parse_args(argv)
    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]
    report = run_benchmark(backend_names, corpus_path=args.corpus, out_path=args.out)
    if not args.out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote report to {args.out}")


if __name__ == "__main__":
    main()
