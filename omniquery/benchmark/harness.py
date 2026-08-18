"""OmniQuery NL-parser accuracy/latency benchmark harness.

Runs one or more parser backends (plus, optionally, the full `search`
policy -- nlq with nl2sql refinement) against `corpus.jsonl` and reports,
per backend:

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

`search` additionally reports refinement stats: which backend ultimately
answered each entry, and what fraction of entries consulted the model.

CLI: `python -m omniquery.benchmark.harness --backends nlq,sqlsearch --out report.json`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

try:
    import resource  # POSIX-only; absent on Windows
except ImportError:
    resource = None
import ctypes
import os
import tempfile
from collections import Counter
from ctypes import wintypes
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from omniquery.ast import canonicalize
from omniquery.benchmark.fixtures import ANCHOR_EPOCH, FIXTURE_BASE_PATH, build_fixture_db
from omniquery.engine import OmniQueryEngine
from omniquery.parsers import ParserBackend, ParserOutcome, get_backend, try_validate
from omniquery.parsers.nl2sql import SqlSearch
from omniquery.parsers.nlq import NlqParser
from omniquery.validation import AuthContext

_DEFAULT_CORPUS_PATH = Path(__file__).with_name("corpus.jsonl")

# Maximally privileged context, so no corpus entry is rejected on
# authorization grounds -- the benchmark measures parsing, not access control.
BENCH_CTX = AuthContext(role="ADMIN", user_id="bench", client_uuid="bench", ai_enabled=True)

# Fixed-id stand-ins for the model-backed similarity searches, keeping
# execution comparisons deterministic and model-free.
_STUB_AI_RESOLVERS = {
    "similar_to_semantic": lambda _v: ["f001", "f002"],
    "similar_to_visual": lambda _v: ["f001"],
    "near_dup_of": lambda _v: ["f003"],
}

_THETA_STEPS = [round(i * 0.1, 1) for i in range(10)]  # 0.0 .. 0.9


# ---------------------------------------------------------------------------
# Corpus / fixture plumbing
# ---------------------------------------------------------------------------


def load_corpus(corpus_path: Path) -> list[dict]:
    """Parse a JSONL corpus: one entry per non-blank line, in file order."""
    entries: list[dict] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _date_placeholder_map(now_epoch: float) -> dict[str, str]:
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


def _resolve_date_placeholders(obj: Any, mapping: dict[str, str]) -> Any:
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
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="omniquery_bench_")
    os.close(fd)
    build_fixture_db(db_path, seed=42)
    return OmniQueryEngine(db_path=db_path, base_path=FIXTURE_BASE_PATH, ai_resolvers=_STUB_AI_RESOLVERS)


def _exec_result(engine: OmniQueryEngine, query: Any, now_epoch: float) -> tuple[str, Any] | None:
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
    confidence: float | None  # backend's self-reported confidence, if any


def _score_entry(expected: dict, outcome: ParserOutcome, engine: OmniQueryEngine, now_epoch: float) -> _EntryRecord:
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

    return _EntryRecord(
        expected_unsupported=expected_unsupported,
        produced_ast=produced_ast_dict is not None,
        valid=valid,
        ast_exact=ast_exact,
        exec_match=exec_match,
        confidence=outcome.confidence,
    )


def _latency_stats(latencies_ms: list[float]) -> dict[str, float | None]:
    """p50/p95/mean over millisecond latencies; all None when empty."""
    if not latencies_ms:
        return {"p50": None, "p95": None, "mean": None}
    s = sorted(latencies_ms)

    def _pct(p: float) -> float:
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    return {"p50": _pct(0.50), "p95": _pct(0.95), "mean": sum(s) / len(s)}


def _false_confident_sweep(records: list[_EntryRecord]) -> dict[str, float | None]:
    """Per theta in _THETA_STEPS (keyed by its string form): among entries
    where the backend produced an AST and reported confidence >= theta, the
    fraction that are execution-mismatched; None where nothing clears theta."""
    scored = [(r.confidence, not r.exec_match) for r in records if r.produced_ast and r.confidence is not None]
    result: dict[str, float | None] = {}
    for theta in _THETA_STEPS:
        confident = [mismatched for conf, mismatched in scored if conf >= theta]
        result[str(theta)] = (sum(confident) / len(confident)) if confident else None
    return result


def _aggregate(records: list[_EntryRecord], latencies_ms: list[float]) -> dict[str, Any]:
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
        return peak // 1024 if sys.platform == "darwin" else peak

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
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) // 1024 if ok else 0


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------


def _run_backend(
    backend: ParserBackend, entries: list[dict], engine: OmniQueryEngine, now_epoch: float
) -> dict[str, Any]:
    """Run one backend over every entry -- timing only the .parse() call --
    and aggregate the scores."""
    records: list[_EntryRecord] = []
    latencies: list[float] = []
    for entry in entries:
        t0 = time.monotonic()
        outcome = backend.parse(entry["nl"], now_epoch)
        latencies.append((time.monotonic() - t0) * 1000.0)
        records.append(_score_entry(entry["expected"], outcome, engine, now_epoch))
    return _aggregate(records, latencies)


def _run_fusion(entries: list[dict], engine: OmniQueryEngine, now_epoch: float) -> dict[str, Any]:
    """Measure the SHIPPED endpoint policy: nlq answers entries it fully
    consumes (no leftover text terms); free-language entries go to the
    SqlSearch agentic loop, and a model failure falls back to the nlq
    answer. This is the product's acceptance number."""

    nlq = NlqParser()
    search = SqlSearch(db_path=engine.db_path)
    records: list[_EntryRecord] = []
    latencies: list[float] = []
    model_used = model_correct = 0
    for entry in entries:
        expected_query, _ = try_validate(entry["expected"]["ast"])
        want = _exec_result(engine, expected_query, now_epoch)

        def _rules_ok() -> bool:
            query, err = try_validate(out.ast)
            got = _exec_result(engine, query, now_epoch) if err is None else None
            return got is not None and got == want

        t0 = time.monotonic()
        out = nlq.parse(entry["nl"], now_epoch)
        if not (out.raw or {}).get("text_terms"):
            ok = _rules_ok()  # fully consumed: the rules answer is final
        else:
            model_used += 1
            ids, sql, _err = search.search(entry["nl"])
            if ids is None:
                ok = _rules_ok()  # model hard-failed: rules answer stands
            elif sql and re.match(r"\s*SELECT\s+(COUNT|SUM|AVG|MIN|MAX)", sql, re.IGNORECASE):
                got_count = int(float(ids[0])) if ids else 0
                want_count = want[1] if want[0] == "count" else len(want[1])
                ok = got_count == want_count
                model_correct += int(ok)
            else:
                ok = want is not None and want[0] == "ids" and frozenset(ids) == want[1]
                model_correct += int(ok)
        latencies.append((time.monotonic() - t0) * 1000.0)
        records.append(
            _EntryRecord(
                expected_unsupported=False,
                produced_ast=True,
                valid=True,
                ast_exact=False,
                exec_match=ok,
                confidence=None,
            )
        )
    metrics = _aggregate(records, latencies)
    metrics["fusion"] = {"model_used": model_used, "model_correct": model_correct}
    return metrics


def _run_sqlsearch(entries: list[dict], engine: OmniQueryEngine, now_epoch: float) -> dict[str, Any]:
    """Measure the nl2sql SQL path (omniquery.parsers.nl2sql.SqlSearch):
    the model's agentic generate/execute/read-results loop runs against
    the fixture DB, and its id set is compared with the expected AST's
    engine result. exec-match is the only meaningful metric here (there is
    no AST to compare); COUNT answers match when the count equals the
    expected id-set size."""

    search = SqlSearch(db_path=engine.db_path)
    records: list[_EntryRecord] = []
    latencies: list[float] = []
    fail_reasons: Counter = Counter()
    for entry in entries:
        expected = entry["expected"]
        expected_query, _ = try_validate(expected["ast"])
        want = _exec_result(engine, expected_query, now_epoch)

        t0 = time.monotonic()
        ids, sql, err = search.search(entry["nl"])
        latencies.append((time.monotonic() - t0) * 1000.0)

        exec_match = False
        if ids is not None and want is not None:
            if sql and re.match(r"\s*SELECT\s+COUNT", sql, re.IGNORECASE):
                got_count = int(ids[0]) if ids else 0
                want_count = want[1] if want[0] == "count" else len(want[1])
                exec_match = got_count == want_count
            else:
                got: Any = frozenset(ids)
                want_set = want[1] if want[0] == "ids" else None
                exec_match = want_set is not None and got == want_set
        if ids is None:
            fail_reasons[(err or "unknown")[:60]] += 1
        records.append(
            _EntryRecord(
                expected_unsupported=False,
                produced_ast=ids is not None,
                valid=ids is not None,
                ast_exact=False,
                exec_match=exec_match,
                confidence=None,
            )
        )
    metrics = _aggregate(records, latencies)
    metrics["sql_fail_reasons"] = dict(fail_reasons)
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    backend_names: list[str],
    corpus_path: Any = _DEFAULT_CORPUS_PATH,
    out_path: str | None = None,
    now_epoch: float = ANCHOR_EPOCH,
) -> dict[str, Any]:
    """Run each of `backend_names` (any of 'nlq', 'nl2sql',
    'sqlsearch') against the corpus at `corpus_path`. Writes a
    JSON report to `out_path` if given, and always returns the report dict.
    """
    entries = load_corpus(Path(corpus_path))
    entries = _resolve_date_placeholders(entries, _date_placeholder_map(now_epoch))
    engine = _build_fixture_engine()

    instances: dict[str, ParserBackend] = {
        name: get_backend(name) for name in backend_names if name not in ("sqlsearch", "fusion")
    }

    report: dict[str, Any] = {
        "now_epoch": now_epoch,
        "corpus_path": str(corpus_path),
        "corpus_size": len(entries),
        "backends": {},
    }

    for name in backend_names:
        if name == "sqlsearch":
            report["backends"]["sqlsearch"] = _run_sqlsearch(entries, engine, now_epoch)
        elif name == "fusion":
            report["backends"]["fusion"] = _run_fusion(entries, engine, now_epoch)
        else:
            report["backends"][name] = _run_backend(instances[name], entries, engine, now_epoch)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI argument parsing; `argv=None` reads sys.argv."""
    parser = argparse.ArgumentParser(description="OmniQuery v2 NL-parser benchmark")
    parser.add_argument(
        "--backends",
        type=str,
        default="nlq",
        help="comma-separated backend names, e.g. nlq,nl2sql,search",
    )
    parser.add_argument("--corpus", type=str, default=str(_DEFAULT_CORPUS_PATH))
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
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
