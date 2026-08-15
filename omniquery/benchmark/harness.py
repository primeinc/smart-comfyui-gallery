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
  - peak_rss_kb: `resource.getrusage(RUSAGE_SELF).ru_maxrss` sampled after
    the backend's entries all ran (process-wide high-water mark).

`router` additionally reports escalation stats: which backend ultimately
answered each entry, and what fraction of entries needed to escalate past
the heuristic's outright-accept rule.

CLI: `python -m omniquery.benchmark.harness --backends heuristic,needle2,fallback_qwen,router --out report.json`
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omniquery.ast import canonicalize
from omniquery.benchmark.fixtures import ANCHOR_EPOCH, FIXTURE_BASE_PATH, build_fixture_db
from omniquery.engine import OmniQueryEngine
from omniquery.parsers import ParserBackend, ParserOutcome, get_backend, try_validate
from omniquery.parsers.router import Router, load_thresholds
from omniquery.validation import AuthContext

_DEFAULT_CORPUS_PATH = Path(__file__).with_name("corpus.jsonl")

BENCH_CTX = AuthContext(role="ADMIN", user_id="bench", client_uuid="bench", ai_enabled=True)

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
    entries: List[dict] = []
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _build_fixture_engine() -> OmniQueryEngine:
    import tempfile
    db_path = tempfile.mktemp(suffix=".db", prefix="omniquery_bench_")
    build_fixture_db(db_path, seed=42)
    return OmniQueryEngine(db_path=db_path, base_path=FIXTURE_BASE_PATH,
                            ai_resolvers=_STUB_AI_RESOLVERS)


def _exec_result(engine: OmniQueryEngine, query: Any, now_epoch: float) -> Optional[Tuple[str, Any]]:
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
    expected_unsupported: bool
    produced_ast: bool
    valid: bool
    ast_exact: bool
    exec_match: bool
    confidence: Optional[float]


def _score_entry(expected: dict, outcome: ParserOutcome, engine: OmniQueryEngine,
                  now_epoch: float) -> _EntryRecord:
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
    if not latencies_ms:
        return {"p50": None, "p95": None, "mean": None}
    s = sorted(latencies_ms)

    def _pct(p: float) -> float:
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    return {"p50": _pct(0.50), "p95": _pct(0.95), "mean": sum(s) / len(s)}


def _false_confident_sweep(records: List[_EntryRecord]) -> Dict[str, Optional[float]]:
    scored = [(r.confidence, not r.exec_match) for r in records if r.produced_ast and r.confidence is not None]
    result: Dict[str, Optional[float]] = {}
    for theta in _THETA_STEPS:
        confident = [mismatched for conf, mismatched in scored if conf >= theta]
        result[str(theta)] = (sum(confident) / len(confident)) if confident else None
    return result


def _aggregate(records: List[_EntryRecord], latencies_ms: List[float]) -> Dict[str, Any]:
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
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


# ---------------------------------------------------------------------------
# Backend / router runners
# ---------------------------------------------------------------------------

def _run_backend(backend: ParserBackend, entries: List[dict], engine: OmniQueryEngine,
                  now_epoch: float) -> Dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="OmniQuery v2 NL-parser benchmark")
    parser.add_argument(
        "--backends", type=str, default="heuristic",
        help="comma-separated backend names, e.g. heuristic,needle2,fallback_qwen,router",
    )
    parser.add_argument("--corpus", type=str, default=str(_DEFAULT_CORPUS_PATH))
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]
    report = run_benchmark(backend_names, corpus_path=args.corpus, out_path=args.out)
    if not args.out:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"wrote report to {args.out}")


if __name__ == "__main__":
    main()
