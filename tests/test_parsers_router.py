"""Router policy tests using scripted fake backends -- no needle/llama_cpp
loaded anywhere in this file, per the "unit tests must not load
needle2/llama models" rule. Real-engine behavior is exercised separately
via the CLI harness (omniquery/benchmark/harness.py), not pytest.
"""

from __future__ import annotations

import pytest

from omniquery.parsers import ParserBackend, ParserOutcome, get_backend, make_default_router
from omniquery.parsers.router import Router, load_thresholds

NOW = 1735689600.0


class FakeBackend(ParserBackend):
    """Scripted backend: always returns the outcome/availability it was
    constructed with, and records how many times parse() was called."""

    def __init__(self, name: str, outcome: ParserOutcome, avail: bool = True):
        self.name = name
        self._outcome = outcome
        self._avail = avail
        self.call_count = 0

    def available(self) -> bool:
        return self._avail

    def parse(self, text: str, now_epoch: float) -> ParserOutcome:
        self.call_count += 1
        return self._outcome


def _outcome(backend: str, ast=None, confidence=None, coverage=None, unsupported=False,
             reason=None) -> ParserOutcome:
    return ParserOutcome(ast=ast, confidence=confidence, backend=backend, unsupported=unsupported,
                          reason=reason, coverage=coverage)


@pytest.fixture()
def thresholds():
    return load_thresholds()


# ---------------------------------------------------------------------------
# 1. Heuristic full parse (coverage == 1.0) wins outright, no escalation
# ---------------------------------------------------------------------------

def test_heuristic_full_coverage_wins_without_calling_others(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", ast={"a": 1}, confidence=1.0, coverage=1.0))
    primary = FakeBackend("needle2", _outcome("needle2", ast={"b": 2}, confidence=0.9, coverage=1.0))
    fallback = FakeBackend("fallback_qwen", _outcome("fallback_qwen", ast={"c": 3}, coverage=1.0))

    router = Router(primary=primary, fallback=fallback, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "heuristic"
    assert outcome.ast == {"a": 1}
    assert len(trace) == 1
    assert primary.call_count == 0
    assert fallback.call_count == 0


# ---------------------------------------------------------------------------
# 2. Needle2 accepted only when BOTH coverage and confidence clear threshold
# ---------------------------------------------------------------------------

def test_needle2_accepted_when_both_thresholds_cleared(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", ast={"a": 1}, coverage=0.5))
    primary = FakeBackend(
        "needle2",
        _outcome("needle2", ast={"b": 2},
                 confidence=thresholds["needle2_min_confidence"] + 0.01,
                 coverage=thresholds["needle2_min_coverage"]),
    )
    router = Router(primary=primary, fallback=None, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "needle2"
    assert len(trace) == 2
    assert primary.call_count == 1


@pytest.mark.parametrize("confidence,coverage", [
    (0.0, 1.0),   # coverage clears, confidence does not
    (1.0, 0.5),   # confidence clears, coverage does not
])
def test_needle2_rejected_when_either_threshold_misses(thresholds, confidence, coverage):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="nothing"))
    primary = FakeBackend("needle2", _outcome("needle2", ast={"b": 2}, confidence=confidence, coverage=coverage))
    router = Router(primary=primary, fallback=None, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "router"
    assert outcome.unsupported


def test_needle2_unavailable_is_skipped_entirely(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="nothing"))
    primary = FakeBackend("needle2", _outcome("needle2", ast={"b": 2}, confidence=1.0, coverage=1.0), avail=False)
    router = Router(primary=primary, fallback=None, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert primary.call_count == 0
    assert len(trace) == 1  # only heuristic ran
    assert outcome.unsupported


# ---------------------------------------------------------------------------
# 3. Fallback path: reached only after needle2 declines/is absent
# ---------------------------------------------------------------------------

def test_fallback_accepted_after_needle2_rejected(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="nothing"))
    primary = FakeBackend("needle2", _outcome("needle2", ast={"b": 2}, confidence=0.0, coverage=1.0))
    fallback = FakeBackend("fallback_qwen",
                            _outcome("fallback_qwen", ast={"c": 3}, coverage=thresholds["fallback_min_coverage"]))
    router = Router(primary=primary, fallback=fallback, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "fallback_qwen"
    assert outcome.ast == {"c": 3}
    assert len(trace) == 3
    assert fallback.call_count == 1


def test_fallback_rejected_below_coverage_threshold(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="nothing"))
    fallback = FakeBackend("fallback_qwen",
                            _outcome("fallback_qwen", ast={"c": 3},
                                     coverage=thresholds["fallback_min_coverage"] - 0.1))
    router = Router(primary=None, fallback=fallback, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "router"
    assert outcome.unsupported


def test_fallback_unavailable_is_skipped(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="nothing"))
    fallback = FakeBackend("fallback_qwen", _outcome("fallback_qwen", ast={"c": 3}, coverage=1.0), avail=False)
    router = Router(primary=None, fallback=fallback, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert fallback.call_count == 0
    assert outcome.unsupported


# ---------------------------------------------------------------------------
# 4. Heuristic partial-floor acceptance (last resort before aggregate fail)
# ---------------------------------------------------------------------------

def test_heuristic_partial_accepted_above_floor_when_nothing_else_worked(thresholds):
    heuristic = FakeBackend(
        "heuristic",
        _outcome("heuristic", ast={"a": 1}, coverage=thresholds["heuristic_partial_floor"], reason="some tokens"),
    )
    primary = FakeBackend("needle2", _outcome("needle2", unsupported=True, reason="bad"))
    fallback = FakeBackend("fallback_qwen", _outcome("fallback_qwen", unsupported=True, reason="bad2"))
    router = Router(primary=primary, fallback=fallback, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "heuristic"
    assert not outcome.unsupported
    assert outcome.ast == {"a": 1}
    assert "partial" in outcome.reason
    assert "some tokens" in outcome.reason  # original reason preserved
    assert len(trace) == 3


def test_heuristic_partial_rejected_below_floor(thresholds):
    heuristic = FakeBackend(
        "heuristic",
        _outcome("heuristic", ast={"a": 1}, coverage=thresholds["heuristic_partial_floor"] - 0.1),
    )
    router = Router(primary=None, fallback=None, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "router"
    assert outcome.unsupported


# ---------------------------------------------------------------------------
# 5. Aggregate unsupported: every backend's reason is collected
# ---------------------------------------------------------------------------

def test_aggregate_unsupported_collects_every_reason(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="no rules matched"))
    primary = FakeBackend("needle2", _outcome("needle2", unsupported=True, reason="engine error: boom"))
    fallback = FakeBackend("fallback_qwen", _outcome("fallback_qwen", unsupported=True, reason="invalid AST"))
    router = Router(primary=primary, fallback=fallback, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "router"
    assert outcome.unsupported
    assert outcome.ast is None
    assert "no rules matched" in outcome.reason
    assert "engine error: boom" in outcome.reason
    assert "invalid AST" in outcome.reason
    assert len(trace) == 3


def test_aggregate_unsupported_with_no_backends_available(thresholds):
    heuristic = FakeBackend("heuristic", _outcome("heuristic", unsupported=True, reason="nothing recognized"))
    router = Router(primary=None, fallback=None, heuristic=heuristic, thresholds=thresholds)
    outcome, trace = router.route("q", NOW)

    assert outcome.backend == "router"
    assert outcome.unsupported
    assert len(trace) == 1


# ---------------------------------------------------------------------------
# 6. load_thresholds / make_default_router wiring (no .route() call -- that
#    would touch the real needle2/fallback_qwen availability checks).
# ---------------------------------------------------------------------------

def test_load_thresholds_has_all_four_keys():
    thresholds = load_thresholds()
    assert set(thresholds) == {
        "needle2_min_confidence", "needle2_min_coverage",
        "fallback_min_coverage", "heuristic_partial_floor",
    }
    assert all(isinstance(v, float) for v in thresholds.values())


def test_make_default_router_wires_real_backend_types(monkeypatch):
    # Benchmark-measured default: the fallback model is NOT wired in
    # (routing_defaults.json fallback_enabled=false).
    monkeypatch.delenv("OMNIQUERY_ENABLE_FALLBACK", raising=False)
    router = make_default_router()
    assert router.heuristic.name == "heuristic"
    assert router.primary.name == "needle2"
    assert router.fallback is None
    assert router.thresholds == load_thresholds()

    # Env override re-enables it without touching the config file.
    monkeypatch.setenv("OMNIQUERY_ENABLE_FALLBACK", "true")
    router_with_fb = make_default_router()
    assert router_with_fb.fallback is not None
    assert router_with_fb.fallback.name == "fallback_qwen"


def test_make_default_router_config_overrides_thresholds():
    router = make_default_router({"needle2_min_confidence": 0.9})
    assert router.thresholds["needle2_min_confidence"] == 0.9
    assert router.thresholds["needle2_min_coverage"] == load_thresholds()["needle2_min_coverage"]


def test_get_backend_unknown_name_raises():
    with pytest.raises(KeyError):
        get_backend("not_a_real_backend")


def test_get_backend_constructs_without_touching_optional_runtime():
    # Constructing must never itself require needle/llama_cpp to be usable
    # -- only .available()/.parse() do.
    backend = get_backend("needle2", topology="family")
    assert backend.name == "needle2"
    backend2 = get_backend("fallback_qwen")
    assert backend2.name == "fallback_qwen"
