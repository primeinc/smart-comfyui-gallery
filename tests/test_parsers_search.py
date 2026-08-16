"""SearchParser policy tests using scripted fake backends -- no model
runtime loaded anywhere in this file. Real-model behavior is exercised
separately via the CLI harness (omniquery/benchmark/harness.py).

The policy under test: nlq always answers; the nl2sql model is consulted
only when nlq's outcome carries model_hint AND the caller allows it AND
the model is available -- and the model's AST replaces nlq's only when it
passes coverage_guard at full coverage.
"""

from __future__ import annotations

from omniquery.parsers import ParserBackend, ParserOutcome, SearchParser

NOW = 1735689600.0

# A trivially coverage-complete AST for fake outcomes: no literals/keywords
# in the query text means coverage_guard scores any AST 1.0, so tests
# control acceptance purely through the query text they pass.
_AST = {"version": 1, "target": "files", "result": "ids",
        "where": {"field": "is_favorite", "op": "eq", "value": True}}


class FakeBackend(ParserBackend):
    """Scripted backend: fixed outcome/availability; counts parse() calls."""

    def __init__(self, name: str, outcome: ParserOutcome, avail: bool = True):
        self.name = name
        self._outcome = outcome
        self._avail = avail
        self.call_count = 0

    def available(self) -> bool:
        return self._avail

    def parse(self, _text: str, _now_epoch: float) -> ParserOutcome:
        self.call_count += 1
        return self._outcome


def _nlq_outcome(hint: bool, ast=_AST) -> ParserOutcome:
    return ParserOutcome(ast=ast, confidence=1.0, backend="nlq", coverage=1.0,
                          raw={"interpretation": [], "text_terms": [], "model_hint": hint})


def _model_outcome(ast=_AST, unsupported=False) -> ParserOutcome:
    return ParserOutcome(ast=None if unsupported else ast, confidence=None,
                          backend="nl2sql", unsupported=unsupported)


def test_no_hint_never_consults_the_model():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=False))
    model = FakeBackend("nl2sql", _model_outcome())
    outcome, trace = SearchParser(nlq=nlq, model=model).parse("q", NOW)
    assert outcome.backend == "nlq"
    assert model.call_count == 0
    assert len(trace) == 1


def test_hint_consults_model_and_accepts_full_coverage_ast():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=True))
    better = {"version": 1, "target": "files", "result": "ids",
              "where": {"field": "rating_avg", "op": "ge", "value": 4}}
    model = FakeBackend("nl2sql", _model_outcome(ast=better))
    # "rated at least 4": the number 4 appears in the model's AST, so
    # coverage_guard passes at 1.0 and the refinement wins.
    outcome, trace = SearchParser(nlq=nlq, model=model).parse("rated at least 4", NOW)
    assert outcome.backend == "nl2sql"
    assert outcome.ast == better
    assert len(trace) == 2


def test_model_ast_dropping_a_literal_is_rejected():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=True))
    dropped = {"version": 1, "target": "files", "result": "ids",
               "where": {"field": "is_favorite", "op": "eq", "value": True}}
    model = FakeBackend("nl2sql", _model_outcome(ast=dropped))
    # The model dropped the "4" -> coverage < 1.0 -> nlq's parse stands.
    outcome, _trace = SearchParser(nlq=nlq, model=model).parse("favorites rated at least 4", NOW)
    assert outcome.backend == "nlq"
    assert model.call_count == 1


def test_model_unsupported_keeps_nlq_result():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=True))
    model = FakeBackend("nl2sql", _model_outcome(unsupported=True))
    outcome, trace = SearchParser(nlq=nlq, model=model).parse("q", NOW)
    assert outcome.backend == "nlq"
    assert len(trace) == 2


def test_unavailable_model_is_never_called():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=True))
    model = FakeBackend("nl2sql", _model_outcome(), avail=False)
    outcome, trace = SearchParser(nlq=nlq, model=model).parse("q", NOW)
    assert outcome.backend == "nlq"
    assert model.call_count == 0
    assert len(trace) == 1


def test_allow_model_false_is_the_live_typing_path():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=True))
    model = FakeBackend("nl2sql", _model_outcome())
    outcome, trace = SearchParser(nlq=nlq, model=model).parse("q", NOW, allow_model=False)
    assert outcome.backend == "nlq"
    assert model.call_count == 0
    assert len(trace) == 1


def test_none_model_degrades_to_nlq_only():
    nlq = FakeBackend("nlq", _nlq_outcome(hint=True))
    outcome, trace = SearchParser(nlq=nlq, model=None).parse("q", NOW)
    assert outcome.backend == "nlq"
    assert len(trace) == 1


def test_real_default_construction_is_lazy_and_model_free():
    # Constructing the standard SearchParser must not touch llama_cpp; the
    # nl2sql backend only checks its runtime inside available().
    import sys
    from omniquery.parsers import make_search_parser
    parser = make_search_parser()
    outcome, _ = parser.parse("photos of trees", NOW, allow_model=False)
    assert outcome.backend == "nlq"
    assert outcome.ast is not None
