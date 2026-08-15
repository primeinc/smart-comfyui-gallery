"""Corpus integrity tests: every non-unsupported entry in
omniquery/benchmark/corpus.jsonl must parse, validate (permissive ctx), and
execute cleanly against the deterministic fixture DB; every `file_ref`
literal id it names must actually exist in the fixture; unsupported entries
must be excluded from execution. Also a fast smoke test of the benchmark
harness itself, restricted to the heuristic backend so nothing here loads
needle2/llama_cpp.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omniquery.ast import ASTError, parse_query
from omniquery.benchmark.fixtures import (
    ANCHOR_EPOCH, FIXTURE_BASE_PATH, FIXTURE_FILES, build_fixture_db,
)
from omniquery.benchmark.harness import (
    _date_placeholder_map, _resolve_date_placeholders, load_corpus, run_benchmark,
)
from omniquery.engine import OmniQueryEngine
from omniquery.validation import AuthContext, ValidationError, validate

CORPUS_PATH = Path(__file__).resolve().parent.parent / "omniquery" / "benchmark" / "corpus.jsonl"

PERM_CTX = AuthContext(role="ADMIN", user_id="test", client_uuid="test", ai_enabled=True)

_STUB_RESOLVERS = {
    "similar_to_semantic": lambda _v: ["f001", "f002"],
    "similar_to_visual": lambda _v: ["f001"],
    "near_dup_of": lambda _v: ["f003"],
}

_FIXTURE_IDS = {f["id"] for f in FIXTURE_FILES}


@pytest.fixture(scope="module")
def corpus():
    entries = load_corpus(CORPUS_PATH)
    # Calendar-vocabulary entries carry date placeholders resolved against
    # the same clock the parsers receive (here: the fixture anchor).
    entries = _resolve_date_placeholders(entries, _date_placeholder_map(ANCHOR_EPOCH))
    assert len(entries) >= 60, f"corpus must have >= 60 entries, has {len(entries)}"
    return entries


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("corpus_fixture") / "fixture.db")
    build_fixture_db(db_path, seed=42)
    return OmniQueryEngine(db_path=db_path, base_path=FIXTURE_BASE_PATH, ai_resolvers=_STUB_RESOLVERS)


# ---------------------------------------------------------------------------
# Structural integrity
# ---------------------------------------------------------------------------

def test_corpus_has_unique_ids(corpus):
    ids = [e["id"] for e in corpus]
    assert len(ids) == len(set(ids)), "duplicate corpus entry ids"


def test_corpus_entries_have_required_keys(corpus):
    for entry in corpus:
        assert set(entry) == {"id", "nl", "expected", "tags"}, entry["id"]
        assert isinstance(entry["nl"], str) and entry["nl"] or entry["nl"] == ""
        assert isinstance(entry["tags"], list)
        expected = entry["expected"]
        assert ("ast" in expected) ^ ("unsupported" in expected), (
            f"{entry['id']}: expected must have exactly one of 'ast'/'unsupported'"
        )


def test_corpus_covers_required_categories(corpus):
    all_tags = set()
    for entry in corpus:
        all_tags.update(entry["tags"])
    required = {"simple", "boolean", "negation", "disjunction", "join", "count",
                "privileged", "date", "duration", "ambiguous", "adversarial"}
    missing = required - all_tags
    assert not missing, f"corpus is missing tag categories: {missing}"


def test_corpus_has_unsupported_entries_for_ambiguous_and_adversarial(corpus):
    ambiguous = [e for e in corpus if "ambiguous" in e["tags"]]
    adversarial = [e for e in corpus if "adversarial" in e["tags"]]
    assert ambiguous and all(e["expected"].get("unsupported") for e in ambiguous)
    # adversarial has a mix: SQL-injection-as-literal (supported, escaped as
    # a plain value) and instruction-injection (unsupported).
    assert any(e["expected"].get("unsupported") for e in adversarial)
    assert any("ast" in e["expected"] for e in adversarial)


def test_corpus_adversarial_sql_literals_land_as_plain_text_values(corpus):
    sql_entries = [e for e in corpus if "adversarial" in e["tags"] and "ast" in e["expected"]]
    assert sql_entries
    for entry in sql_entries:
        query = parse_query(entry["expected"]["ast"])
        values = [c.value for c in _iter_all_conds(query.where)]
        assert any(isinstance(v, str) and ("DROP" in v or "UPDATE" in v or "OR '1'='1" in v)
                   for v in values), entry["id"]


def _iter_all_conds(node):
    from omniquery.ast import Cond, Not, Group
    if node is None:
        return
    if isinstance(node, Cond):
        yield node
    elif isinstance(node, Not):
        yield from _iter_all_conds(node.child)
    elif isinstance(node, Group):
        for c in node.children:
            yield from _iter_all_conds(c)


# ---------------------------------------------------------------------------
# parse + validate + execute every supported entry; unsupported ones are
# explicitly excluded from this round-trip (there is no AST to run).
# ---------------------------------------------------------------------------

def test_every_supported_entry_parses_validates_and_executes(corpus, engine):
    n_supported = 0
    n_unsupported = 0
    for entry in corpus:
        expected = entry["expected"]
        if expected.get("unsupported"):
            n_unsupported += 1
            assert "ast" not in expected
            continue
        n_supported += 1
        try:
            query = parse_query(expected["ast"])
        except ASTError as exc:  # failure path, informative message
            pytest.fail(f"{entry['id']}: AST failed to parse: {exc}")
        try:
            validate(query, PERM_CTX)
        except ValidationError as exc:
            pytest.fail(f"{entry['id']}: AST failed validation: {exc}")
        out = engine.run(query, PERM_CTX, now_epoch=ANCHOR_EPOCH)
        assert out.ok, f"{entry['id']}: execution failed: {out.error}"

    assert n_supported > 0 and n_unsupported > 0


def test_file_ref_literal_ids_exist_in_fixture(corpus):
    from omniquery.ast import iter_conditions

    checked_any = False
    for entry in corpus:
        expected = entry["expected"]
        if "ast" not in expected:
            continue
        query = parse_query(expected["ast"])
        for cond in iter_conditions(query.where):
            if cond.field in ("near_dup_of", "similar_to_semantic", "similar_to_visual"):
                value = cond.value
                file_id = value if isinstance(value, str) else value.get("file_id")
                assert file_id in _FIXTURE_IDS, f"{entry['id']}: {file_id!r} not in fixture"
                checked_any = True
    assert checked_any, "no file_ref entries were exercised by this test"


# ---------------------------------------------------------------------------
# Harness smoke test (heuristic backend only -- model-free, fast)
# ---------------------------------------------------------------------------

def test_harness_smoke_heuristic_only(tmp_path):
    out_path = str(tmp_path / "report.json")
    report = run_benchmark(["heuristic"], corpus_path=str(CORPUS_PATH), out_path=out_path)

    assert Path(out_path).exists()
    assert report["corpus_size"] >= 60
    assert "heuristic" in report["backends"]

    metrics = report["backends"]["heuristic"]
    for key in ("ast_exact_rate", "execution_match_rate", "invalid_rate",
                "unsupported_correct", "unsupported_incorrect", "unsupported_correct_rate",
                "false_confident_rate", "latency_ms", "peak_rss_kb",
                "n_entries", "n_supported_expected", "n_unsupported_expected"):
        assert key in metrics, key

    assert 0.0 <= metrics["ast_exact_rate"] <= 1.0
    assert 0.0 <= metrics["execution_match_rate"] <= 1.0
    assert 0.0 <= metrics["invalid_rate"] <= 1.0
    assert metrics["invalid_rate"] == 0.0  # heuristic must never emit an invalid AST
    assert metrics["n_entries"] == report["corpus_size"]

    assert set(metrics["false_confident_rate"]) == {
        str(round(i * 0.1, 1)) for i in range(10)
    }
    assert metrics["latency_ms"]["p50"] is not None
    assert metrics["peak_rss_kb"] > 0


def test_harness_router_not_requested_means_no_router_key():
    report = run_benchmark(["heuristic"], corpus_path=str(CORPUS_PATH))
    assert "router" not in report["backends"]
