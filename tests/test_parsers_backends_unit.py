"""Model-free unit tests for the two optional-runtime parser backends,
needle2.Needle2Backend and fallback_qwen.FallbackQwenBackend: availability
checks (and their caching), the never-raise parse() contract, and the
coverage/negation guard interplay -- exercised through injected fake
engines only.

IMPORTANT: on this machine the real `needle` and `llama_cpp` packages ARE
installed and the fallback GGUF exists at its default path, so these tests
never rely on an ImportError happening naturally and never touch the
default model path. Unavailability is forced hermetically by stubbing
sys.modules (a None entry makes `import x` raise ImportError); fake engines
are injected through the backends' own seams (`Needle2Backend._agent`,
`fallback_qwen._MODEL_CACHE`). No real engine is ever constructed.
"""

from __future__ import annotations

import sys
import types

import pytest

from omniquery import fields
from omniquery.parsers import ParserOutcome
from omniquery.parsers import fallback_qwen as fallback_module
from omniquery.parsers.fallback_qwen import DEFAULT_MODEL_PATH, FallbackQwenBackend, _example_for
from omniquery.parsers.needle2 import Needle2Backend

NOW = 1735689600.0  # unused by both backends, but required by the interface


def _iter_conds(node):
    if "children" in node:
        for c in node["children"]:
            yield from _iter_conds(c)
    elif "child" in node:
        yield from _iter_conds(node["child"])
    else:
        yield node


class FakeNeedleAgent:
    """Scripted stand-in for needle.Needle: returns a canned response dict
    (or raises) from complete()."""

    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    def reset(self):
        pass

    def complete(self, _text, max_new_tokens=None):
        del max_new_tokens  # accepted only for complete()'s call-signature compatibility (kwarg call)
        if self._exc is not None:
            raise self._exc
        return self._resp


def _needle_backend_with(resp=None, exc=None, topology="frame") -> Needle2Backend:
    backend = Needle2Backend(topology=topology)
    # _get_agent() returns self._agent when set, so injecting here means the
    # optional `needle` runtime is never imported.
    backend._agent = FakeNeedleAgent(resp=resp, exc=exc)
    return backend


def _needle_resp(frame_args, confidence=0.9, validation=None, calls=None):
    if calls is None:
        calls = [{"name": "query_gallery", "arguments": frame_args}]
    resp = {"success": True, "confidence": confidence, "function_calls": calls}
    if validation is not None:
        resp["validation"] = validation
    return resp


def _fake_needle_module(resp):
    mod = types.ModuleType("needle")

    class Needle:
        def __init__(self, tools=None):
            self.tools = tools

        def reset(self):
            pass

        def complete(self, _text, max_new_tokens=None):
            del max_new_tokens  # accepted only for complete()'s call-signature compatibility (kwarg call)
            return resp

    mod.Needle = Needle
    return mod


class FakeLlama:
    """Scripted stand-in for llama_cpp.Llama: returns canned completion
    content (or raises) from create_chat_completion()."""

    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    def create_chat_completion(self, messages, grammar=None, temperature=None,
                               max_tokens=None):
        # all four accepted only for create_chat_completion()'s call-signature
        # compatibility (the real call passes each of them by keyword).
        del messages, grammar, temperature, max_tokens
        if self._exc is not None:
            raise self._exc
        return {"choices": [{"message": {"content": self._content}}]}


def _fallback_backend_with(monkeypatch, tmp_path, content=None, exc=None) -> FallbackQwenBackend:
    """Backend whose _load_model() hits an injected _MODEL_CACHE entry, so
    neither the file check nor the llama_cpp import ever runs."""
    path = str(tmp_path / "fake-injected.gguf")  # deliberately does not exist
    backend = FallbackQwenBackend(model_path=path)
    monkeypatch.setitem(fallback_module._MODEL_CACHE, path,
                        (FakeLlama(content=content, exc=exc), object()))
    return backend


# ---------------------------------------------------------------------------
# 1. Needle2: constructor + availability (and its caching)
# ---------------------------------------------------------------------------

def test_needle2_unknown_topology_raises_value_error():
    """Constructing with a topology outside frame/family raises ValueError naming the bad value."""
    with pytest.raises(ValueError, match="unknown topology 'grid'"):
        Needle2Backend(topology="grid")


def test_needle2_available_false_when_runtime_missing_and_cached(monkeypatch):
    """available() is False when `import needle` fails, and the False answer is cached across calls."""
    monkeypatch.setitem(sys.modules, "needle", None)  # forces ImportError
    backend = Needle2Backend()
    assert backend.available() is False

    # Runtime "appears" afterwards: the cached answer must stick.
    monkeypatch.setitem(sys.modules, "needle", _fake_needle_module({}))
    assert backend.available() is False


def test_needle2_available_true_is_cached_along_with_the_agent(monkeypatch):
    """A successful available() caches both the answer and the constructed agent, surviving runtime removal."""
    resp = _needle_resp({"media_type": "image"}, confidence=0.7)
    monkeypatch.setitem(sys.modules, "needle", _fake_needle_module(resp))
    backend = Needle2Backend()
    assert backend.available() is True

    # Break the runtime: cached availability AND cached agent keep working.
    monkeypatch.setitem(sys.modules, "needle", None)
    assert backend.available() is True
    out = backend.parse("images", NOW)
    assert not out.unsupported
    assert {"field": "type", "op": "eq", "value": "image"} in list(_iter_conds(out.ast["where"]))


# ---------------------------------------------------------------------------
# 2. Needle2: parse() never raises -- engine failures become outcomes
# ---------------------------------------------------------------------------

def test_needle2_parse_on_unavailable_engine_returns_engine_error_outcome(monkeypatch):
    """parse() with no usable runtime returns an unsupported outcome with an 'engine error' reason, never raises."""
    monkeypatch.setitem(sys.modules, "needle", None)
    backend = Needle2Backend()
    out = backend.parse("favorite images", NOW)
    assert isinstance(out, ParserOutcome)
    assert out.unsupported is True
    assert out.ast is None
    assert out.confidence is None
    assert out.backend == "needle2"
    assert out.reason is not None and out.reason.startswith("engine error:")
    assert out.latency_ms is not None and out.latency_ms >= 0.0


def test_needle2_parse_complete_exception_becomes_engine_error_outcome():
    """An exception thrown inside the engine's complete() is converted to an unsupported outcome carrying the message."""
    backend = _needle_backend_with(exc=ValueError("truncated at token budget"))
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is True
    assert out.ast is None
    assert out.reason == "engine error: truncated at token budget"


def test_needle2_parse_engine_declining_reports_no_function_call_with_engine_error_text():
    """A success=False engine response yields an unsupported outcome quoting the engine's error and confidence."""
    resp = {"success": False, "error": "no tool selected", "confidence": 0.15}
    backend = _needle_backend_with(resp=resp)
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is True
    assert out.ast is None
    assert out.reason == "no function call: no tool selected"
    assert out.confidence == 0.15
    assert out.raw == resp


def test_needle2_parse_non_dict_engine_response_is_malformed_not_a_crash():
    """A non-dict engine response yields an unsupported 'malformed response' outcome instead of an AttributeError."""
    backend = _needle_backend_with(resp="garbage-not-a-dict")
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is True
    assert out.ast is None
    assert out.confidence is None
    assert out.reason == "no function call: malformed response"


# ---------------------------------------------------------------------------
# 3. Needle2: coverage / negation guard interplay on frame-shaped results
# ---------------------------------------------------------------------------

def test_needle2_dropped_literal_lowers_coverage_and_names_the_miss():
    """A frame that silently dropped the query's rating literal yields coverage < 1.0 with the missing '4' named in reason."""
    backend = _needle_backend_with(
        resp=_needle_resp({"media_type": "video"}, confidence=0.95))
    out = backend.parse("favorite videos rated at least 4", NOW)
    # The AST itself is valid, so the outcome is NOT unsupported -- the
    # router rejects it on coverage; here we pin the guard's signal.
    assert out.unsupported is False
    assert out.coverage is not None and out.coverage < 1.0
    assert out.reason is not None
    assert "4" in out.reason
    assert "favorite" in out.reason
    assert out.confidence == 0.95


def test_needle2_negation_flag_without_not_node_rejects_the_parse():
    """When the engine flags negation but the frame-expanded AST has no Not node, the outcome is rejected with the guard reason."""
    resp = _needle_resp({"media_type": "video", "status_flag": "Approved"},
                        confidence=0.8, validation={"negation": True})
    backend = _needle_backend_with(resp=resp)
    out = backend.parse("videos that are not approved", NOW)
    assert out.unsupported is True
    assert out.ast is None
    assert out.reason == "negation not expressible in frame"
    assert out.confidence == 0.8
    # Coverage is still computed and carried on the rejection (both the
    # 'video' and 'approved' literals ARE reflected -- negation is the
    # semantic failure coverage_guard deliberately cannot see).
    assert out.coverage == 1.0


def test_needle2_full_frame_parse_succeeds_with_validated_ast_and_full_coverage():
    """A complete multi-call family response merges into one validated AST with coverage 1.0 and reason None."""
    calls = [
        {"name": "filter_kind", "arguments": {"media_type": "video"}},
        {"name": "filter_rating", "arguments": {"favorite": True}},
        {"name": "present", "arguments": {"result": "count"}},
    ]
    backend = _needle_backend_with(
        resp=_needle_resp(None, confidence=0.9, calls=calls), topology="family")
    out = backend.parse("how many favorite videos", NOW)
    assert out.unsupported is False
    assert out.backend == "needle2"
    assert out.confidence == 0.9
    assert out.coverage == 1.0
    assert out.reason is None
    assert out.ast["result"] == "count"
    conds = list(_iter_conds(out.ast["where"]))
    assert {"field": "type", "op": "eq", "value": "video"} in conds
    assert {"field": "is_favorite", "op": "eq", "value": True} in conds


def test_needle2_merge_is_last_call_wins_per_key():
    """Two calls setting the same frame key merge last-call-wins, so only the later media_type reaches the AST."""
    calls = [
        {"name": "query_gallery", "arguments": {"media_type": "image"}},
        {"name": "query_gallery", "arguments": {"media_type": "video"}},
    ]
    backend = _needle_backend_with(resp=_needle_resp(None, calls=calls))
    out = backend.parse("videos", NOW)
    assert out.unsupported is False
    conds = list(_iter_conds(out.ast["where"]))
    assert conds == [{"field": "type", "op": "eq", "value": "video"}]


# ---------------------------------------------------------------------------
# 4. Fallback Qwen: model path resolution + availability
# ---------------------------------------------------------------------------

def test_fallback_model_path_resolution_order(monkeypatch, tmp_path):
    """model_path resolves explicit arg over OMNIQUERY_FALLBACK_GGUF over the built-in default."""
    monkeypatch.delenv("OMNIQUERY_FALLBACK_GGUF", raising=False)
    assert FallbackQwenBackend().model_path == DEFAULT_MODEL_PATH

    env_path = str(tmp_path / "env-model.gguf")
    monkeypatch.setenv("OMNIQUERY_FALLBACK_GGUF", env_path)
    assert FallbackQwenBackend().model_path == env_path

    explicit = str(tmp_path / "explicit.gguf")
    assert FallbackQwenBackend(model_path=explicit).model_path == explicit


def test_fallback_available_false_when_model_file_absent(monkeypatch, tmp_path):
    """available() is False when the GGUF file is missing, even with the runtime importable."""
    # Stub the runtime so the check isolates the file-existence branch (and
    # never imports the real llama_cpp installed on this machine).
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    backend = FallbackQwenBackend(model_path=str(tmp_path / "missing.gguf"))
    assert backend.available() is False


def test_fallback_available_true_when_runtime_and_file_both_present(monkeypatch, tmp_path):
    """available() is True when the runtime imports and the model file exists -- without loading anything."""
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not really a gguf")
    backend = FallbackQwenBackend(model_path=str(model))
    assert backend.available() is True


def test_fallback_available_false_when_runtime_missing_despite_model_file(monkeypatch, tmp_path):
    """available() is False when llama_cpp cannot be imported, even though the model file exists."""
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not really a gguf")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # forces ImportError
    backend = FallbackQwenBackend(model_path=str(model))
    assert backend.available() is False


# ---------------------------------------------------------------------------
# 4b. Fallback Qwen: _example_for's exhaustive-over-Kind fallback branch
# ---------------------------------------------------------------------------

def test_example_for_unrecognized_kind_falls_through_to_ellipsis():
    """_example_for's if-chain matches every real fields.Kind member; an
    unrecognized kind (only reachable if Kind is ever extended without
    updating this function) must still fall through to the "..." default
    instead of raising."""
    spec = fields.FieldSpec(name="bogus", kind="not_a_real_kind",
                             ops=frozenset(), strategy=fields.Strategy.COLUMN)
    assert _example_for(spec) == "..."


# ---------------------------------------------------------------------------
# 5. Fallback Qwen: parse() fail-closed contract
# ---------------------------------------------------------------------------

def test_fallback_parse_absent_model_fails_closed_without_importing_llama_cpp(monkeypatch, tmp_path):
    """parse() with a missing model file returns a 'model load error' outcome before ever importing llama_cpp."""
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    missing = str(tmp_path / "missing.gguf")
    backend = FallbackQwenBackend(model_path=missing)

    out = backend.parse("favorite images", NOW)

    assert out.unsupported is True
    assert out.ast is None
    assert out.backend == "fallback_qwen"
    assert out.reason is not None and out.reason.startswith("model load error:")
    assert missing in out.reason
    assert out.latency_ms is not None and out.latency_ms >= 0.0
    # The isfile check runs BEFORE the llama_cpp import, so the runtime was
    # never pulled in by this failure path.
    assert "llama_cpp" not in sys.modules


def test_fallback_parse_generation_exception_becomes_outcome(monkeypatch, tmp_path):
    """An exception from the llama engine during generation yields a 'generation error' outcome with no raw content."""
    backend = _fallback_backend_with(monkeypatch, tmp_path,
                                     exc=RuntimeError("context overflow"))
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is True
    assert out.ast is None
    assert out.reason == "generation error: context overflow"
    assert out.raw is None  # engine died before any content existed


def test_fallback_parse_malformed_json_content_becomes_outcome_with_raw(monkeypatch, tmp_path):
    """Non-JSON generation output yields a 'generation error' outcome that preserves the raw content for debugging."""
    backend = _fallback_backend_with(monkeypatch, tmp_path, content="{not json at all")
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is True
    assert out.ast is None
    assert out.reason is not None and out.reason.startswith("generation error:")
    assert out.raw == {"raw_content": "{not json at all"}


def test_fallback_parse_grammar_legal_but_invalid_ast_fails_closed(monkeypatch, tmp_path):
    """A shape-valid AST that fails semantic validation (op not allowed for field) is rejected, never returned."""
    # is_favorite only allows 'eq'; 'contains' is grammar-legal but invalid.
    content = ('{"target":"files","where":'
               '{"field":"is_favorite","op":"contains","value":"x"}}')
    backend = _fallback_backend_with(monkeypatch, tmp_path, content=content)
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is True
    assert out.ast is None
    # The reason must come from the SEMANTIC validation layer (op-vs-field
    # rules), not the structural parser: both share the "invalid AST:"
    # prefix, so pin the semantic message itself.
    assert out.reason is not None and out.reason.startswith("invalid AST:")
    assert "op 'contains' not supported" in out.reason
    assert out.raw == {"raw_content": content}


def test_fallback_parse_success_has_no_confidence_and_full_coverage(monkeypatch, tmp_path):
    """A valid generation returns the validated AST with confidence None (backend has no signal) and coverage 1.0."""
    content = ('{"target":"files","where":{"op":"and","children":['
               '{"field":"is_favorite","op":"eq","value":true},'
               '{"field":"type","op":"eq","value":"image"}]}}')
    backend = _fallback_backend_with(monkeypatch, tmp_path, content=content)
    out = backend.parse("favorite images", NOW)
    assert out.unsupported is False
    assert out.backend == "fallback_qwen"
    assert out.confidence is None
    assert out.coverage == 1.0
    assert out.reason is None
    assert out.raw == {"raw_content": content}
    conds = list(_iter_conds(out.ast["where"]))
    assert {"field": "is_favorite", "op": "eq", "value": True} in conds
    assert {"field": "type", "op": "eq", "value": "image"} in conds


def test_fallback_parse_dropped_quoted_literal_lowers_coverage(monkeypatch, tmp_path):
    """A valid AST that dropped the query's quoted literal reports coverage < 1.0 naming the lost string."""
    content = '{"target":"files","where":{"field":"type","op":"eq","value":"image"}}'
    backend = _fallback_backend_with(monkeypatch, tmp_path, content=content)
    out = backend.parse("images named 'dragon'", NOW)
    assert out.unsupported is False
    assert out.coverage is not None and out.coverage < 1.0
    assert out.reason is not None and "dragon" in out.reason
