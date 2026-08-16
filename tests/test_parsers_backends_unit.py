"""Model-free unit tests for the optional-runtime GGUF parser backends,
fallback_qwen.FallbackQwenBackend and nl2sql.Nl2SqlBackend: availability
checks, model-path resolution, and the never-raise parse() contract --
exercised through injected fake engines only.

IMPORTANT: on this machine the real `llama_cpp` package IS installed and
GGUF files exist at real paths, so these tests never rely on an
ImportError happening naturally and never touch a default model path.
Unavailability is forced hermetically by stubbing sys.modules (a None
entry makes `import x` raise ImportError); fake engines are injected
through the module's own seam (`fallback_qwen._MODEL_CACHE`). No real
engine is ever constructed.
"""

from __future__ import annotations

import sys
import types

from omniquery import fields
from omniquery.parsers import fallback_qwen as fallback_module
from omniquery.parsers.fallback_qwen import DEFAULT_MODEL_PATH, FallbackQwenBackend, _example_for
from omniquery.parsers.nl2sql import Nl2SqlBackend

NOW = 1735689600.0  # unused by the backends, but required by the interface


def _iter_conds(node):
    if "children" in node:
        for c in node["children"]:
            yield from _iter_conds(c)
    elif "child" in node:
        yield from _iter_conds(node["child"])
    else:
        yield node


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
# 1. Model path resolution + availability
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


def test_nl2sql_model_path_resolution_order(monkeypatch, tmp_path):
    """Nl2SqlBackend resolves explicit arg over OMNIQUERY_NL2SQL_GGUF over
    OMNIQUERY_FALLBACK_GGUF (shared install) over the built-in default --
    and reports as its own backend name."""
    monkeypatch.delenv("OMNIQUERY_NL2SQL_GGUF", raising=False)
    monkeypatch.delenv("OMNIQUERY_FALLBACK_GGUF", raising=False)
    assert Nl2SqlBackend().model_path == DEFAULT_MODEL_PATH
    assert Nl2SqlBackend.name == "nl2sql"

    shared = str(tmp_path / "shared.gguf")
    monkeypatch.setenv("OMNIQUERY_FALLBACK_GGUF", shared)
    assert Nl2SqlBackend().model_path == shared

    own = str(tmp_path / "nl2sql.gguf")
    monkeypatch.setenv("OMNIQUERY_NL2SQL_GGUF", own)
    assert Nl2SqlBackend().model_path == own

    explicit = str(tmp_path / "explicit.gguf")
    assert Nl2SqlBackend(model_path=explicit).model_path == explicit


def test_fallback_available_false_when_model_file_absent(monkeypatch, tmp_path):
    """available() is False when the GGUF file is missing, even with the runtime importable."""
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


def test_nl2sql_availability_follows_same_contract(monkeypatch, tmp_path):
    """Nl2SqlBackend inherits the runtime+file availability contract."""
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    assert Nl2SqlBackend(model_path=str(tmp_path / "missing.gguf")).available() is False
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not really a gguf")
    assert Nl2SqlBackend(model_path=str(model)).available() is True


# ---------------------------------------------------------------------------
# 2. _example_for's exhaustive-over-Kind fallback branch
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
# 3. parse() fail-closed contract
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
