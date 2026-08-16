"""Grammar-constrained Qwen2.5-Coder-0.5B fallback: the single optional
constrained-decoding backend. It emits the SAME typed AST schema as every
other backend, enforced structurally by decoding under a grammar built
straight from `omniquery.ast.json_schema()` (not a hand-rolled schema) --
so "the fallback might drift from the real AST shape" is not a failure mode
that needs separate handling here.

The grammar guarantees *shape* (valid JSON matching the schema: known
field/op enums, typed values, well-formed and/or/not nesting, no unknown
keys) but not *semantics* -- a small model can still legally emit
`{"field": "is_favorite", "op": "eq", "value": "yes"}`: a schema-legal
scalar of the wrong kind for the field, which will fail
`omniquery.validation.validate`. That's expected and handled the same way
as every other backend failure: a failed ParserOutcome, never a raised
exception and never an unvalidated AST returned to the caller.

Model + grammar are loaded lazily and cached at module scope (once per
process); `available()` only checks that the runtime imports and the model
file exists -- it never triggers the load.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from omniquery import ast as ast_module
from omniquery import fields
from omniquery.parsers import ParserBackend, ParserOutcome, coverage_guard, try_validate

# Model file resolution precedence: constructor argument, then the
# ENV_MODEL_PATH environment variable, then DEFAULT_MODEL_PATH -- the
# provisioned nl2sql GGUF inside the AI models directory (AI_DAM_MODELS_DIR
# env, .AImodels fallback; provision.py's 'omniquery' group downloads it).
DEFAULT_MODEL_FILENAME = "distil-qwen3-4b-text2sql-4bit.gguf"
DEFAULT_MODEL_PATH = os.path.join(
    os.environ.get("AI_DAM_MODELS_DIR", ".AImodels"), DEFAULT_MODEL_FILENAME)
ENV_MODEL_PATH = "OMNIQUERY_FALLBACK_GGUF"  # name of the env var, not a path itself

# (llama_instance, grammar_instance), keyed by resolved model path -- loaded
# once per process no matter how many FallbackQwenBackend instances exist.
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _resolve_model_path(model_path: Optional[str]) -> str:
    """Apply the model-path precedence: explicit argument, then
    $OMNIQUERY_FALLBACK_GGUF, then DEFAULT_MODEL_PATH."""
    if model_path:
        return model_path
    return os.environ.get(ENV_MODEL_PATH, DEFAULT_MODEL_PATH)


def _prepare_dll_path() -> None:
    """Prebuilt CUDA llama-cpp-python wheels need the pip-installed NVIDIA
    runtime DLLs on PATH (the wheel's loader uses winmode=RTLD_GLOBAL, the
    legacy search that ignores add_dll_directory — see
    llama_cpp/_ctypes_extensions.py). smartgallery_ai owns the shared
    bootstrap; standalone omniquery use falls through harmlessly."""
    try:
        from smartgallery_ai.llama_runtime import prepare_llama_runtime
        prepare_llama_runtime()
    except ImportError:
        pass


def _example_for(spec: fields.FieldSpec) -> str:
    """One representative JSON literal for a field's kind, shown in the
    system prompt's field listing so the model sees each value shape."""
    if spec.kind == fields.Kind.TEXT:
        return '"dragon"'
    if spec.kind == fields.Kind.NUMBER:
        return "4"
    if spec.kind == fields.Kind.BOOL:
        return "true"
    if spec.kind == fields.Kind.ENUM:
        values = sorted(spec.enum_values or ())
        return json.dumps(values[0]) if values else '"..."'
    if spec.kind == fields.Kind.DATETIME:
        return '{"days_ago": 7}'
    if spec.kind == fields.Kind.FILE_REF:
        return '"f001"'
    return "..."  # exhaustive over Kind


def _build_system_prompt() -> str:
    """Compose the fixed system prompt: output contract, the field catalog
    (generated from omniquery.fields so it can never lag the live schema),
    and few-shot examples covering and/or/not, count, and days_ago."""
    lines = [
        "You translate a natural-language SmartGallery media query into a single "
        "JSON query AST. Output ONLY the JSON object -- no prose, no markdown "
        "fences. Every condition is {\"field\":..., \"op\":..., \"value\":...}; "
        "combine several with {\"op\":\"and\"/\"or\",\"children\":[...]}; negate "
        "one with {\"op\":\"not\",\"child\":...}. Only emit predicates the query "
        "actually asks for.",
        "",
        "Fields (name: kind, example value):",
    ]
    for name in fields.field_names():
        spec = fields.FIELDS[name]
        lines.append(f"- {name}: {spec.kind.value}, ops={sorted(spec.ops)}, example={_example_for(spec)}")
    lines += [
        "",
        "Examples:",
        "Q: favorite images from the last 7 days",
        'A: {"target":"files","where":{"op":"and","children":['
        '{"field":"type","op":"eq","value":"image"},'
        '{"field":"is_favorite","op":"eq","value":true},'
        '{"field":"mtime","op":"ge","value":{"days_ago":7}}]}}',
        "Q: not approved",
        'A: {"target":"files","where":{"op":"not","child":'
        '{"field":"status_flag","op":"eq","value":"Approved"}}}',
        "Q: favorite images or 4 star videos",
        'A: {"target":"files","where":{"op":"or","children":['
        '{"field":"is_favorite","op":"eq","value":true},'
        '{"field":"rating_avg","op":"ge","value":4}]}}',
        "Q: how many favorites",
        'A: {"target":"files","result":"count","where":'
        '{"field":"is_favorite","op":"eq","value":true}}',
    ]
    return "\n".join(lines)


def _load_model(model_path: str, n_ctx: int, n_threads: int) -> Tuple[Any, Any]:
    """Return the process-wide (llama, grammar) pair for `model_path`,
    loading and caching on first use. The grammar is built from
    omniquery.ast.json_schema(), so decoding is constrained to the real
    AST shape by construction."""
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"fallback GGUF model not found at {model_path}")
    _prepare_dll_path()
    from llama_cpp import Llama, LlamaGrammar  # local import: optional runtime

    # Full GPU offload by default: a CUDA build uses it, a CPU-only build
    # ignores every GPU knob, so this is safe everywhere.
    llama = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                  n_gpu_layers=int(os.environ.get("OMNIQUERY_FALLBACK_GPU_LAYERS", "-1")),
                  verbose=False)
    schema = ast_module.json_schema(field_names=fields.field_names(), operator_names=fields.all_ops())
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    _MODEL_CACHE[model_path] = (llama, grammar)
    return llama, grammar


def _ms(t0: float) -> float:
    """Milliseconds elapsed since monotonic timestamp `t0`."""
    return (time.monotonic() - t0) * 1000.0


class FallbackQwenBackend(ParserBackend):
    """ParserBackend wrapper around the grammar-constrained GGUF model; the
    module docstring states the shape-vs-semantics contract."""

    name = "fallback_qwen"

    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 2048,
                 n_threads: int = 4, max_tokens: int = 512):
        """Store decode settings only; the model itself is loaded (once per
        process) on the first parse() call, never here."""
        self.model_path = _resolve_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens
        self._system_prompt = _build_system_prompt()

    def available(self) -> bool:
        """True when llama_cpp imports and the model file exists;
        deliberately never triggers the (expensive) model load."""
        try:
            _prepare_dll_path()
            import llama_cpp
        except Exception:
            # CUDA builds can fail at DLL load, not just plain ImportError.
            return False
        return bool(llama_cpp) and os.path.isfile(self.model_path)

    def parse(self, text: str, _now_epoch: float) -> ParserOutcome:
        """Constrained-decode an AST from `text` at temperature 0, gated by
        coverage_guard and validation. `_now_epoch` is unused: relative
        dates stay symbolic as {"days_ago": N}. Never raises."""
        t0 = time.monotonic()
        try:
            llama, grammar = _load_model(self.model_path, self.n_ctx, self.n_threads)
        except Exception as exc:
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=f"model load error: {exc}", latency_ms=_ms(t0))

        content: Optional[str] = None
        try:
            resp = llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
                grammar=grammar,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
            content = resp["choices"][0]["message"]["content"]
            ast_dict = json.loads(content)
        except Exception as exc:
            # A truncated/malformed generation
            # must never propagate as an exception out of parse().
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=f"generation error: {exc}", latency_ms=_ms(t0),
                                  raw={"raw_content": content} if content is not None else None)

        latency_ms = _ms(t0)
        coverage, missing = coverage_guard(text, ast_dict)

        query, err = try_validate(ast_dict)
        if err is not None:
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=f"invalid AST: {err}", coverage=coverage,
                                  latency_ms=latency_ms, raw={"raw_content": content})

        # This backend emits no confidence signal at all (grammar-constrained
        # decoding produces none); the router leans entirely on `coverage`
        # for it -- see router.py's fallback acceptance rule.
        return ParserOutcome(ast=query.to_dict(), confidence=None, backend=self.name,
                              unsupported=False, reason=("; ".join(missing) or None),
                              coverage=coverage, latency_ms=latency_ms, raw={"raw_content": content})
