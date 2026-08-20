# AI/ML debug surfaces. Every diagnostic that matters is a recipe here,
# not shell history: hardware coverage, decode canaries, search probes,
# and the acceptance benchmarks.

set positional-arguments

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
py := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# bare `just ai` lists instead of executing
default:
    @just --list ai

# Every AI runtime (transformers, torch, onnxruntime, faiss) exercised on
# CPU and on EACH GPU, crash-contained per cell, with output verification.
# Model/ONNX paths default to the provisioned artifacts.
[doc('Exercise generate/torch/ort/faiss on CPU and each GPU, verified')]
hardware-matrix out="hw_matrix.json":
    {{ py }} probes/hardware_matrix_probe.py --out "$1"

# One tiny generation on the resolved nl2sql model: proves the (build,
# device) pair actually decodes. CUDA_VISIBLE_DEVICES selects the GPU;
# AI_DAM_DEVICE=cpu forces CPU decode.
[doc('One tiny decode on the resolved nl2sql model (device via env)')]
nl2sql-canary:
    {{ py }} -c "from omniquery.parsers.nl2sql import SqlSearch; from smartgallery_ai import models; s = SqlSearch(db_path=':memory:'); print('model:', s.model_ref, '| provisioned:', s.available()); chat = models.Chat(s.model_ref, models_dir=s.models_dir); print('decoded:', repr(chat.ask('Say OK.', max_new_tokens=8)))"

# The full agentic loop for one question against the deterministic
# fixture DB: generated SQL, execution outcome, refinement rounds.
[doc('Run the agentic NL-to-SQL loop for one question (fixture DB)')]
nl2sql-search question:
    {{ py }} -c "import sys; from omniquery.benchmark.harness import _build_fixture_engine; from omniquery.parsers.nl2sql import SqlSearch; e = _build_fixture_engine(); s = SqlSearch(db_path=e.db_path); ids, sql, err = s.search(sys.argv[1]); print('ids:', len(ids) if ids is not None else None, '| err:', err); print('sql:', sql)" "$1"

# The deterministic parser alone: what one query parses to, with chips.
nlq-parse question:
    {{ py }} -c "import json, sys, time; from omniquery.parsers.nlq import NlqParser; o = NlqParser().parse(sys.argv[1], time.time()); print(json.dumps(o.ast, indent=1)); print('chips:', [c['label'] for c in o.raw['interpretation']])" "$1"

# Acceptance benchmarks over the committed corpus (fixture DB).
# fusion = the shipped endpoint policy; nlq = rules alone (sub-ms);
# sqlsearch = the model alone.
[doc('Corpus benchmark: rules alone')]
bench-nlq out="omni_nlq.json":
    {{ py }} -m omniquery.benchmark.harness --backends nlq --out "$1"

[doc('Corpus benchmark: the shipped fusion policy')]
bench-fusion out="omni_fusion.json":
    {{ py }} -m omniquery.benchmark.harness --backends nlq,fusion --out "$1"

[doc('Corpus benchmark: the model alone')]
bench-sqlsearch out="omni_sqlsearch.json":
    {{ py }} -m omniquery.benchmark.harness --backends sqlsearch --out "$1"
