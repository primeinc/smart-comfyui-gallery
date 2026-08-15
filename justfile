# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# Full test suite in the dev venv
test:
    ./.venv/Scripts/python.exe -m pytest tests/ -q

# FAISS graph-backend evidence: cross-backend equivalence + timings on
# seeded synthetic data at production shape -> benchmarks/results/
bench-faiss:
    ./.venv/Scripts/python.exe benchmarks/faiss_graph_evidence.py

# Same evidence run over REAL embeddings from a gallery cache DB (read-only)
bench-faiss-db db:
    ./.venv/Scripts/python.exe benchmarks/faiss_graph_evidence.py --source db --db "$1"
