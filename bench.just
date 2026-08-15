# Repeatable benchmarks with live load monitoring. `check-idle` refuses a
# busy machine; the harness watches external CPU load DURING timing, warns
# live, and stamps load stats into benchmarks/results/ so a contaminated
# run cannot pass as clean evidence.

set export
set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

python := './.venv/Scripts/python.exe'
omp := '8'

# bare `just bench` lists instead of executing
default:
    @just --list bench

# preflight: exit 1 unless CPU and GPUs are idle
check-idle:
    "$python" benchmarks/faiss_graph_evidence.py --check-idle

# synthetic corpus at production shape (seeded, reproducible)
faiss: check-idle
    "$python" benchmarks/faiss_graph_evidence.py --omp-threads "$omp"

# real embeddings from a gallery cache DB (opened read-only)
faiss-db db: check-idle
    "$python" benchmarks/faiss_graph_evidence.py --source db --db "$1" --omp-threads "$omp"

# real image corpus end to end (faces detected+embedded, cached in the npz)
faiss-images dir models cache thr='0.5': check-idle
    "$python" benchmarks/faiss_graph_evidence.py --source images --dir "$1" --models-dir "$2" --cache "$3" --threshold "$4" --omp-threads "$omp"
