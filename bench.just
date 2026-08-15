# Repeatable benchmarks under real load. The harness measures external CPU
# load live DURING timing, prints it, and stamps the summary into every
# record in benchmarks/results/ — numbers always carry their measured load
# context. Best-of-N timing takes the least-loaded window. `load` is a
# standalone diagnostic snapshot of current CPU/GPU load.

set export
set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

python := './.venv/Scripts/python.exe'
omp := '8'

# bare `just bench` lists instead of executing
default:
    @just --list bench

# diagnostic: current CPU busy fraction and per-GPU utilization
load:
    @-"$python" benchmarks/faiss_graph_evidence.py --check-idle

# synthetic corpus at production shape (seeded, reproducible)
faiss:
    @"$python" benchmarks/faiss_graph_evidence.py --omp-threads "$omp"

# real embeddings from a gallery cache DB (opened read-only)
faiss-db db:
    @"$python" benchmarks/faiss_graph_evidence.py --source db --db "$1" --omp-threads "$omp"

# real image corpus end to end (faces detected+embedded, cached in the npz)
faiss-images dir models cache thr='0.5':
    @"$python" benchmarks/faiss_graph_evidence.py --source images --dir "$1" --models-dir "$2" --cache "$3" --threshold "$4" --omp-threads "$omp"
