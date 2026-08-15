# Repeatable benchmarks under real load, through the PRODUCTION pipeline:
# get_face_backend(AIConfig.from_env) resolves detection gates from the
# same env knobs the worker uses, replace_faces_for_file writes through the
# real schema, and cluster_faces() itself is timed. External CPU load is
# measured live during timing and stamped into every record in
# benchmarks/results/. `load` is a standalone CPU/GPU load snapshot.

set export
set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

python := './.venv/Scripts/python.exe'
omp := '8'
corpus := 'C:/Users/will/dev/sg-lab/hf-datasets'
models := 'C:/Users/will/dev/sg-lab/models'
cache := 'C:/Users/will/dev/sg-lab/hf-datasets-embeddings.npz'

# bare `just bench` lists instead of executing
default:
    @just --list bench

# diagnostic: current CPU busy fraction and per-GPU utilization
load:
    @-"$python" benchmarks/faiss_graph_evidence.py --check-idle

# image corpus through the production pipeline (detect -> embed -> DB -> cluster)
faiss thr='0.5':
    @"$python" benchmarks/faiss_graph_evidence.py --source images --dir "$corpus" --models-dir "$models" --cache "$cache" --threshold "$1" --omp-threads "$omp"

# real embeddings from a gallery cache DB (opened read-only)
faiss-db db:
    @"$python" benchmarks/faiss_graph_evidence.py --source db --db "$1" --omp-threads "$omp"

# detection recall by ground-truth face-size band (production detect path)
face-recall *args:
    @"$python" benchmarks/face_detection_recall.py --dir "$corpus" --models-dir "$models" "$@"
