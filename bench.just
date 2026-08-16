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
models_w600k := models + '-w600k'
insightface_root := 'C:/Users/will/dev/sg-lab/insightface-root'
face_ab_corpora := 'kyc=caucasian-people-kyc-photo-dataset/files/*/*.jpg;bpfr=Black_People_Face_Recognition/Kaggle - samples_Black/*/*'

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

# face-pipeline A/B (sface | arcface glintr100 | w600k | insightface scrfd+glintr100) on labeled corpora
face-ab:
    @FACE_AB_MODELS="$models" FACE_AB_MODELS_W600K="$models_w600k" FACE_AB_INSIGHTFACE_ROOT="$insightface_root" FACE_AB_CORPUS_ROOT="$corpus" FACE_AB_CORPORA="$face_ab_corpora" "$python" benchmarks/face_embedder_ab.py
