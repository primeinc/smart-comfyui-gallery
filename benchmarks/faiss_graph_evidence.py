"""Repeatable end-to-end evidence for the faces neighbor-graph backends.

Runs every available backend (torch-cuda, faiss-cpu, numpy) over the same
vectors, verifies they produce the identical exhaustive cosine-threshold
edge set AND the identical chinese-whispers clustering, times each, and
writes the whole record to benchmarks/results/faiss_graph_evidence.json.

Data sources:
  --source synthetic          seeded clustered vectors at production shape
                              (default; runs anywhere, byte-reproducible)
  --source db --db PATH       real embeddings from a gallery cache DB,
                              opened read-only (mode=ro)
  --source hf --hf-dataset D --hf-column C
                              any Hugging Face dataset with an embedding
                              column (huggingface/datasets
                              docs/source/faiss_es.mdx pattern); requires
                              `pip install datasets`

Usage:  python benchmarks/faiss_graph_evidence.py [--threshold 0.6] [...]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time

# Must precede any BLAS-loading import: with OpenBLAS (what pip faiss-cpu and
# numpy wheels bundle), active OpenMP spin-waiting degrades search seriously
# (faiss wiki Troubleshooting.md "Slow brute-force search with OpenBLAS").
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "faiss_graph_evidence.json"
)


def _system_times():
    """(idle, kernel, user) CPU seconds summed over all cores, via win32
    GetSystemTimes; None on other platforms. kernel INCLUDES idle."""
    if sys.platform != "win32":
        return None
    import ctypes

    class _FT(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

    idle, kern, user = _FT(), _FT(), _FT()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)
    )
    if not ok:
        return None

    def _s(ft):
        return ((ft.hi << 32) | ft.lo) / 1e7

    return _s(idle), _s(kern), _s(user)


class _LoadWatch(threading.Thread):
    """Live external-CPU-load monitor for benchmark runs.

    Each interval: system busy time minus THIS process's CPU time, as a
    fraction of total capacity. Crossing `warn_frac` prints a warning the
    moment contamination happens; `stop()` returns a summary for the
    results record so a dirty run can never pass as clean evidence.
    """

    def __init__(self, warn_frac: float = 0.10, interval: float = 1.0):
        super().__init__(daemon=True)
        self.warn_frac = warn_frac
        self.interval = interval
        self.samples: list = []
        self._stop_evt = threading.Event()

    def run(self):
        prev = _system_times()
        if prev is None:
            return
        prev_proc = time.process_time()
        while not self._stop_evt.wait(self.interval):
            cur = _system_times()
            cur_proc = time.process_time()
            didle, dkern, duser = (c - p for c, p in zip(cur, prev))
            total = dkern + duser
            prev, prev_proc_old = cur, prev_proc
            prev_proc = cur_proc
            if total <= 0:
                continue
            external = max(0.0, (dkern - didle) + duser - (cur_proc - prev_proc_old))
            frac = external / total
            self.samples.append(frac)

    def stop(self) -> dict:
        self._stop_evt.set()
        self.join(timeout=5)
        if not self.samples:
            return {"supported": _system_times() is not None, "samples": 0}
        return {
            "supported": True,
            "samples": len(self.samples),
            "max_external_frac": round(max(self.samples), 3),
            "mean_external_frac": round(sum(self.samples) / len(self.samples), 3),
            "warn_frac": self.warn_frac,
            "contaminated": max(self.samples) > self.warn_frac,
        }


def check_idle(seconds: int = 3, busy_frac: float = 0.15) -> int:
    """Preflight gate: refuse to benchmark on a busy machine.

    Prints one live line per second for CPU (all processes) and one per GPU;
    returns 1 when mean CPU busy exceeds `busy_frac` or any GPU is >20%
    utilized.
    """
    fracs = []
    prev = _system_times()
    if prev is None:
        print("[load] CPU preflight unsupported on this platform; proceeding")
    else:
        for _ in range(seconds):
            time.sleep(1.0)
            cur = _system_times()
            didle, dkern, duser = (c - p for c, p in zip(cur, prev))
            prev = cur
            total = dkern + duser
            busy = ((dkern - didle) + duser) / total if total > 0 else 0.0
            fracs.append(busy)
            print(f"[load] cpu busy {busy:.0%}", flush=True)
    gpu_busy = False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            for i, line in enumerate(out.stdout.split()):
                util = int(line)
                print(f"[load] gpu{i} util {util}%", flush=True)
                gpu_busy = gpu_busy or util > 20
    except (OSError, ValueError, subprocess.TimeoutExpired):
        print("[load] nvidia-smi unavailable; skipping GPU preflight")
    cpu_busy = bool(fracs) and sum(fracs) / len(fracs) > busy_frac
    if cpu_busy or gpu_busy:
        print("[load] FAIL: machine is busy — benchmark refused", flush=True)
        return 1
    print("[load] ok: machine is idle", flush=True)
    return 0


def load_faces_module():
    """Import smartgallery_ai.faces with the repo root on sys.path, so the
    script runs from any cwd without installation."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module("smartgallery_ai.faces")


def load_synthetic(n: int, d: int, centers: int, seed: int) -> np.ndarray:
    """Clustered vectors mimicking generated-face edge density; random
    uniform vectors would concentrate cosines near 0 and test nothing."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((centers, d)).astype(np.float32)
    m = c[rng.integers(0, centers, n)] + 0.35 * rng.standard_normal((n, d)).astype(
        np.float32
    )
    return m


def load_db(path: str) -> np.ndarray:
    """Real face embeddings, read-only; majority dim wins if mixed."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT embedding, dim FROM ai_face_instances WHERE embedding IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise SystemExit(f"no embeddings in {path}")
    dims: dict = {}
    for _, dim in rows:
        dims[dim] = dims.get(dim, 0) + 1
    dim = max(dims.items(), key=lambda kv: kv[1])[0]
    return np.stack(
        [np.frombuffer(blob, dtype="<f4") for blob, d in rows if d == dim]
    )


def load_hf(dataset: str, column: str, split: str) -> np.ndarray:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    return np.asarray(ds[column], dtype=np.float32)


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Datasets whose immediate parent directory is a ground-truth identity.
_LABELED_DATASETS = ("Black_People_Face_Recognition", "caucasian-people-kyc-photo-dataset")


def load_images(root: str, models_dir: str, cache: str) -> tuple:
    """Embed every face in every image under `root` with the production
    YuNet+SFace backend (the pixels->embeddings leg IS part of the system
    under test). Returns (embeddings, face_datasets, face_identities).

    The result is cached to `cache` (.npz) keyed by nothing fancier than
    the file's existence — delete it to force a re-embed.
    """
    if cache and os.path.isfile(cache):
        data = np.load(cache, allow_pickle=False)
        if "paths" not in data:
            print(f"cache {cache} lacks per-face paths; re-embedding")
        else:
            print(f"loaded embedding cache {cache} ({data['embeddings'].shape[0]} faces)")
            return data["embeddings"], list(data["datasets"]), list(data["identities"])

    from PIL import Image

    from smartgallery_ai.faces import OpenCVFaceBackend

    backend = OpenCVFaceBackend(models_dir)
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(_IMAGE_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    print(f"embedding {len(paths)} images under {root} via {backend.model_id} ...")

    embs, ds_names, identities = [], [], []
    face_paths, face_bboxes = [], []
    decode_failures = 0
    t0 = time.perf_counter()
    for i, path in enumerate(paths):
        rel = os.path.relpath(path, root)
        parts = rel.replace("\\", "/").split("/")
        dataset = parts[0]
        identity = (
            f"{dataset}/{parts[-2]}"
            if dataset in _LABELED_DATASETS and len(parts) >= 3
            else ""
        )
        try:
            with Image.open(path) as img:
                detections = backend.detect(img)
        except Exception:
            decode_failures += 1
            continue
        for det in detections:
            if det.embedding is not None:
                embs.append(det.embedding)
                ds_names.append(dataset)
                identities.append(identity)
                face_paths.append(rel)
                face_bboxes.append(det.bbox)
        if (i + 1) % 500 == 0:
            print(
                f"  {i + 1}/{len(paths)} images, {len(embs)} faces, "
                f"{time.perf_counter() - t0:.0f}s"
            )
    print(
        f"embedded {len(embs)} faces from {len(paths)} images "
        f"({decode_failures} undecodable) in {time.perf_counter() - t0:.0f}s"
    )
    embeddings = np.stack(embs) if embs else np.zeros((0, 128), dtype=np.float32)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(
            cache,
            embeddings=embeddings,
            datasets=np.array(ds_names),
            identities=np.array(identities),
            paths=np.array(face_paths),
            bboxes=np.array(face_bboxes, dtype=np.float32).reshape(-1, 4),
        )
        print(f"wrote embedding cache {cache}")
    return embeddings, ds_names, identities


def sanity_report(labels: list, ds_names: list, identities: list) -> dict:
    """Does the clustering still make sense on a messy mixed corpus?

    - faces per source dataset (scene dataset should contribute few)
    - top-cluster share (a mega-cluster is the historical failure mode)
    - for every ground-truth identity with >= 2 faces: purity = largest
      single-cluster fraction of that identity's faces.
    """
    n = len(labels)
    per_dataset: dict = {}
    for d in ds_names:
        per_dataset[d] = per_dataset.get(d, 0) + 1

    cluster_sizes: dict = {}
    for lab in labels:
        cluster_sizes[lab] = cluster_sizes.get(lab, 0) + 1
    top_share = max(cluster_sizes.values()) / n if n else 0.0

    by_identity: dict = {}
    for lab, ident in zip(labels, identities):
        if ident:
            by_identity.setdefault(ident, []).append(lab)
    purities = {}
    for ident, labs in by_identity.items():
        if len(labs) < 2:
            continue
        counts: dict = {}
        for lab in labs:
            counts[lab] = counts.get(lab, 0) + 1
        purities[ident] = max(counts.values()) / len(labs)
    return {
        "faces_per_dataset": per_dataset,
        "clusters": len(cluster_sizes),
        "top_cluster_share": round(top_share, 4),
        "labeled_identities": len(purities),
        "identities_fully_grouped": sum(1 for p in purities.values() if p == 1.0),
        "mean_identity_purity": (
            round(sum(purities.values()) / len(purities), 4) if purities else None
        ),
    }


def sorted_edges(graph) -> tuple:
    """CSR graph as (rows, cols, weights) sorted by (row, col) so two graphs
    can be compared element-wise regardless of emission order."""
    indptr, cols, weights = graph
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], weights[order]


def canonical_partition(labels: list) -> list:
    """Cluster memberships as sorted tuples, order-independent."""
    groups: dict = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    return sorted(tuple(v) for v in groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("synthetic", "db", "hf", "images"), default="synthetic")
    ap.add_argument("--db", help="gallery cache sqlite path (source=db)")
    ap.add_argument("--hf-dataset", help="HF dataset name (source=hf)")
    ap.add_argument("--hf-column", default="embeddings")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--dir", help="image corpus root (source=images)")
    ap.add_argument("--models-dir", help="dir with YuNet+SFace ONNX (source=images)")
    ap.add_argument("--cache", default="", help="embedding cache .npz (source=images)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=22712)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--centers", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--omp-threads",
        type=int,
        default=0,
        help="OpenMP threads for faiss (0 = library default). Upstream "
        "guidance (faiss wiki How-to-make-Faiss-run-faster.md) is the number "
        "of PHYSICAL cores, not the hyperthread default.",
    )
    ap.add_argument(
        "--check-idle",
        action="store_true",
        help="preflight only: exit 1 if the machine is busy, run nothing",
    )
    args = ap.parse_args()

    if args.check_idle:
        raise SystemExit(check_idle())

    faces = load_faces_module()

    ds_names: list = []
    identities: list = []
    if args.source == "synthetic":
        m = load_synthetic(args.n, args.dim, args.centers, args.seed)
        source = f"synthetic(n={args.n},d={args.dim},centers={args.centers},seed={args.seed})"
    elif args.source == "db":
        if not args.db:
            raise SystemExit("--source db requires --db PATH")
        m = load_db(args.db)
        source = f"db({args.db})"
    elif args.source == "images":
        if not args.dir or not args.models_dir:
            raise SystemExit("--source images requires --dir and --models-dir")
        m, ds_names, identities = load_images(args.dir, args.models_dir, args.cache)
        source = f"images({args.dir})"
    else:
        if not args.hf_dataset:
            raise SystemExit("--source hf requires --hf-dataset")
        m = load_hf(args.hf_dataset, args.hf_column, args.hf_split)
        source = f"hf({args.hf_dataset}:{args.hf_column}:{args.hf_split})"

    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    m = (m / norms).astype(np.float32)
    print(f"{m.shape[0]:,} vectors ({m.shape[1]}-d), threshold {args.threshold} — {source}")

    backends = {}
    try:
        import torch

        if torch.cuda.is_available():
            faces._neighbor_graph_torch_cuda(m[: min(2048, len(m))], args.threshold)
            backends["torch-cuda"] = (
                faces._neighbor_graph_torch_cuda,
                f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}",
            )
    except ImportError:
        pass
    try:
        import faiss

        if args.omp_threads > 0:
            faiss.omp_set_num_threads(args.omp_threads)
        backends["faiss-cpu"] = (
            faces._neighbor_graph_faiss,
            f"faiss {getattr(faiss, '__version__', '?')} "
            f"omp={faiss.omp_get_max_threads()} "
            f"wait_policy={os.environ.get('OMP_WAIT_POLICY', 'default')}",
        )
    except ImportError:
        pass
    backends["numpy"] = (faces._neighbor_graph_numpy, f"numpy {np.__version__}")

    record = {
        "source": source,
        "threshold": args.threshold,
        "vectors": list(m.shape),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        "backends": {},
        "equivalence": {},
    }

    watch = _LoadWatch()
    watch.start()
    graphs, partitions, labels_by_backend = {}, {}, {}
    for name, (fn, runtime) in backends.items():
        best = float("inf")
        graph = None
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            graph = fn(m, args.threshold)
            best = min(best, time.perf_counter() - t0)
        graphs[name] = graph
        t0 = time.perf_counter()
        labels = faces._chinese_whispers(graph)
        cw_time = time.perf_counter() - t0
        labels_by_backend[name] = labels
        partitions[name] = canonical_partition(labels)
        sizes = sorted((len(g) for g in partitions[name]), reverse=True)
        record["backends"][name] = {
            "runtime": runtime,
            "graph_ms": round(best * 1e3, 1),
            "chinese_whispers_ms": round(cw_time * 1e3, 1),
            "edges": int(len(graph[1])),
            "clusters": len(sizes),
            "largest_cluster": sizes[0] if sizes else 0,
        }
        print(
            f"  {name:10s} {best * 1e3:7.0f} ms graph + {cw_time * 1e3:5.0f} ms clustering",
            flush=True,
        )

    # Float contract, not bitwise fantasy: cuBLAS/BLAS reduce dot products in
    # different orders even in IEEE mode, so similarities differ in final
    # float32 ULPs. Therefore:
    #   - edges present in BOTH backends must agree in weight to <= 1e-5
    #   - an edge present in only ONE backend is legal ONLY if its similarity
    #     sits within 1e-5 of the threshold (the ULP flipped its membership)
    #   - the chinese-whispers clustering must be identical
    tol = 1e-5
    n_vec = m.shape[0]
    names = list(graphs)
    ref = names[0]
    ref_rows, ref_cols, ref_w = sorted_edges(graphs[ref])
    ref_key = ref_rows * n_vec + ref_cols
    ref_part = partitions[ref]
    for other in names[1:]:
        o_rows, o_cols, o_w = sorted_edges(graphs[other])
        o_key = o_rows * n_vec + o_cols
        in_both_ref = np.isin(ref_key, o_key, assume_unique=True)
        in_both_o = np.isin(o_key, ref_key, assume_unique=True)
        max_w_delta = (
            float(
                np.max(
                    np.abs(
                        ref_w[in_both_ref].astype(np.float64)
                        - o_w[in_both_o].astype(np.float64)
                    )
                )
            )
            if in_both_ref.any()
            else 0.0
        )
        divergent_w = np.concatenate([ref_w[~in_both_ref], o_w[~in_both_o]])
        n_divergent = int(len(divergent_w))
        boundary_ok = bool(
            np.all(np.abs(divergent_w.astype(np.float64) - args.threshold) <= tol)
        )
        weights_ok = max_w_delta <= tol
        same_clusters = partitions[other] == ref_part
        record["equivalence"][f"{ref}=={other}"] = {
            "shared_edges_max_weight_delta": max_w_delta,
            "divergent_edges": n_divergent,
            "divergent_edges_all_at_threshold_boundary": boundary_ok,
            "clustering": same_clusters,
        }
        if not (weights_ok and boundary_ok and same_clusters):
            print(
                f"FAILED: {ref} and {other} disagree — max weight delta "
                f"{max_w_delta:.2e}, {n_divergent} divergent edges "
                f"(boundary-only: {boundary_ok}), clustering equal: {same_clusters}"
            )
            raise SystemExit(1)

    sizes = sorted((len(g) for g in ref_part), reverse=True)
    agreement = f" — identical across all {len(names)} backends" if len(names) > 1 else ""
    print(
        f"  result: {len(ref_cols):,} edges, {len(sizes):,} clusters "
        f"(largest {sizes[0] if sizes else 0}){agreement}"
    )

    record["load"] = watch.stop()
    if record["load"].get("samples"):
        print(
            f"  background load: avg {record['load']['mean_external_frac']:.0%}, "
            f"peak {record['load']['max_external_frac']:.0%}"
        )

    if ds_names:
        record["sanity"] = sanity_report(labels_by_backend[ref], ds_names, identities)
        s = record["sanity"]
        print(
            f"  sanity: top cluster holds {s['top_cluster_share']:.1%} of faces; "
            f"{s['labeled_identities']} labeled identities, "
            f"purity {s['mean_identity_purity']}"
        )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  details: {os.path.relpath(RESULTS_PATH)}")


if __name__ == "__main__":
    main()
