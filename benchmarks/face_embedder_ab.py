"""Three-way face-embedder A/B through the production pipeline (repo
OpenCVFaceBackend: same YuNet policy, gates, and per-embedder alignment
contract) on identity-labeled corpora: SFace vs ArcFace w600k_r50 vs
ArcFace glintr100.

Per variant:
  verification -- genuine/impostor cosine distributions + best-F1 threshold
  clustering   -- repo _neighbor_graph + _chinese_whispers threshold sweep,
                  scored as pairwise precision/recall/F1 vs identity labels

Corpus globs resolve <identity-folder>/<image>; identity ground truth is
the parent folder name. Configuration is env-only (no baked-in paths;
`just bench face-ab` carries the machine defaults):
  FACE_AB_MODELS        required; dir with yunet + sface + the shipped
                        arcface (glintr100) weights
  FACE_AB_MODELS_W600K  optional; dir where w600k_r50 is staged under the
                        arcface filename (variant skipped when unset)
  FACE_AB_INSIGHTFACE_ROOT  optional; insightface root (root/models/antelopev2)
                        enabling the upstream FaceAnalysis pipeline variant
                        (needs the insightface + onnxruntime packages)
  FACE_AB_CORPUS_ROOT   required; root the corpus globs resolve against
  FACE_AB_CORPORA       required; "name=relative-glob;name=relative-glob"
  FACE_AB_OUT           optional; results JSON path (defaults to
                        benchmarks/results/face_embedder_ab.json)
Writes the results JSON and prints a table.
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AI_DAM_FACE_GRAPH_BACKEND", "numpy")

import cv2
from insightface.app import FaceAnalysis
from PIL import Image

from smartgallery_ai.faces import OpenCVFaceBackend, _chinese_whispers, _neighbor_graph


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"{name} is required (see module docstring; "
                 "`just bench face-ab` supplies the machine defaults)")
    return value


MODELS = _required_env("FACE_AB_MODELS")
MODELS_W600K = os.environ.get("FACE_AB_MODELS_W600K", "").strip()
_ROOT = _required_env("FACE_AB_CORPUS_ROOT")
# one relative glob per labeled corpus: "name=rel-glob;name=rel-glob"
CORPORA = {name: os.path.join(_ROOT, rel)
           for name, rel in (pair.split("=", 1)
                             for pair in _required_env("FACE_AB_CORPORA").split(";"))}


def _corpus_images():
    for corpus, pattern in CORPORA.items():
        for path in sorted(glob.glob(pattern)):
            if not path.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            yield f"{corpus}/{os.path.basename(os.path.dirname(path))}", path


def _accumulate(detect_best):
    """Run `detect_best(path) -> unit vector or None` over the corpus."""
    labels, vecs, skipped = [], [], 0
    t0 = time.perf_counter()
    for identity, path in _corpus_images():
        try:
            v = detect_best(path)
        except Exception:
            v = None
        if v is None:
            skipped += 1
            continue
        labels.append(identity)
        vecs.append(v)
    return labels, np.vstack(vecs), skipped, time.perf_counter() - t0


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n else None


def collect(embedder, models_dir=MODELS):
    backend = OpenCVFaceBackend(models_dir, embedder=embedder)

    def detect_best(path):
        with Image.open(path) as img:
            dets = backend.detect(img)
        if not dets:
            return None
        return _unit(max(dets, key=lambda d: d.det_score).embedding)

    labels, vecs, skipped, elapsed = _accumulate(detect_best)
    return labels, vecs, skipped, elapsed, backend.model_version


def collect_insightface(pack, root):
    """insightface's own quickstart pipeline: SCRFD joint 128+640 detect,
    upstream landmark alignment, pack recognizer -- their code end to end."""

    app = FaceAnalysis(name=pack, root=root,
                       allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    def detect_best(path):
        img = cv2.imread(path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        return _unit(max(faces, key=lambda f: f.det_score).embedding)

    labels, vecs, skipped, elapsed = _accumulate(detect_best)
    return labels, vecs, skipped, elapsed, f"insightface-{pack}"


def pairwise_f1(pred_labels, true_labels):
    true = np.asarray(true_labels)
    pred = np.asarray(pred_labels)
    n = len(true)
    iu = np.triu_indices(n, k=1)
    same_true = (true[iu[0]] == true[iu[1]])
    same_pred = (pred[iu[0]] == pred[iu[1]])
    tp = int(np.sum(same_true & same_pred))
    fp = int(np.sum(~same_true & same_pred))
    fn = int(np.sum(same_true & ~same_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def verification(labels, vecs):
    true = np.asarray(labels)
    sims = vecs @ vecs.T
    iu = np.triu_indices(len(true), k=1)
    s = sims[iu]
    genuine = s[true[iu[0]] == true[iu[1]]]
    impostor = s[true[iu[0]] != true[iu[1]]]
    best = (0.0, 0.0)
    for thr in np.arange(0.05, 0.95, 0.01):
        tp = int(np.sum(genuine >= thr))
        fp = int(np.sum(impostor >= thr))
        fn = int(np.sum(genuine < thr))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[1]:
            best = (float(thr), f1)
    return {
        "n_genuine": int(genuine.size), "n_impostor": int(impostor.size),
        "genuine_mean": float(genuine.mean()), "genuine_p05": float(np.percentile(genuine, 5)),
        "impostor_mean": float(impostor.mean()), "impostor_p95": float(np.percentile(impostor, 95)),
        "separation": float(np.percentile(genuine, 5) - np.percentile(impostor, 95)),
        "best_f1_threshold": best[0], "best_f1": best[1],
    }


VARIANTS = {
    "sface": lambda: collect("sface", MODELS),
    "arcface-glintr100": lambda: collect("arcface", MODELS),
}
if MODELS_W600K:
    VARIANTS["arcface-w600k-r50"] = lambda: collect("arcface", MODELS_W600K)
_IF_ROOT = os.environ.get("FACE_AB_INSIGHTFACE_ROOT", "").strip()
if _IF_ROOT:
    VARIANTS["insightface-antelopev2"] = (
        lambda: collect_insightface("antelopev2", _IF_ROOT))
results = {}
for variant, run in VARIANTS.items():
    labels, vecs, skipped, elapsed, version = run()
    version = variant
    embedder = variant
    ver = verification(labels, vecs)
    sweep = []
    for thr in [0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70]:
        graph, _backend = _neighbor_graph(vecs, thr)
        pred = _chinese_whispers(graph)
        prec, rec, f1 = pairwise_f1(pred, labels)
        sweep.append({"threshold": thr, "clusters": len(set(pred)),
                      "precision": round(prec, 4), "recall": round(rec, 4),
                      "f1": round(f1, 4)})
    results[embedder] = {
        "model_version": version, "faces": len(labels),
        "identities": len(set(labels)), "skipped": skipped,
        "detect_embed_seconds": round(elapsed, 1),
        "verification": ver, "cluster_sweep": sweep,
    }
    print(f"\n== {embedder} ({version}) — {len(labels)} faces, "
          f"{len(set(labels))} identities, {skipped} skipped, {elapsed:.0f}s")
    v = ver
    print(f"   genuine mean {v['genuine_mean']:.3f} (p05 {v['genuine_p05']:.3f})  "
          f"impostor mean {v['impostor_mean']:.3f} (p95 {v['impostor_p95']:.3f})  "
          f"separation {v['separation']:+.3f}")
    print(f"   verification best-F1 {v['best_f1']:.4f} @ thr {v['best_f1_threshold']:.2f}")
    for row in sweep:
        print(f"   CW thr {row['threshold']:.2f}: {row['clusters']:4d} clusters  "
              f"P {row['precision']:.3f}  R {row['recall']:.3f}  F1 {row['f1']:.3f}")

out_path = os.environ.get("FACE_AB_OUT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "face_embedder_ab.json"))
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {out_path}")
