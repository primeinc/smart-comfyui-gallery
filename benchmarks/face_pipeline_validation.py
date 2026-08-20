"""Both sample datasets through the production face pipeline, every method judged.

KYC: 7 known people, photos copied into ONE flat folder under anonymous
shuffled names -- truth lives only in this process's memory, never in a
path, so folder layout cannot leak the answer. i2i: 55 renders of the app's
own output, no truth, judged by shape and by `choose_primary`.

The pipeline is the shipped one end to end: `scan.scan` -> `ingest.one` ->
`detect.harvest_all` (which forces orientation) -> `derived.cluster` for
every method x threshold -> `derived.health` / `derived.agreement` ->
`derived.choose_primary`. Nothing here re-implements a production step.

A JSON record of every run lands in benchmarks/results/, like the other
benchmarks in this directory.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import tempfile
import time
from itertools import combinations

import numpy as np

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape faiss_graph_evidence.py uses (:176-181).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
METHODS = ("chinese-whispers", "connected-components", "spherical-kmeans", "consensus")
THRESHOLDS = (0.60, 0.55, 0.48, 0.40)
DATASETS = "C:/ComfyUI/output/sample-datasets"
MODELS = "C:/ComfyUI/output/.AImodels"


def build(paths, models_dir):
    """Scan + ingest + harvest a flat folder of images, the production way."""
    from db import detect, ingest, library, scan
    from vision.faces import OpenCVFaceBackend

    root = os.path.join(tempfile.mkdtemp(), "lib")
    os.makedirs(root)
    for src, name in paths:
        shutil.copy(src, os.path.join(root, name))
    conn = sqlite3.connect(":memory:")
    conn.executescript((REPO / "db" / "schema.sql").read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = library.add_root(conn, root, "library", 0.0)
    scan.scan(conn, root_id, root, 0.0)
    for file_id, name in conn.execute("SELECT id,name FROM file ORDER BY id"):
        ingest.one(conn, file_id, os.path.join(root, name), 0.0)
    backend = OpenCVFaceBackend(models_dir)
    started = time.perf_counter()
    counts = detect.harvest_all(conn, backend, 0.0)
    took = time.perf_counter() - started
    print(f"  harvest: {counts} in {took:.0f}s")
    return conn, backend


def matrix_stats(conn):
    """FAISS's own diagnostic over the embedding matrix (faiss.wiki/FAQ.md,
    'How can I get constructive criticism about my data?')."""
    from vision.faiss_runtime import import_faiss

    faiss = import_faiss()
    rows = conn.execute("SELECT embedding FROM derived_face_instance WHERE embedding IS NOT NULL").fetchall()
    matrix = np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    print("  MatrixStats (faiss's own diagnostic):")
    for line in faiss.MatrixStats(matrix).comments.splitlines():
        if line.strip():
            print(f"    {line}")


def pair_f1(conn, run_id, person_of):
    rows = conn.execute(
        "SELECT f.name, m.cluster_id FROM derived_face_membership m"
        " JOIN derived_face_instance fi ON fi.id = m.face_id"
        " JOIN file f ON f.id = fi.file_id"
        " JOIN derived_face_cluster c ON c.id = m.cluster_id AND c.run_id = ?",
        (run_id,),
    ).fetchall()
    tp = fp = fn = 0
    for (name_a, cluster_a), (name_b, cluster_b) in combinations(rows, 2):
        same_truth = person_of[name_a] == person_of[name_b]
        same_found = cluster_a == cluster_b
        tp += same_truth and same_found
        fp += same_found and not same_truth
        fn += same_truth and not same_found
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    balance = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, balance


def sweep(conn, backend, person_of=None):
    from db import derived

    header = (
        f"  {'method':22} {'thr':>5} {'people':>6} {'big':>4} {'med':>5} "
        f"{'alone':>6} {'silh':>6}" + (f" {'F1':>6}" if person_of else "") + f" {'agree':>6} {'via'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    records = []
    for method in METHODS:
        for threshold in THRESHOLDS:
            derived.cluster(
                conn,
                backend.model_id,
                backend.model_version,
                0.0,
                method=method,
                threshold=threshold,
            )
            run_id = derived.run_for(conn, backend.model_id, backend.model_version, method, threshold, 0.0)
            shape = derived.health(conn, run_id)
            said = derived.agreement(conn, run_id)
            agree = said["held_together"] - said["split_apart"] - said["clusters_mixing_people"]
            via = conn.execute("SELECT backend FROM derived_face_run WHERE id = ?", (run_id,)).fetchone()[0]
            record = {
                "method": method,
                "threshold": threshold,
                "backend": via,
                "clusters": shape["clusters"],
                "largest": shape["largest"],
                "median": shape["median"],
                "alone_share": shape["alone_share"],
                "silhouette": shape["silhouette"],
                "agreement": said,
            }
            line = (
                f"  {method:22} {threshold:>5.2f} {shape['clusters']:>6} "
                f"{shape['largest']:>4} {shape['median']:>5.1f} "
                f"{shape['alone_share']:>6.2f} {shape['silhouette']:>6.3f}"
            )
            if person_of:
                _, _, f1 = pair_f1(conn, run_id, person_of)
                record["pair_f1"] = f1
                line += f" {f1:>6.3f}"
            print(line + f" {agree:>6} {via}")
            records.append(record)
        print()
    return records


def assert_some(conn, person_of, per_person=2):
    """Simulate a user naming a couple of photos per person."""
    from db import scan

    by_person: dict[str, list[str]] = {}
    for name, person in person_of.items():
        by_person.setdefault(person, []).append(name)
    for person, names in sorted(by_person.items()):
        pid = scan.mint(conn, "person", f"kyc-{person}")
        conn.execute(
            "INSERT INTO person(id,name,created_at) VALUES(?,?,0)",
            (pid, f"KYC {person}"),
        )
        for name in sorted(names)[:per_person]:
            row = conn.execute("SELECT id FROM file WHERE name = ?", (name,)).fetchone()
            conn.execute(
                "INSERT INTO person_assertion(person_id,file_id,created_at) VALUES(?,?,0)",
                (pid, row[0]),
            )


def kyc(datasets: str, models_dir: str) -> dict:
    from db import derived

    print("=" * 72)
    print("KYC: 7 real people, flat shuffled pile, truth only in the manifest")
    print("=" * 72)
    base = os.path.join(datasets, "caucasian-people-kyc-photo-dataset")
    lookup = {}
    with open(os.path.join(base, "caucasian_kyc_dataset.csv"), newline="") as handle:
        for row in csv.DictReader(handle):
            for value in row.values():
                if value and value.startswith("/"):
                    src = os.path.join(base, "files" + value)
                    if os.path.exists(src):
                        lookup[src] = row["id"]
    sources = sorted(lookup)
    np.random.default_rng(11).shuffle(sources)
    person_of = {f"h{n:03}.jpg": lookup[src] for n, src in enumerate(sources)}
    flat = [(src, f"h{n:03}.jpg") for n, src in enumerate(sources)]

    conn, backend = build(flat, models_dir)
    matrix_stats(conn)
    print("\n  before anybody is named (agreement has nothing to read):")
    sweep(conn, backend, person_of)

    print("  a user names 2 photos of each of the 7 people, then choose_primary:")
    assert_some(conn, person_of)
    records = sweep(conn, backend, person_of)
    chosen = derived.choose_primary(conn)
    picked = (
        conn.execute("SELECT method, threshold FROM derived_face_run WHERE id = ?", (chosen,)).fetchone()
        if chosen
        else None
    )
    best = None
    for run in derived.runs(conn):
        _, _, f1 = pair_f1(conn, run["id"], person_of)
        if best is None or f1 > best[1]:
            best = ((run["method"], run["threshold"]), f1)
    print(f"  choose_primary  -> {picked}")
    print(f"  labels say best -> {best[0] if best else None} (F1 {best[1]:.3f})" if best else "")
    return {
        "runs": records,
        "chosen": picked,
        "labels_best": best[0] if best else None,
        "labels_best_f1": best[1] if best else None,
    }


def i2i(datasets: str, models_dir: str) -> dict:
    from db import derived

    print("=" * 72)
    print("i2i: 55 renders of the app's own output, no manifest")
    print("=" * 72)
    sources = sorted(glob.glob(os.path.join(datasets, "i2i-test-output", "*.png")))
    conn, backend = build([(s, os.path.basename(s)) for s in sources], models_dir)
    matrix_stats(conn)
    records = sweep(conn, backend)
    chosen = derived.choose_primary(conn)
    picked = (
        conn.execute(
            "SELECT method, threshold, clusters FROM derived_face_run WHERE id = ?",
            (chosen,),
        ).fetchone()
        if chosen
        else None
    )
    print(f"  choose_primary (no assertions) -> {picked}")
    return {"runs": records, "chosen": picked}


def main() -> None:
    parser = argparse.ArgumentParser(description="Both sample datasets through the production face pipeline.")
    parser.add_argument("--datasets", default=DATASETS, help="folder holding the sample datasets")
    parser.add_argument("--models-dir", default=MODELS, help="local ONNX face model weights")
    args = parser.parse_args()

    report = {
        "recorded_at": time.time(),
        "kyc": kyc(args.datasets, args.models_dir),
        "i2i": i2i(args.datasets, args.models_dir),
    }
    out = REPO / "benchmarks" / "results" / "face_pipeline_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nrecord: {out}")


if __name__ == "__main__":
    main()
