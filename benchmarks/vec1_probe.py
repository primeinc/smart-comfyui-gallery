"""The vec1-vs-FAISS evidence, repeatable: docs/SIMILARITY_ENGINE.md's table.

Usage:
    python benchmarks/vec1_probe.py <path-to-vec1-dll-without-extension>

The extension is first-party SQLite vec1 (sqlite.org/vec1), compiled per
its own docs from a trunk checkout:

    gcc -O3 -DNDEBUG -mavx2 -mfma -I<sqlite-headers> vec1.c -shared -o vec1.dll

Data is 512-d unit vectors in planted clusters -- the shape face
embeddings have, with real neighbourhood structure so radius queries
return real edges. The GPU manager is constructed before anything else
on purpose: the process's faiss build is decided by the first import
(vision/faiss_runtime.py), and a CPU-first ordering would silently
measure the wrong device.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from db import connect as connect_module
from vision.faiss_index import IndexManager, SpaceSpec

DIM = 512
RADIUS = 0.5  # cosine similarity floor; vec1 speaks distance = 1 - similarity

GPU = IndexManager(gpu=True)


def connect(dll: str, path=":memory:"):
    conn = connect_module.memory() if path == ":memory:" else connect_module.connect(path)
    conn.enable_load_extension(True)
    conn.load_extension(dll)
    conn.enable_load_extension(False)
    return conn


def planted(n: int, seed: int = 7):
    """Tight clusters of 8, so every vector has ~7 neighbours above the floor."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(max(n // 8, 1), DIM)).astype(np.float32)
    rows = centres[np.arange(n) % len(centres)] + 0.15 * rng.normal(size=(n, DIM)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return np.ascontiguousarray(rows.astype(np.float32))


def fill(conn, vectors, mode="flat"):
    conn.execute("CREATE VIRTUAL TABLE v1 USING vec1(v)")
    conn.executemany(
        "INSERT INTO v1(rowid, v) VALUES(?, ?)",
        ((i, memoryview(vectors[i].tobytes())) for i in range(len(vectors))),
    )
    conn.execute("INSERT INTO v1(cmd, arg) VALUES('rebuild', ?)", (f'{{"index":"{mode}","distance":"cos"}}',))
    conn.commit()


RANGED = "SELECT rowid, distance FROM v1(?, ?) WHERE distance <= ?"


def say(msg: str) -> None:
    print(msg, flush=True)


def main(dll: str) -> None:

    say("== correctness: 2000 planted vectors, floor 0.5, vs numpy oracle ==")
    V = planted(2000)
    sims = V @ V.T
    keep = sims >= RADIUS
    np.fill_diagonal(keep, False)
    a, b = np.nonzero(keep)
    want = set(zip(a.tolist(), b.tolist(), strict=True))

    conn = connect(dll)
    fill(conn, V)
    t0 = time.perf_counter()
    got = set()
    for i in range(len(V)):
        for rid, _ in conn.execute(RANGED, (memoryview(V[i].tobytes()), len(V), 1.0 - RADIUS)):
            if rid != i:
                got.add((i, rid))
    t_v = time.perf_counter() - t0

    spec = SpaceSpec("planted.2000", "float32", DIM, "cosine")
    GPU.load(spec, list(range(2000)), V)
    t0 = time.perf_counter()
    lims, labels, _ = GPU.range(spec.key, RADIUS)
    t_g = time.perf_counter() - t0
    rows = np.repeat(np.arange(2000), np.diff(np.asarray(lims)))
    mask = rows != labels
    ours = set(zip(rows[mask].tolist(), labels[mask].tolist(), strict=True))
    say(f"  oracle={len(want)} vec1={len(got)} equal={got == want}")
    say(f"  faiss[{GPU.served_by(spec.key)}]={len(ours)} equal={ours == want}")
    say(f"  vec1 exact K=n: {t_v:.2f}s | faiss: {t_g:.2f}s")

    say("== transactions: insert searchable in-txn, gone after rollback ==")
    conn.execute("BEGIN")
    conn.execute("INSERT INTO v1(rowid, v) VALUES(?, ?)", (999999, memoryview(V[0].tobytes())))
    seen = conn.execute("SELECT rowid FROM v1(?, 1)", (memoryview(V[0].tobytes()),)).fetchone()[0]
    conn.rollback()
    gone = conn.execute("SELECT rowid FROM v1(?, 1)", (memoryview(V[0].tobytes()),)).fetchone()[0]
    say(f"  in-txn nearest={seen}, post-rollback nearest={gone}, rides-transaction={gone != 999999}")
    conn.close()

    say("== mutation: UPDATE visibility per index mode (v0.7 flat corrupts) ==")
    for mode in ("none", "flat"):
        mconn = connect(dll)
        base = np.eye(3, DIM, dtype=np.float32)
        fill(mconn, base, mode)
        probe = memoryview(base[0].tobytes())
        mconn.execute("UPDATE v1 SET v = ? WHERE rowid = 2", (probe,))
        top = [r[0] for r in mconn.execute("SELECT rowid FROM v1(?, 3)", (probe,))]
        ok = mconn.execute("PRAGMA integrity_check").fetchone()[0]
        say(f"  mode={mode}: updated row found={2 in top}; integrity_check={ok!r}")
        mconn.close()

    say("== restart: cold reopen of a file database ==")
    dbfile = pathlib.Path(__file__).resolve().parent / "results" / "vec1_restart.db"
    dbfile.parent.mkdir(parents=True, exist_ok=True)
    dbfile.unlink(missing_ok=True)
    fconn = connect(dll, str(dbfile))
    fill(fconn, V)
    fconn.close()
    t0 = time.perf_counter()
    fconn = connect(dll, str(dbfile))
    hit = fconn.execute("SELECT rowid FROM v1(?, 1)", (memoryview(V[7].tobytes()),)).fetchone()[0]
    say(
        f"  reopen+query: {(time.perf_counter() - t0) * 1000:.1f}ms, self-hit={hit == 7}, "
        f"file x{dbfile.stat().st_size / (2000 * DIM * 4):.2f} of raw"
    )
    fconn.close()
    dbfile.unlink(missing_ok=True)

    say("== the graph workload at scale ==")
    for n in (8000, 20000):
        W = planted(n, seed=11)
        spec = SpaceSpec(f"planted.{n}", "float32", DIM, "cosine")
        GPU.load(spec, list(range(n)), W)
        GPU.range(spec.key, RADIUS)
        t0 = time.perf_counter()
        lims, labels, _ = GPU.range(spec.key, RADIUS)
        t_g = time.perf_counter() - t0
        served = GPU.served_by(spec.key)
        cpu = IndexManager(gpu=False)
        cpu.load(spec, list(range(n)), W)
        t0 = time.perf_counter()
        cpu.range(spec.key, RADIUS)
        t_c = time.perf_counter() - t0
        if n <= 8000:
            vconn = connect(dll)
            fill(vconn, W)
            t0 = time.perf_counter()
            for i in range(n):
                vconn.execute(RANGED, (memoryview(W[i].tobytes()), 1024, 1.0 - RADIUS)).fetchall()
            cell = f"vec1 {time.perf_counter() - t0:.2f}s"
            vconn.close()
        else:
            cell = "vec1 skipped (quadratic; ~4x its 8000 figure)"
        say(f"  n={n}: faiss {served} {t_g:.2f}s | faiss-cpu {t_c:.2f}s | {cell}")

    say("== binary representability ==")
    bconn = connect(dll)
    bconn.execute("CREATE VIRTUAL TABLE b USING vec1(v)")
    bconn.execute("INSERT INTO b(rowid, v) VALUES(1, ?)", (memoryview((0x0123456789ABCDEF).to_bytes(8, "big")),))
    seen = bconn.execute("SELECT vec1_to_json(v) FROM b").fetchone()[0]
    say(f"  phash64 blob parsed as {seen} -- two floats, not 64 hamming bits; no hamming metric exists")
    bconn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
