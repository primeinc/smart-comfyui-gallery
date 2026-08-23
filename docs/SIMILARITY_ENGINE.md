# Why similarity runs on FAISS over SQLite blobs, measured against vec1

SQLite's first-party vector extension (`vec1`, sqlite.org/vec1, first
checkin 2026-02-04) was evaluated as a candidate to own durable float
vector state, with FAISS demoted to a hot projection. The decision was
made on measurements against the real runtimes, not on roadmap
sympathy. Verdict: **SQLite BLOB columns stay the durable truth and the
resident FAISS layer (vision/faiss_index.py) serves every similarity
workload.** The numbers and the reproduction path are below so the
decision can be re-taken when the inputs change.

Probe environment: this application's venv (Python 3.13, sqlite 3.47.1),
vec1 trunk 2026-08-20 compiled `gcc -O3 -DNDEBUG -mavx2 -mfma` per its
own docs, the vendored CUDA FAISS (built from faiss v1.15.0,
docs/FAISS_GPU_WINDOWS.md; `just faiss-verify` prints the build the
process loads), 512-d unit vectors in planted
clusters (the shape face embeddings have). Reproduce with
`benchmarks/vec1_probe.py` (it prints the build command).

## What was measured

**Correctness** (n=2000, cosine floor 0.5, 14,000 true edges): vec1
flat-exact with K=n and FAISS (GPU path) both matched the numpy oracle
edge-for-edge. Nobody is wrong; the question is cost and coverage.

**The application's dominant workload** -- the all-pairs radius graph
that dupe grouping and face clustering consume:

| n      | FAISS gpu | FAISS cpu | vec1 flat exact (per-row KNN loop) |
|--------|-----------|-----------|-------------------------------------|
| 2,000  | 0.22s     | 0.03s     | 2.05s                               |
| 8,000  | 0.07s     | 0.15s     | 34.95s                              |
| 20,000 | 0.21s     | 0.93s     | ~140s (quadratic, extrapolated)     |

vec1 has no range query and no self-join form: the graph must be
emulated as n separate K-overfetch KNN calls through the table-valued
function, single-threaded (multi-threaded queries are on vec1's post-1.0
roadmap). At the library's scale that is minutes against FAISS's
sub-second -- two to three orders of magnitude, growing quadratically.

**Single-query KNN** (n=8000, k=10): vec1 4.09ms, FAISS cpu 1.37ms, and
FAISS batched 200 queries in 16ms. Both are fine for an interactive
find-similar page; neither side wins anything decisive here.

**Transactions**: vec1's genuine advantage. An insert inside an open
transaction is searchable immediately and a rollback removes it from
both table and index (probed; also upstream's vec1rollback.test). But
this advantage does not transfer: our durable truth (BLOB columns) is
ALREADY transactional, and the tier that is not -- the resident RAM/VRAM
index -- would remain outside SQLite transactions under vec1 too,
because vec1 has no GPU and no graph workload. The reconciliation layer
(db/similarity.align) is compensation for GPU residency, not for the
storage format, and would survive unchanged in a vec1 world.

**Mutation** -- and a disqualifying defect in vec1 0.7: with the "flat"
(exact) index, `UPDATE` of a vector makes the row unfindable by search
while the base table still holds it, and vec1's own
`PRAGMA integrity_check` then reports `wrong number of entries in
%_base - have 3, expect 2`. The "none" mode handles the same update
correctly. Upstream's vec1update.test never checks post-update search
visibility on a flat index, consistent with the author's own roadmap
note "Testing is insufficient". An exact-mode index that corrupts on
UPDATE cannot own durable state for mutable vectors.

**Representation coverage**: vec1's native format is float32-only; an
8-byte phash64 inserted as a blob is silently parsed as two garbage
floats. The perceptual space (64-bit hamming) is unrepresentable --
hamming distance and non-float elements are both post-1.0 roadmap
items. Adopting vec1 for floats would split similarity across two
engines, which is the bespoke-consumer disease this architecture was
built to end.

**Restart**: vec1 reopens cold in 4.2ms with no rebuild (its index IS
the database file) at 2.01x raw vector size on disk. Our snapshot
restore serves the same purpose (proven cross-process in
tests/test_faiss_index.py) at 1x plus a sidecar.

## The decision, stated as invariants

- Durable truth: SQLite rows (BLOB representations + the
  `similarity_space` identity row). Already transactional;
  that was never the gap.
- Serving: the one resident FAISS layer, CPU canonical + optional GPU
  clone, for every space -- float and binary alike.
- `align` stays: it reconciles the non-transactional resident tier with
  committed truth, a need that exists in every topology that keeps a
  RAM/VRAM index.

## What would reopen this decision

Recorded as evidence conditions, not a schedule: vec1 gaining (a) a
radius/self-join query path or multi-threaded queries fast enough for
the graph workload at 100k vectors, (b) non-float elements with hamming
distance, and (c) an exact mode whose UPDATE survives its own
integrity_check. Re-run `benchmarks/vec1_probe.py` against the newer
vec1 and let the same table decide. Note vec1's own repo treats FAISS
exactly as this one does -- vec1_faiss.cpp trains models WITH FAISS and
serializes into vec1's own format, engine and durable format kept
separate.
