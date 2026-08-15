# Face Clustering

Face identity grouping for browsing (`smartgallery_ai/faces.py`). Not
real-world identity recognition.

## Pipeline

```
image -> YuNet detect -> SFace embed (128-d) -> L2 normalize
      -> cosine-threshold graph (CSR) -> Chinese Whispers -> ai_face_clusters
```

## Models

| Stage | Model | File | License |
|---|---|---|---|
| Detect | YuNet 2023mar | `face_detection_yunet_2023mar.onnx` | MIT |
| Embed | SFace 2021dec | `face_recognition_sface_2021dec.onnx` | Apache-2.0 |

Constraints (opencv_zoo model cards):

- YuNet detects faces of **~10x10 to ~300x300 pixels**. Faces outside that
  band produce degenerate boxes (partial-face crops on close-ups, blur/noise
  detections at the small end).
- OpenCV's SFace samples decide same-identity at cosine >= **0.363**
  (`opencv/samples/dnn/face_detect.py:113`).

## Neighbor graph backends

`_neighbor_graph(normed, threshold)` returns a CSR triple
`(indptr, cols, weights)` plus the backend name, recorded in every cluster's
`params.graph_backend`.

| Backend | Implementation | Requires |
|---|---|---|
| `torch-cuda` | blocked CUDA matmul, TF32 off | torch + CUDA device |
| `faiss-cpu` | `IndexFlatIP.range_search` | faiss |
| `numpy` | chunked matmul | always available |

Selection: `AI_DAM_FACE_GRAPH_BACKEND` = `auto` (default) \| `torch-cuda` \|
`faiss` \| `numpy`. A named backend that is unavailable raises; there is no
silent fallback. `auto` tries torch-cuda, faiss, numpy in that order.

faiss GPU indexes implement only k-NN `search`; `range_search` is CPU-only
on every platform (faiss wiki Special-operations; `GpuIndexIVF.cu` throws).
The GPU path therefore uses the tensor runtime directly.

### Equivalence contract

Backends compute the same exhaustive edge set. Floating-point reduction
order differs between BLAS/cuBLAS kernels (faiss wiki Comparing-GPU-vs-CPU),
so:

- edges present in both backends: weights agree within 1e-5
- edges present in one backend only: legal only when the similarity lies
  within 1e-5 of the threshold
- Chinese Whispers clustering: identical

Enforced by `benchmarks/faiss_graph_evidence.py` (exits nonzero on
divergence) and `tests/test_faces.py` backend tests.

## Clustering

Deterministic Chinese Whispers (dlib's update rule: each node adopts the
label with the highest summed edge weight among its neighbors), fixed
ascending sweeps, ties to the lowest label. Output is a pure function of
the graph. Single-linkage connected components are unsuitable for this
data: transitive chaining merges dense look-alike sets into one cluster.

## Measured operating points

12,713 SFace embeddings from 5,775 mixed real images (six HF datasets:
labeled identities, unlabeled people, scenes). AMD 5800X (8C/16T),
RTX 3070 Ti. `benchmarks/results/faiss_graph_evidence.json` holds the
machine-readable record.

| threshold | edges | clusters | top-cluster share | labeled-identity purity |
|---|---|---|---|---|
| 0.363 | 1.76M | 368 | 0.963 | 0.99 |
| 0.45 | 376k | 5,291 | 0.462 | 0.63 |
| 0.5 | 133k | 7,217 | 0.186 | 0.52 |
| 0.6 | 12.8k | 10,517 | 0.036 | 0.35 |

Visual sampling of cluster contact sheets (crops of members) shows:

- Large clusters at loose thresholds are dominated by low-quality
  detections (blur, hands, textures) whose embeddings occupy one region;
  they are a detection-quality artifact, not identity confusion.
- The purity metric undercounts: identity-labeled datasets contain group
  photos, and secondary faces inherit the folder label, so splitting them
  is scored as error while being correct.
- Coherent single-person clusters form at every threshold tested.

Graph-build timings at that shape, threshold 0.5, idle machine,
`OMP_WAIT_POLICY=PASSIVE` (faiss wiki Troubleshooting), 8 OMP threads
(physical cores, faiss wiki How-to-make-Faiss-run-faster):

| backend | graph build |
|---|---|
| torch-cuda | 24 ms |
| faiss-cpu | 270 ms |
| numpy | 1.27 s |

Chinese Whispers on the same graph: 0.9-1.5 s (all backends; CPU-bound).

## Reproduce

```
just bench load             # current CPU/GPU load snapshot
just bench faiss [thr]      # image corpus through the production pipeline
just bench faiss-db <db>    # real embeddings from a gallery DB, read-only
```

`bench faiss` preconditions exactly as production does: backend from
`get_face_backend(AIConfig.from_env(...))` (gates + env knobs identical to
the worker), storage via `replace_faces_for_file`, and `cluster_faces()`
itself timed. Corpus/model/cache paths are module variables
(`just bench::corpus=... faiss`). The embedding cache stores its gate
config and re-embeds automatically when the config changes.

Benchmarks run under real load. The harness measures external CPU load
(system busy minus its own CPU time) live during timing, warns past 10%,
and writes the summary into the results record — every number carries its
measured load context (`load.contaminated` = external load exceeded 10%
at some point during timing). Best-of-N timing takes the least-loaded
window.

## Detection policy

Detection input is capped at `face_detect_max_side` = 1600 px (env
`AI_DAM_FACE_DETECT_MAX_SIDE`, 0 disables): larger images are downscaled
before detection so large faces stay inside YuNet's ~10-300px band.
Measured by `just bench face-recall` (103 trusted single-face sources,
per-band synthesized variants, IoU >= 0.5, production detect path):

| ground-truth min-side | native recall | max-side-1600 recall |
|---|---|---|
| <16 px | 51.5% | 51.5% |
| 16-23 | 92.2% | 92.2% |
| 24-39 | 93.2% | 93.2% |
| 40-79 | 94.2% | 94.2% |
| 80-159 | 99.0% | 99.0% |
| 160-299 | 100% | 100% |
| >=300 | 55.3% | **97.1%** |

Native policy also costs precision (66.3% vs 93.8%; 0.41 vs 0.06 false
positives/image) and speed (62 vs 33 ms/image). The policy change bumps
`model_version` to `yunet-2023mar+sface-2021dec-v2-ms1600`; instances from
the two policies never mix. Records:
`benchmarks/results/face_detection_recall_{native,ms1600}.json`.

Detector replacement (SCRFD etc.) is not justified by this data: every
band from 24-299px sits at 93-100% under the deployed detector, and the
only weak bands are sub-gate junk (<24px) and the large-face hole the
policy closes.

## Detection quality gate

`face_min_px` (default 24, env `AI_DAM_FACE_MIN_PX`) drops detections whose
box min-side is under N source pixels. Measured on the mixed corpus:
junk-blob members have median min-side 15.5 px (YuNet's noise floor),
labeled identities 525 px. At 24 px the gate keeps 96.6% of labeled-identity
faces, removes 74% of blob members, and improves both top-cluster share
(0.46 -> 0.20 at threshold 0.45) and identity purity simultaneously.

Measured at detect time through the production pipeline: 8,210 of 12,713
detections pass, corpus embedding drops 483s -> 161s (SFace never runs on
gated detections), top-cluster share 18.6% -> 8.3% at threshold 0.5, and
`cluster_faces()` end to end takes 0.32s on 8,210 faces (torch-cuda
backend auto-selected).

## Scale limits

Exact flat search is the correct faiss index while corpora stay small
(faiss wiki Guidelines-to-choose-an-index: exact results -> `Flat`). At
~1M vectors, `IVF{4*sqrt(N)},Flat` with `nprobe` tuning is the documented
escape hatch; GPU faiss becomes material at hundreds of thousands to
millions of vectors (faiss wiki Comparing-GPU-vs-CPU).
