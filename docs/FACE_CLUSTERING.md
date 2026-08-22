# Face clustering

Faces that recur across a library are grouped so a person can be browsed
(`vision/faces.py`, `db/derived.py`). This is clustering for browsing: a
cluster is a bucket of similar-looking faces, a person's name is a
nickname a human attached, and nothing here is real-world identity
recognition or a claim about who a face resembles.

## Pipeline

```
picture -> detect + 5-point landmarks -> aligned per-face embedding -> L2 normalise
        -> cosine-threshold graph (resident FAISS, vision/faiss_index.py)
        -> Chinese Whispers -> derived_face_cluster (one run per embedder, method, threshold)
```

Two backends ship; the `face_backend` setting picks one and the faces
job provisions its weights from the registry that owns them
(`docs/AI_MODELS.md`, `vision/weights.py`):

| Backend | Detect | Embed | Device | License |
|---|---|---|---|---|
| `InsightFaceBackend` (`auto`) | SCRFD-10GF, multi-scale, CPU | glintr100 (ResNet100@Glint360K, 512-d) on `ort_providers` -- CUDA when the build offers it | per stage, measured | non-commercial research (insightface) |
| `OpenCVFaceBackend` (`opencv`, or `auto` without the insightface runtime) | YuNet 2023mar, detection input capped at 1600 px | glintr100 via cv2.dnn when the pack is present, else SFace (128-d) | CPU (OpenCV DNN default) | MIT / Apache with SFace |

A backend whose runtime or weights are missing reports
`BackendUnavailable`; a forced embedder whose weights are missing raises.
Each backend is its own embedding space: switching starts a fresh space
and never rewrites what the other found.

## Runs

A run is `(model_id, model_version, method, threshold)` and several are
live at once (`db/schema.sql` `derived_face_run`): trying a second
threshold never destroys the first. `primary_run` names the one the
People page shows; `POST /clusterings/choose` adopts the soundest.
Naming writes a `person_assertion` record, so a re-cluster re-applies
the name from the record instead of losing it with a dissolved cluster.

## Thresholds

The embedding spaces have different cosine scales; one number cannot
serve all. `db/derived.py threshold_for` holds the measured operating
point per embedder: insightface 0.40, opencv/arcface 0.48, opencv/sface
0.55; an unmeasured embedder gets a deliberately tight default.

Labelled A/B, 175-177 faces / 31 identities (`benchmarks/results/face_embedder_ab.json`):

| variant | verification best-F1 | cluster peak F1 | P/R at shipped threshold |
|---|---|---|---|
| yunet+sface (128-d) | 0.878 @ 0.38 | 0.897 @ 0.45 | 1.000/0.759 @ 0.55 |
| yunet+arcface w600k_r50 | 0.923 @ 0.27 | 0.932 @ 0.30 | — |
| yunet+arcface glintr100 | 0.933 @ 0.33 | 0.933 @ 0.30-0.35 | 0.968/0.896 @ 0.48 |
| **scrfd+glintr100 (insightface)** | **0.997 @ 0.35** | **0.999 @ 0.35** | **1.000/0.995 @ 0.40** |

Same recogniser in the last two rows; the gap is SCRFD's detection and
alignment. SFace's own sample decides same-identity at cosine 0.363
(`opencv/samples/dnn/face_detect.py:113`) -- a threshold for SFace, wrong
for every other space.

## Chaining

Chinese Whispers (dlib's update rule, fixed ascending sweeps, ties to the
lowest label) replaces connected components, whose transitive chaining
merged 97% of a real 22k-face library into one cluster. Operating points
on 12,713 SFace embeddings from 5,775 mixed real images
(`benchmarks/results/faiss_graph_evidence.json`):

| threshold | edges | clusters | top-cluster share | labelled-identity purity |
|---|---|---|---|---|
| 0.363 | 1.76M | 368 | 0.963 | 0.99 |
| 0.45 | 376k | 5,291 | 0.462 | 0.63 |
| 0.5 | 133k | 7,217 | 0.186 | 0.52 |
| 0.6 | 12.8k | 10,517 | 0.036 | 0.35 |

A run whose top cluster holds more than half the library has chained
(`db/derived.py CHAINED`); the run chooser refuses it.

## Detection policy

YuNet detects faces of ~10 to ~300 px. Detection input is capped at 1600
px on the longest side (`detect_max_side`), so large faces stay inside
that band (`benchmarks/results/face_detection_recall_{native,ms1600}.json`):

| ground-truth min-side | native recall | max-side-1600 recall |
|---|---|---|
| <16 px | 51.5% | 51.5% |
| 16-23 | 92.2% | 92.2% |
| 24-39 | 93.2% | 93.2% |
| 40-79 | 94.2% | 94.2% |
| 80-159 | 99.0% | 99.0% |
| 160-299 | 100% | 100% |
| >=300 | 55.3% | **97.1%** |

Native policy also costs precision (66.3% vs 93.8%) and speed (62 vs
33 ms/image). Detections whose box is under 24 px on its short side
(`min_face_px`) are dropped: they are YuNet's noise floor (median 15.5
px), and the gate keeps 96.6% of labelled faces while removing 74% of
the blobs that chain clusters.

## Reproduce

```
just bench faces-validate   # both sample datasets through db/, every clustering method judged
                            # -> benchmarks/results/face_pipeline_validation.json
```

The A/B, the operating points and the detection-policy tables above are
the records under `benchmarks/results/` named beside each.
