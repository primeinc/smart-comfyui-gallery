# MINIMUM-REQ — swappable components, face pipeline + AI stack

Every pipeline component, its current implementation, and at least two viable
swap options. **Seam** states whether the swap is a config flip today or needs
code.

Licenses: models marked `NC` are InsightFace pretrained releases —
non-commercial research only (code is MIT; commercial model licensing is
offered separately). Ship-safe options are MIT/Apache.

## Face pipeline

Two full pipelines are deployed and config-swappable
(`AI_DAM_FACE_BACKEND=insightface|opencv`, `auto` prefers insightface —
best measured, `benchmarks/results/face_embedder_ab.json`; opencv+sface
is the MIT/Apache ship-safe lane). `/faces/compare/<file_id>` runs every
installed lane live, side by side, with the inventory + swap selectors.

| # | Component | Deployed option A | Deployed option B | Further options | Seam today |
|---|---|---|---|---|---|
| 1 | Face detector | SCRFD-10GF `NC` (insightface lane) | YuNet 2023mar (MIT, opencv lane) | RetinaFace `NC`; MediaPipe BlazeFace (unverified here) | **config** — `AI_DAM_FACE_BACKEND` |
| 2 | Detector runtime | onnxruntime CPU (insightface lane) | OpenCV DNN CPU (opencv lane) | ORT CUDA EP; OpenCV DNN CUDA | **config** picks the runtime with the lane; GPU EPs = code |
| 3 | Detection policy | multi-scale joint 128+640 + unified NMS (upstream default) | single pass, ms1600 cap (`AI_DAM_FACE_DETECT_MAX_SIDE`) | two-pass tiled re-detect | **config** — rides on #1 |
| 4 | Landmarks | SCRFD_KPS 5-pt | YuNet 5-pt | dense 2d106 `NC` (in the provisioned pack, unused) | rides on #1 |
| 5 | Alignment | InsightFace `norm_crop` (upstream) | same contract in numpy (Umeyama onto `arcface_dst`, `faces.py:_arcface_norm_crop`); SFace `alignCrop` for sface | raw bbox crop (baseline only) | rides on #1/#6 |
| 6 | Face embedder | glintr100 ResNet100@Glint360K `NC`, 512-d (both lanes read one provisioned copy) | SFace 2021dec, 128-d (Apache-2.0) — `AI_DAM_FACE_EMBEDDER` | MBF `NC`; any ONNX ArcFace-template model | **config** — per-pipeline `model_version`, embeddings never mix |
| 7 | Quality gate | `det_score>=0.5` + `face_min_px=24` (purity-calibrated) — `faces.py:196-199` | embedding-quality gate (norm/consistency of aligned crop) | landmark-stability gate across re-detection | **config** for thresholds (`AI_DAM_FACE_MIN_PX`); new gate types = code |
| 8 | Neighbor graph | `torch-cuda` \| `faiss-cpu` \| `numpy` — `faces.py:404-443` | — already 3 backends, equivalence-contracted | | **config** — `AI_DAM_FACE_GRAPH_BACKEND` |
| 9 | Clustering | deterministic Chinese Whispers — `faces.py:446` | DBSCAN (InsightFace GUI default, cosine thr 0.48) | connected components (rejected: transitive chaining, documented) | **code** — pure function of the graph, trivial to parameterize |
| 10 | Label persistence | centroid cosine match > 0.9, greedy 1:1 — `faces.py:477` | member-overlap voting (face-ID intersection) | manual pin table | **code** |

## Shared AI stack (existing config knobs)

| # | Component | Current | Option 2 | Option 3 | Seam today |
|---|---|---|---|---|---|
| 11 | Semantic embedder | open_clip ViT-B-32 laion2b — `embedders.py:195` | open_clip ViT-L or SigLIP | stub (test only) | **config** — `AI_DAM_SEMANTIC_BACKEND`; one real impl, second = new class |
| 12 | Visual embedder | dinov2-small — `embedders.py:263` | dinov2-base/large | open_clip image tower | **config** — `AI_DAM_VISUAL_BACKEND`; one real impl |
| 13 | Reviewer | vlm — `review.py` `get_reviewer` | any checkpoint via `AI_DAM_CRITIC_MODEL` | stub | **config** — one code path, the model is a string |
| 14 | Segmenter | MobileSAM — `review.py:623` | SAM / SAM2 | stub | **config** — one real impl |
| 15 | Retrieval (topk) | faiss `IndexFlatIP` — `vectors.py:192` | numpy exact fallback (automatic when faiss absent) — `vectors.py:186` | faiss IVF (`IVF{4*sqrt(N)},Flat`) at ~1M vectors; GPU faiss | **runtime** — two paths live now |

## Minimum bar

A component meets the bar when swapping it is a config value plus provisioned
weights, with `model_id`/`model_version` provenance keeping outputs from
different options unmixed. Meets it now: #1-#6 (two deployed pipelines,
`AI_DAM_FACE_BACKEND` + `AI_DAM_FACE_EMBEDDER`), #7 (thresholds), #8,
#13, #15. Code-level pure-function swaps remain: #9, #10; GPU execution
providers for #2 remain code.
