# MINIMUM-REQ — swappable components, face pipeline + AI stack

Every pipeline component, its current implementation, and at least two viable
swap options. **Seam** states whether the swap is a config flip today or needs
code.

Licenses: models marked `NC` are InsightFace pretrained releases —
non-commercial research only (code is MIT; commercial model licensing is
offered separately). Ship-safe options are MIT/Apache.

## Face pipeline

| # | Component | Current | Option 2 | Option 3 | Seam today |
|---|---|---|---|---|---|
| 1 | Face detector | YuNet 2023mar (MIT) — `faces.py:168` | SCRFD 500MF/2.5G/10G `NC` (5-pt KPS variants exist) | RetinaFace `NC`; MediaPipe BlazeFace (unverified here) | **code** — single `face_backend` knob covers detect+embed; detector-only swap needs the interface split |
| 2 | Detector runtime | OpenCV DNN, CPU | OpenCV DNN CUDA / CUDA-FP16 (`opencv_zoo` demo backend pairs) | int8 block-quantized YuNet; OpenCV 5 ORT engine + `2026may` dynamic-shape export (weight-identical) | **code** — no backend/target parameter passed at create |
| 3 | Detection policy | single pass, native resolution — `faces.py:187` | multi-scale joint pass + unified NMS (InsightFace 1.0 default: 128+640) | two-pass tiled re-detect on small/ambiguous candidates | **none** — hardcoded inside `detect()` |
| 4 | Landmarks | YuNet 5-pt (detector output rows) | SCRFD_KPS 5-pt | dense 2d106 `NC` | rides on #1 |
| 5 | Alignment | `FaceRecognizerSF.alignCrop` (5-pt similarity transform) — `faces.py:202` | InsightFace `norm_crop` (same 5-pt ArcFace template) | raw bbox crop (no alignment; baseline only) | **code** |
| 6 | Face embedder | SFace 2021dec, 128-d (Apache-2.0) — `faces.py:171` | ArcFace ResNet50@WebFace600K `NC` (buffalo_l, 512-d) | MBF@WebFace600K `NC` (buffalo_s); any ONNX ArcFace-template model | **code** — fused with detector in `OpenCVFaceBackend`; embedder swap = new `model_version`, embeddings never mix |
| 7 | Quality gate | `det_score>=0.5` + `face_min_px=24` (purity-calibrated) — `faces.py:196-199` | embedding-quality gate (norm/consistency of aligned crop) | landmark-stability gate across re-detection | **config** for thresholds (`AI_DAM_FACE_MIN_PX`); new gate types = code |
| 8 | Neighbor graph | `torch-cuda` \| `faiss-cpu` \| `numpy` — `faces.py:404-443` | — already 3 backends, equivalence-contracted | | **config** — `AI_DAM_FACE_GRAPH_BACKEND` |
| 9 | Clustering | deterministic Chinese Whispers — `faces.py:446` | DBSCAN (InsightFace GUI default, cosine thr 0.48) | connected components (rejected: transitive chaining, documented) | **code** — pure function of the graph, trivial to parameterize |
| 10 | Label persistence | centroid cosine match > 0.9, greedy 1:1 — `faces.py:477` | member-overlap voting (face-ID intersection) | manual pin table | **code** |

## Shared AI stack (existing config knobs)

| # | Component | Current | Option 2 | Option 3 | Seam today |
|---|---|---|---|---|---|
| 11 | Semantic embedder | open_clip ViT-B-32 laion2b — `embedders.py:195` | open_clip ViT-L or SigLIP | stub (test only) | **config** — `AI_DAM_SEMANTIC_BACKEND`; one real impl, second = new class |
| 12 | Visual embedder | dinov2-small — `embedders.py:263` | dinov2-base/large | open_clip image tower | **config** — `AI_DAM_VISUAL_BACKEND`; one real impl |
| 13 | Critic | qwen-vl — `review.py:587` | smolvlm — `review.py:606` | stub | **config** — two real backends today |
| 14 | Segmenter | MobileSAM — `review.py:623` | SAM / SAM2 | stub | **config** — one real impl |
| 15 | Retrieval (topk) | faiss `IndexFlatIP` — `vectors.py:192` | numpy exact fallback (automatic when faiss absent) — `vectors.py:186` | faiss IVF (`IVF{4*sqrt(N)},Flat`) at ~1M vectors; GPU faiss | **runtime** — two paths live now |

## Minimum bar

A component meets the bar when swapping it is a config value plus provisioned
weights, with `model_id`/`model_version` provenance keeping outputs from
different options unmixed. Meets it now: #7 (thresholds), #8, #13, #15.
Blocked on the detector/embedder interface split: #1, #2, #4, #5, #6.
Blocked on a policy seam: #3. Code-level pure-function swaps: #9, #10.
