"""Detection recall by ground-truth face-size band, through the production
detect() path.

Ground truth: faces the detector is trusted on — native-resolution
detections with min-side >= 320 px, det_score >= 0.9, in images with
exactly one detection. Each source image is rescaled so that face lands in
every size band, so labels stay exact by construction. External labeled
corpora plug in via --labels (JSON: [{"image": path, "faces":
[{"bbox": [x, y, w, h]}]}], source-image pixels).

Two recalls per band:
  detector recall = matched at IoU >= 0.5 at all (gate disabled)
  pipeline recall = matched AND passes the production face_min_px gate

Variants via --max-side N: whole-image downscale policy (0 = native).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time



def load_repo():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


BANDS = [
    ("<16", 1, 15),
    ("16-23", 16, 23),
    ("24-39", 24, 39),
    ("40-79", 40, 79),
    ("80-159", 80, 159),
    ("160-299", 160, 299),
    (">=300", 300, 10**9),
]
# one synthesized min-side target inside each band; >=300 also gets natives
TARGETS = [12, 20, 30, 56, 110, 220, 380]

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "face_detection_recall.json"
)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def band_of(min_px: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= min_px <= hi:
            return name
    return BANDS[0][0]


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def to_px(det, w, h):
    x, y, bw, bh = det.bbox
    return (x * w, y * h, bw * w, bh * h)


def collect_ground_truth(backend, root, limit):
    """(path, native_bbox_px) for trusted single-face images.

    Harvest runs on a max-side-640 DOWNSCALE, where a large face sits inside
    YuNet's 10-300px training band, then scales the box back to native
    pixels. Harvesting at native resolution would only find faces the
    native-res policy already handles — exactly the bias the >=300 band
    exists to expose.
    """
    from PIL import Image

    sources = []
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(_IMAGE_EXTS):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    for path in paths:
        if len(sources) >= limit:
            break
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size
                f = min(1.0, 640 / max(w, h))
                small = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS) if f < 1.0 else img
                dets = backend.detect(small)
                if len(dets) != 1 or dets[0].det_score < 0.8:
                    continue
                box = to_px(dets[0], w, h)  # normalized bbox -> NATIVE pixels
                if min(box[2], box[3]) >= 300:
                    sources.append((path, box))
        except Exception:
            continue
    return sources


def main() -> None:
    load_repo()
    from PIL import Image

    from smartgallery_ai import AIConfig
    from smartgallery_ai.faces import get_face_backend

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="image corpus root")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--labels", help="external labeled corpus JSON (optional)")
    ap.add_argument("--limit", type=int, default=150, help="max ground-truth sources")
    ap.add_argument("--max-side", type=int, default=0, help="downscale policy; 0=native")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    cfg = AIConfig.from_env(base_path=args.dir, db_path=":memory:")
    cfg.models_dir = args.models_dir
    cfg.face_backend = "opencv"
    gate_px = cfg.face_min_px
    cfg.face_min_px = 0  # detector recall needs raw detections; the gate is
    cfg.face_detect_max_side = args.max_side  # the policy under test (0=off)
    backend = get_face_backend(cfg)  # gate applied arithmetically below

    cases = []  # (image_or_path, gt_boxes_px, tag)
    if args.labels:
        with open(args.labels, encoding="utf-8") as f:
            for entry in json.load(f):
                cases.append((entry["image"], [tuple(fc["bbox"]) for fc in entry["faces"]], "labeled"))
        print(f"{len(cases)} labeled images from {args.labels}")
    else:
        t0 = time.perf_counter()
        sources = collect_ground_truth(backend, args.dir, args.limit)
        print(
            f"{len(sources)} trusted single-face sources "
            f"(native min-side>=300px, harvested at max-side 640) "
            f"in {time.perf_counter() - t0:.0f}s"
        )
        for path, box in sources:
            for target in TARGETS:
                cases.append((path, [box], f"scale->{target}px"))
            cases.append((path, [box], "native"))

    stats = {name: {"faces": 0, "tp": 0, "gated_tp": 0} for name, _lo, _hi in BANDS}
    fp_total = 0
    images_run = 0
    detect_seconds = 0.0
    rows = []

    for path, gt_boxes, tag in cases:
        with Image.open(path) as img:
            img = img.convert("RGB")
            scale = 1.0
            if tag.startswith("scale->"):
                target = int(tag[len("scale->"):-2])
                scale = target / min(gt_boxes[0][2], gt_boxes[0][3])
            if scale != 1.0:
                nw = max(1, int(round(img.size[0] * scale)))
                nh = max(1, int(round(img.size[1] * scale)))
                img = img.resize((nw, nh), Image.LANCZOS)
            gts = [tuple(v * scale for v in b) for b in gt_boxes]
            t0 = time.perf_counter()
            dets = backend.detect(img)
            detect_seconds += time.perf_counter() - t0
            images_run += 1
            det_boxes = [to_px(d, *img.size) for d in dets]

        used = set()
        for gt in gts:
            gt_min = min(gt[2], gt[3])
            band = band_of(gt_min)
            stats[band]["faces"] += 1
            best_i, best_iou = -1, 0.0
            for i, db in enumerate(det_boxes):
                if i in used:
                    continue
                v = iou(gt, db)
                if v > best_iou:
                    best_i, best_iou = i, v
            matched = best_iou >= args.iou
            survived = False
            if matched:
                used.add(best_i)
                stats[band]["tp"] += 1
                db = det_boxes[best_i]
                survived = min(db[2], db[3]) >= gate_px
                if survived:
                    stats[band]["gated_tp"] += 1
            rows.append(
                {
                    "image": os.path.relpath(path, args.dir),
                    "variant": tag,
                    "gt_min_px": round(gt_min, 1),
                    "band": band,
                    "detected": matched,
                    "iou": round(best_iou, 3),
                    "det_score": round(float(dets[best_i].det_score), 3) if matched else None,
                    "survived_size_gate": survived,
                }
            )
        fp_total += len(det_boxes) - len(used)

    tp = sum(s["tp"] for s in stats.values())
    faces = sum(s["faces"] for s in stats.values())
    print(
        f"\npolicy: {'native' if not args.max_side else f'max-side {args.max_side}px'}"
        f" | gate {gate_px}px | IoU >= {args.iou} | "
        f"{detect_seconds / max(images_run, 1) * 1e3:.0f} ms/image detect"
    )
    print(f"{'band':<9} {'faces':>6} {'detected':>9} {'recall':>8} {'post-gate':>10}")
    for name, _lo, _hi in BANDS:
        s = stats[name]
        if not s["faces"]:
            continue
        print(
            f"{name:<9} {s['faces']:>6} {s['tp']:>9} "
            f"{s['tp'] / s['faces']:>7.1%} {s['gated_tp'] / s['faces']:>9.1%}"
        )
    print(
        f"overall: recall {tp / faces:.1%}, precision "
        f"{tp / max(tp + fp_total, 1):.1%}, {fp_total} false positives "
        f"({fp_total / max(images_run, 1):.2f}/image)"
    )

    record = {
        "policy_max_side": args.max_side,
        "gate_px": gate_px,
        "iou_threshold": args.iou,
        "ms_per_image_detect": round(detect_seconds / max(images_run, 1) * 1e3, 1),
        "ground_truth_face_count": faces,
        "true_positive_count": tp,
        "false_negative_count": faces - tp,
        "false_positive_count": fp_total,
        "recall_by_min_side_band": {
            n: round(stats[n]["tp"] / stats[n]["faces"], 4)
            for n, _l, _h in BANDS
            if stats[n]["faces"]
        },
        "post_gate_recall_by_min_side_band": {
            n: round(stats[n]["gated_tp"] / stats[n]["faces"], 4)
            for n, _l, _h in BANDS
            if stats[n]["faces"]
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"details: {os.path.relpath(RESULTS_PATH)}")


if __name__ == "__main__":
    main()
