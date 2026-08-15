#!/usr/bin/env python3
"""Calibrate the critic's grounding gate on adversarial description classes.

The v1 gate (absolute CLIP cosine >= 0.20) was shown by adversarial review
to have no separating power: vacuous descriptions ("an image with some
shapes and colors") score ABOVE genuinely grounded ones, and paraphrases
of the schema example pass on unrelated images. This probe measures the
v2 CONTRASTIVE gate:

    margin(desc, img) = cos(desc, img) - cos(GENERIC_BASELINE, img)

A vacuous description IS the baseline, so its margin ~ 0 regardless of the
image; a grounded description names image-specific content the baseline
does not, earning a positive margin. The probe sweeps margin thresholds
over grounded / vacuous / parroted / unrelated description classes across
a diverse image set and reports FAR/FRR per threshold, writing JSON to
benchmarks/results/grounding_calibration.json.

Usage: python3 probes/grounding_calibration.py [--models-dir DIR]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from smartgallery_ai import AIConfig  # noqa: E402
from smartgallery_ai.embedders import get_semantic_backend  # noqa: E402

GENERIC_BASELINE = "an image with some shapes and colors"

# Fixed portrait input, committed with the repo (public-domain NASA
# photograph of Eileen Collins, distributed as scikit-image's `astronaut`
# sample). CAL_PORTRAIT_IMAGE overrides it, and the override's file hash is
# recorded in the report manifest so a changed population is visible.
DEFAULT_PORTRAIT = os.path.join(REPO, "probes", "data", "calibration_portrait.png")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_pixels(img):
    return hashlib.sha256(
        img.size[0].to_bytes(4, "big") + img.size[1].to_bytes(4, "big")
        + img.convert("RGB").tobytes()).hexdigest()

VACUOUS = [
    "This is an image. It contains some shapes and colors.",
    "A picture showing various elements and details.",
    "An image with a subject in it.",
]
PARROTED = [
    "Good portrait with one artifact. The image shows a red square artifact "
    "in the lower right and slightly flat lighting.",
    "A portrait with a red square artifact and slightly flat lighting.",
]
UNRELATED = [
    "A golden retriever puppy sits on a wooden porch in afternoon sunlight.",
    "A bowl of ramen with chopsticks on a restaurant table.",
    "A snow-covered mountain range under a clear blue sky.",
]


def build_images():
    """Returns (imgs, manifest): the calibration population plus one
    manifest row per image pinning its exact pixels (and, for file-backed
    inputs, the source file's hash)."""
    rng = np.random.default_rng(23)
    imgs = {}
    manifest = []

    def _add(name, img, desc, source, file_path=None):
        imgs[name] = (img, desc)
        row = {"image": name, "source": source,
               "pixels_sha256": _sha256_pixels(img)}
        if file_path is not None:
            row["file"] = os.path.relpath(file_path, REPO)
            row["file_sha256"] = _sha256_file(file_path)
        manifest.append(row)

    astro_path = os.environ.get("CAL_PORTRAIT_IMAGE", DEFAULT_PORTRAIT)
    if astro_path and os.path.isfile(astro_path):
        astro = Image.open(astro_path).convert("RGB").resize((512, 512))
        _add("portrait", astro,
             "a person wearing an orange astronaut space suit with patches, "
             "in front of a flag",
             "portrait file, resized to 512x512", file_path=astro_path)
        flawed = astro.copy()
        ImageDraw.Draw(flawed).rectangle([300, 300, 419, 419], fill=(255, 20, 20))
        _add("portrait-defect", flawed,
             "a person in an orange space suit, with a solid red square "
             "artifact in the lower right",
             "portrait + planted 120px red square at (300,300)",
             file_path=astro_path)
        _add("portrait-dark", ImageEnhance.Brightness(astro).enhance(0.12),
             "a very dark, underexposed photo of a person in a space suit",
             "portrait at brightness 0.12", file_path=astro_path)
    noise = Image.fromarray((rng.random((512, 512, 3)) * 255).astype("uint8"))
    _add("noise", noise, "dense multicolored static noise with no subject",
         "np.random.default_rng(23) uniform noise 512x512")
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32) / 512.0
    _add("gradient",
         Image.fromarray(np.stack([xx * 255, yy * 255, (1 - xx) * 255],
                                  axis=-1).astype("uint8")),
         "a smooth colorful gradient from orange to blue with no objects",
         "deterministic RGB gradient 512x512")
    _add("red", Image.new("RGB", (512, 512), (220, 20, 20)),
         "a plain solid red image", "solid RGB(220,20,20) 512x512")
    for name in ("filter_panel.png", "compare.png"):
        p = os.path.join(REPO, "assets", name)
        if os.path.isfile(p):
            _add(f"screenshot:{name}", Image.open(p).convert("RGB"),
                 "a screenshot of a software user interface with panels and text",
                 "repo asset", file_path=p)
    return imgs, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir",
                    default=os.path.join(REPO, ".AImodels"))
    args = ap.parse_args()

    sem = get_semantic_backend(AIConfig(enabled=True, models_dir=args.models_dir,
                                        semantic_backend="open_clip"))
    if sem is None:
        print("FAIL: OpenCLIP weights not provisioned")
        return 2

    imgs, manifest = build_images()

    def cos(a, b):
        return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))

    base_t = sem.embed_text(GENERIC_BASELINE)
    rows = []
    for img_name, (img, grounded_desc) in imgs.items():
        iv = sem.embed_image(img)
        base_cos = cos(base_t, iv)
        cases = [("grounded", grounded_desc)]
        cases += [("vacuous", d) for d in VACUOUS]
        cases += [("parroted", d) for d in PARROTED]
        cases += [("unrelated", d) for d in UNRELATED]
        for cls, desc in cases:
            c = cos(sem.embed_text(desc), iv)
            rows.append({"image": img_name, "class": cls, "cos": round(c, 4),
                         "margin": round(c - base_cos, 4)})

    sweep = []
    for thr in [round(0.01 * t, 2) for t in range(0, 16)]:
        far = [r for r in rows if r["class"] != "grounded" and r["margin"] >= thr]
        frr = [r for r in rows if r["class"] == "grounded" and r["margin"] < thr]
        n_bad = sum(1 for r in rows if r["class"] != "grounded")
        n_good = sum(1 for r in rows if r["class"] == "grounded")
        sweep.append({"margin_threshold": thr,
                      "false_accept_rate": round(len(far) / n_bad, 3),
                      "false_reject_rate": round(len(frr) / n_good, 3)})

    report = {
        "baseline_text": GENERIC_BASELINE,
        "backend": {"model_id": sem.model_id, "model_version": sem.model_version},
        "inputs": manifest,
        "description_classes": {
            "vacuous": VACUOUS, "parroted": PARROTED, "unrelated": UNRELATED,
        },
        "pairs": rows,
        "sweep": sweep,
    }
    out = os.path.join(REPO, "benchmarks", "results",
                       "grounding_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"{'thr':>5} {'FAR':>6} {'FRR':>6}")
    for s in sweep:
        print(f"{s['margin_threshold']:>5} {s['false_accept_rate']:>6} "
              f"{s['false_reject_rate']:>6}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
