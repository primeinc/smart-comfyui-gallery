"""Hardware matrix probe: every AI/ML runtime exercised on CPU and on EACH
NVIDIA GPU, with real computation and output verification -- because "it
loaded" proves nothing (observed live: a wedged CUDA context decoded
'!!!!' garbage while loading fine, then crashed the grammar sampler with a
C++ exception).

Components x devices:

  generate  transformers decode canary (tiny generation, alnum check)
  torch     matmul verified against the CPU result (garbage detection)
  ort       onnxruntime inference on a provisioned ONNX model, finite check
  faiss     top-k neighbors verified against brute-force numpy

Every (component, device) cell runs in a SUBPROCESS with
CUDA_VISIBLE_DEVICES pinned, so a hard crash (access violation, sm
mismatch abort) is contained and reported as that cell's FAIL instead of
killing the probe. Missing runtimes/models are honest SKIPs, never
passes.

Usage:
    python probes/hardware_matrix_probe.py [--model REF] [--out report.json]

Exit code 0 when no cell FAILed (SKIPs allowed), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each snippet prints one JSON line {"status": "PASS"|"FAIL"|"SKIP",
# "detail": ...} and exits 0; any other termination is a FAIL.

_GENERATE_SNIPPET = r"""
import json, os, sys
sys.path.insert(0, os.environ["PROBE_REPO"])
try:
    from smartgallery_ai import models as ai_models
except Exception as exc:
    print(json.dumps({"status": "SKIP", "detail": f"runtime unavailable: {exc}"})); sys.exit(0)
ref, models_dir = os.environ["PROBE_MODEL"], os.environ["PROBE_MODELS_DIR"]
if not ai_models.is_provisioned(ref, models_dir):
    print(json.dumps({"status": "SKIP",
                      "detail": f"{ref} not provisioned under {models_dir}"})); sys.exit(0)
try:
    chat = ai_models.Chat(ref, models_dir=models_dir,
                          device=os.environ["PROBE_DEVICE"])
    text = chat.ask("Say OK.", max_new_tokens=8)
except Exception as exc:
    print(json.dumps({"status": "FAIL", "detail": f"generate raised: {exc}"})); sys.exit(0)
ok = any(c.isalnum() for c in text)
print(json.dumps({"status": "PASS" if ok else "FAIL",
                  "detail": f"decoded {text!r}" + ("" if ok else " (garbage logits)")}))
"""

_TORCH_SNIPPET = r"""
import json, os, sys
try:
    import torch
except Exception as exc:
    print(json.dumps({"status": "SKIP", "detail": f"torch unavailable: {exc}"})); sys.exit(0)
device = os.environ["PROBE_DEVICE"]
if device != "cpu" and not torch.cuda.is_available():
    print(json.dumps({"status": "FAIL", "detail": "cuda not available to torch"})); sys.exit(0)
torch.manual_seed(0)
a = torch.randn(256, 256); b = torch.randn(256, 256)
want = (a @ b).sum().item()
got = (a.to(device) @ b.to(device)).sum().item()
ok = abs(want - got) < 1.0 and got == got  # NaN check via self-equality
print(json.dumps({"status": "PASS" if ok else "FAIL",
                  "detail": f"matmul sum cpu={want:.2f} dev={got:.2f}"}))
"""

_ORT_SNIPPET = r"""
import json, os, sys
try:
    import numpy as np
    import onnxruntime as ort
except Exception as exc:
    print(json.dumps({"status": "SKIP", "detail": f"onnxruntime unavailable: {exc}"})); sys.exit(0)
model = os.environ["PROBE_ONNX"]
if not os.path.isfile(model):
    print(json.dumps({"status": "SKIP", "detail": f"no ONNX model at {model}"})); sys.exit(0)
provider = os.environ["PROBE_ORT_PROVIDER"]
if provider != "CPUExecutionProvider" and provider not in ort.get_available_providers():
    print(json.dumps({"status": "SKIP", "detail": f"{provider} not in this ort build"})); sys.exit(0)
sess = ort.InferenceSession(model, providers=[provider])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
out = sess.run(None, {inp.name: x})[0]
ok = bool(np.isfinite(out).all())
print(json.dumps({"status": "PASS" if ok else "FAIL",
                  "detail": f"{provider} output shape {list(out.shape)}, finite={ok}"}))
"""

_FAISS_SNIPPET = r"""
import json, os, sys
sys.path.insert(0, os.environ["PROBE_REPO"])
try:
    import numpy as np
    from smartgallery_ai import faiss_runtime
    faiss = faiss_runtime.import_faiss()
except Exception as exc:
    print(json.dumps({"status": "SKIP", "detail": f"faiss unavailable: {exc}"})); sys.exit(0)
rng = np.random.default_rng(0)
xb = rng.standard_normal((500, 32)).astype(np.float32)
xq = xb[:5]
index = faiss.IndexFlatIP(32)
index.add(xb)
use_gpu = os.environ["PROBE_FAISS_GPU"] == "1"
if use_gpu:
    if not hasattr(faiss, "StandardGpuResources"):
        print(json.dumps({"status": "SKIP", "detail": "faiss build has no GPU support"})); sys.exit(0)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
_, got = index.search(xq, 3)
want = np.argsort(-(xq @ xb.T), axis=1)[:, :3]
ok = bool((got[:, 0] == want[:, 0]).all())
print(json.dumps({"status": "PASS" if ok else "FAIL",
                  "detail": f"top1 agreement with numpy: {ok}"}))
"""


def _detect_gpus():
    """[(index, name)] from nvidia-smi; empty when no NVIDIA driver."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    gpus = []
    for line in proc.stdout.strip().splitlines():
        idx, _, name = line.partition(",")
        gpus.append((idx.strip(), name.strip()))
    return gpus


def _run_cell(snippet: str, env_extra: dict, timeout: int = 300) -> dict:
    env = {**os.environ, "PROBE_REPO": REPO, **env_extra}
    try:
        proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "detail": f"timeout after {timeout}s"}
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout or "").strip()[-200:]
        return {"status": "FAIL", "detail": f"process died rc={proc.returncode}: {tail}"}
    return json.loads(lines[-1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI/ML hardware matrix probe")
    parser.add_argument(
        "--model", default=os.environ.get("OMNIQUERY_NL2SQL_MODEL", "distil-labs/distil-qwen3-4b-text2sql")
    )
    parser.add_argument("--models-dir", default=os.environ.get("AI_DAM_MODELS_DIR", os.path.join(REPO, ".AImodels")))
    parser.add_argument(
        "--onnx", default=os.path.join(REPO, ".AImodels", "insightface", "models", "antelopev2", "glintr100.onnx")
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    gpus = _detect_gpus()
    devices = [("cpu", "CPU")] + [(idx, name) for idx, name in gpus]
    results = {}

    for dev, name in devices:
        is_cpu = dev == "cpu"
        pin = {"CUDA_VISIBLE_DEVICES": ""} if is_cpu else {"CUDA_VISIBLE_DEVICES": dev}

        results[f"generate/{name}"] = _run_cell(
            _GENERATE_SNIPPET,
            {
                **pin,
                "PROBE_MODEL": args.model,
                "PROBE_MODELS_DIR": args.models_dir,
                # CUDA_VISIBLE_DEVICES is already pinned to this card, so the
                # child process sees exactly one GPU and it is index 0.
                "PROBE_DEVICE": "cpu" if is_cpu else "cuda:0",
            },
            timeout=600,
        )
        results[f"torch/{name}"] = _run_cell(_TORCH_SNIPPET, {**pin, "PROBE_DEVICE": "cpu" if is_cpu else "cuda:0"})
        results[f"ort/{name}"] = _run_cell(
            _ORT_SNIPPET,
            {
                **pin,
                "PROBE_ONNX": args.onnx,
                "PROBE_ORT_PROVIDER": "CPUExecutionProvider" if is_cpu else "CUDAExecutionProvider",
            },
        )
        results[f"faiss/{name}"] = _run_cell(_FAISS_SNIPPET, {**pin, "PROBE_FAISS_GPU": "0" if is_cpu else "1"})

    width = max(len(k) for k in results)
    failed = False
    for key, r in results.items():
        print(f"{key:<{width}}  {r['status']:<5}  {r['detail']}")
        failed = failed or r["status"] == "FAIL"

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "gpus": gpus}, fh, indent=2)
        print(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
