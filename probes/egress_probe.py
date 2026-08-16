#!/usr/bin/env python3
"""Runtime egress probe (WI-31).

Proves — at runtime, not by static inspection — that SmartGallery with the
AI DAM layer enabled operates with public Internet egress denied.

Method: re-exec the whole probe inside an isolated network namespace
(`unshare -n`, requires root or CAP_SYS_ADMIN) where only loopback exists,
then start the real server (waitress, ENABLE_AI_DAM=true) against a temp
gallery, exercise the gallery view, the AI DAM API surface, and the local
OmniQuery parse path, and assert every request succeeds. Inside the
namespace any attempted egress fails at the kernel level (ENETUNREACH), so
a functioning server IS the evidence of local-only operation.

Run AFTER model provisioning (the needle engine cache and any local model
files must already exist). Exit code 0 = PASS.

Usage: sudo python3 probes/egress_probe.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (parent of probes/)
PORT = 18911  # loopback-only port the probe's server listens on
MARK = "SG_EGRESS_PROBE_STAGE2"  # env var whose presence means "this process is the stage-2 re-exec"


def stage1() -> int:
    """Re-exec this script inside a fresh network namespace (unshare -n).
    Returns the child's exit code, or 2 when unshare is unavailable."""
    if shutil.which("unshare") is None:
        print("FAIL: 'unshare' not available; cannot build isolated netns")
        return 2
    env = dict(os.environ, **{MARK: "1"})
    # -n: new network namespace (no interfaces but lo, which stage2 brings
    # up itself via ioctl -- the 'ip' binary may not exist on minimal hosts).
    cmd = ["unshare", "-n", sys.executable, os.path.abspath(__file__)]
    return subprocess.call(cmd, env=env)


def _loopback_up() -> None:
    """Bring 'lo' up inside the fresh namespace without the iproute2 tools."""
    import fcntl
    import socket
    import struct

    SIOCSIFFLAGS = 0x8914
    IFF_UP, IFF_RUNNING = 0x1, 0x40
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ifr = struct.pack("16sh", b"lo", IFF_UP | IFF_RUNNING)
        fcntl.ioctl(s, SIOCSIFFLAGS, ifr)
    finally:
        s.close()


def wait_for(url: str, timeout: float = 60.0) -> None:
    """Poll `url` until it answers; after `timeout` seconds, raise
    RuntimeError carrying the most recent connection error."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                resp.read()
                return
        except Exception as exc:  # retry loop
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"server never came up: {last}")


def get(url: str):
    """GET `url`, returning (HTTP status, body bytes)."""
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.status, resp.read()


def stage2() -> int:
    """Namespace-side body: verify egress really is denied, start the real
    server with the AI layer enabled, exercise it plus the local OmniQuery
    parse path, print JSON evidence and the verdict, then terminate via
    os._exit(0 on PASS, 1 on FAIL) -- it never returns."""
    _loopback_up()
    # Prove the namespace actually denies egress before trusting anything.
    try:
        urllib.request.urlopen("https://example.com", timeout=3)
        print("FAIL: egress unexpectedly possible inside the namespace")
        return 2
    except Exception:
        pass  # good: no route out

    tmp = tempfile.mkdtemp(prefix="sg_egress_probe_")
    gallery = os.path.join(tmp, "gallery")
    os.makedirs(gallery, exist_ok=True)
    # A tiny real image so the scanner has something to chew on.
    try:
        from PIL import Image
        Image.new("RGB", (64, 64), (120, 40, 200)).save(
            os.path.join(gallery, "probe_img.png"))
    except Exception:
        pass

    env = dict(os.environ)
    env.update({
        "BASE_OUTPUT_PATH": gallery,
        "BASE_SMARTGALLERY_PATH": gallery,
        "BASE_INPUT_PATH": os.path.join(tmp, "input"),
        "SERVER_PORT": str(PORT),
        "ENABLE_AI_DAM": "true",
        # Explicit stub backends: heavy models are a provisioning concern;
        # the probe proves the *service layer* needs no network.
        "AI_DAM_SEMANTIC_BACKEND": "stub",
        "AI_DAM_VISUAL_BACKEND": "stub",
        # Egress must be denied by the netns, not by proxy settings:
        "HTTP_PROXY": "", "HTTPS_PROXY": "", "http_proxy": "", "https_proxy": "",
    })
    os.makedirs(env["BASE_INPUT_PATH"], exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, os.path.join(REPO, "smartgallery.py")],
        cwd=REPO, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    evidence = {"netns_egress_denied": True, "requests": []}
    try:
        base = f"http://127.0.0.1:{PORT}"
        wait_for(f"{base}/galleryout/")
        checks = [
            ("gallery_view", f"{base}/galleryout/"),
            ("aidam_status", f"{base}/galleryout/api/aidam/status"),
        ]
        ok = True
        for name, url in checks:
            try:
                status, body = get(url)
                evidence["requests"].append({"name": name, "status": status,
                                             "bytes": len(body)})
                if status != 200:
                    ok = False
            except Exception as exc:
                evidence["requests"].append({"name": name, "error": str(exc)})
                ok = False

        # Local OmniQuery parse path (no server round-trip needed): the
        # deterministic nlq parser and validator/compiler must work with
        # zero egress.
        sys.path.insert(0, REPO)
        try:
            from omniquery.parsers.nlq import NlqParser
            outcome = NlqParser().parse(
                "favorite videos from the last 7 days", now_epoch=time.time())
            evidence["omniquery_nlq"] = {
                "ast_produced": outcome.ast is not None,
                "confidence": outcome.confidence,
            }
            ok = ok and outcome.ast is not None
        except Exception as exc:
            evidence["omniquery_nlq"] = {"error": str(exc)}
            ok = False

        print(json.dumps(evidence, indent=2))
        print("PASS" if ok else "FAIL", flush=True)
        # Teardown inside the ephemeral namespace can be reaped ungracefully
        # by sandboxed hosts (observed: SIGKILL during graceful shutdown after
        # the verdict). The verdict is already out; exit hard so the process
        # reports the true result instead of the reaper's 137.
        try:
            server.kill()
            server.wait(timeout=5)
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
        os._exit(0 if ok else 1)
    except Exception as exc:
        print(f"FAIL: {exc}", flush=True)
        try:
            server.kill()
        except Exception:
            pass
        os._exit(1)


if __name__ == "__main__":
    sys.exit(stage2() if os.environ.get(MARK) else stage1())
