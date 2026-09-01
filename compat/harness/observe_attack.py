from __future__ import annotations

import contextlib
import ctypes
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import proc
from compat.harness import observe

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


_EXTRA: list[Path] = []


@dataclass
class Probe:
    name: str
    observed: bool
    expected: bool
    detail: str

    @property
    def held(self) -> bool:
        return self.observed == self.expected

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _weight(where: Path) -> Path:
    probe = where / "_observer_probe.onnx"
    probe.write_bytes(b"not a real model, written by observe_attack")
    return probe


def probe_python_open(probe: Path) -> Probe:
    with observe.recording(tuple(_EXTRA)) as rec, probe.open("rb") as handle:
        handle.read()
    seen = any(one["identity"] == probe.name for one in rec.rows())
    return Probe("1 python_open", seen, True, f"rows={len(rec.rows())}")


def probe_alias_before_install(probe: Path) -> Probe:
    from builtins import open as aliased

    with observe.recording(tuple(_EXTRA)) as rec, aliased(probe, "rb") as handle:
        handle.read()
    seen = any(one["identity"] == probe.name for one in rec.rows())
    return Probe("2 alias_before_install", seen, True, f"rows={len(rec.rows())}")


def probe_native_open(probe: Path) -> Probe:
    with observe.recording(tuple(_EXTRA)) as rec:
        crt = ctypes.CDLL("msvcrt", use_errno=True)
        crt._wfopen.restype = ctypes.c_void_p
        crt._wfopen.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        handle = crt._wfopen(str(probe), "rb")
        opened = bool(handle)
        if handle:
            crt.fclose.argtypes = [ctypes.c_void_p]
            crt.fclose(ctypes.c_void_p(handle))
    seen = any(one["identity"] == probe.name for one in rec.rows())
    return Probe("3 native_open", seen, True, f"the CRT opened it={opened}; observer rows={len(rec.rows())}")


def probe_subprocess_open(probe: Path) -> Probe:
    script = f"import pathlib; pathlib.Path({str(probe)!r}).read_bytes()"
    with observe.recording(tuple(_EXTRA)) as rec:
        out = probe.parent / "child_observations.json"
        proc.run(
            [sys.executable, "-c", script],
            timeout=proc.LOCAL_SECONDS,
            env=observe.child_env(out, tuple(_EXTRA)),
        )
        folded = observe.absorb(out)
    seen = any(one["identity"] == probe.name for one in rec.rows())
    return Probe("4 subprocess_open", seen, True, f"child returned {folded} row(s); observer rows={len(rec.rows())}")


def probe_native_model_loader() -> Probe:
    with observe.recording() as rec:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="antelopev2", root="C:/ComfyUI/output/.AImodels/insightface")
        app.prepare(ctx_id=0, det_size=(640, 640))
    graphs = sorted({one["identity"] for one in rec.rows() if one["loader"] == "onnxruntime.InferenceSession"})
    wanted = {"1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"}
    return Probe("5 native_model_loader", wanted <= set(graphs), True, f"{len(graphs)} graph(s): {graphs}")


def probe_native_open_is_noticed(probe: Path) -> Probe:
    with observe.recording(tuple(_EXTRA)) as rec:
        crt = ctypes.CDLL("msvcrt", use_errno=True)
        crt._wfopen.restype = ctypes.c_void_p
        crt._wfopen.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        handle = crt._wfopen(str(probe), "rb")
        if handle:
            crt.fclose.argtypes = [ctypes.c_void_p]
            crt.fclose(ctypes.c_void_p(handle))
    armed = [one for one in rec.rows() if one["loader"] == observe.NATIVE_UNSEEN]
    return Probe(
        "7 native_open_noticed",
        bool(armed),
        True,
        f"armed native openers recorded: {sorted({one['identity'] for one in armed})}",
    )


def probe_wrapping_preserves_dispatch() -> Probe:

    class Base:
        @classmethod
        def from_pretrained(cls, where: str) -> str:
            return f"{cls.__name__}:{where}"

    class Derived(Base):
        pass

    seen: list[str] = []
    with observe.recording(tuple(_EXTRA)), contextlib.ExitStack() as stack:
        observe.wrap_recording(stack, Base, "from_pretrained", lambda args, kwargs: seen.append(str(args)))
        wrapped = Derived.from_pretrained("weights")
    restored = Derived.from_pretrained("weights")
    held = wrapped == "Derived:weights" and restored == "Derived:weights" and seen == ["('weights',)"]
    return Probe(
        "8 wrapping_preserves_dispatch",
        held,
        True,
        f"wrapped -> {wrapped!r}, restored -> {restored!r}, recorded {seen}",
    )


def probe_hub_identity() -> Probe:
    with observe.recording() as rec:
        observe._RECORDER.note("snapshot_download", "org/model", "", "deadbeef")
    rows = [one for one in rec.rows() if one["loader"] == "snapshot_download"]
    kept = bool(rows) and rows[0]["identity"] == "org/model" and rows[0]["revision"] == "deadbeef"
    return Probe("6 hub_identity", kept, True, f"rows={rows}")


def run_all() -> list[Probe]:
    with tempfile.TemporaryDirectory(prefix="observe_attack_") as raw:
        _EXTRA.clear()
        _EXTRA.append(Path(raw))
        probe = _weight(Path(raw))
        try:
            return [
                probe_python_open(probe),
                probe_alias_before_install(probe),
                probe_native_open(probe),
                probe_subprocess_open(probe),
                probe_native_open_is_noticed(probe),
                probe_wrapping_preserves_dispatch(),
                probe_native_model_loader(),
                probe_hub_identity(),
            ]
        finally:
            probe.unlink(missing_ok=True)


def main() -> int:
    probes = run_all()
    print("dynamic observer controls\n")
    for one in probes:
        print(f"{one.mark} {one.name:<26} observed={one.observed!s:<6} required={one.expected}")
        print(f"       {one.detail[:110]}")

    missed = [one.name for one in probes if not one.held]
    print(f"\n{len(probes)} probes, {len(missed)} failing: {missed or 'none'}")
    if missed:
        print("the observer is NOT a completeness oracle: artifacts loaded this way are invisible to it")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "observer_controls.json"
    payload: dict[str, Any] = {"probes": [vars(one) for one in probes], "failing": missed}
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
