"""Attack the dynamic observer. It is a completeness oracle or it is nothing.

`observe.py` claims to record what a run resolved. It records what CPython
audited plus the handful of APIs wrapped by hand, and those are different
contracts: a native extension, an inference runtime or a child process can open
model bytes below CPython's audited `open`.

Eight controls, in increasing order of how badly they hurt:

    1 python_open          `open(path)`                     must be observed
    2 alias_before_install a reference bound pre-hook        must be observed
    3 native_open          CRT `_wfopen` through ctypes      must be observed
    4 subprocess_open      a child process reading the file  must be observed
    5 native_model_loader  a real ORT session                must be observed
    6 hub_identity         repo/revision preserved, not just a path
    7 native_open_noticed  a native opener being ARMED       must be observed
    8 wrapping_preserves_dispatch  recording changes nothing about the call

3 and 4 were red, and each was treated for a while as a limit rather than a
defect. Neither is:

  * an audit hook is per-PROCESS, so a child gets one of its own through
    `sitecustomize` instead of the parent being asked to see what it cannot;
  * a native open fires no `open` event, but ctypes resolves its symbols
    through `CDLL.__getattr__` in PYTHON, so the call is reachable and its
    first argument is the path.

Probe 7 covers what remains of the second: a symbol armed and never called is
still recorded, from the audited `ctypes.dlsym`.

Probe 8 is the other direction, and it caught a real one. An observer that
records correctly and CHANGES the run is worse than no observer: wrapping a
classmethod with a plain function dropped the `cls` binding and sent every
subclass to the base implementation.

What none of this reaches is a compiled extension calling `fopen` inside its
own binary, with no Python frame anywhere. Probe 5 is why that is not a hole
in practice for this suite: ONNX Runtime is exactly such an extension, and it
is covered by instrumenting the loader it exposes rather than the syscall it
makes. A new native loader needs the same treatment and gets no free pass.
"""

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

#: The control's own directory, added to the observer's scope for the run.
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
    """A probe file, in the control's OWN directory. Never a real model root.

    The first version wrote into the live antelopev2 pack because that is a
    directory `model_roots()` returns -- and insightface globs that pack, so
    it tried to parse the dummy as a graph and the run died. Scope is proven
    instead by passing the directory to `recording(extra_roots=...)`.
    """
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
    """The CRT's own file open, through ctypes. No Python `open` runs.

    This is the class the audit hook cannot see: any C extension, inference
    runtime or vendored library doing its own I/O takes this path.
    """
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
    """A child process reading the file, observed by its OWN hook.

    The parent's audit hook is per-process and cannot see this; that is not a
    defect to work around but the reason the child gets a hook of its own,
    installed by `sitecustomize` before any user code runs. The child here is
    given no knowledge of the observer beyond its environment.
    """
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
    """A real ORT session over a real pack. Semantic instrumentation only."""
    with observe.recording() as rec:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="antelopev2", root="C:/ComfyUI/output/.AImodels/insightface")
        app.prepare(ctx_id=0, det_size=(640, 640))
    graphs = sorted({one["identity"] for one in rec.rows() if one["loader"] == "onnxruntime.InferenceSession"})
    wanted = {"1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"}
    return Probe("5 native_model_loader", wanted <= set(graphs), True, f"{len(graphs)} graph(s): {graphs}")


def probe_native_open_is_noticed(probe: Path) -> Probe:
    """A native open cannot be SEEN, and must not be silent.

    Probe 3 stays red: the path a C extension opens is not recoverable from
    inside the process. But `ctypes.dlsym` IS audited, so arming a native
    opener is detectable, and a run in which one was armed can say so instead
    of leaving the reader to assume the blindness never mattered.

    This is what turns that blindness from a standing assumption into a
    per-run measurement, which is what `reconcile` reads before deciding
    whether an unobserved artifact may be called absent.
    """
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
    """Recording may not change what a call resolves to.

    `from_pretrained` is a classmethod, and replacing one with a plain
    function drops the `cls` binding: every subclass then runs the base
    class's implementation. `Sam3VideoModel.from_pretrained` became
    `PreTrainedModel`'s and raised "PreTrainedModel does not define
    `config_class`" -- four id_v2v cases failing on the observer rather than
    on anything they measure.

    A stand-in hierarchy rather than transformers, because the question is
    whether the WRAPPER preserves dispatch. It runs against the production
    `wrap_recording`, never a copy of it.
    """

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
    """A repository identity must survive as a repo, not decay to a path."""
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
