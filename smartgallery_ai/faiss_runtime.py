"""Import-time faiss selection: vendored Windows CUDA build first,
installed faiss-cpu as the fallback.

vendor/faiss-gpu-win64/faiss is the self-contained GPU build from
docs/FAISS_GPU_WINDOWS.md (CUDA 13, sm_86 + sm_120). Its CUDA runtime
DLLs (cublas64_13, cublasLt64_13, cudart64_13, nvJitLink_130_0) are not
vendored -- GitHub rejects >100MB files -- and come instead from the
nvidia-cublas / nvidia-cuda-runtime / nvidia-nvjitlink pip wheels
(nvidia/<pkg>/bin/x86_64/), which the provisioner installs when an
NVIDIA GPU is present. Python extension modules resolve dependent DLLs
through os.add_dll_directory (PATH is ignored for extensions on 3.8+),
so those wheel dirs are registered before the import.

import_faiss() is the one sanctioned way to import faiss in this
codebase: it prefers the vendored GPU package on Windows boxes with an
NVIDIA driver and falls back to the installed faiss-cpu package when
the vendored import cannot load. AI_DAM_FAISS_GPU=0 opts out.
"""

import contextlib
import glob
import importlib
import os
import shutil
import sys
import sysconfig

_VENDOR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "faiss-gpu-win64")

# Directories already handed to the DLL loader and prepended to PATH.
_REGISTERED_DLL_DIRS: set = set()


def _register_cuda_dll_dirs() -> int:
    """Register every nvidia wheel bin dir with the DLL loader; returns
    how many dirs were registered.

    Repeat calls add nothing. import_faiss runs this whenever faiss is not
    already imported, so on a machine where BOTH the vendored GPU build and
    faiss-cpu fail to import it ran again on every similarity query -- and
    each run prepended the same directories to PATH again. PATH is the
    environment every child process inherits, and Windows caps that block
    at about 32k characters: a few dozen repeats and ffmpeg stops being
    spawnable at all, which reads as video breaking rather than as faiss
    being missing.
    """
    purelib = sysconfig.get_paths().get("purelib") or ""
    registered = 0
    for pattern in ("bin", os.path.join("bin", "x86_64")):
        for d in sorted(glob.glob(os.path.join(purelib, "nvidia", "*", pattern))):
            if not os.path.isdir(d):
                continue
            if d in _REGISTERED_DLL_DIRS:
                continue
            _REGISTERED_DLL_DIRS.add(d)
            try:
                os.add_dll_directory(d)
                registered += 1
            except OSError:
                pass
            current = os.environ.get("PATH", "")
            if d not in current.split(os.pathsep):
                os.environ["PATH"] = d + os.pathsep + current
    return registered


def vendored_faiss_dir() -> str:
    return os.path.join(_VENDOR_ROOT, "faiss")


def import_faiss():
    """Import and return the faiss module (vendored GPU build when
    eligible, else the installed faiss-cpu). Raises ImportError when no
    faiss is available at all."""
    if "faiss" in sys.modules:
        return sys.modules["faiss"]
    use_vendor = (
        sys.platform == "win32"
        and os.path.isdir(vendored_faiss_dir())
        and shutil.which("nvidia-smi") is not None
        and os.environ.get("AI_DAM_FAISS_GPU", "1") == "1"
    )
    if use_vendor:
        _register_cuda_dll_dirs()
        sys.path.insert(0, _VENDOR_ROOT)
        try:
            import faiss

            return faiss
        except (Exception, SystemExit):
            # missing CUDA wheel DLLs, wrong arch, partial vendor dir --
            # purge the half-imported package and fall back to faiss-cpu
            for name in [m for m in sys.modules if m == "faiss" or m.startswith("faiss.")]:
                del sys.modules[name]
            importlib.invalidate_caches()
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(_VENDOR_ROOT)

    # Either the vendored build was not eligible or it failed to load. Both
    # land here, on the installed faiss-cpu, and both need their own import:
    # the one above binds `faiss` as a local, so reaching this line without
    # it raised UnboundLocalError rather than importing anything. That is
    # every install without the vendored GPU build -- all of Linux and
    # macOS, any Windows box with no nvidia-smi, and anyone setting
    # AI_DAM_FAISS_GPU=0 -- so the documented fallback never once happened.
    import faiss

    return faiss
