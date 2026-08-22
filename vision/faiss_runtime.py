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
the vendored import cannot load. `gpu=False` opts out -- the caller's
choice, carried from the `faiss_gpu` setting (db/settings.py).
"""

import contextlib
import glob
import importlib
import logging
import os
import shutil
import sys
import sysconfig

_logger = logging.getLogger(__name__)

_VENDOR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "faiss-gpu-win64")

# Directories already handed to the DLL loader and prepended to PATH.
_REGISTERED_DLL_DIRS: set = set()


def _cuda_dll_dirs() -> list:
    """Directories that may hold the CUDA runtime DLLs the vendored build
    needs (cublas64_13, cublasLt64_13, cudart64_13, nvJitLink_130_0).

    TORCH FIRST, because on a CUDA torch install it already has them:
    `torch/lib` ships the same four DLLs, so a GPU box that can run the
    embedders can run GPU faiss with nothing extra installed. Looking only
    in the `nvidia/*/bin` wheel layout meant the vendored build failed for
    want of files that were already on disk, and silently became faiss-cpu.
    The nvidia wheel dirs stay in the list for installs that have them and
    no torch.
    """
    purelib = sysconfig.get_paths().get("purelib") or ""
    dirs = [os.path.join(purelib, "torch", "lib")]
    for pattern in ("bin", os.path.join("bin", "x86_64")):
        dirs.extend(sorted(glob.glob(os.path.join(purelib, "nvidia", "*", pattern))))
    return dirs


def _register_cuda_dll_dirs() -> int:
    """Register every candidate CUDA DLL dir with the loader; returns how
    many were registered.

    Repeat calls add nothing. import_faiss runs this whenever faiss is not
    already imported, so on a machine where BOTH the vendored GPU build and
    faiss-cpu fail to import it ran again on every similarity query -- and
    each run prepended the same directories to PATH again. PATH is the
    environment every child process inherits, and Windows caps that block
    at about 32k characters: a few dozen repeats and ffmpeg stops being
    spawnable at all, which reads as video breaking rather than as faiss
    being missing.
    """
    registered = 0
    for d in _cuda_dll_dirs():
        if not os.path.isdir(d) or d in _REGISTERED_DLL_DIRS:
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


def import_faiss(gpu: bool = True):
    """Import and return the faiss module (vendored GPU build when
    eligible, else the installed faiss-cpu). Raises ImportError when no
    faiss is available at all.

    `gpu` is consulted only on the first import in a process -- a module
    once loaded stays loaded, so changing the setting applies from the
    next start, and that is a fact about Python, not a policy here."""
    if "faiss" in sys.modules:
        # already loaded: the import is a dictionary lookup, and it hands
        # back the module AS the module rather than as sys.modules' Any
        import faiss

        return faiss
    use_vendor = (
        gpu
        and sys.platform == "win32"
        and os.path.isdir(vendored_faiss_dir())
        and shutil.which("nvidia-smi") is not None
    )
    if use_vendor:
        _register_cuda_dll_dirs()
        sys.path.insert(0, _VENDOR_ROOT)
        try:
            import faiss

        except (Exception, SystemExit):
            # missing CUDA wheel DLLs, wrong arch, partial vendor dir --
            # purge the half-imported package and fall back to faiss-cpu.
            # WARNING, not debug: the worker sets this package's logger to
            # INFO, so a debug line here made the one thing worth saying
            # invisible. The GPU index quietly becomes a CPU one and only
            # the speed says so, on a machine that shipped 102 MB of
            # binaries specifically to avoid that.
            _logger.warning(
                "the vendored GPU faiss did not load; using faiss-cpu. Its CUDA runtime DLLs "
                "normally come from the installed torch's lib directory (a CUDA build ships "
                "cublas64_13, cublasLt64_13, cudart64_13 and nvJitLink_130_0), else the nvidia "
                "wheels or a system CUDA 13 toolkit on PATH; the faiss_gpu setting silences "
                "this by choosing faiss-cpu deliberately",
                exc_info=True,
            )
            for name in [m for m in sys.modules if m == "faiss" or m.startswith("faiss.")]:
                del sys.modules[name]
            importlib.invalidate_caches()
        else:
            # Say what actually loaded, next to the noise: the vendored
            # package's own upstream loader first probes swigfaiss_avx2 and
            # logs the miss at INFO ("Could not load library with AVX2
            # support") before loading its real binary. That reads like a
            # degraded fallback; it is not -- the CUDA wheel simply ships
            # no AVX2 sub-module, because its main binary IS the build.
            _logger.info(
                "vendored GPU faiss loaded from %s (%d CUDA device(s)); the loader's AVX2 lines "
                "above are its probe order, not a fallback",
                vendored_faiss_dir(),
                faiss.get_num_gpus(),
            )
            return faiss
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(_VENDOR_ROOT)

    # Either the vendored build was not eligible or it failed to load. Both
    # land here, on the installed faiss-cpu, and both need their own import:
    # the one above binds `faiss` as a local, so reaching this line without
    # it raised UnboundLocalError rather than importing anything. That is
    # every install without the vendored GPU build -- all of Linux and
    # macOS, any Windows box with no nvidia-smi, and anyone who turned
    # faiss_gpu off -- so the documented fallback never once happened.
    import faiss

    return faiss
