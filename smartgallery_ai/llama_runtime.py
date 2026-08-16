"""llama.cpp runtime bootstrap: DLL search paths, official-binary override,
and dynamic backend registration.

Two supported library sources, resolved in this order:

1. **Official upstream binaries** (ggml-org/llama.cpp release zips,
   provisioned by the 'llama-cuda' group into <models>/llama-cpp-cuda/).
   These are built for every current GPU architecture -- including sm_120
   (Blackwell), which the llama-cpp-python wheels do not ship -- and are
   selected by exporting LLAMA_CPP_LIB_PATH (the binding's documented
   override, llama_cpp/llama_cpp.py) before `import llama_cpp`. Official
   builds register compute backends DYNAMICALLY (ggml-backend-reg.cpp
   searches the executable dir -- python.exe's, not the DLL's -- so a
   ctypes host must register explicitly): after importing llama_cpp, call
   activate_llama_backends() or zero devices exist and every model load
   fails.
2. **The wheel's own bundled DLLs** (static backend registration; no
   activation call needed). The cu12x wheels link cudart64_12/cublas64_12
   from the pip nvidia-* packages, and open llama.dll with
   winmode=RTLD_GLOBAL -- the LEGACY PATH-based DLL search that ignores
   os.add_dll_directory (llama_cpp/_ctypes_extensions.py) -- so those bin
   dirs must sit on PATH before import.

Call prepare_llama_runtime() before every llama_cpp import and
activate_llama_backends() right after it. Both idempotent; both no-ops
when they don't apply.
"""

import ctypes
import glob
import logging
import os
import sys
import sysconfig

_logger = logging.getLogger(__name__)

_prepared = False
_activated = False

# Directory name the 'llama-cuda' provisioning artifact unpacks into,
# relative to the AI models directory.
OFFICIAL_LIB_DIRNAME = "llama-cpp-cuda"


def _provisioned_lib_dir() -> str:
    """The provisioned official-binaries directory, or '' when absent."""
    models_dir = os.environ.get("AI_DAM_MODELS_DIR", ".AImodels")
    lib_dir = os.path.join(models_dir, OFFICIAL_LIB_DIRNAME)
    return lib_dir if os.path.isfile(os.path.join(lib_dir, "llama.dll")) else ""


def prepare_llama_runtime() -> None:
    """Set up DLL resolution BEFORE `import llama_cpp`. Idempotent."""
    global _prepared
    if _prepared:
        return
    _prepared = True
    if sys.platform != "win32":
        return

    # Official upstream binaries, when provisioned and not overridden by
    # the user's own LLAMA_CPP_LIB_PATH.
    if not os.environ.get("LLAMA_CPP_LIB_PATH"):
        lib_dir = _provisioned_lib_dir()
        if lib_dir:
            os.environ["LLAMA_CPP_LIB_PATH"] = os.path.abspath(lib_dir)
            _logger.info("[AI] llama.cpp libraries: official binaries at %s",
                         os.environ["LLAMA_CPP_LIB_PATH"])
            # The CUDA runtime redistributables land in a sibling dir
            # (their own provisioning artifact); the legacy PATH search
            # must be able to see them from ggml-cuda.dll.
            cudart_dir = os.path.abspath(lib_dir + "-cudart")
            if os.path.isdir(cudart_dir):
                os.environ["PATH"] = cudart_dir + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(cudart_dir)
                except OSError:
                    pass

    purelib = sysconfig.get_paths().get("purelib") or ""
    for bin_dir in sorted(glob.glob(os.path.join(purelib, "nvidia", "*", "bin"))):
        if not os.path.isdir(bin_dir):
            continue
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bin_dir)  # newer loaders honor this too
        except OSError:
            pass


def activate_llama_backends() -> None:
    """Register ggml compute backends AFTER `import llama_cpp` when running
    on official upstream binaries (LLAMA_CPP_LIB_PATH set). Static wheel
    builds export no loader symbol and fall through untouched. Idempotent."""
    global _activated
    if _activated:
        return
    _activated = True
    lib_dir = os.environ.get("LLAMA_CPP_LIB_PATH")
    if not lib_dir or sys.platform != "win32":
        return
    # The registry API lives in ggml.dll in official builds (verified
    # b9976); ggml-base.dll is checked as a fallback for other layouts.
    fn = None
    for name in ("ggml.dll", "ggml-base.dll"):
        dll_path = os.path.join(lib_dir, name)
        if not os.path.isfile(dll_path):
            continue
        try:
            lib = ctypes.CDLL(dll_path, winmode=ctypes.RTLD_GLOBAL)
            fn = lib.ggml_backend_load_all_from_path
            break
        except (OSError, AttributeError):
            continue
    if fn is None:
        return  # static build or unexpected layout: nothing to register
    fn.argtypes = [ctypes.c_char_p]
    fn.restype = None
    fn(os.fsencode(lib_dir))
    lib.ggml_backend_dev_count.restype = ctypes.c_size_t
    _logger.info("[AI] llama.cpp dynamic backends registered from %s "
                 "(%d devices)", lib_dir, lib.ggml_backend_dev_count())
