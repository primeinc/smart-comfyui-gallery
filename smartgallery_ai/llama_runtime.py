"""Windows DLL-path bootstrap for the prebuilt CUDA llama-cpp-python wheel.

The cu12x wheels link cudart64_12.dll / cublas64_12.dll, which pip provides
via the nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12 wheels — but
llama-cpp-python opens llama.dll with winmode=RTLD_GLOBAL (0), the LEGACY
PATH-based DLL search that ignores os.add_dll_directory
(llama_cpp/_ctypes_extensions.py). So the wheel bin dirs must sit on PATH
before `import llama_cpp`. Observed live: ggml-cuda.dll unresolvable and
the critic dead until these dirs were prepended.

Call prepare_llama_runtime() before every llama_cpp import. Idempotent,
no-op off Windows.
"""

import glob
import os
import sys
import sysconfig

_prepared = False


def prepare_llama_runtime() -> None:
    global _prepared
    if _prepared:
        return
    _prepared = True
    if sys.platform != "win32":
        return
    purelib = sysconfig.get_paths().get("purelib") or ""
    for bin_dir in sorted(glob.glob(os.path.join(purelib, "nvidia", "*", "bin"))):
        if not os.path.isdir(bin_dir):
            continue
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(bin_dir)  # newer loaders honor this too
        except OSError:
            pass
