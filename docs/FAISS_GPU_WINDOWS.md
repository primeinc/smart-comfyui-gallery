# GPU Faiss on Windows (from source)

No distributed faiss GPU package exists for Windows (faiss INSTALL.md:
conda `faiss-gpu` is Linux x86-64 only). The source build works; upstream
declares Windows a supported source platform (INSTALL.md "Building from
source") and its CMake contains no Windows exclusion for GPU. Upstream CI
does not test the Windows GPU combination.

Verified on: Windows 11, MSVC 19.51 (VS 18 Insiders), CUDA toolkit 13.2,
RTX 3070 Ti (sm_86) + RTX 5060 Ti (sm_120), faiss commit `0f0f728`.

## Prerequisites

- Visual Studio with C++ toolchain (`vcvars64.bat`)
- CUDA toolkit 13.x (`nvcc`)
- ninja
- swig (winget `SWIG.SWIG`)
- OpenBLAS prebuilt release (OpenMathLib/OpenBLAS `-x64.zip`)
- python with numpy (build target interpreter)

## Configure

Under `vcvars64.bat`, with `<SRC>` a faiss checkout at tag `v1.15.0`
(the version `uv.lock` pins for `faiss-cpu`):

```
cmake -S <SRC> -B build -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DFAISS_ENABLE_GPU=ON
  -DFAISS_ENABLE_PYTHON=ON
  -DFAISS_ENABLE_C_API=OFF
  -DBUILD_TESTING=OFF
  -DBUILD_SHARED_LIBS=ON
  -DFAISS_OPT_LEVEL=avx2
  "-DCMAKE_CXX_FLAGS=/Zc:preprocessor /DWIN32_LEAN_AND_MEAN /DNOMINMAX"
  "-DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor -DWIN32_LEAN_AND_MEAN -DNOMINMAX"
  "-DCMAKE_CUDA_ARCHITECTURES=86-real;120-real"
  "-DCUDAToolkit_ROOT=<CUDA_ROOT>"
  "-DCMAKE_PREFIX_PATH=<OPENBLAS_DIR>"
  "-DPython_EXECUTABLE=<PYTHON>"
  "-DSWIG_EXECUTABLE=<SWIG_EXE>"
  "-DSWIG_DIR=<SWIG_LIB_DIR>"
cmake --build build --target swigfaiss -j
```

Required Windows-specific flags:

- `/Zc:preprocessor` — CUDA 13's CCCL headers reject MSVC's traditional
  preprocessor (fatal C1189).
- `WIN32_LEAN_AND_MEAN` — `rpcndr.h` defines `small` as a macro;
  `faiss/gpu/utils/MergeNetworkWarp.cuh` uses `small` as a variable name.
- One source patch: `faiss/gpu/impl/PQCodeDistances-inl.cuh:545` needs
  `.template view<2>(` (EDG dependent-name disambiguation).

## Package

`setup.py build` only copies files; equivalent manually:

```
mkdir faiss-pkg/faiss
cp build/faiss/python/{__init__.py,__init__.pyi,loader.py,swigfaiss.py,
   _swigfaiss.pyd,class_wrappers.py,array_conversions.py,extra_wrappers.py,
   gpu_wrappers.py,py.typed} faiss-pkg/faiss/
cp -r build/faiss/python/contrib faiss-pkg/faiss/
```

Do NOT copy `_gpu_build.py`: it triggers a Linux-wheel-only CUDA preload in
`__init__.py` (`libcudart.so.*` paths) that fails on Windows. Without the
marker, DLLs resolve through `os.add_dll_directory`.

## Runtime

```python
import os, sys
os.add_dll_directory("<BUILD>/build/faiss")            # faiss.dll
os.add_dll_directory("<OPENBLAS_DIR>/bin")             # libopenblas.dll
os.add_dll_directory("<CUDA_ROOT>/bin/x64")            # cudart64_13, cublas64_13
sys.path.insert(0, "<PKG_PARENT>")                     # dir containing faiss/
import faiss
```

CUDA 13 moved runtime DLLs from `bin/` to `bin/x64/`.

## Verified behavior

`sg-lab/gpu_faiss_proof.py` run, 12,713 real 128-d embeddings and
500k x 512 synthetic, 1000 batched queries, k=10, best-of-5:

| dataset | CPU flat | GPU dev0 (sm_86) | GPU dev1 (sm_120) | both GPUs |
|---|---|---|---|---|
| 12,713 x 128 | 10.8 ms | 0.8 ms | 0.9 ms | 0.5 ms |
| 500k x 512 | 1568 ms | 46 ms | 76 ms | 39 ms |

- recall@10 overlap vs CPU: 1.0000 everywhere; max distance delta
  4.77e-07 / 1.25e-06 (reduction-order ULPs, per faiss wiki
  Comparing-GPU-vs-CPU)
- `index_cpu_to_all_gpus` (IndexReplicas) works across both cards
- spherical GPU k-means (`faiss.Kmeans(gpu=True, spherical=True)`):
  12,713 -> 256 unit-norm centroids in 0.2 s
- GPU indexes are k-NN only: `range_search` remains CPU-only on all
  platforms

## Constraints

- GPU indexes and `StandardGpuResources` are not thread-safe, even
  read-only: one resources object per calling thread (faiss wiki Threads).
- `k` and `nprobe` <= 2048 on GPU.
- GPU indexes cannot be serialized; convert with `index_gpu_to_cpu` first.
