# faiss GPU build — Windows x64, CUDA 13

Self-built faiss python package with CUDA support (`get_num_gpus() > 0`),
selected at runtime by `smartgallery_ai/faiss_runtime.py` on Windows
boxes with an NVIDIA driver; the installed `faiss-cpu` wheel is the
fallback everywhere else. `AI_DAM_FAISS_GPU=0` opts out.

- Build recipe, toolchain, and verified-behavior table: `docs/FAISS_GPU_WINDOWS.md`
- Arches: `sm_86` + `sm_120` (`CMAKE_CUDA_ARCHITECTURES=86-real;120-real`)
- The CUDA runtime DLLs (`cublas64_13`, `cublasLt64_13`, `cudart64_13`,
  `nvJitLink_130_0`) are NOT vendored (GitHub's 100MB file cap;
  cublasLt alone is 453MB). They ship in the `nvidia-cublas`,
  `nvidia-cuda-runtime`, and `nvidia-nvjitlink` pip wheels
  (`nvidia/<pkg>/bin/x86_64/`), installed by the auto-provisioner when
  a GPU is present, and are also found from a system CUDA 13 toolkit on
  `PATH`.
- Verify selection: `just faiss-verify`
