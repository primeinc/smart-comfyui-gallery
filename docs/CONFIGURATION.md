# Configuration Reference

Every setting is an environment variable. Set them in your launcher
(`run_smartgallery.bat` / `run_smartgallery.sh`), your Docker `-e` flags, or
your shell before starting `python smartgallery.py`.

**A blank value means "use the default."** `set "BASE_OUTPUT_PATH="`,
`export BASE_OUTPUT_PATH=`, and an empty field in a Docker or Unraid
template all define the variable as an empty string; SmartGallery treats
that exactly like leaving it unset. Values are trimmed, an unparseable
number or yes/no word warns on the console and falls back to the default,
and nothing here can stop the app from starting.

Numbers that parse but cannot work — a zero or a negative where a count or
a size is wanted — do the same. `BATCH_SIZE=0` warns and uses 500 rather
than stopping the scan. Where zero is meaningful it is kept:
`STREAM_THRESHOLD_MB=0` streams everything, and in the AI layer
`AI_DAM_FACE_DETECT_MAX_SIDE=0` disables the cap.

Yes/no settings accept `1 / true / yes / on` and `0 / false / no / off`, in
any case.

---

## Paths

| Variable | Default | What it does |
|---|---|---|
| `BASE_OUTPUT_PATH` | `C:/ComfyUI/output` | The folder the gallery shows. Use forward slashes even on Windows. |
| `BASE_INPUT_PATH` | `C:/ComfyUI/input` | ComfyUI's `input` folder, for the Remix workflow features. |
| `BASE_SMARTGALLERY_PATH` | same as `BASE_OUTPUT_PATH` | Where service folders live (database, thumbnail cache, zips). Point it off your output folder to keep system files out of the gallery. |
| `BASE_MODELS_PATH` | `<parent of output>/models` | ComfyUI models root, for Remix model pickers. |
| `LORAS_PATH` / `CHECKPOINTS_PATH` / `UNET_PATH` | `<models>/loras`, `/checkpoints`, `/unet` | Override individually for split model layouts (Stability Matrix, `extra_model_paths.yaml`). |
| `DELETE_TO` | unset (permanent delete) | When set, deletes move files **and folders** to `DELETE_TO/SmartGallery/<timestamp>_<name>` instead of removing them, so a mistake is recoverable. Links and shortcuts are only unlinked, never relocated. The folder is created on first start; point it at a path the app can write to. |
| `FFPROBE_MANUAL_PATH` | `C:/ffmpeg/bin/ffprobe.exe` | Full path to **ffprobe** (not ffmpeg — the app verifies which one it is and falls back to your `PATH` if it is the wrong tool). |

## Server and access

| Variable | Default | What it does |
|---|---|---|
| `SERVER_PORT` | `8189` | Web server port. Must differ from ComfyUI's (usually 8188). |
| `ADMIN_PASSWORD` | unset | Sets or resets the admin password at startup; equivalent to `--admin-pass`. |
| `SECRET_KEY` | random per start | Flask session signing key. Set a fixed value to keep users logged in across restarts. |
| `COMFYUI_SERVER_URL` | `http://127.0.0.1:8188` | Where ComfyUI is. Used by Remix to queue generations, and by the page for the ComfyUI and LoRA Manager links. |
| `COMFYUI_MAX_UPLOAD_MB` | `2000` | Maximum upload size in MB. The web server's own ceiling is derived from this, so values above 2048 take effect. |

## Media handling and performance

| Variable | Default | What it does |
|---|---|---|
| `GENERATE_THUMBNAILS` | `true` | Server-side thumbnail generation. The Tools-menu toggle stored in the database overrides this at runtime. |
| `THUMBNAIL_WIDTH` | `300` | Thumbnail width in pixels (height is capped at 2×). |
| `GENERATE_WAVEFORMS` | `false` | Audio waveform images (needs ffmpeg). |
| `WEBP_ANIMATED_FPS` | `16.0` | Assumed frame rate when computing animated-WebP duration. |
| `PAGE_SIZE` | `100` | Files per page / infinite-scroll batch. |
| `BATCH_SIZE` | `500` | Database batch size during scans. |
| `MAX_PARALLEL_WORKERS` | auto (CPU count) | Parallel workers for scanning and thumbnailing. |
| `STREAM_THRESHOLD_MB` | `20` | Videos above this size stream with range requests instead of being sent whole. |
| `GENPARAMS_BACKFILL` | `true` | Startup backfill of typed generation parameters. Set false to skip it on very large libraries. |

## AI layer

The AI layer is **on by default** and provisions itself; see
[INSTALL_AI.md](INSTALL_AI.md) for what it downloads and
[AI_MODELS.md](AI_MODELS.md) for the models and measured numbers.

| Variable | Default | What it does |
|---|---|---|
| `ENABLE_AI_DAM` | `true` | Master switch for Similar / Faces / Review / the search palette. |
| `ENABLE_AI_SEARCH` | `false` | Shows the legacy AI Search box and AI Manager panel. **Leave this off.** See below. |
| `AI_DAM_AUTO_PROVISION` | `true` | Download missing weights and install missing runtimes on startup. Set false for strict no-egress hosts. |
| `AI_DAM_MODELS_DIR` | `<gallery>/.AImodels` | Where model weights live. |
| `AI_DAM_CACHE_DIR` | `<gallery>/.ai_cache` | Derived caches (vector index, masks). |
| `AI_DAM_DEVICE` | auto (most-VRAM GPU) | `cpu`, `cuda`, or `cuda:N` for every backend. |
| `AI_DAM_SEMANTIC_BACKEND` / `AI_DAM_VISUAL_BACKEND` / `AI_DAM_FACE_BACKEND` / `AI_DAM_CRITIC_BACKEND` / `AI_DAM_SEGMENTER_BACKEND` | `auto` | Per-feature selector: `auto`, `none` to disable, or an explicit backend name. |
| `AI_DAM_WORKER_POLL` | `25` | Seconds between background indexing cycles. |
| `AI_DAM_WORKER_BATCH` | `150` | Per-cycle file ceiling. The worker measures its own throughput and shrinks batches below this on slow or busy hardware. |
| `AI_DAM_EMBED_BATCH` | `16` | Images per embedding call. |
| `AI_DAM_NEAR_DUP_DISTANCE` | `8` | Maximum perceptual-hash Hamming distance counted as a near-duplicate. |
| `AI_DAM_SIMILAR_K` | `24` | Default neighbour count for similarity queries. |
| `AI_DAM_FACE_CLUSTER_THRESHOLD` | per-embedder default | Cosine similarity required to group two faces. |
| `AI_DAM_FACE_MIN_PX` / `AI_DAM_FACE_DETECT_MAX_SIDE` | `24` / `1600` | Smallest kept face box, and the detection input cap. |
| `AI_DAM_FACE_EMBEDDER` | `auto` | `arcface` (512-d) or `sface` (128-d). |
| `AI_DAM_GPU_LAYERS` | `-1` (all) | Critic layers offloaded to GPU. |
| `AI_DAM_TENSOR_SPLIT` | unset | Proportions for splitting the critic across GPUs, e.g. `0.6,0.4`. |
| `AI_DAM_EPHEMERAL_INDEX` | `false` | Keep the vector index in memory only. |
| `OMNIQUERY_NL2SQL_GGUF` | provisioned model | Override the model the search palette uses to turn a question into SQL. |
| `OMNIQUERY_FALLBACK_GGUF` | provisioned model | Override the model behind the palette's plain-language fallback. Also used for nl2sql when `OMNIQUERY_NL2SQL_GGUF` is unset, so setting this alone changes both. |
| `OMNIQUERY_FALLBACK_GPU_LAYERS` | `-1` (all) | Layers of the fallback model offloaded to GPU. |

### `ENABLE_AI_SEARCH` is inert

Turning it on adds the old AI Search box and AI Manager panel to the
interface, and the gallery then writes to two queues — one of searches, one
of files to index. **Nothing processes either of them.** The component that
did was replaced by the AI layer above (`ENABLE_AI_DAM`), which indexes
directly from the library and needs no queue.

What you see if you enable it anyway: a search that stays on "pending" and
never returns a result, an indexing count that only ever grows, and one
queue row per file written on every scan. Nothing breaks, and nothing
happens.

Semantic search over your library is the search palette
(<kbd>Ctrl</kbd>+<kbd>P</kbd>) and the Similar view, both part of the AI
layer, which is on by default.

Additional low-level switches (`AI_DAM_VISION_GPU`, `AI_DAM_VISION_FA`,
`AI_DAM_FAISS_GPU`, `AI_DAM_VECTOR_GPU`, `AI_DAM_ORT_PROVIDERS`,
`AI_DAM_FACE_GRAPH_BACKEND`, `AI_DAM_CUDA_INDEX`,
`AI_DAM_LLAMA_CUDA_INDEX`, `AI_DAM_LLAMA_VERBOSE`, `LLAMA_CPP_LIB_PATH`)
exist for diagnosing hardware problems and are described in
[INSTALL_AI.md](INSTALL_AI.md).

---

## Command-line flags

These are flags, not environment variables:

| Flag | What it does |
|---|---|
| `--port N` | Override `SERVER_PORT`. |
| `--admin-pass PASSWORD` | Set or reset the admin password. |
| `--force-login` | Require login on the main interface. |
| `--enable-guest-login` | Allow passwordless guest access. |
| `--exhibition` | Start in Exhibition Mode. |
| `--blind-rating` | Hide global averages so raters are not anchored. |
