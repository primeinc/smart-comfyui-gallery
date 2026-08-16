# OmniQuery — Search Your Gallery in Plain English

Press **Ctrl+P** (or **Alt+P**, or the ⚡ toolbar button) anywhere in the
gallery. A search field opens over a live masonry of your newest images.
Type anything — the tiles morph as you type; pause for a moment and the
AI's deeper answer swaps in; press **Enter** to open the full result set
in the gallery. Press **Esc** to close. No SQL is ever shown, nothing
leaves your machine.

## Ways you can search

Structured criteria are understood instantly (under a millisecond,
deterministic), and anything else is treated as a search of your prompts,
captions, filenames, models, and LoRA names:

| You type | It finds |
|---|---|
| `girlnextdoor` | every file whose prompt, LoRA, caption, name, or path carries the term |
| `favorite videos` | starred files of one media type |
| `pngs over 20 MB` | file formats by extension (png, jpg, webp, mp4, ...) |
| `4+ star images from last week` | ratings and calendar words combined |
| `not approved images` | status flags, including negations |
| `seed 424242` / `images with 30 steps cfg 7.5` | exact generation settings |
| `lora girlnextdoor` / `model flux` | generation provenance |
| `videos longer than 2 minutes` | durations, sizes, resolutions |
| `images with faces` / `face cluster 1` | the face pipeline's results |
| `how many favorites` | count questions get a number |
| `files rated at least 4 by more than one person` | free language — the AI writes the database query itself |

## How it works (and why it's safe)

Two answerers cooperate per query:

- A **deterministic rules engine** handles everything it fully
  recognizes. It is exact, instant, and what live typing always uses.
- The **local nl2sql model** (a 2.5 GB text2sql GGUF, Apache-2.0) handles
  free language: it reads your database's actual schema, writes a
  read-only query, **looks at what came back**, and refines before
  answering — broadening a search that found nothing, repairing its own
  mistakes. If it fails, the rules answer stands; search never breaks.

Everything the model writes executes through one sandbox: SELECT-only,
a read-only database connection, and SQLite's engine-level authorizer —
it is impossible for a generated query to modify anything.

## Setup

The rules engine needs nothing. The AI answerer is one command:

```bash
python -m smartgallery_ai provision omniquery
```

See [INSTALL_AI.md](INSTALL_AI.md) for GPU notes and environment knobs.
Everything runs locally; no external AI service is involved.

## For developers

`POST /galleryout/api/omniquery/nlq` with `{"query": "...", "live": true}`
is the palette's endpoint (live = rules-only, no writes; non-live may
consult the model and stores a result session). Raw read-only SQL remains
available at `POST /galleryout/api/omniquery/execute` through the same
sandbox. Diagnostics and acceptance benchmarks are `just ai` recipes;
current measured numbers live in [AI_MODELS.md](AI_MODELS.md).
