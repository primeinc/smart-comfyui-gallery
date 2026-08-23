# Smarter Gallery — alpha 1

A local gallery for a library of generated and captured media: one
process, one SQLite file, every picture addressable by name. Scans
folders, reads what generators and cameras wrote into files, groups
faces, finds duplicates, searches by meaning, and tells the story of a
session -- all as explicit jobs you start and watch.

Alpha: the schema, the addresses and the jobs are the product; the
surface is a server-rendered shell over them. Expect change.

## Run

```
uv sync
uv run python -m sg_web            # http://127.0.0.1:8777
uv run python -m sg_web --home D:/runs/two --port 8000
```

A run lives wholly in its home directory (`~/.smartgallery` by default):
the database, model weights, the thumbnail cache. Delete the directory,
delete the run. Media is never under it -- libraries are roots you
register. A home with no database gets one from `db/schema.sql` on
first start (`db.connect.create`).

First library:

1. Open `/operations`, add a root, press **Scan**.
2. Press the sweeps you want: **ingest** (metadata), **phash** /
   **dupes**, **faces** then **cluster**, **embed** (semantic search,
   downloads weights once), **annotate** (a caption per picture and per sampled moment of a video,
   searchable and shown on its page; downloads weights once),
   **context** then **events** (the timeline and stories). Also there:
   **verify** (every present file's bytes against its hash), **thumbs**
   (every missing grid thumb and preview), **embed_prompts** (every
   prompt's vector per space -- the substrate stories and
   `/prompts/N/neighbours` read).

Every sweep is a job row; the activity surface on every page shows it
live over `/ws/jobs`. Nothing expensive runs by itself. The ingest, phash,
faces, embed, annotate and context sweeps queue only what is still
missing -- a file already read, fingerprinted, looked at for faces,
embedded, captioned or interpreted for its current bytes is not an item
again -- and answer 204 when nothing is left (embed, one job per
space, answers an empty list instead); `?everything=true` on the route (or
`{"everything": true}` in the faces and annotate bodies), or the
console's **again** button beside the sweep, redoes all of it.

## Addresses

```
/g                  the gallery: one question, one ordered answer
/i/<slug>           a picture, a video, a document
/p/<slug>           a person           /people
/places             everywhere a person said a picture happened; each a door into the gallery
/t/<slug>           an album or a smart collection   /albums
/f/<slug>           a folder           /folders
/m/<slug> /l/<slug> /w/<slug>   a model, a LoRA, a workflow
/timeline           the library on its human axis: every month, day, bin and session a door; stories told from sessions;
                    any gallery question scopes it (?folder= ?person= ?f=place.id:eq:N ...)
/stories            every story told, newest first
/stories/renders/N  a story: frozen evidence, a plan, words the plan supports; /stories/plans/N/evolution beside it
/search?q=          by meaning, across the configured spaces
/operations         the console: worker, queue, every job's row and ledger, live; roots, sweeps, settings
```

Every entity address (`/g`, `/i`, `/p`, `/t`, `/f`, `/m /l /w`,
`/timeline`, `/stories`, `/operations`) answers JSON to a machine, a
fragment to htmx, and a page to a browser (`Vary: Accept, HX-Request`);
the index and job routes answer JSON only. A renamed thing 301s from
its old slug forever.

Writes, bytes and machine reads:

```
POST /i/<slug>/{favorite,rating,place,collections/<t>}      one picture's authored state
POST /g/selection/{favorite,rating,place,collections/<t>}   a selection, proved against the answer it was made on (409 on a race)
POST /albums  POST /albums/smart  PATCH /t/<slug>  PUT /t/<slug>/rule  POST /t/<slug>/convert  POST /t/<slug>/{add,remove}
POST /jobs/<kind>  POST /jobs/N/cancel       GET /jobs  /jobs/N
POST /stories/snapshots  /stories/plans  /stories/renders    GET the same, and /N
POST /settings/<key>  POST /roots  POST /roots/N/scan        GET /settings  /roots
/media/<slug>  /thumb/<slug>  /preview/<slug>  /avatar/<slug>   bytes (GET and HEAD, Range)
/g/grid  /g/peek  /g/locate/<slug>  /timeline/density          the gallery's and the timeline's own reads
/health  /models  /loras  /workflows  /clusterings  /ways  /dupes  /prompts/N/neighbours
/operations/{overview,job/N,job/N/items,events,events/before}  the console's reads
```

## Settings

Rows in the `setting` table, changed over `POST /settings/<key>` while
the app runs; the vocabulary is `db/settings.py REGISTRY`
(`models_dir`, `semantic_model`, `worker`, `faiss_gpu`,
`dupe_threshold`, `dupe_dhash_verify`, `thumbnail_precache`,
`ort_providers`, `face_backend`, `caption_model`). No environment variables.

## Develop

```
just check          ruff, sglint, format, pyright, repo hygiene -- no tests
just test           the fast tests, one module per worker (~20s)
just test-slow      the tests marked slow (real libraries, real browsers), four at a time (~30s)
just check-all      the gate plus both test lanes
just audit          sglint code rules, `--repo` hygiene, the linter self-tests -- seconds
just serve          the app
just bench          faces-validate, browser-report: the measured evidence behind the docs
just faiss-verify   which faiss build the process loads
```

`sglint` is this repository's own linter for rules no stock tool holds:
SQL built from structure only, the schema contract (STRICT, foreign
keys, migration steps), adapters that own no semantics, and no test
that starts a program.

## Docs

- `docs/ARCHITECTURE.md` -- the process, the shell, the feed, the one negotiation
- `docs/AI_MODELS.md` -- which weights, where they live, how they arrive
- `docs/FACE_CLUSTERING.md` -- the face pipeline and its measured thresholds
- `docs/SIMILARITY_ENGINE.md` -- why similarity runs on FAISS over SQLite blobs
- `docs/FAISS_GPU_WINDOWS.md` -- building the vendored GPU FAISS

## License

MIT. Forked from biagiomaf/smart-comfyui-gallery; the schema, the
application and the tests here are a rewrite.
