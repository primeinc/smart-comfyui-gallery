# Architecture

The application is `sg_web` over `db`: a server-rendered Litestar
application, one process, SQLite as the only truth. There is no other
runtime.

```
sg_web/app.py        Litestar: routes, lifespan, the worker thread, ChannelsPlugin
sg_web/*_view.py     one Module per address: assemble a view, negotiate its shape
sg_web/presenting.py the one negotiation: JSON / htmx fragment / full page
sg_web/templates/    Jinja: base.html is the shell; pages extend it; _*.html are fragments
sg_web/activity.py   the activity surface: persisted jobs rendered, deltas as oob fragments
sg_web/operations.py the runtime's page, one Router under /operations
sg_web/submitting.py what every job submit does after commit: announce, wake
sg_web/worker.py     the in-process worker thread; publishes every change
db/                  schema, queries, jobs, runner -- every fact lives here
```

## Stack

| Concern | Owner |
|---|---|
| routing, lifecycle, DI | Litestar 2.24 |
| rendered presentation | Jinja, one engine (`TemplateConfig`), `base.html` inheritance |
| server-driven interaction | htmx 2.0.7 (`hx-get`/`hx-post`), `htmx-ext-ws` 2.0.3 for the feed |
| interactions HTML cannot express | small vanilla JS per page (`static/*.js`) |
| realtime transport | `ChannelsPlugin` + `MemoryChannelsBackend`, channel `jobs` |
| truth | SQLite (`db/schema.sql`); job rows in `job` / `job_item` |

## Representation

`presenting.py` decides once, for every entity address:

```
Accept: application/json   -> the view (JSON)
HX-Request: true           -> the fragment (lightbox, drawer, grid)
Accept: text/html          -> the page, extending base.html
otherwise                  -> JSON
```

Every response carries `Vary: Accept, HX-Request`.

## Shell

`base.html` renders navigation (gallery, people, albums, folders,
timeline, stories, operations) and mounts the activity surface through the
`activity()` Jinja global installed by `sg_web/activity.py` on the
application's engine (`engine_callback`). Story and evolution pages
render through `story_view._story_env` (StrictUndefined) and pass
`request` so the same shell works there.

## Realtime

```
worker / submit  --publish-->  channel "jobs"  --/ws/jobs-->  subscriber
                                                   |
                              subscribe FIRST, then read the job rows,
                              send the snapshot, then every delta
```

`/ws/jobs` sends JSON: `{"type":"snapshot","jobs":[...]}` then
`{"job","kind","state","done","total"}` per change. `/ws/jobs?as=html`
sends the same feed rendered: `_activity_list.html` (oob replacement of
`#activity-jobs`), then `_job.html` per delta -- appended when the
connection has not shown the job, replaced by id otherwise. The htmx ws
extension swaps them; no JavaScript holds job state.

A submit announces its committed `queued` row (`submitting.submitted`);
the worker announces `running` per item and the terminal state. Every
delta describes committed rows. Reconnection re-reads the rows.

**One process.** `MemoryChannelsBackend` fans out inside the process
that published. `python -m sg_web` starts one uvicorn process with the
worker thread inside it; `uvicorn.run` is never given `workers`. Splitting
publisher and subscribers across processes requires a broker-backed
Channels backend first (`litestar.channels.backends.redis` /
`.asyncpg` / `.psycopg`). `tests/test_the_shell_mounts_every_surface.py`
pins this.

## Operations

`/operations` is a Litestar `Router`: the console, url-encoded forms for
roots, scan, settings and clustering choice, and one `POST
/operations/jobs/{kind}` per sweep. Each form receives its section back
as a fragment with a notice swapped out-of-band. The JSON routes in
`app.py` keep serving machines unchanged. The gallery header carries no
operational control.

### Sweeps

A sweep is `POST /jobs/<kind>` or its console button; every sweep is a
job row. The phash, faces, embed, annotate and context sweeps queue
only what is missing and answer 204 (the console: "nothing to do") when
nothing is; `everything` redoes all of it. Each reads its own record of
having been done against the file's CURRENT bytes:

```
phash     derived_file_hash     space = current PHASH space, source_sha256
faces     derived_face_scan     any model's pass, source_sha256
embed     derived_embedding     space = the checkpoint the cache pins, source_sha256
annotate  derived_annotation    kind caption, model = caption_model, source_sha256
context   derived_media_context policy_version = context.POLICY_VERSION
```

Ingest and verify read every present file by design: there is no
record of a read that would make skipping honest. A scan or ingest that
changes a source claim stales the interpretation (db/context.py
`stale`), which is what puts a file back into the context sweep.

### The console

```
job + job_item        current truth        db/jobs.py
job_event             historical ledger    db/ledger.py  append-only, monotonic id, never sampled
channel "events"      transport            sg_web/app.py publish_event -> /ws/events
```

Three depths of the same rows: `jobs.active` (the shell's list),
`jobs.snapshot` (the ordinary client), `db/inspecting.py` (the console:
every column of the row, items paged, ledger paged, derived numbers
that say what they derive from). Payloads are redacted by key at the
read model.

Every transition is a typed row (`db/ledger.py TYPES`, mirrored by the
schema CHECK): submit, claim/reclaim, pause, cancel asked/cancelled,
done/failed, item started/done/failed, handler phases and observations,
checkpoint, worker turn crashed. `sg_web/console.py` renders every type
to words; the contract test fails on a type without a renderer.

Handlers report phases through `db/runner.py report()` -- `phase`,
`progress`, `observe` -- a seam that knows no web. A report is spoken
at once as a `pending` frame (no id; presentation) and written to the
ledger at the item boundary in the commit that settles the item, so a
failed item's phases survive its rollback.

`/ws/events?after=N` subscribes first, then sends `backlog` frames from
the rows, then `event` and `pending` frames. The client holds every
event it is given, renders a window, and fetches `GET /operations/events`
for any id it finds skipped. Pause and filters touch painting only.

An item failure (`item.failed`, the job continues) and a worker defect
(`worker.turn_failed`, traceback, lease lapses, reclaimable) are
distinct conditions on every surface.

## Timeline

`/timeline` draws the library over the human moment
(`context.HUMAN_MOMENT`: the wall clock where one is claimed, the
instant otherwise) at a zoom the URL owns (`bin`, `start`, `end`):
pictures per bin split by clock domain and origin, claims too coarse
for the bin as spans, thumbnails per bin while the bins are few, and
the sessions touching the range (`db/pages.py TIMELINE_*`,
`SESSION_*`). Every bin, span and session is a door into the gallery
through the facets (`context.local_day`, `context.moment`,
`context.origin`, `context.disputed`, `event.id`) ordered by `moment`.
A session card names who is in it (primary clustering), carries a
bounded strip of its pictures, and offers the story chain behind one
button. The people, folders and albums shelves carry the span of their
pictures' moments so the same axis reads across the product.

## Stories

```
derived_event  --freeze-->  story_snapshot  --plan-->  story_plan  --render-->  story_render
                            db/stories.py              db/planning.py          db/rendering.py
```

One planner per session kind -- `generation_history`, `capture_history`,
`file_history` -- each reading only the evidence its kind carries; the
plan grammar is versioned and frozen (`STORY_PLAN_V1..V5`), a stored
document is judged by its own version forever. `story_renderers` words
every claim kind through a closed registry; a render cites what
supports it and `violations()` proves the chain on every read. The
timeline offers the whole chain behind one button per session.

## Captions

The `annotate` job (`db/runner.py submit_annotate`, `POST /jobs/annotate`)
runs the BLIP checkpoint the `caption_model` setting names
(`vision/captions.py`) over every present picture and video's poster
frame and writes `derived_annotation` rows through `derived.annotate`:
one caption per model per file, the same model replacing its own, two
models kept side by side. `annotation_fts` indexes the text; retrieval
fuses a bm25 ranking named `captions` with the semantic spaces once any
caption exists (`derived.rank_by_annotation`). Surfaces: the media page
and lightbox say what was said and by which model; the grid says it on
hover; a story page shows a hero's current caption beside the frozen
name, labelled as today's and never part of the snapshot.

## Coverage

`tests/test_the_shell_mounts_every_surface.py::SURFACES` names every
HTML-rendering handler, its owning Module and its browser door;
`HEADLESS` names the rest with a reason. The sweep fails on a handler
missing from both.
