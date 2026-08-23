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

`base.html` renders navigation (gallery, people, places, albums,
folders, timeline, stories, operations) and mounts the activity surface through the
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
job row. The ingest, phash, faces, embed, annotate and context sweeps
queue only what is missing and answer 204 (the console: "nothing to
do") when nothing is; `everything` -- the console's "again" button --
redoes all of it. The console shows beside each such button how many
present files the sweep would still queue (`inspecting.coverage`),
counted by the same predicate the sweep uses. Each reads its own record
of having been done against the file's CURRENT bytes:

```
ingest    file.ingested_sha256  the bytes the last metadata read was of
phash     derived_file_hash     space = current PHASH space, source_sha256
faces     derived_face_scan     any model's pass, source_sha256
embed     derived_embedding     space = the checkpoint the cache pins, source_sha256
annotate  derived_annotation    kind caption, model = caption_model, source_sha256
context   derived_media_context policy_version = context.POLICY_VERSION
```

Verify reads every present file by design: checking bytes against
their recorded hash is the whole job. Bytes that rot behind the
scanner's back are caught by a scan (which records the new hash) or by
"again"; a scan or ingest that changes a source claim stales the
interpretation (db/context.py `stale`), which is what puts a file back
into the context sweep.

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

## Collections

A listed collection's membership is its filed rows; a smart collection's
is a typed rule (`db/collection_rules.py`) evaluated through the same
ResultSet every gallery question goes through. The rule is authored from
a GalleryQuery (`from_gallery_query`) -- the gallery's save-view button
sends the mounted answer's canonical parameters, every repeated `f`
included, and never a rule shape -- and read back per version:

```
v1  folder, person, kind, favorite, rating_min; select sort/text/take
v2  + artifact (a checkpoint, LoRA or workflow, by entity uuid)
v3  + facets (registered metadata predicates, db/facets.py, by spelling)
```

Reading is fail-closed: an unknown key, a missing key, a facet whose
key this build no longer registers, or a version stamp that does not
equal its column is `BrokenCollectionRule`, never an evaluated empty
collection. Entity references are uuids, so a rename moves the address
and not the membership. A rule never holds `event.id`: a session is a
run's hypothesis over one interpretation and would answer nothing the
day the runs regroup -- save its day or moment window instead. The
collection's page names the rule's words and opens the same question
in the gallery.

## Places

A place is an entity (`place`, nested by kind: country .. poi). Nothing
resolves coordinates to one -- no gazetteer ships, and GPS alone names
no place -- so `derived_media_context.place_id` comes from a person's
word: `POST /i/{slug}/place {"name", "kind"}` finds or mints the place
by name, kind and parent (`places.named`, under the writer lane; the
`place_identity` index is the database's own word that there is one
Lisbon), records the claim as authored
desired state (`file_place`, one per file, `authored.set_place`) and
re-interprets the file at once, so the context reads
`location_basis = 'authored'`. The claim survives every rebuild. A
picture's page says where, or "nowhere said", and offers the form; a
selection can be placed at once (`POST /g/selection/place`); `/places`
lists every place named with its count and span; the `place.id` facet
is the door to everything there (its chip says the name); a session
whose placed members agree carries the place on its card; a person's,
a folder's and an album's page say where their pictures are; a story's
`located` claim says it in words.

## Timeline

`/timeline` draws the library over the human moment
(`context.HUMAN_MOMENT`: the wall clock where one is claimed, the
instant otherwise) at a zoom the URL owns (`bin`, `start`, `end`):
pictures per bin split by clock domain and origin, claims too coarse
for the bin as spans, thumbnails per bin while the bins are few, and
the sessions touching the range (`db/pages.py TIMELINE_*`,
`SESSION_*`). Any gallery question scopes the whole surface --
`?folder=`, `?album=`, `?person=`, `?kind=`, `?f=` -- through
`resultset.scope_of`: the same membership predicates the gallery walks,
appended to every timeline statement (HEAD + conjunct + TAIL), and
every door is that question plus a moment and the precision the bar
counted (`context.granule`), so what is drawn is what opens. A session is a door, never a scope; a rule-defined collection
scopes through its materialized membership (`f.id IN (...)`, the one
engine) and refuses with why when its rule cannot be answered. A
person's, a folder's and an album's page open their own timeline, and
the gallery opens its question on the timeline. Every bin, span and session is a door into the gallery
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
plan grammar is versioned and frozen (`STORY_PLAN_V1..V7`), a stored
document is judged by its own version forever. v6 added `seen`: what a
captioning model said about a phase's members, read from the captions
the snapshot froze -- every planner emits it, the renderer quotes one
sentence and names the models under the technical profile. v7 added
`located`: where a phase's members happened, from the place each
member froze (a person's word), named in the render. `story_renderers` words
every claim kind through a closed registry; a render cites what
supports it and `violations()` proves the chain on every read. The
timeline offers the whole chain behind one button per session.

## Captions

The `annotate` job (`db/runner.py submit_annotate`, `POST /jobs/annotate`)
runs the BLIP checkpoint the `caption_model` setting names
(`vision/captions.py`) over every present picture, and over every
video's poster frame and each of its sampled moments (`db/sample.py`,
the rows detection looks at), and writes `derived_annotation` rows
through `derived.annotate`:
one caption per model per file, the same model replacing its own, two
models kept side by side. `annotation_fts` indexes the text; retrieval
fuses a bm25 ranking named `captions` with the semantic spaces once any
caption exists (`derived.rank_by_annotation`). Surfaces: the media page
and lightbox say what was said and by which model; the grid says it on
hover; a snapshot freezes every caption its members carry, the plan's
`seen` claim cites them, and a story page also shows a hero's CURRENT
caption beside the frozen name, labelled as today's.

## Coverage

`tests/test_the_shell_mounts_every_surface.py::SURFACES` names every
HTML-rendering handler, its owning Module and its browser door;
`HEADLESS` names the rest with a reason. The sweep fails on a handler
missing from both.
