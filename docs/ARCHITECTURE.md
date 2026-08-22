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
timeline, operations) and mounts the activity surface through the
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

`/operations` is a Litestar `Router`: the page, url-encoded forms for
roots, scan, settings and clustering choice, and one `POST
/operations/jobs/{kind}` per sweep. Each form receives its section back
as a fragment with a notice swapped out-of-band. The JSON routes in
`app.py` keep serving machines unchanged. The gallery header carries no
operational control.

## Coverage

`tests/test_the_shell_mounts_every_surface.py::SURFACES` names every
HTML-rendering handler, its owning Module and its browser door;
`HEADLESS` names the rest with a reason. The sweep fails on a handler
missing from both.
