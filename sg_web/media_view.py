"""One media item, one address, three ways of looking at it.

`/i/{slug}` is the item's ONLY address, whatever its kind -- image,
animated image, video, audio, document are peers here exactly as they
are in the ResultSet; a picture-shaped module would reintroduce the
special-case worldview the schema deleted. The path is the entity; the
query string is the browsing context -- the GalleryQuery the viewer is
walking, carried whole so a reload, a pasted link or a fresh tab can
reconstruct previous/next without any client state owning what "next"
means.

The presentations of that one address, negotiated deterministically
(and declared with `Vary: Accept, HX-Request`, because caches punish
ambiguity):

    Accept names application/json      -> the MediaView itself
    else HX-Request: true              -> the lightbox fragment
    else Accept names text/html        -> the full detail page
    else (wildcard, machine default)   -> the MediaView itself

All from ONE view assembly. There is no /lightbox/ URL and no second
definition of the item.

Previous and next come from `resultset.locate` -- position in the
answer being walked, never a folder walk: a bare `/i/{slug}` uses the
default GalleryQuery (the whole library, newest first), so the arrows
always mean something and always mean what the grid meant. The context
carries the ResultSet currency as concurrency evidence and the
computed return-to-results URL, so closing a directly-opened item goes
back to its results page instead of wherever the browser had been.
The `X-SG-Expect` header works as the rail's `expect` does: a context
from a superseded generation is refused with 409, never silently
swapped under the gallery still on screen.
"""

from __future__ import annotations

import pathlib
import time

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import HTTPException, NotFoundException
from litestar.response import Redirect, Response, Template

from db import connect, naming, pages, resultset, settings
from sg_web import home
from sg_web.gallery import _asked, canonical

VARIES = {"vary": "Accept, HX-Request"}


def view(conn, models_dir: str, file_id: int, slug: str, query: resultset.GalleryQuery, now: float) -> dict:
    """The MediaView: everything every presentation shows, assembled
    once, inside ONE database snapshot. Keys carried by the old JSON
    page keep their names, so the machine consumers keep their
    assertions.

    Ordering inside the snapshot is load-bearing: the ResultSet context
    comes FIRST, because its currency read (the monitor connection)
    must precede the read that pins this connection's snapshot -- a
    metadata read first would pin the snapshot before the currency was
    taken, and a commit in the gap would label pre-commit data with a
    post-commit currency: exactly the mislabeling the caller's 409
    comparison exists to catch. The context's currency is therefore
    always present -- from `locate` when the item is in the answer,
    from `describe` (same projection, same snapshot) when it is not.
    """
    with resultset.snapshot(conn):
        found = resultset.locate(conn, models_dir, query, file_id, now)
        generation = (
            found["currency"] if found is not None else resultset.describe(conn, models_dir, query, now)["currency"]
        )
        told = pages.picture(conn, file_id)
        if told is None:
            raise NotFoundException(f"/i/{slug} has no file row")
        return _assembled(conn, file_id, slug, query, found, generation, told)


def _assembled(conn, file_id: int, slug: str, query, found, generation: str, told) -> dict:
    name, folder, width, height, duration, asked_w, checkpoint, missing_since, prompt, seed, fields, kind = told
    asked = canonical(query)
    back = f"/g?{asked}" if asked else "/g"
    if found is not None and found["page"] > 1:
        back += ("&" if asked else "?") + f"page={found['page']}"
    context = {
        "qs": asked,
        "in_answer": found is not None,
        "return_url": back,
        "currency": generation,
        **({k: found[k] for k in ("ordinal", "page", "total")} if found else {}),
    }
    return {
        "slug": slug,
        "name": name,
        "folder": folder,
        "kind": kind,
        "present": missing_since is None,
        "width": width,
        "height": height,
        "duration": duration,
        "asked_for_width": asked_w,
        "checkpoint": checkpoint,
        "loras": pages.file_loras(conn, file_id),
        "prompt": prompt,
        "seed": seed,
        "fields": fields,
        "params": [
            {"source": source, "key": key, "value": value} for source, key, value in pages.fields_of(conn, file_id)
        ],
        "copies": [
            {"slug": s, "name": n, "distance": d, "is_best": bool(b)} for s, n, d, b in pages.dupe_copies(conn, file_id)
        ],
        "previous": found["previous"] if found else None,
        "next": found["next"] if found else None,
        "parents": [{"slug": s, "name": n, "kind": k} for s, n, k in pages.parents(conn, file_id)],
        "children": [{"slug": s, "name": n, "kind": k} for s, n, k in pages.children(conn, file_id)],
        "context": context,
    }


@get("/i/{slug:str}", sync_to_thread=True)
def media_page(
    state: State,
    request: Request,
    slug: str,
    folder: str | None = None,
    album: str | None = None,
    kind: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
) -> Template | Response | Redirect:
    """One media item at its address, presented for whoever is asking.

    The overlay's currency expectation arrives OUT-OF-BAND in the
    `X-SG-Expect` header -- never in the URL, which stays the canonical
    context the browser may push, share or reload."""
    query = _asked(folder, album, kind, q, sort, size)
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "file", slug)
        if found is None:
            raise NotFoundException(f"no file at /i/{slug}")
        file_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, file_id)
            if live is not None:
                # The context survives the rename: the address moved, the
                # walk the viewer was on did not.
                asked = canonical(query)
                return Redirect(path=f"/i/{live[1]}" + (f"?{asked}" if asked else ""), status_code=301)
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        told = view(conn, weights, file_id, slug, query, time.time())
        conn.commit()  # a semantic context may have minted registry rows
        # Compared AFTER assembly, against the currency the view was
        # actually located in: a commit landing mid-request would
        # otherwise pass a pre-assembly check and hand back arrows from
        # a newer answer under the old mounted gallery.
        expected = request.headers.get("x-sg-expect")
        if expected is not None and told["context"]["currency"] != expected:
            raise HTTPException(status_code=409, detail="the result set has changed; redraw the gallery")
    finally:
        connect.close(conn)
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return Response(told, headers=VARIES)
    if request.headers.get("hx-request") == "true":
        return Template(template_name="_media_lightbox.html", context={"item": told}, headers=VARIES)
    if "text/html" in accept:
        return Template(template_name="media.html", context={"item": told}, headers=VARIES)
    return Response(told, headers=VARIES)
