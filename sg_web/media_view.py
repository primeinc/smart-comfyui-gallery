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

import dataclasses
import pathlib
import time

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import HTTPException, NotFoundException
from litestar.response import Redirect, Response, Template

from db import authored, connect, derived, naming, pages, resultset, settings
from db.resultset import canonical
from sg_web import home
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import presented


def view(
    conn, models_dir: str, file_id: int, slug: str, query: resultset.GalleryQuery, now: float, actor_id: int
) -> dict:
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
        found = resultset.locate(conn, models_dir, query, file_id, now, actor_id=actor_id)
        if found is not None:
            generation, asked, answer = found["currency"], found["qs"], found["answer"]
        else:
            shape = resultset.describe(conn, models_dir, query, now, actor_id=actor_id)
            generation, asked, answer = shape["currency"], shape["qs"], shape["answer"]
        told = pages.picture(conn, file_id)
        if told is None:
            raise NotFoundException(f"/i/{slug} has no file row")
        made = _assembled(conn, file_id, slug, found, generation, asked, told)
        made["context"]["answer"] = answer
        # The actor's authored facts ride the SAME snapshot as everything
        # else -- a favorite committed mid-request must not appear under
        # the metadata of the generation before it.
        made["authored"] = dataclasses.asdict(authored.media_state(conn, file_id, actor_id))
        return made


def _assembled(conn, file_id: int, slug: str, found, generation: str, asked: str, told) -> dict:
    name, folder, width, height, duration, asked_w, checkpoint, missing_since, prompt, seed, fields, kind = told
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
        "when": _when(conn, file_id),
        "said": derived.said_about(conn, file_id),
        "faces": _faces(conn, file_id),
        "context": context,
    }


def _faces(conn, file_id: int) -> dict:
    """Who is in the picture, by the primary clustering, and whether any
    detector has looked at its current bytes -- so the page can tell
    "nobody here" from "nobody looked"."""
    return {
        "people": [
            {"slug": slug, "name": name, "href": f"/p/{slug}", "faces": int(count)}
            for slug, name, count in pages.media_people(conn, file_id)
        ],
        "looked": [
            {"model_id": model_id, "model_version": version, "faces": int(faces), "at": at}
            for model_id, version, faces, at in pages.media_face_scans(conn, file_id)
        ],
    }


def _when(conn, file_id: int) -> dict | None:
    """The picture's place on the human timeline, with the evidence
    behind it (db/pages.py MEDIA_WHEN), and the current sessions it
    belongs to -- each a door to the timeline, the day, the session's
    pictures and its story. None while the file is uninterpreted."""
    import json
    import urllib.parse

    from db import facets

    row = pages.media_when(conn, file_id)
    if row is None:
        return None
    local_at, instant_at, tz, basis, certainty, supports, conflicts, precision, origin, moment, local_day = row
    day_qs = urllib.parse.urlencode(
        [("f", facets.spell(facets.facet("context.local_day", "eq", local_day))), ("sort", "moment")]
    )
    return {
        "moment": moment,
        "local_at": local_at,
        "instant_at": instant_at,
        "tz_offset_min": tz,
        "domain": "wall" if local_at is not None else "instant",
        "precision": precision,
        "basis": basis,
        "certainty": certainty,
        "supports": json.loads(supports) if supports else [],
        "conflicts": json.loads(conflicts) if conflicts else [],
        "origin": origin,
        "local_day": local_day,
        "day_qs": day_qs,
        "timeline": "/timeline?"
        + urllib.parse.urlencode(
            {"bin": "hour", "start": int(moment // 86400) * 86400, "end": int(moment // 86400) * 86400 + 86400}
        ),
        "sessions": [
            {
                "id": event_id,
                "kind": kind,
                "start": start,
                "end": end,
                "pictures": pictures,
                "qs": urllib.parse.urlencode(
                    [("f", facets.spell(facets.facet("event.id", "eq", str(event_id)))), ("sort", "moment")]
                ),
                "story": f"/stories/renders/{render_id}" if render_id is not None else None,
            }
            for event_id, kind, start, end, pictures, render_id in pages.media_sessions(conn, file_id)
        ],
    }


@get("/i/{slug:str}", sync_to_thread=True)
def media_page(
    state: State,
    request: Request,
    slug: str,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    artifact: str | None = None,
    kind: str | None = None,
    favorite: str | None = None,
    rating_min: int | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
) -> Template | Response | Redirect:
    """One media item at its address, presented for whoever is asking.

    The overlay's currency expectation arrives OUT-OF-BAND in the
    `X-SG-Expect` header -- never in the URL, which stays the canonical
    context the browser may push, share or reload."""
    query = _asked(
        folder, album, kind, q, sort, size, person=person, artifact=artifact, favorite=favorite, rating_min=rating_min
    )
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
        told = view(conn, weights, file_id, slug, query, time.time(), state.actor_id)
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
    return presented(request, told, page="media.html", fragment="_media_lightbox.html", name="item")
