"""One folder, one address: the physical axis rendered as an entity.

`/f/{slug}` is the folder's only address, with the 301 contract every
entity address carries. The FolderView owns IDENTITY and HIERARCHY --
the name, the breadcrumb walked by parent, the immediate child folders,
the presence state -- and never the media answer: the pictures are ONE
ResultSet page of the folder-faceted GalleryQuery, the same membership
`/g?folder=` serves, so order, paging, arrows and semantics cannot fork
between the entity page and the gallery.

`folder=` means the folder ITSELF -- direct children only, exactly the
predicate db/resultset.py runs. Subfolders are listed as entities to
navigate into, never silently folded into the answer; a recursive
subtree question would be its own spelled facet with its own semantics.

Presence is a state, never an inference: a folder whose directory was
not found where it was last seen renders and says "missing"; a folder
whose root is unreachable right now is "offline"; only an address
nothing lives at is a 404. The disk path stays server-side -- durable
identity is the slug and the parent chain, not a host path.
"""

from __future__ import annotations

import os.path
import pathlib
import time

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import NotFoundException
from litestar.response import Redirect, Response, Template

from db import connect, library, naming, pages, resultset, settings
from sg_web import home
from sg_web.presenting import VARIES, wants_json


@get("/folders", sync_to_thread=True)
def folders_index(state: State, request: Request) -> Template | Response:
    """Where physical navigation enters: each registered root as a shelf
    -- its kind, whether it is reachable RIGHT NOW (db/library.py
    check_roots, which verifies the marker rather than trusting a
    directory to be the same one), and its depth-0 folder entities to
    walk into. No root ids and no host paths: addresses are slugs, and
    the operational /roots route keeps the management shape."""
    conn = connect.connect(state.db_path)
    try:
        online = {root_id: reachable for root_id, _, reachable in library.check_roots(conn)}
        told = [
            {
                "kind": kind,
                "online": online.get(root_id, False),
                "folders": [{"slug": s, "name": n, "pictures": p} for s, n, p in pages.folder_tops(conn, root_id)],
            }
            for root_id, kind in pages.roots_shelf(conn)
        ]
    finally:
        connect.close(conn)
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return Template(template_name="folders.html", context={"shelves": told}, headers=VARIES)
    return Response(told, headers=VARIES)


def view(conn, models_dir: str, folder_id: int, slug: str, now: float, *, legacy: bool) -> dict:
    """The FolderView, assembled inside ONE database snapshot -- a scan
    or a move committing between the reads must not hand back a grid
    from one generation under the hierarchy of another. The ResultSet
    page comes FIRST, so its currency is read before the snapshot pins.

    The unbounded legacy `files` list is the machine Adapter's shape
    only; the rendered page never enumerates a directory to draw a
    bounded grid."""
    query = resultset.parse(folder=slug)
    with resultset.snapshot(conn):
        grid = resultset.page(conn, models_dir, query, 1, now)
        name, _parent_id, missing_since, root_path = pages.folder_card(conn, folder_id)
        crumbs = []
        for crumb_id, crumb_name in pages.breadcrumb(conn, folder_id):
            addressed = naming.entity_slug(conn, crumb_id)
            crumbs.append({"name": crumb_name, "slug": addressed[1] if addressed else None})
        if missing_since is not None:
            state = "missing"
        elif os.path.isdir(root_path):
            state = "present"
        else:
            state = "offline"
        told = {
            "slug": slug,
            "name": name,
            "state": state,
            "breadcrumb": crumbs,
            "folders": [{"slug": s, "name": n, "pictures": p} for s, n, p in pages.folder_children(conn, folder_id)],
            "count": grid["total"],
            "gallery": {
                "items": grid["items"],
                "total": grid["total"],
                "pages": grid["pages"],
                "qs": grid["qs"],
            },
        }
        if legacy:
            told["files"] = [{"slug": s, "name": n} for s, n in pages.folder_files(conn, folder_id)]
        return told


@get("/f/{slug:str}", sync_to_thread=True)
def folder_page(state: State, request: Request, slug: str) -> Template | Response | Redirect:
    """One folder at its address, presented for whoever is asking. A
    retired slug redirects to the live one."""
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "folder", slug)
        if found is None:
            raise NotFoundException(f"no folder at /f/{slug}")
        folder_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, folder_id)
            if live is not None:
                return Redirect(path=f"/f/{live[1]}", status_code=301)
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        told = view(conn, weights, folder_id, slug, time.time(), legacy=wants_json(request))
    finally:
        connect.close(conn)
    if wants_json(request):
        return Response(told, headers=VARIES)
    return Template(template_name="folder.html", context={"folder": told}, headers=VARIES)
