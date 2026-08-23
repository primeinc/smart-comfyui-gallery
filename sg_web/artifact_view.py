"""One artifact, one address: the recipe axis rendered as an entity.

An artifact describes a media predicate; the ResultSet owns the ordered
media answer to it; this view owns the artifact -- its identity, its
facts (name, kind, architecture, the two hash claims, first sighting),
its LoRA synergy when that applies, and ONE ResultSet page of
`artifact={slug}`: the same membership `/g?artifact=` serves, the same
answer identity, never a private media list. Whether membership means
`file_artifact` or `generation.workflow_id` is the ResultSet's private
knowledge; nothing here knows the difference.

Three shelf addresses, one view: /m, /l and /w are thin adapters, and
the canonical URL is computed ONCE from the resolved entity, its live
slug and its own kind -- a retired slug on the wrong shelf reaches the
canonical artifact page in exactly one 301, never a redirect chain.
"""

from __future__ import annotations

import pathlib
import time

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import NotFoundException
from litestar.params import FromPath
from litestar.response import Redirect, Response, Template

from db import connect, naming, pages, resultset, settings
from sg_web import home
from sg_web.presenting import presented_page

#: Which shelf each addressable artifact kind lives on. Kinds outside
#: this map have rows and identity but no page yet.
_SHELVES = {"checkpoint": "/m", "lora": "/l", "workflow": "/w"}


def view(conn, models_dir: str, artifact_id: int, slug: str, now: float) -> dict:
    """The ArtifactView, assembled inside ONE database snapshot: the
    artifact's facts and its first ResultSet page describe the same
    generation of the library. `count` IS the ResultSet total -- there
    is no second arithmetic for a shelf to disagree with."""
    with resultset.snapshot(conn):
        grid = resultset.page(conn, models_dir, resultset.parse(artifact=slug), 1, now)
        name, kind, architecture, sha, quoted, first_seen = pages.artifact_card(conn, artifact_id)
        told = {
            "slug": slug,
            "name": name,
            "kind": kind,
            "architecture": architecture,
            "content_sha256": sha,
            "quoted_hash": quoted,
            "first_seen_at": first_seen,
            "count": grid["total"],
            "gallery": {
                "items": grid["items"],
                "total": grid["total"],
                "pages": grid["pages"],
                "qs": grid["qs"],
                "answer": grid["answer"],
                "currency": grid["currency"],
            },
        }
        if kind == "lora":
            told["used_with"] = [
                {"name": n, "slug": s, "together": together} for n, s, together in pages.lora_synergy(conn, artifact_id)
            ]
        return told


def _artifact_page(state: State, request: Request, slug: str, shelf: str) -> Template | Response | Redirect:
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "artifact", slug)
        if found is None:
            raise NotFoundException(f"no artifact at {shelf}/{slug}")
        artifact_id, is_current = found
        live = slug
        if not is_current:
            fresh = naming.entity_slug(conn, artifact_id)
            if fresh is not None:
                live = fresh[1]
        kind = pages.artifact_card(conn, artifact_id)[1]
        home_shelf = _SHELVES.get(kind)
        if home_shelf is None:
            raise NotFoundException(f"a {kind} has no page yet")
        if home_shelf != shelf or not is_current:
            # The canonical address, computed once: one redirect covers a
            # retired slug, a wrong shelf, and both at once.
            return Redirect(path=f"{home_shelf}/{live}", status_code=301)
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        told = view(conn, weights, artifact_id, slug, time.time())
    finally:
        connect.close(conn)
    return presented_page(request, told, page="artifact.html", context={"artifact": told, "shelf": shelf})


@get("/m/{slug:str}", sync_to_thread=True)
def model_page(state: State, request: Request, slug: FromPath[str]) -> Template | Response | Redirect:
    return _artifact_page(state, request, slug, "/m")


@get("/l/{slug:str}", sync_to_thread=True)
def lora_page(state: State, request: Request, slug: FromPath[str]) -> Template | Response | Redirect:
    return _artifact_page(state, request, slug, "/l")


@get("/w/{slug:str}", sync_to_thread=True)
def workflow_page(state: State, request: Request, slug: FromPath[str]) -> Template | Response | Redirect:
    return _artifact_page(state, request, slug, "/w")
