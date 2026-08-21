"""The gallery grid and its rail: presentation adapters over db/resultset.py.

Every route here maps request-shaped inputs onto ONE GalleryQuery and
reads the module's materialized answer. No SQL, no sorting, no
membership opinions -- a route that grew any of those would be the
second notion of truth the ResultSet contract exists to forbid.

The URL owns canonical query state: `/g?q=...&folder=...&sort=...&page=N`
renders complete from cold, so reload, back, forward and bookmarks all
work without client state. The fragments and JSON exist for the parts a
running page swaps live -- the grid on a page change, the rail popover
on hover -- and answer from the same query the shell was drawn from.
"""

from __future__ import annotations

import pathlib
import time

from litestar import get
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.response import Template

from db import connect, naming, resultset, settings
from sg_web import home
from sg_web.asking import gallery_query as _asked


def _grid_context(state: State, query: resultset.GalleryQuery, page: int) -> dict:
    conn = connect.connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            shape = resultset.page(conn, weights, query, page, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()  # a semantic answer may have minted registry rows
    finally:
        connect.close(conn)
    provenance = shape["provenance"] or {}
    return {
        "items": shape["items"],
        "page": shape["page"],
        "pages": shape["pages"],
        "total": shape["total"],
        "size": query.size,
        "sort": query.sort,
        "currency": shape["currency"],
        "missing_spaces": provenance.get("missing") or {},
        "q": query.text or "",
        "folder": query.folder or "",
        "album": query.album or "",
        "person": query.person or "",
        "kind": query.kind or "",
        "qs": shape["qs"],
        "kinds": resultset.KINDS,
        "sorts": resultset.SORTS,
    }


@get("/g", sync_to_thread=True)
def gallery(
    state: State,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
    page: int = 1,
) -> Template:
    """The gallery, whole, from nothing but the URL."""
    query = _asked(folder, album, kind, q, sort, size, person=person)
    return Template(template_name="gallery.html", context=_grid_context(state, query, page))


@get("/g/grid", sync_to_thread=True)
def grid_fragment(
    state: State,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
    page: int = 1,
) -> Template:
    """One page of cells, for the running page to swap in place."""
    query = _asked(folder, album, kind, q, sort, size, person=person)
    return Template(template_name="_grid.html", context=_grid_context(state, query, page))


@get("/g/peek", sync_to_thread=True)
def rail_peek(
    state: State,
    page: int,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
    count: int = resultset.PEEK_MOST,
    expect: str | None = None,
) -> dict:
    """The rail popover: real members of exactly the page a jump would
    land on, never a guess from scroll height.

    `expect` is the currency of the grid the rail is drawn beside. When
    the library has moved on, a preview from the NEW ordering shown
    beside the OLD grid would present two generations as one answer --
    the response is 409 and the client redraws instead of pretending.
    """
    query = _asked(folder, album, kind, q, sort, size, person=person)
    conn = connect.connect(state.db_path)
    try:
        if expect is not None and resultset.currency(conn) != expect:
            raise HTTPException(status_code=409, detail="the result set has changed; redraw the gallery")
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            told = resultset.peek(conn, weights, query, page, time.time(), count=count)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return told
    finally:
        connect.close(conn)


@get("/g/locate/{slug:str}", sync_to_thread=True)
def locate_in_answer(
    state: State,
    slug: str,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
) -> dict:
    """Where one picture sits in this answer -- ordinal, page, and its
    previous/next in ANSWER order, which is what the arrows mean while
    a result set is being walked."""
    query = _asked(folder, album, kind, q, sort, size, person=person)
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "file", slug)
        if found is None:
            raise NotFoundException(f"no file at /i/{slug}")
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            told = resultset.locate(conn, weights, query, found[0], time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        conn.commit()
        if told is None:
            return {"in_answer": False}
        return {"in_answer": True, **told}
    finally:
        connect.close(conn)
