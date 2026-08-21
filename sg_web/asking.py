"""Request-shaped values into the GalleryQuery Interface, once.

db/resultset.py owns what a question means and how it is spelled; this
is the one Litestar-facing translation of request values and errors
into that Interface, so no presentation Adapter depends on another to
understand a GalleryQuery.
"""

from __future__ import annotations

from litestar.exceptions import ClientException

from db import resultset


def gallery_query(
    folder: str | None,
    album: str | None,
    kind: str | None,
    q: str | None,
    sort: str | None,
    size: int | None,
    person: str | None = None,
    artifact: str | None = None,
    favorite: str | None = None,
    rating_min: int | None = None,
) -> resultset.GalleryQuery:
    try:
        return resultset.parse(
            folder=folder,
            album=album,
            person=person,
            artifact=artifact,
            kind=kind,
            favorite=favorite,
            rating_min=rating_min,
            text=q,
            sort=sort,
            size=size,
        )
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
