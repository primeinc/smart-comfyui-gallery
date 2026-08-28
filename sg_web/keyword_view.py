"""The keyword shelf: every word this library has been called, and the
two gestures that keep the list honest.

Read-only surfaces like `/places` show a vocabulary. This one EDITS it,
which is the difference and the reason it exists: a keyword vocabulary
is only worth having if one picture-idea gets one word, and a year of
typing produces "beach", "Beaches" and "beech" whether or not anybody
meant it to. `db/authored.py rename_tag` has folded one word into
another since the tables shipped, correctly and under test, and nothing
called it -- so the fixing was possible and unreachable.

Both writes name the keyword in the BODY rather than the path, for the
reason the tagging route already gives: an album is addressed by a slug
this application minted, and a keyword is whatever somebody typed,
spaces and all. A free sentence squeezed into a path segment is an
encoding argument nobody asked to have.

Both answer with the WHOLE refreshed vocabulary rather than the row they
touched, because a fold changes two rows and a forget removes one: the
page redraws from the authoritative answer instead of reasoning about
what its click must have done.
"""

from __future__ import annotations

import time
import urllib.parse

from litestar import MediaType, Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException
from litestar.openapi.datastructures import ResponseSpec
from litestar.response import Response, Template

from db import authored, connect, facets
from sg_web.presenting import VARIES, presented_page
from sg_web.wire import Wire


class KeywordListed(Wire):
    """One keyword on the shelf."""

    #: The normalised identity a filter is built from.
    tag: str
    #: The spelling somebody typed, which is what the page shows.
    label: str
    pictures: int
    #: The gallery question this keyword asks, ready to hang on a link.
    qs: str


def _shelf(conn) -> list[KeywordListed]:
    return [
        KeywordListed(
            tag=tag,
            label=label,
            pictures=count,
            qs=urllib.parse.urlencode([("f", facets.spell(facets.facet("tag", "eq", tag)))]),
        )
        for tag, label, count in authored.vocabulary(conn)
    ]


@get(
    "/keywords",
    # The return annotation can only say `Template | Response`, and a union
    # mixing a page with a JSON answer reaches OpenAPI as the empty schema
    # however precisely the arms are written. So the shape a client
    # actually parses is declared here, where the document reads it.
    responses={
        200: ResponseSpec(
            data_container=list[KeywordListed],
            description="Every keyword, commonest first, with how many pictures wear it",
            media_type=MediaType.JSON,
            generate_examples=False,
        )
    },
    sync_to_thread=True,
)
def keywords_index(state: State, request: Request) -> Template | Response:
    """Every keyword, commonest first -- the page for a browser, a JSON
    list for everything else.

    Commonest first rather than alphabetical: the question this page
    answers is "what have I actually been calling things", and the
    answer to that is a shape, not an index. The three-picture typo
    sitting under the four-hundred-picture word is exactly the row
    somebody came here to fix.
    """
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = _shelf(conn)
    finally:
        connect.close(conn)
    return presented_page(request, told, page="keywords.html", context={"keywords": told})


class Renamed(Wire):
    """The body of POST /keywords/rename.

    A collision FOLDS rather than refusing: somebody typing "Beaches"
    when "beach" exists is saying they were always one word, which is
    the ordinary case and the whole reason to be on this page.
    """

    name: str
    to: str


class Forgotten(Wire):
    """The body of POST /keywords/forget.

    `pictures` is the count the page was SHOWING. It is checked against
    the count now, and a mismatch is refused: this is the one gesture
    here that destroys authored work, and retyping a word onto two
    hundred pictures is not a recovery. Same doctrine as removing a
    root -- prove what you are acting on -- rather than a lighter rule
    invented for keywords.
    """

    name: str
    pictures: int


def _answered(conn) -> Response[list[KeywordListed]]:
    return Response(_shelf(conn), headers=VARIES)


@post("/keywords/rename", sync_to_thread=True)
def rename_keyword(state: State, data: Renamed) -> Response[list[KeywordListed]]:
    """Say two words were always one, or fix a spelling."""
    conn = connect.connect(state.db_path)
    try:
        try:
            authored.rename_tag(conn, data.name, data.to, time.time())
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return _answered(conn)
    finally:
        connect.close(conn)


@post("/keywords/forget", sync_to_thread=True)
def forget_keyword(state: State, data: Forgotten) -> Response[list[KeywordListed]]:
    """Take a keyword off every picture that wears it."""
    conn = connect.connect(state.db_path)
    try:
        try:
            authored.forget_tag(conn, data.name, expecting=data.pictures)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return _answered(conn)
    finally:
        connect.close(conn)
