"""Representation negotiation, decided once for every entity address.

Two Adapters (media, person) independently carried the same contract
before this module existed; a third would have reproduced it from
memory and subtly broken a cache. The rules, deterministic and
declared:

    Accept names application/json      -> the view itself (JSON)
    else HX-Request: true              -> the overlay fragment
    else Accept names text/html        -> the full page
    else (wildcard, machine default)   -> the view itself (JSON)

Every response carries `Vary: Accept, HX-Request`, because a cache
handed ambiguity will replay one representation as another.
`wants_json` is exposed separately so an Adapter can decide BEFORE
assembly whether the machine-only parts of its view are worth
computing.
"""

from __future__ import annotations

from litestar import Request
from litestar.response import Response, Template

VARIES = {"vary": "Accept, HX-Request"}


def wants_json(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return True
    return request.headers.get("hx-request") != "true" and "text/html" not in accept


def presented(request: Request, told: dict, *, page: str, fragment: str, name: str) -> Template | Response:
    """The view, in whichever representation the request negotiated.
    `name` is the template context key the page and fragment share."""
    if wants_json(request):
        return Response(told, headers=VARIES)
    if request.headers.get("hx-request") == "true":
        return Template(template_name=fragment, context={name: told}, headers=VARIES)
    return Template(template_name=page, context={name: told}, headers=VARIES)
