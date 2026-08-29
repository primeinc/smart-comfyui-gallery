"""What this library can do, and where each of those things is.

A capability that ships with tests, styling and no visible entry point is
unshipped -- and the way that happened here was never malice. Somebody
built the compare tray, wired it, styled it and tested it, and the only
way to open it was a letter nothing on screen mentioned. Nobody decided
that. It just never got a door, and nothing in the application was in a
position to notice.

So this is the door register, as a page. Every way in this application
offers, the sweeps it can run and who starts them, the settings it
answers to, and -- the part that matters -- the things it can do that
NOTHING on screen reaches, each with the reason it is not surfaced.

READ FROM THE APPLICATION wherever the application can answer. The sweeps
come from `sg_web/operations.py LAUNCHERS`, so the list IS the console's
buttons; the settings from `db/settings.py REGISTRY`; the gaps from the
same registers `sglint` holds the tree to -- one register, two readers,
which cannot disagree because there is only one of them.

The surfaces are the exception: a route is not a surface, and no rule the
application can state tells the two apart. `SAID` below is therefore AUTHORED --
a person decides what counts as a surface -- and every entry is checked against
what is served, so an entry whose address stops being served disappears rather
than 404ing.

Being served is not enough to be a surface: /clusterings and /views answer 200 to
a browser with raw JSON, and are recorded as machine reads in sglint/policy.py
UNSURFACED instead, with the page a person goes to.
tests/test_the_shell_mounts_every_surface.py crawls every link every page emits
and requires it to land a person on a page. A page added with no entry in `SAID`
does not appear; sglint's SG010 is what catches an unreachable surface, and it
reads the templates and the browser source, not this list.

Importing `sglint.policy` from the application looks backwards and is
deliberate. The alternative is a second copy of the decisions, which
would drift -- and a register that drifts is worse than none, because it
reads as an answer. `policy` is plain data with no imports of its own.
"""

from __future__ import annotations

import dataclasses
import typing

from sglint import policy


@dataclasses.dataclass(frozen=True, slots=True)
class Way:
    """One thing a person can do, and where."""

    name: str
    about: str
    at: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class Gap:
    """Something this application can do that no surface reaches, and
    why. Never a bug list: every one of these is a decision, and the
    reason is what makes it checkable rather than merely stated."""

    name: str
    why: str


#: The surfaces: what each is called, what it answers, and where. Authored, and
#: checked against what is served -- see the module docstring.
SAID: dict[str, tuple[str, str]] = {
    "/g": ("The library", "every picture, newest first, and the question box over it"),
    "/field": ("The field", "the whole answer on one canvas, placed in time, with a board of kept questions"),
    "/timeline": ("Timeline", "every sitting in order, with the quiet stretches drawn as quiet stretches"),
    "/people": ("People", "everyone this library has found, named or waiting to be"),
    "/places": ("Places", "where the pictures with a place were taken"),
    "/albums": ("Collections", "albums, and questions saved as albums"),
    "/keywords": ("Keywords", "the words put on pictures by hand"),
    "/folders": ("Folders", "the library as it really sits on disk"),
    "/dupes": ("Duplicates", "near-identical copies, with the difference measured rather than guessed"),
    "/stories": ("Stories", "what the library has been told to say about a sitting"),
    "/operations": ("Operations", "what is running, what it did, and every setting"),
    "/models": ("Models", "the checkpoints these pictures were made with"),
    "/loras": ("LoRAs", "the adapters, and what each one was used on"),
    "/workflows": ("Workflows", "the graphs behind the generated files"),
    "/what": ("What this can do", "this page"),
}

#: Addresses that are served to a browser but are not places to go: the
#: front door, and anything the shell already carries.
NOT_A_SURFACE = frozenset({"/"})


def surfaces(served: typing.Iterable[str]) -> list[Way]:
    """The surfaces, each one CHECKED against what is being served.

    `served` is the application's own route paths -- handed over rather
    than imported, because `sg_web/app.py` imports this module and asking
    it for its routes from in here would be a cycle.

    An entry whose address stops being served drops out of the page. That
    is the failure worth designing against: a register that keeps
    offering a door onto a room that was demolished is worse than one
    that is merely incomplete, because it is confidently wrong and a
    person finds out by clicking.
    """
    held = set(served)
    return [Way(name, about, at) for at, (name, about) in SAID.items() if at in held and at not in NOT_A_SURFACE]


def sweeps() -> list[Way]:
    """The work this application can start, from the console's own table.

    `LAUNCHERS` is what the console draws its buttons from, so this list
    is the buttons -- not a description of them.
    """
    from sg_web import operations

    return [
        Way(kind.replace("_", " "), about, "/operations#panel-running")
        for kind, (about, _how) in operations.LAUNCHERS.items()
    ]


def knobs() -> list[Way]:
    """Every setting, in the words the console gives it."""
    from db import settings
    from sg_web.operations import SETTING_WORDS

    said: list[Way] = []
    for key in settings.REGISTRY:
        name, does = SETTING_WORDS.get(key, (key, ""))
        said.append(Way(name, does, "/operations#panel-setup"))
    return said


def gaps() -> list[Gap]:
    """What this application can do that nothing on screen reaches.

    Three kinds, and they are different in a way worth keeping apart. A
    job kind with no button is reached by doing the thing it belongs to.
    An address with no link is usually a machine's, answered for a person
    somewhere else. And a capability that is not an address at all cannot
    be surfaced by drawing anything -- each of those says what would have
    to happen first, which is the useful part: "no affordance" reads like
    a decision somebody could reverse this afternoon, and none of them
    are.
    """
    found = [Gap(f"the {kind} sweep", why) for kind, why in sorted(policy.STARTED_ELSEWHERE.items())]
    found += [Gap(where, why) for where, why in sorted(policy.UNSURFACED.items())]
    found += [Gap(what, why) for what, why in sorted(policy.UNSURFACED_BEYOND_ROUTES.items())]
    return found


def held(served: typing.Iterable[str]) -> dict[str, typing.Any]:
    """The whole inventory, for the page and for a machine asking."""
    return {
        "surfaces": surfaces(served),
        "sweeps": sweeps(),
        "settings": knobs(),
        "gaps": gaps(),
    }
