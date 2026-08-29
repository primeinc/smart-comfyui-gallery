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

READ FROM THE APPLICATION, never written out here. The ways in come from
the route table, the sweeps from `sg_web/operations.py LAUNCHERS`, the
settings from `db/settings.py REGISTRY`, and the recorded gaps from the
same registers `sglint` holds the tree to. One register, two readers: the
linter refuses an unrecorded gap at commit time, and this draws the same
list for a person. They cannot disagree, because there is only one of
them.

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


#: The surfaces, in the order a person meets them. The addresses are
#: real; a name here whose address stopped being served would 404 on
#: click, which is the loudest possible way for this list to be wrong.
SURFACES: tuple[tuple[str, str, str], ...] = (
    ("The library", "every picture, newest first, and the question box over it", "/g"),
    ("The field", "the whole answer on one canvas, placed in time, with a board of kept questions", "/field"),
    ("Timeline", "every sitting in order, with the quiet stretches drawn as quiet stretches", "/timeline"),
    ("People", "everyone this library has found, named or waiting to be", "/people"),
    ("Places", "where the pictures with a place were taken", "/places"),
    ("Collections", "albums, and questions saved as albums", "/albums"),
    ("Keywords", "the words put on pictures by hand", "/keywords"),
    ("Folders", "the library as it really sits on disk", "/folders"),
    ("Duplicates", "near-identical copies, with the difference measured rather than guessed", "/dupes"),
    ("Stories", "what the library has been told to say about a sitting", "/stories"),
    ("Operations", "what is running, what it did, and every setting", "/operations"),
    ("Models and LoRAs", "the checkpoints and adapters these pictures were made with", "/models"),
    ("Workflows", "the graphs behind the generated files", "/workflows"),
)


def surfaces() -> list[Way]:
    return [Way(name, about, at) for name, about, at in SURFACES]


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


def held() -> dict[str, typing.Any]:
    """The whole inventory, for the page and for a machine asking."""
    return {
        "surfaces": surfaces(),
        "sweeps": sweeps(),
        "settings": knobs(),
        "gaps": gaps(),
    }
