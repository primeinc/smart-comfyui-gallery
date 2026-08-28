"""Every job this worker can run, and what starts it.

A capability that ships with tests, styling and no visible entry point is
unshipped. Three enumerations decide whether that is true of this
application, and each is held somewhere:

  routes      sglint SG010 (sglint/rules.py UNSURFACED). Both halves --
              an unrecorded gap is a finding, and so is a record for a
              route that no longer exists.
  settings    tests/test_the_operations_console_is_expert.py, holding
              sg_web/operations.py SETTING_GROUPS against
              db/settings.py REGISTRY.
  job kinds   here.

THE ROUTE HALF IS DELIBERATELY NOT REPEATED HERE. SG010 already reads
the route table and the interface and refuses an unrecorded gap; a
second check over the same question, written differently, is a second
answer free to disagree with the first. It would also be the WORSE
answer: half the addresses in this application are built by the server
and handed to the client (`story.href`, `view.links.search`), so a
search of the templates alone reports surfaces as unreachable that are
reached every day.

What nothing covered is `db/runner.py HANDLERS`. A job kind is a
capability in the fullest sense -- it has a handler, a queue, a ledger
and tests -- and until now nothing held it against a way to start one.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

pytestmark = pytest.mark.slow

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "sg_web" / "templates"

#: A job kind no console button starts, and what starts it instead.
#:
#: Not every capability belongs on a button. A kind here is reached by
#: doing the thing it belongs to, and the entry names that thing so the
#: claim can be checked -- which the two tests at the foot of this file
#: do, rather than trusting the sentence.
STARTED_ELSEWHERE: dict[str, str] = {
    "walk": (
        "queued by `catch_up` as its first step (db/runner.py). Walking the roots alone finds "
        "files and reads none of them, which settles `done` having apparently done nothing; "
        "'bring the library up to date' is the affordance, and it walks first."
    ),
    "story_plan": (
        "queued by opening a sitting that has no story yet -- /stories/sessions/{id}, which IS "
        "the session card's link on the timeline (sg_web/templates/_timeline_session.html). "
        "Telling a story is something done to one sitting, never a sweep over the library."
    ),
}


def _queued_by() -> dict[str, set[str]]:
    """Which job kind each `submit_*` puts on the queue.

    Read from the source rather than declared, because the console's
    vocabulary and the worker's are different ON PURPOSE: the console
    offers "faces" and "thumbs", the worker runs `detect_faces` and
    `hash`. Comparing those two name sets directly would assert a
    contract this application has never had and never should.

    Both modules that submit are read. `db/prompts.py` has its own
    `submit_embed`, and reading only `db/runner.py` made `embed_prompts`
    look unstartable when its button is right there beside the others.
    """
    from db import prompts, runner

    found: dict[str, set[str]] = {}
    for module in (runner, prompts):
        for name in dir(module):
            if not name.startswith("submit_"):
                continue
            try:
                body = inspect.getsource(getattr(module, name))
            except (OSError, TypeError):
                continue
            kinds = set(re.findall(r"""jobs\.submit\(\s*conn,\s*["']([a-z_]+)["']""", body))
            kinds |= set(re.findall(r"""kind\s*=\s*["']([a-z_]+)["']""", body))
            found[f"{module.__name__.rsplit('.', 1)[-1]}.{name}"] = kinds
    return found


def test_every_job_kind_can_be_started():
    """A job the worker can run and nobody can start is a job nobody has.

    Enumerated from `db/runner.py HANDLERS`, so a new handler fails here
    until something offers a way to run it.
    """
    from db import runner
    from sg_web import operations

    queued = _queued_by()

    # POSITIVE CONTROL. If no submit appears to queue anything, this is
    # reading the source wrongly -- which is a different fact from "the
    # sweeps are gone", and the two must never be confused.
    assert any(queued.values()), (
        "the control failed: no `submit_*` in db/runner.py or db/prompts.py appears to queue a "
        "kind, so this test is misreading the source rather than finding a real gap"
    )

    reachable: set[str] = set()
    for _label, launch in operations.LAUNCHERS.values():
        try:
            body = inspect.getsource(launch)
        except (OSError, TypeError):
            continue
        for module, called in re.findall(r"\b(runner|prompts)\.(submit_\w+|catch_up)\b", body):
            if called == "catch_up":
                # The ordered run of all of them: it reaches whatever
                # every submit reaches.
                reachable |= {kind for kinds in queued.values() for kind in kinds}
            else:
                reachable |= queued.get(f"{module}.{called}", set())

    unstartable = sorted(set(runner.HANDLERS) - reachable - set(STARTED_ELSEWHERE))
    assert not unstartable, (
        f"{unstartable} are job kinds the worker can run with no way to start them. Add a "
        "launcher to sg_web/operations.py LAUNCHERS, or record in STARTED_ELSEWHERE what does "
        "start each one and how a person gets to it."
    )


def test_no_recorded_job_gap_outlives_its_handler():
    """A kind recorded as started elsewhere must still be a kind.

    The other half of the register, and the reason it can be trusted: a
    reason kept for something that no longer exists is a claim nobody
    has checked in a long time.
    """
    from db import runner

    stale = sorted(set(STARTED_ELSEWHERE) - set(runner.HANDLERS))
    assert not stale, (
        f"{stale} are recorded in STARTED_ELSEWHERE but are not job kinds any more; drop them so "
        "the register describes this application rather than an older one"
    )


def test_what_starts_a_kind_elsewhere_is_still_there():
    """The thing each recorded gap points at has to exist.

    Without this the register decays into prose: "the session card
    starts this" stays readable and reassuring long after the session
    card stops doing it.
    """
    from sg_web import operations

    markup = "\n".join(p.read_text(encoding="utf-8") for p in TEMPLATES.rglob("*.html"))
    assert "/stories/sessions/" in markup, (
        "STARTED_ELSEWHERE says story_plan is queued by opening a tellable sitting, but nothing "
        "in the markup builds /stories/sessions/ any more"
    )
    assert "catch_up" in operations.LAUNCHERS, (
        "STARTED_ELSEWHERE says `walk` is queued as catch_up's first step, but there is no catch_up launcher to press"
    )
