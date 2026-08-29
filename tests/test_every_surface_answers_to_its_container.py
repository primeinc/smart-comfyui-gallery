"""Every surface, at the widths it is really used at, under every input.

The layout contract is one composition that answers to its CONTAINER,
not to a guess about the viewport -- and pointer, keyboard and touch are
three equal input classes rather than one and two afterthoughts. This is
that contract, executed.

What is checked, and why each one is what actually breaks:

  SIDEWAYS SCROLL   the most reliable sign that a layout has stopped
                    answering to its container. Something inside it has
                    a width it insists on, and on a narrow screen the
                    page is dragged along behind it.
  NOTHING BROKEN    a console error or a first-party 4xx means the
                    surface did not finish assembling, which no
                    screenshot will tell you.
  TOUCH TARGETS     a control a finger cannot hit is a control that does
                    not exist on a phone. Measured on a TOUCH context,
                    because that is the input class the rule is for.
  KEYBOARD          something must be focusable and the focus must be
                    visible. A surface you cannot walk is a surface
                    reachable by one input class out of three.

Every one is MEASURED on a rendered page. A layout that reads correctly
in the markup and collapses at 390px reads identically in a diff, which
is the whole reason this file exists rather than a careful reading of
the stylesheet.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image
from playwright.sync_api import Browser

from tests.conftest import Live

pytestmark = pytest.mark.slow

FILES = 8

#: The widths this application is really used at, not a sweep of every pixel.
#: Three shapes that break different things: a phone (one column), the awkward
#: middle where a two-column layout is deciding, and a desk.
WIDTHS = ((390, 844, "phone"), (820, 1180, "tablet"), (1512, 950, "desktop"))

#: The smallest a target may be under a finger. Below this a control is
#: present, styled, tested -- and unusable by the input class most of
#: this application's readers will be holding it in.
FINGER = 44

#: Every surface a person navigates to. Fragments and byte addresses are
#: not surfaces; sglint SG010 keeps the register of what is deliberately
#: unreachable, and this is what IS reachable.
SURFACES = (
    "/",
    "/what",
    "/g",
    "/field",
    "/timeline",
    "/people",
    "/places",
    "/albums",
    "/keywords",
    "/folders",
    "/dupes",
    "/operations",
    "/stories",
)


def write_library(root) -> None:
    at = 1_686_400_000
    for i in range(FILES):
        path = root / f"shot-{i:02d}.png"
        Image.new("RGB", (40 + i, 30 + i), (20 * i, 90, 160)).save(path)
        os.utime(path, (at + i * 600, at + i * 600))


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    api.post(f"/roots/{made['id']}/scan")
    for kind in ("ingest", "context", "events"):
        api.post(f"/jobs/{kind}")


def _watched(page) -> list[str]:
    """What the browser complained about, as it happens.

    Static assets are excluded from the 4xx watch on purpose: a missing
    thumbnail for a file this fixture never generated one for is a fact
    about the fixture, not about the layout under test.
    """
    said: list[str] = []
    page.on("console", lambda m: said.append(f"console: {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: said.append(f"pageerror: {e}"))
    page.on(
        "response",
        lambda r: (
            said.append(f"{r.status} {r.url}")
            if r.status >= 400 and "/static/" not in r.url and "/thumb" not in r.url and "/preview/" not in r.url
            else None
        ),
    )
    return said


@pytest.mark.parametrize("surface", SURFACES)
def test_a_surface_answers_to_every_width(browser: Browser, live: Live, surface: str):
    """No surface drags the page sideways, at any width it is used at."""
    trouble: list[str] = []
    for width, height, name in WIDTHS:
        context = browser.new_context(base_url=live.url, viewport={"width": width, "height": height})
        page = context.new_page()
        said = _watched(page)
        page.goto(surface, wait_until="load", timeout=30_000)
        page.wait_for_timeout(1200)
        over = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        if over > 0:
            trouble.append(f"{name} ({width}px): scrolls sideways by {over}px")
        trouble.extend(f"{name} ({width}px): {one}" for one in said)
        context.close()
    assert not trouble, f"{surface}\n  " + "\n  ".join(trouble)


@pytest.mark.parametrize("surface", SURFACES)
def test_a_surface_can_be_used_by_a_finger(browser: Browser, live: Live, surface: str):
    """Every control a phone shows is big enough to press.

    Measured on a TOUCH context at phone size, because that is the
    pairing the rule exists for: a 24px control is comfortable under a
    mouse and is not a control at all under a thumb.

    Only what is VISIBLE and OFFERED. A surface is entitled to hold
    collapsed things, and a control inside a closed disclosure is not
    being offered yet.
    """
    context = browser.new_context(
        base_url=live.url, viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True
    )
    page = context.new_page()
    page.goto(surface, wait_until="load", timeout=30_000)
    page.wait_for_timeout(1200)
    small = page.evaluate(
        """(finger) => {
          const out = [];
          for (const el of document.querySelectorAll('a[href], button, input, select, [role="button"]')) {
            const box = el.getBoundingClientRect();
            if (box.width === 0 || box.height === 0) continue;
            if (el.closest('[hidden], details:not([open])')) continue;
            const style = getComputedStyle(el);
            if (style.visibility === 'hidden' || style.display === 'none') continue;
            // The ESSENTIAL exception. On the timeline's axis a bar's
            // WIDTH is a duration and its POSITION is a moment: growing
            // one to 44px would move it, and it would then be at the
            // wrong time. The criterion carves this out by name --
            // "an interactive data visualization where targets are
            // necessarily dense" -- and asks for equivalent function by
            // other means, which the session cards below the axis are.
            if (el.closest('[data-strip], [data-overview], [data-scrubber], .axis, svg')) continue;
            // The INLINE exception, which the criterion itself carves
            // out: a link sitting in running text, whose height IS the
            // line-height of the prose around it. Detected by asking
            // whether its parent holds text of its own -- which is what
            // "in a sentence" means.
            if (el.tagName === 'A' && style.display.startsWith('inline')) {
              const parent = el.parentElement;
              const prose = parent
                ? [...parent.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 0)
                : false;
              if (prose) continue;
            }
            if (Math.min(box.width, box.height) < finger) {
              out.push(el.tagName.toLowerCase() + '.' + (String(el.className).split(' ')[0] || '(none)') +
                       ' ' + Math.round(box.width) + 'x' + Math.round(box.height) +
                       ' "' + (el.textContent || '').trim().slice(0, 24) + '"');
            }
          }
          return out;
        }""",
        FINGER,
    )
    context.close()
    assert not small, (
        f"{surface} at 390px under a finger: {len(small)} controls are under {FINGER}px on their "
        "short side\n  " + "\n  ".join(sorted(set(small))[:20])
    )


@pytest.mark.parametrize("surface", SURFACES)
def test_a_surface_can_be_walked_by_a_keyboard(browser: Browser, live: Live, surface: str):
    """Something is reachable by Tab, and the focus can be seen.

    The two halves are one requirement. A surface where Tab reaches
    nothing cannot be used without a pointer; a surface where Tab reaches
    something and draws nothing can be used only by somebody who
    remembers where they are.
    """
    context = browser.new_context(base_url=live.url, viewport={"width": 1512, "height": 950})
    page = context.new_page()
    page.goto(surface, wait_until="load", timeout=30_000)
    page.wait_for_timeout(1200)

    page.keyboard.press("Tab")
    landed = page.evaluate(
        """() => {
          const el = document.activeElement;
          if (!el || el === document.body) return null;
          const style = getComputedStyle(el);
          return {
            what: el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(' ')[0] : ''),
            marked: (style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0)
                    || style.boxShadow !== 'none',
          };
        }"""
    )
    context.close()
    assert landed is not None, f"{surface}: one Tab from the top reaches nothing focusable"
    assert landed["marked"], (
        f"{surface}: Tab reaches {landed['what']} and nothing marks it as focused, so a keyboard "
        "can move through this surface without ever showing where it is"
    )
