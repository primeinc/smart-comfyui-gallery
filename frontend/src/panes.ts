/**
 * Panes: the application's older surfaces, brought to the canvas.
 *
 * The canvas is the world and it fills the viewport. Everything that
 * used to be a PAGE -- the grid, operations, settings, the album shelf,
 * the duplicate review -- is a pane: it lives off-screen, slides in when
 * asked for, and leaves the same way. Navigating away from the canvas to
 * read a list and navigating back is the thing this removes; the world
 * does not move, the panel comes to you.
 *
 * Three presentations, because they answer different questions:
 *
 *   OVERLAY  on top, dimming the world. For something you read, decide,
 *            and dismiss -- settings, a confirmation, a shelf.
 *   DOCK     beside the world, which shrinks to make room. For something
 *            you work ALONGSIDE -- the grid while you place pictures,
 *            the job list while you watch it move.
 *   WINDOW   floating and movable. For something you want to keep in
 *            view while you do something else, wherever it suits you.
 *
 * A pane's content is not reimplemented. Every surface in this
 * application renders its content into `<main class="stage">` -- checked
 * across all fourteen -- so a pane is that element, fetched from the
 * address that already serves it. There is no second copy of the
 * duplicates page to drift out of step with the first, and a surface
 * that gains a feature gains it in its pane on the same deploy.
 */

import { closestFrom, findElement } from "./dom";
import { spellDays } from "./spelling";
import { pin as keep, board as pinned, unpin } from "./workspace";

/** How a pane is presented. */
type Mode = "overlay" | "dock" | "window";

/** htmx, when the page has it. Content arriving through `innerHTML` has
 *  never been seen by htmx's own load handler, so its `hx-` attributes
 *  are inert until something processes them. */
interface Htmx {
  process: (node: Element) => void;
}

const htmxOf = (): Htmx | null => {
  const held = (window as unknown as { htmx?: Htmx }).htmx;
  return held && typeof held.process === "function" ? held : null;
};

export function mountPanes(root: ParentNode): void {
  const found = findElement(root, "[data-panes]", HTMLElement);
  if (!found) return;
  // Re-bound after the guard: the functions below are declarations and
  // are hoisted, so TypeScript will not carry a narrowing from above
  // into a body that could have been called first.
  const deck = found;

  /** Every pane on screen, oldest first: Escape closes the newest. */
  const open: HTMLElement[] = [];

  function frame(title: string, mode: Mode, href: string): HTMLElement {
    const pane = document.createElement("aside");
    pane.className = "pane";
    pane.dataset.paneMode = mode;
    pane.setAttribute("role", mode === "overlay" ? "dialog" : "region");
    pane.setAttribute("aria-label", title);
    // Focusable, so the keyboard follows the pane in. Without this the
    // reader's focus stays behind on the canvas: Tab walks the surface
    // underneath, and Escape -- which the deck listens for -- never
    // reaches the deck at all.
    pane.tabIndex = -1;
    if (mode === "overlay") pane.setAttribute("aria-modal", "true");
    pane.innerHTML = `
      <header class="pane-bar" data-pane-bar>
        <b class="pane-title"></b>
        <span class="pane-modes" role="group" aria-label="how to show this">
          <button type="button" data-pane-mode-set="dock" title="beside the canvas">Dock</button>
          <button type="button" data-pane-mode-set="overlay" title="over the canvas">Overlay</button>
          <button type="button" data-pane-mode-set="window" title="as a movable window">Window</button>
        </span>
        <a class="pane-away" data-pane-away>Open as a page</a>
        <button type="button" class="pane-shut" data-pane-shut aria-label="close">&times;</button>
      </header>
      <div class="pane-body" data-pane-body>
        <p class="pane-waiting">fetching&hellip;</p>
      </div>`;
    const named = pane.querySelector(".pane-title");
    if (named) named.textContent = title;
    const away = pane.querySelector("[data-pane-away]");
    if (away instanceof HTMLAnchorElement) away.href = href;
    return pane;
  }

  /**
   * Bring a surface in.
   *
   * The address is the same one the surface serves itself at, so the
   * pane and the page can never disagree about what that surface is.
   */
  async function show(href: string, title: string, mode: Mode): Promise<void> {
    const pane = frame(title, mode, href);
    deck.append(pane);
    open.push(pane);
    // A frame on screen before the fetch returns: a control that appears
    // to do nothing for half a second reads as a control that is broken.
    requestAnimationFrame(() => {
      pane.setAttribute("data-pane-in", "");
      pane.focus({ preventScroll: true });
      // AFTER the attribute, not before: `settle` counts panes that are
      // actually in, so calling it first found none and the canvas never
      // made room for a dock.
      settle();
    });

    const body = pane.querySelector("[data-pane-body]");
    if (!(body instanceof HTMLElement)) return;
    try {
      const answer = await fetch(href, { headers: { accept: "text/html" } });
      if (!answer.ok) throw new Error(`${answer.status}`);
      // Parsed, not assigned: `DOMParser` builds an inert document, so
      // nothing in the fetched page runs while it is being read.
      const told = new DOMParser().parseFromString(await answer.text(), "text/html");
      const stage = told.querySelector("main.stage");
      if (!stage) throw new Error("that surface has no stage to show");
      body.replaceChildren(document.importNode(stage, true));
      // htmx binds on load and has never seen this markup, so every
      // `hx-` control inside it would be inert. This is htmx's own
      // answer for content that arrived some other way.
      htmxOf()?.process(body);
      // The application's own passes over the markup, re-run on markup
      // they have never seen. They mount once at page load, and a pane's
      // content arrives long after -- which is why a pinned person's row
      // showed `1784368047.71758` where a date belongs. A surface is not
      // the same surface if half of what makes it readable never ran.
      spellDays(body);
      // A `#section` in the address means the caller asked for a PART of
      // that surface. The browser cannot honour it -- the fragment never
      // reached the address bar -- so the pane scrolls to it itself.
      // Without this, "Settings" opened the operations console at the
      // top and left the reader to find the settings themselves.
      const wanted = href.includes("#") ? href.slice(href.indexOf("#") + 1) : "";
      if (wanted) {
        const part = body.querySelector(`#${CSS.escape(wanted)}`);
        if (part instanceof HTMLElement) part.scrollIntoView({ block: "start" });
      }
      offerPins(body);
    } catch (why) {
      body.replaceChildren(said(why, href));
    }
  }

  /** What a pane says when it could not fetch its surface. Never a blank
   *  panel: the address is offered as a link, so the reader can still
   *  get where they were going. */
  function said(why: unknown, href: string): HTMLElement {
    const told = document.createElement("p");
    told.className = "pane-waiting";
    told.textContent = `could not open this here — ${why instanceof Error ? why.message : "it did not answer"}. `;
    const link = document.createElement("a");
    link.href = href;
    link.textContent = "open it as a page instead";
    told.append(link);
    return told;
  }

  /**
   * What one address stands for, as a pin.
   *
   * A person, an album and a folder are all QUESTIONS about the library
   * -- `person=ada` is the same kind of thing as `q=dog`, and the field
   * already knows how to answer it. So a pin for one is a query pin with
   * a different badge, not a different mechanism, and the only thing
   * that has to be written per kind is how its address spells the
   * question. A picture is the exception: there is nothing to arrange
   * about one photograph, so it opens as its own page.
   */
  function asPin(href: string): { kind: "person" | "album" | "folder" | "picture"; at: string } | null {
    const path = href.replace(/^https?:\/\/[^/]+/, "").split(/[?#]/)[0] ?? "";
    const slug = decodeURIComponent(path.slice(3));
    if (!slug) return null;
    if (path.startsWith("/p/")) return { kind: "person", at: `person=${encodeURIComponent(slug)}` };
    if (path.startsWith("/t/")) return { kind: "album", at: `album=${encodeURIComponent(slug)}` };
    if (path.startsWith("/f/")) return { kind: "folder", at: `folder=${encodeURIComponent(slug)}` };
    if (path.startsWith("/i/")) return { kind: "picture", at: slug };
    return null;
  }

  /**
   * What to call the thing a row stands for.
   *
   * Read from the markup that NAMES it, never from the row's text. Every
   * list in this application puts the name in its own element -- a
   * person is `.person-name`, an album `.shelf-name` -- and the rest of
   * the row is everything else known about it. Reading the text instead
   * produced a pin called "Shelia / 72 pictures / 1784368047.71758" and
   * an album called "rule", which is the aria-hidden glyph standing in
   * for a missing cover.
   *
   * A person with no name has no name element, because nobody has told
   * this library who they are. Saying so is better than papering over it
   * with a picture count: "60 pictures" is not a person.
   */
  function named(link: HTMLAnchorElement, kind: string): string {
    const own = link.querySelector('[class$="-name"]');
    const said = own?.textContent?.trim();
    if (said) return said;
    // Nothing decorative: a cover glyph is drawn for the eye and hidden
    // from assistive technology, and it is not what this is called.
    const visible = [...link.childNodes]
      .filter((one) => !(one instanceof Element && one.getAttribute("aria-hidden") === "true"))
      .map((one) => one.textContent ?? "")
      .join(" ");
    const first = visible
      .split("\n")
      .map((one) => one.trim())
      .find((one) => one.length > 0);
    if (first && !/^[\d,]+\s+pictures?$/i.test(first)) return first;
    return kind === "person" ? "Someone not named yet" : (first ?? "");
  }

  /**
   * Put a "keep this" on every row a pane shows that the board can hold.
   *
   * The lists this application already has -- people, albums, folders --
   * are exactly the things somebody would want on their board, and they
   * are already rendered. Rather than building a second people list with
   * pins on it, the pin is added to the one that exists, from its own
   * links. A surface that gains a person gains a pinnable person.
   */
  function offerPins(body: HTMLElement): void {
    const held = new Set(pinned().map((one) => one.at));
    for (const link of body.querySelectorAll("a[href]")) {
      if (!(link instanceof HTMLAnchorElement)) continue;
      if (link.dataset.pinOffered !== undefined) continue;
      const what = asPin(link.getAttribute("href") ?? "");
      if (!what) continue;
      link.dataset.pinOffered = "";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "pin-offer";
      button.dataset.pinAt = what.at;
      button.dataset.pinKind = what.kind;
      button.dataset.pinName = named(link, what.kind);
      button.textContent = held.has(what.at) ? "on the board" : "pin";
      button.title = "keep this on the board";
      link.insertAdjacentElement("afterend", button);
    }
  }

  function shut(pane: HTMLElement): void {
    pane.removeAttribute("data-pane-in");
    const at = open.indexOf(pane);
    if (at >= 0) open.splice(at, 1);
    settle();
    // Removed after it has left, so it slides out instead of vanishing.
    // `transitionend` alone would strand it on a reader who has asked
    // for no motion, where the transition never runs.
    window.setTimeout(() => pane.remove(), 260);
  }

  /**
   * What the deck as a whole is doing, so the canvas can make room.
   *
   * `panesOpen`, not `paneOpen`. The singular is already taken by
   * `[data-pane-open]`, which is an ACTION carrying the address to
   * fetch -- and writing the plural state under the singular name put
   * `data-pane-open="no"` on the `<body>`, so every click in the
   * document walked up to the body, matched the action selector, and
   * fetched `/no`. One name, two meanings, and the one that lost was
   * silent.
   */
  function settle(): void {
    const docked = open.filter((p) => p.dataset.paneMode === "dock" && p.hasAttribute("data-pane-in"));
    document.body.dataset.panesDocked = docked.length ? "yes" : "no";
    document.body.dataset.panesOpen = open.length ? "yes" : "no";
  }

  // One listener for the whole deck: panes arrive and leave constantly,
  // and a listener per pane is a listener per pane to remove.
  deck.addEventListener("click", (event) => {
    const pane = closestFrom(event.target, ".pane", HTMLElement);
    if (!pane) return;
    if (closestFrom(event.target, "[data-pane-shut]", HTMLButtonElement)) {
      shut(pane);
      return;
    }
    const offer = closestFrom(event.target, "[data-pin-at]", HTMLButtonElement);
    if (offer) {
      const at = offer.dataset.pinAt ?? "";
      const already = pinned().find((one) => one.at === at);
      if (already) {
        unpin(already.id);
        offer.textContent = "pin";
      } else {
        const count = pinned().length;
        keep({
          id: `pin-${Date.now().toString(36)}`,
          kind: (offer.dataset.pinKind as "person" | "album" | "folder" | "picture") ?? "person",
          name: offer.dataset.pinName ?? at,
          at,
          x: (count % 4) * 340,
          y: Math.floor(count / 4) * 226,
        });
        offer.textContent = "on the board";
      }
      return;
    }
    const wanted = closestFrom(event.target, "[data-pane-mode-set]", HTMLButtonElement);
    if (wanted) {
      pane.dataset.paneMode = wanted.dataset.paneModeSet ?? "overlay";
      pane.setAttribute("role", pane.dataset.paneMode === "overlay" ? "dialog" : "region");
      settle();
    }
  });

  // Anything, anywhere, can ask for a pane.
  document.addEventListener("click", (event) => {
    const asked = closestFrom(event.target, "[data-pane-open]", HTMLElement);
    if (!asked) return;
    const href = asked.dataset.paneOpen;
    if (!href) return;
    // A plain click only: a middle click, or a held modifier, means the
    // reader asked for a real page and this must not eat that.
    if (event instanceof MouseEvent && (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey)) return;
    event.preventDefault();
    const mode = asked.dataset.paneMode;
    void show(
      href,
      asked.dataset.paneTitle ?? asked.textContent?.trim() ?? "",
      mode === "dock" ? "dock" : mode === "window" ? "window" : "overlay",
    );
  });

  // A movable window is moved by its bar. Pointer events rather than
  // mouse events, so a finger drags it too.
  let held: HTMLElement | null = null;
  let fromX = 0;
  let fromY = 0;
  let atX = 0;
  let atY = 0;
  deck.addEventListener("pointerdown", (event) => {
    const bar = closestFrom(event.target, "[data-pane-bar]", HTMLElement);
    const pane = bar && closestFrom(event.target, ".pane", HTMLElement);
    if (!bar || !pane || pane.dataset.paneMode !== "window") return;
    if (closestFrom(event.target, "button, a", HTMLElement)) return;
    held = pane;
    fromX = event.clientX;
    fromY = event.clientY;
    atX = Number(pane.dataset.paneX ?? 0);
    atY = Number(pane.dataset.paneY ?? 0);
    bar.setPointerCapture(event.pointerId);
  });
  deck.addEventListener("pointermove", (event) => {
    if (!held) return;
    const x = atX + event.clientX - fromX;
    const y = atY + event.clientY - fromY;
    held.dataset.paneX = String(x);
    held.dataset.paneY = String(y);
    held.style.translate = `${x}px ${y}px`;
  });
  const drop = (): void => {
    held = null;
  };
  deck.addEventListener("pointerup", drop);
  deck.addEventListener("pointercancel", drop);

  // Escape closes the pane the key was pressed in, the way every layered
  // surface behaves.
  //
  // ON THE DECK, not on the document. A second document keydown listener
  // is the exact defect the key registry exists to prevent -- two
  // keyboards that cannot see each other's claims -- and this one was
  // excused in a comment rather than fixed. It does not need the
  // document: a pane takes focus when it opens, so the keystroke starts
  // inside the pane and reaches the deck by bubbling, and the pane it
  // reaches from is the pane to close. That is also more correct than
  // "the newest": with two panes open, Escape now shuts the one you are
  // actually in.
  deck.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const pane = closestFrom(event.target, ".pane", HTMLElement);
    if (!pane) return;
    event.preventDefault();
    shut(pane);
  });

  settle();
}
