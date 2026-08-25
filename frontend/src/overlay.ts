// AddressableOverlay: the mechanics the media lightbox and the person
// drawer BOTH proved before this file existed. An overlay is a way of
// looking at an ADDRESS while the page underneath stays mounted: open
// fetches the address's HX fragment and PUSHES the URL once; requested
// re-opens REPLACE it; dismissal -- close button, Escape, a click on
// the backdrop (never on content) -- IS history Back; popstate makes
// the screen agree with whatever the URL now names; any non-OK response
// falls back to full navigation, because the address always works as a
// page.
//
// What makes it deep rather than moved code:
//   sequencing  every open takes a ticket; a response that arrives
//               after a newer open or a dismissal is DISCARDED, so a
//               slow fragment can never overwrite a newer view or
//               resurrect a closed one.
//   real clicks a modified click (ctrl/cmd/shift/alt, middle button,
//               target=_blank) is an ordinary link -- humans want tabs.
//   focus       focus moves into the overlay, the underlay goes inert,
//               and dismissal returns focus to the element that opened
//               it. The overlay roots sit directly under <body> so
//               "everything else" is exactly the underlay.
//
// The one hook is `generation`: an adapter walking an ordered ResultSet
// supplies its view data version, sent out-of-band and compared against the
// fragment's own generation -- a mismatch redraws whole rather than
// mounting one generation over another. Everything an adapter does
// beyond this (media's arrows, person's rename) is its own file's
// business.
//
// Adapters import what they need. An overlay whose root is not on the
// page returns null, so a surface that renders no drawer wires nothing
// -- the absence is a fact about the DOM, not about which scripts a
// template happened to list.
import { closestFrom, findElement } from "./dom";
import { register } from "./keys";

/** How an open should touch history: a new stop, the current one, or neither. */
export type OpenMode = "push" | "replace" | "none";

export interface OverlaySpec {
  /** Selector for the overlay root, which sits directly under <body>. */
  root: string;
  /** Selector for the links that open it. */
  trigger: string;
  /** The prefix the overlay's addresses share, for popstate. */
  pathPrefix: string;
  /** The adapter's view data version, compared against the fragment's own. */
  generation?: () => string | null;
  /** Refresh the generation evidence; true means the mounted content is proven unchanged. */
  recover?: () => Promise<boolean>;
  /**
   * The adapter's chance to spend a dismissal on its own state first.
   *
   * True means "I used it, stay open" -- a zoomed picture returning to
   * fit, an inspector closing. False means dismissal stands. The shell
   * ASKS rather than the adapter installing a competing Escape listener,
   * because two listeners on one key is a race whose winner depends on
   * which script loaded first.
   */
  dismiss?: () => boolean;
  /**
   * Called after each fragment lands, and after a dismissal with null.
   *
   * An overlay's contents are replaced wholesale on every open, so
   * anything an adapter bound over the old DOM is pointing at nodes that
   * no longer exist. This is where it rebinds.
   */
  mounted?: (root: HTMLElement | null) => void;
}

export interface Overlay {
  readonly root: HTMLElement;
  open(href: string, mode: OpenMode): Promise<void>;
  close(): void;
}

/** A click the browser should handle itself is not ours to intercept. */
export function isPlainClick(event: MouseEvent, link: Element | null): boolean {
  return (
    event.button === 0 &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey &&
    link?.getAttribute("target") !== "_blank"
  );
}

export function addressableOverlay(spec: OverlaySpec): Overlay | null {
  const root = findElement(document, spec.root, HTMLElement);
  if (!root) return null;
  root.tabIndex = -1;

  let flight = 0;
  let opener: HTMLElement | null = null;

  const underlay = (frozen: boolean) => {
    for (const el of document.body.children) {
      if (el !== root && el.tagName !== "SCRIPT" && el instanceof HTMLElement) el.inert = frozen;
    }
  };

  const open = async (href: string, mode: OpenMode): Promise<void> => {
    const ticket = ++flight;
    // Every exit below obeys the ticket FIRST: a request that lost to
    // a newer open or a dismissal lands nowhere however it ends --
    // success, HTTP error, transport failure, or a body that dies
    // after the headers. Only a CURRENT failure earns the full-page
    // fallback; a stale one navigating the browser would hijack it.
    //
    // The loop is the 409 recovery, under the SAME ticket: the
    // library generation moves on every commit, but most commits move
    // nothing the adapter has mounted. An adapter that can PROVE its
    // mounted content is unchanged (spec.recover: refresh the
    // generation evidence, true = proven) earns exactly one retry; a
    // real change, an unprovable one, or a second refusal falls back
    // to the whole page.
    let mended = false;
    while (true) {
      const headers: Record<string, string> = { "HX-Request": "true" };
      const expected = spec.generation ? spec.generation() : null;
      if (expected) headers["X-SG-Expect"] = expected;
      let answer: Response;
      try {
        answer = await fetch(href, { headers });
      } catch {
        if (ticket !== flight) return;
        window.location.assign(href);
        return;
      }
      if (ticket !== flight) return;
      if (!answer.ok) {
        if (answer.status === 409 && spec.recover && !mended) {
          let proven = false;
          try {
            proven = await spec.recover();
          } catch {
            proven = false;
          }
          if (ticket !== flight) return;
          if (proven) {
            mended = true;
            continue;
          }
        }
        window.location.assign(href);
        return;
      }
      let fragment: string;
      try {
        fragment = await answer.text();
      } catch {
        if (ticket !== flight) return;
        window.location.assign(href);
        return;
      }
      if (ticket !== flight) return;
      if (expected) {
        // Fail CLOSED: an adapter that expects a generation gets a
        // fragment that proves one, or the whole page. A fragment with
        // no data-currency at all is a template regression, not a pass.
        const got = /data-currency="([^"]*)"/.exec(fragment);
        if (!got?.[1] || got[1] !== expected) {
          window.location.assign(href);
          return;
        }
      }
      root.innerHTML = fragment;
      spec.mounted?.(root);
      if (root.hidden) {
        root.hidden = false;
        underlay(true);
      }
      if (mode === "push") history.pushState({ sgOverlay: true }, "", href);
      else if (mode === "replace") history.replaceState({ sgOverlay: true }, "", href);
      root.focus();
      return;
    }
  };

  const close = () => {
    flight += 1; // anything still in the air lands nowhere
    root.hidden = true;
    root.replaceChildren();
    spec.mounted?.(null);
    underlay(false);
    if (opener?.isConnected) opener.focus();
    opener = null;
  };

  document.addEventListener("click", (event) => {
    const trigger = closestFrom(event.target, spec.trigger, HTMLElement);
    if (trigger) {
      if (!isPlainClick(event, trigger)) return; // the browser's link, not ours
      const href = trigger.getAttribute("href");
      if (!href) return; // a trigger with no address opens nothing
      event.preventDefault();
      opener = trigger;
      void open(href, "push");
      return;
    }
    if (event.target === root || closestFrom(event.target, "[data-close]", Element)) {
      event.preventDefault();
      history.back();
    }
  });

  // Escape goes through the one keyboard registry like every other key
  // (frontend/src/keys.ts), so a surface that grew a second meaning for it
  // is refused at registration rather than dismissing twice.
  register([
    {
      key: "Escape",
      by: `overlay: ${spec.pathPrefix}`,
      run: () => {
        if (root.hidden) return;
        // The adapter unwinds its own state first, one rung per press.
        if (spec.dismiss?.()) return;
        history.back();
      },
    },
  ]);

  window.addEventListener("popstate", () => {
    if (window.location.pathname.startsWith(spec.pathPrefix)) void open(window.location.href, "none");
    else if (!root.hidden) close();
  });

  return { root, open, close };
}
