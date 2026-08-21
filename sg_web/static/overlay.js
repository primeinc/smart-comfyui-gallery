// AddressableOverlay: the mechanics the media lightbox and the person
// drawer BOTH proved before this file existed. An overlay is a way of
// looking at an ADDRESS while the page underneath stays mounted: open
// fetches the address's HX fragment and PUSHES the URL once; requested
// re-opens REPLACE it; dismissal -- close button, Escape, a click on
// the backdrop (never on content) -- IS history Back; popstate makes
// the screen agree with whatever the URL now names; any non-OK answer
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
// supplies its view currency, sent out-of-band and compared against the
// fragment's own generation -- a mismatch redraws whole rather than
// mounting one generation over another. Everything an adapter does
// beyond this (media's arrows, person's rename) is its own file's
// business.
(() => {
  "use strict";

  window.sgPlainClick = (event, link) =>
    event.button === 0 &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey &&
    (!link || link.target !== "_blank");

  window.sgAddressableOverlay = (spec) => {
    const root = document.querySelector(spec.root);
    if (!root) return null;
    root.tabIndex = -1;

    let flight = 0;
    let opener = null;

    const underlay = (frozen) => {
      for (const el of document.body.children) {
        if (el !== root && el.tagName !== "SCRIPT") el.inert = frozen;
      }
    };

    const open = async (href, mode) => {
      const ticket = ++flight;
      const headers = { "HX-Request": "true" };
      const expected = spec.generation ? spec.generation() : null;
      if (expected) headers["X-SG-Expect"] = expected;
      let answer;
      try {
        answer = await fetch(href, { headers });
      } catch {
        window.location.assign(href);
        return;
      }
      if (ticket !== flight) return; // a newer open or a dismissal won
      if (!answer.ok) {
        window.location.assign(href);
        return;
      }
      const fragment = await answer.text();
      if (ticket !== flight) return;
      if (expected) {
        const got = /data-currency="([^"]*)"/.exec(fragment);
        if (got && got[1] && got[1] !== expected) {
          window.location.assign(href);
          return;
        }
      }
      root.innerHTML = fragment;
      if (root.hidden) {
        root.hidden = false;
        underlay(true);
      }
      if (mode === "push") history.pushState({ sgOverlay: true }, "", href);
      else if (mode === "replace") history.replaceState({ sgOverlay: true }, "", href);
      root.focus();
    };

    const close = () => {
      flight += 1; // anything still in the air lands nowhere
      root.hidden = true;
      root.replaceChildren();
      underlay(false);
      if (opener && opener.isConnected) opener.focus();
      opener = null;
    };

    document.addEventListener("click", (event) => {
      const trigger = event.target.closest(spec.trigger);
      if (trigger) {
        if (!window.sgPlainClick(event, trigger)) return; // the browser's link, not ours
        event.preventDefault();
        opener = trigger;
        open(trigger.getAttribute("href"), "push");
        return;
      }
      if (event.target === root || event.target.closest("[data-close]")) {
        event.preventDefault();
        history.back();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (!root.hidden && event.key === "Escape") history.back();
    });

    window.addEventListener("popstate", () => {
      if (window.location.pathname.startsWith(spec.pathPrefix)) open(window.location.href, "none");
      else if (!root.hidden) close();
    });

    return { root, open, close };
  };
})();
