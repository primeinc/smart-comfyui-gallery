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
  window.sgPlainClick = (event, link) =>
    event.button === 0 &&
    !event.metaKey &&
    !event.ctrlKey &&
    !event.shiftKey &&
    !event.altKey &&
    link?.target !== "_blank";

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
      // Every exit below obeys the ticket FIRST: a request that lost to
      // a newer open or a dismissal lands nowhere however it ends --
      // success, HTTP error, transport failure, or a body that dies
      // after the headers. Only a CURRENT failure earns the full-page
      // fallback; a stale one navigating the browser would hijack it.
      //
      // The loop is the 409 recovery, under the SAME ticket: the
      // library generation moves on every commit, but most commits move
      // no answer. An adapter that can PROVE its mounted answer is
      // unchanged (spec.recover: refresh the generation evidence, true
      // = proven) earns exactly one retry; a real change, an unprovable
      // one, or a second refusal falls back to the whole page.
      let mended = false;
      while (true) {
        const headers = { "HX-Request": "true" };
        const expected = spec.generation ? spec.generation() : null;
        if (expected) headers["X-SG-Expect"] = expected;
        let answer;
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
        let fragment;
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
      underlay(false);
      if (opener?.isConnected) opener.focus();
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
