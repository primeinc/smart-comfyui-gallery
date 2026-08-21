// Interaction only. Membership, order, counts and previews all come from
// the server's ResultSet answers; this file maps pointer geometry onto
// page numbers and renders what it is told. Hooks are semantic data
// attributes, never style classes.
(() => {
  "use strict";

  // The form must ask a question the server can answer: a phrase orders
  // by similarity (the seam refuses the contradiction), and empty fields
  // have no place in a canonical URL.
  const ask = document.querySelector("[data-ask]");
  if (ask) {
    ask.addEventListener("submit", () => {
      const phrase = ask.querySelector('[name="q"]');
      const sort = ask.querySelector('[name="sort"]');
      if (phrase.value.trim()) sort.value = "similarity";
      else if (sort.value === "similarity") sort.value = "newest";
      for (const field of ask.querySelectorAll("input, select")) {
        if (!field.value.trim()) field.disabled = true; // a disabled field is not submitted
      }
    });
    // The back/forward cache restores the DOM as submitted -- fields
    // disabled for the send must come back usable.
    window.addEventListener("pageshow", () => {
      for (const field of ask.querySelectorAll("input, select")) field.disabled = false;
    });
  }

  const grid = () => document.querySelector("[data-grid]");
  const rail = document.querySelector("[data-rail]");
  if (!rail) return;
  const thumb = rail.querySelector("[data-rail-thumb]");
  const pop = rail.querySelector("[data-rail-pop]");
  const popLabel = pop.querySelector("[data-rail-pop-label]");
  const popGrid = pop.querySelector("[data-rail-pop-grid]");

  const shape = () => {
    const g = grid();
    return g
      ? {
          page: +g.dataset.page,
          pages: +g.dataset.pages,
          total: +g.dataset.total,
          size: +g.dataset.size,
          currency: g.dataset.currency,
          qbase: g.dataset.qbase,
        }
      : null;
  };

  // The rail is the ORDERED RESULT SET at full height: a fraction of the
  // track is a fraction of the answer, never of scroll height.
  const pageAt = (clientY, s) => {
    const box = rail.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (clientY - box.top) / box.height));
    return Math.min(s.pages, Math.max(1, Math.round(fraction * (s.pages - 1)) + 1));
  };

  const placeThumb = () => {
    const s = shape();
    if (!s) return;
    const fraction = s.pages > 1 ? (s.page - 1) / (s.pages - 1) : 0;
    thumb.style.top = `${fraction * 100}%`;
  };

  // A preview must belong to the SAME result-set generation as the grid
  // it floats beside: the request carries the grid's currency, the
  // server answers 409 when the library has moved on, and the response
  // currency is checked again in case it moved mid-flight. Either
  // mismatch redraws the whole gallery from the URL -- two generations
  // are never presented as one answer.
  const peeked = new Map(); // `${currency}:${page}` -> peek JSON
  const peek = async (page, s) => {
    const key = `${s.currency}:${page}`;
    if (!peeked.has(key)) {
      const answer = await fetch(`/g/peek?${s.qbase}page=${page}&count=9&expect=${encodeURIComponent(s.currency)}`);
      if (answer.status === 409) {
        window.location.reload();
        return null;
      }
      if (!answer.ok) return null;
      const told = await answer.json();
      if (told.currency !== s.currency) {
        window.location.reload();
        return null;
      }
      peeked.set(key, told);
    }
    return peeked.get(key);
  };

  let hoverPage = null;
  rail.addEventListener("pointermove", async (event) => {
    const s = shape();
    if (!s) return;
    const page = pageAt(event.clientY, s);
    pop.style.top = `${event.clientY - rail.getBoundingClientRect().top}px`;
    pop.hidden = false;
    if (page === hoverPage) return;
    hoverPage = page;
    const told = await peek(page, s);
    if (!told || hoverPage !== page) return;
    popLabel.textContent =
      `page ${told.page} of ${told.pages} · ${told.first_ordinal}–${told.last_ordinal}` +
      ` of ${told.total}`;
    popGrid.replaceChildren(
      ...told.items.map((item) => {
        const img = new Image();
        img.src = `/thumb/${item.slug}`;
        img.alt = item.name;
        return img;
      }),
    );
  });

  rail.addEventListener("pointerleave", () => {
    pop.hidden = true;
    hoverPage = null;
  });

  // A jump is a real navigation: the URL owns the state, the server
  // renders it whole, and the back button needs no special case.
  rail.addEventListener("click", (event) => {
    const s = shape();
    if (!s) return;
    window.location.assign(`/g?${s.qbase}page=${pageAt(event.clientY, s)}`);
  });

  placeThumb();
  document.body.addEventListener("htmx:afterSwap", placeThumb);

  // --- the lightbox: an addressable overlay, never a second resource --
  // Opening PUSHES the item URL over the mounted gallery; the arrows
  // REPLACE it so browsing fifty items is still one Back to leave;
  // Escape and the close button ARE Back here, because the mounted
  // gallery is one step behind by construction. What "next" means
  // always lives in the URL's context, never in here -- and every
  // fetch carries the view's currency out-of-band, so an answer from a
  // newer library generation is refused (409) and the walk restarts
  // whole instead of mixing two generations on one screen.
  const lightbox = document.querySelector("[data-lightbox-root]");
  if (lightbox) {
    const viewCurrency = () => {
      const shown = lightbox.querySelector("[data-lightbox]");
      if (shown && shown.dataset.currency) return shown.dataset.currency;
      const s = shape();
      return s ? s.currency : "";
    };
    const open = async (href, mode) => {
      const expected = viewCurrency();
      const answer = await fetch(href, {
        headers: { "HX-Request": "true", "X-SG-Expect": expected },
      });
      if (!answer.ok) {
        window.location.assign(href);
        return;
      }
      const fragment = await answer.text();
      // The server compares after assembly, but a commit can land in
      // the microsecond between its currency read and its snapshot --
      // the fragment says which generation it REALLY belongs to, and a
      // mismatch redraws whole rather than mounting it over the old
      // gallery.
      const got = /data-currency="([^"]*)"/.exec(fragment);
      if (expected && got && got[1] && got[1] !== expected) {
        window.location.assign(href);
        return;
      }
      lightbox.innerHTML = fragment;
      lightbox.hidden = false;
      if (mode === "push") history.pushState({ sgLightbox: true }, "", href);
      else if (mode === "replace") history.replaceState({ sgLightbox: true }, "", href);
    };
    const close = () => {
      lightbox.hidden = true;
      lightbox.replaceChildren();
    };
    document.addEventListener("click", (event) => {
      const cell = event.target.closest("a.cell");
      if (cell) {
        event.preventDefault();
        open(cell.getAttribute("href"), "push");
        return;
      }
      if (event.target.closest("[data-close]")) {
        event.preventDefault();
        history.back();
        return;
      }
      const nav = event.target.closest("[data-nav]");
      if (nav) {
        event.preventDefault();
        open(nav.getAttribute("href"), "replace");
      }
    });
    document.addEventListener("keydown", (event) => {
      if (lightbox.hidden) return;
      if (event.key === "Escape") history.back();
      const asked = { ArrowRight: "next", ArrowLeft: "previous" }[event.key];
      if (asked) {
        const nav = lightbox.querySelector(`[data-nav="${asked}"]`);
        if (nav) open(nav.getAttribute("href"), "replace");
      }
    });
    // Back closes; Forward re-opens the item the URL names. The URL is
    // the state -- the handler only makes the screen agree with it.
    window.addEventListener("popstate", () => {
      if (window.location.pathname.startsWith("/i/")) open(window.location.href, "none");
      else if (!lightbox.hidden) close();
    });
  }
})();
