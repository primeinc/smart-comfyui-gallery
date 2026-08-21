// Interaction only. Membership, order, counts and previews all come from
// the server's ResultSet answers; this file maps pointer geometry onto
// page numbers and renders what it is told. Hooks are semantic data
// attributes, never style classes.
(() => {
  "use strict";

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

  const peeked = new Map(); // `${currency}:${page}` -> peek JSON
  const peek = async (page, s) => {
    const key = `${s.currency}:${page}`;
    if (!peeked.has(key)) {
      const answer = await fetch(`/g/peek?${s.qbase}page=${page}&count=9`);
      if (!answer.ok) return null;
      peeked.set(key, await answer.json());
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
})();
