// Interaction only. Membership, order, counts and previews all come from
// the server's ResultSet answers; this file maps pointer geometry onto
// page numbers and renders what it is told. Hooks are semantic data
// attributes, never style classes.
(() => {
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

  // Save the CURRENT question as a smart collection: the server
  // reconstructs the typed rule from the canonical spelling -- the
  // browser sends the URL's own parameters and a name, never a rule
  // shape. A semantic view needs a cutoff, because similarity ranks
  // the library and only `take` makes that a membership set.
  const saver = document.querySelector("[data-save-smart]");
  if (saver) {
    saver.addEventListener("click", async () => {
      // The MOUNTED answer's canonical spelling, not the address bar:
      // ResultSet heals retired entity spellings into data-qbase, and a
      // save must persist the identity on screen, not the stale words
      // the URL arrived with.
      const mounted = document.querySelector("[data-grid]");
      const spelled = mounted ? mounted.dataset.qbase.replace(/&$/, "") : window.location.search;
      const question = new URLSearchParams(spelled);
      const params = Object.fromEntries(question);
      delete params.page;
      delete params.size;
      // `f` repeats, one per facet; fromEntries keeps only the last and
      // a two-facet view would save as a one-facet rule
      if (question.has("f")) params.f = question.getAll("f");
      const name = window.prompt("name this smart collection");
      if (!name) return;
      let take = null;
      if (params.q) {
        const asked = window.prompt("how many top results belong to it?", "100");
        if (!asked) return;
        take = +asked;
      }
      const answer = await fetch("/albums/smart", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, take, ...params }),
      });
      if (!answer.ok) {
        window.alert((await answer.json().catch(() => ({}))).detail || "the view could not be saved");
        return;
      }
      const told = await answer.json();
      window.location.assign(`/t/${told.slug}`);
    });
  }

  // The other half of the save-view pair: this view becomes an EXISTING
  // smart collection's whole new rule. The browser still sends only the
  // URL's own parameters; the target's current definition revision comes
  // from its authoritative view, so a concurrent edit is a 409, never a
  // silent overwrite.
  const replacer = document.querySelector("[data-replace-smart]");
  if (replacer) {
    replacer.addEventListener("click", async () => {
      const shelf = await fetch("/albums", { headers: { accept: "application/json" } });
      const smarts = (await shelf.json()).filter((held) => held.kind === "smart");
      if (!smarts.length) {
        window.alert("no smart collection exists yet -- save the view as a new one instead");
        return;
      }
      const named = window.prompt(
        `replace the rule of which smart collection?
${smarts.map((held) => held.slug).join(", ")}`,
        smarts[0].slug,
      );
      if (!named) return;
      const current = await fetch(`/t/${named}`, { headers: { accept: "application/json" } });
      if (!current.ok) {
        window.alert(`no collection at /t/${named}`);
        return;
      }
      const mounted = document.querySelector("[data-grid]");
      const spelled = mounted ? mounted.dataset.qbase.replace(/&$/, "") : window.location.search;
      const question = new URLSearchParams(spelled);
      const params = Object.fromEntries(question);
      delete params.page;
      delete params.size;
      // `f` repeats, one per facet; fromEntries keeps only the last and
      // a two-facet view would save as a one-facet rule
      if (question.has("f")) params.f = question.getAll("f");
      let take = null;
      if (params.q) {
        const asked = window.prompt("how many top results belong to it?", "100");
        if (!asked) return;
        take = +asked;
      }
      const answer = await fetch(`/t/${named}/rule`, {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ expected_rev: (await current.json()).definition_rev, take, ...params }),
      });
      if (!answer.ok) {
        window.alert((await answer.json()).detail || "the rule could not be replaced");
        return;
      }
      window.location.assign(`/t/${(await answer.json()).slug}`);
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
          answer: g.dataset.answer,
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

  // The rail hangs from the header's REAL bottom edge. The header is
  // sticky and wraps with its controls, so a fixed offset left the top
  // of the rail -- the newest pages -- under it, where a hover reached
  // the header and never the rail.
  const bar = document.querySelector("header.bar");
  const placeRail = () => {
    if (bar) rail.style.top = `${bar.getBoundingClientRect().bottom}px`;
  };

  const placeThumb = () => {
    placeRail();
    const s = shape();
    if (!s) return;
    const fraction = s.pages > 1 ? (s.page - 1) / (s.pages - 1) : 0;
    thumb.style.top = `${fraction * 100}%`;
  };
  window.addEventListener("resize", placeRail);

  // A preview must belong to the SAME answer as the grid it floats
  // beside -- the answer identity, not the currency. Currency is the
  // library's data_version, which every commit moves: a running job's
  // bookkeeping moved it once per item, so every hover during a job
  // was a 409 and a whole-page reload. The answer is the sha of the
  // ordering itself; the same answer means the same ordering, and a
  // preview of it is true. A different answer redraws the gallery from
  // the URL -- two orderings are never presented as one.
  const peeked = new Map(); // `${answer}:${page}` -> peek JSON
  const peek = async (page, s) => {
    const key = `${s.answer}:${page}`;
    if (!peeked.has(key)) {
      const answer = await fetch(`/g/peek?${s.qbase}page=${page}&count=9`);
      if (!answer.ok) return null;
      const told = await answer.json();
      if (told.answer !== s.answer) {
        window.location.reload();
        return null;
      }
      peeked.set(key, told);
    }
    return peeked.get(key);
  };

  // A page is asked for once the pointer RESTS on it. A sweep down the
  // rail crosses dozens of pages; nine thumbs for each of them queued
  // the page actually wanted behind eighty pictures nobody will see.
  const REST_MS = 60;
  let hoverPage = null;
  let resting = 0;
  const show = async (page, s) => {
    const told = await peek(page, s);
    if (!told || hoverPage !== page) return;
    popLabel.textContent =
      `page ${told.page} of ${told.pages} · ${told.first_ordinal}–${told.last_ordinal}` + ` of ${told.total}`;
    popGrid.replaceChildren(
      ...told.items.map((item) => {
        const img = new Image();
        img.src = `/thumb/${item.slug}`;
        img.alt = item.name;
        return img;
      }),
    );
    placePop(); // the grid changed the box's height
  };

  // Beside the pointer, clamped to the viewport below the header: a
  // preview at the top of the rail sits fully on screen, not half under
  // the bar; one at the bottom does not run off the page.
  const MARGIN = 8;
  let pointerY = 0;
  const placePop = () => {
    const top = bar ? bar.getBoundingClientRect().bottom + MARGIN : MARGIN;
    const height = pop.offsetHeight;
    const floor = Math.max(top, window.innerHeight - height - MARGIN);
    pop.style.top = `${Math.min(Math.max(pointerY - height / 2, top), floor)}px`;
  };

  rail.addEventListener("pointermove", (event) => {
    const s = shape();
    if (!s) return;
    const page = pageAt(event.clientY, s);
    pointerY = event.clientY;
    pop.hidden = false;
    placePop();
    if (page === hoverPage) return;
    hoverPage = page;
    clearTimeout(resting);
    resting = setTimeout(() => show(page, s), REST_MS);
  });

  rail.addEventListener("pointerleave", () => {
    clearTimeout(resting);
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

  // --- the lightbox: the media adapter over the AddressableOverlay ----
  // The shell (static/overlay.js) owns open/mount, push-replace policy,
  // Back-on-dismiss, popstate and the generation check. What is MEDIA'S
  // alone lives here: which currency the view is walking, and the
  // arrows -- each a REPLACE, so browsing fifty items is one Back out.
  const lightbox = window.sgAddressableOverlay({
    root: "[data-lightbox-root]",
    trigger: "a.cell",
    pathPrefix: "/i/",
    generation: () => {
      const shown = document.querySelector("[data-lightbox]");
      if (shown?.dataset.currency) return shown.dataset.currency;
      const s = shape();
      return s ? s.currency : "";
    },
    // A 409'd arrow proves the generation moved, not that THIS answer
    // did -- a favorite, a background job's bookkeeping, any commit at
    // all moves data_version. Ask locate for the walked context's
    // (currency, answer): the same answer identity means the mounted
    // walk is still true, so adopt the fresh currency and let the shell
    // retry once. A changed or vanished answer stays a full redraw.
    recover: async () => {
      const shown = document.querySelector("[data-lightbox]");
      const g = grid();
      const mounted = shown?.dataset.answer || g?.dataset.answer || "";
      const slug = shown ? shown.dataset.slug : null;
      if (!mounted || !slug) return false;
      const asked = await fetch(`/g/locate/${slug}${window.location.search}`);
      if (!asked.ok) return false;
      const told = await asked.json();
      if (told.in_answer === false || told.answer !== mounted) return false;
      for (const surface of [shown, g]) {
        if (surface) {
          surface.dataset.currency = told.currency;
          surface.dataset.answer = told.answer;
        }
      }
      return true;
    },
  });
  if (lightbox) {
    document.addEventListener("click", (event) => {
      const nav = event.target.closest("[data-nav]");
      if (nav && window.sgPlainClick(event, nav)) {
        event.preventDefault();
        lightbox.open(nav.getAttribute("href"), "replace");
      }
    });
    document.addEventListener("keydown", (event) => {
      if (lightbox.root.hidden) return;
      const asked = { ArrowRight: "next", ArrowLeft: "previous" }[event.key];
      if (asked) {
        const nav = lightbox.root.querySelector(`[data-nav="${asked}"]`);
        if (nav) lightbox.open(nav.getAttribute("href"), "replace");
      }
    });
  }
})();
