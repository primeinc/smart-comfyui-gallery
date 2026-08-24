// Interaction only. Membership, order, counts and previews all come from
// the server's ResultSet answers; this file maps pointer geometry onto
// page numbers and renders what it is told. Hooks are semantic data
// attributes, never style classes.
import { api, refusal } from "./api";
import { closestFrom, everyElement, findElement, requireData, requireElement } from "./dom";
import type { components } from "./generated/api";
import { addressableOverlay, isPlainClick } from "./overlay";
import { type Viewer, mountViewer } from "./viewer";

type PeekView = components["schemas"]["PeekView"];

/**
 * The question a rule is minted from, as both save routes spell it.
 *
 * The shared half of NewSmart and ReplaceRule. Spelled field by field
 * because a rule saved from `Object.fromEntries` keeps only the LAST `f`,
 * and a two-facet view would then become a one-facet collection -- and
 * because a body carrying a stray URL parameter is refused outright now
 * that every field is named.
 */
type Rule = Pick<
  components["schemas"]["NewSmart"],
  "take" | "folder" | "person" | "artifact" | "kind" | "favorite" | "rating_min" | "q" | "sort" | "f"
>;

const asked = (spelled: string, take: number | null): Rule => {
  const question = new URLSearchParams(spelled);
  // null, never undefined: the contract spells an absent parameter
  // `str | None`, and under exactOptionalPropertyTypes a present-and-
  // undefined property is not the same as an absent one.
  const one = (name: string) => question.get(name);
  const counted = (name: string) => {
    const held = question.get(name);
    return held === null ? null : Number(held);
  };
  return {
    take,
    folder: one("folder"),
    person: one("person"),
    artifact: one("artifact"),
    kind: one("kind"),
    favorite: one("favorite"),
    rating_min: counted("rating_min"),
    q: one("q"),
    sort: one("sort"),
    f: question.getAll("f"),
  };
};

(() => {
  // The form must ask a question the server can answer: a phrase orders
  // by similarity (the seam refuses the contradiction), and empty fields
  // have no place in a canonical URL.
  const ask = findElement(document, "[data-ask]", HTMLFormElement);
  if (ask) {
    const fields = () => [
      ...everyElement(ask, "input", HTMLInputElement),
      ...everyElement(ask, "select", HTMLSelectElement),
    ];
    ask.addEventListener("submit", () => {
      const phrase = requireElement(ask, '[name="q"]', HTMLInputElement);
      const sort = requireElement(ask, '[name="sort"]', HTMLSelectElement);
      if (phrase.value.trim()) sort.value = "similarity";
      else if (sort.value === "similarity") sort.value = "newest";
      for (const field of fields()) {
        if (!field.value.trim()) field.disabled = true; // a disabled field is not submitted
      }
    });
    // The back/forward cache restores the DOM as submitted -- fields
    // disabled for the send must come back usable.
    window.addEventListener("pageshow", () => {
      for (const field of fields()) field.disabled = false;
    });
  }

  const grid = () => findElement(document, "[data-grid]", HTMLElement);

  /**
   * The canonical spelling of the answer ON SCREEN.
   *
   * Not the address bar: ResultSet heals retired entity spellings into
   * `data-qbase`, and a save must persist the identity being looked at, not
   * the stale words the URL arrived with. `page` and `size` are paging, not
   * meaning, so they are never part of a rule.
   */
  const spelling = () => {
    const mounted = grid();
    const held = mounted ? requireData(mounted, "qbase").replace(/&$/, "") : window.location.search;
    const question = new URLSearchParams(held);
    question.delete("page");
    question.delete("size");
    return question.toString();
  };

  /** How much of a ranked library belongs to the collection, or null. */
  const cutoff = (spelled: string): number | null | undefined => {
    if (!new URLSearchParams(spelled).get("q")) return null;
    // Similarity ranks the WHOLE library, so only a cutoff makes it a
    // membership set. Undefined means the person cancelled.
    const held = window.prompt("how many top results belong to it?", "100");
    return held === null ? undefined : Number(held);
  };

  // Save the CURRENT question as a smart collection: the server
  // reconstructs the typed rule from the canonical spelling -- the browser
  // sends the URL's own parameters and a name, never a rule shape.
  const saver = findElement(document, "[data-save-smart]", HTMLElement);
  saver?.addEventListener("click", async () => {
    const spelled = spelling();
    const name = window.prompt("name this smart collection");
    if (!name) return;
    const take = cutoff(spelled);
    if (take === undefined) return;
    const { data, error } = await api.POST("/albums/smart", { body: { name, ...asked(spelled, take) } });
    if (!data) {
      window.alert(refusal(error, "the view could not be saved"));
      return;
    }
    window.location.assign(`/t/${data.slug}`);
  });

  // The other half of the save-view pair: this view becomes an EXISTING
  // smart collection's whole new rule. The target's current definition
  // revision comes from its authoritative document, so a concurrent edit
  // is a 409, never a silent overwrite.
  const replacer = findElement(document, "[data-replace-smart]", HTMLElement);
  replacer?.addEventListener("click", async () => {
    const shelf = await api.GET("/albums", { headers: { accept: "application/json" } });
    const smarts = (shelf.data ?? []).filter((held) => held.kind === "smart");
    const first = smarts[0];
    if (first === undefined) {
      window.alert("no smart collection exists yet -- save the view as a new one instead");
      return;
    }
    const named = window.prompt(
      `replace the rule of which smart collection?\n${smarts.map((held) => held.slug).join(", ")}`,
      first.slug,
    );
    if (!named) return;
    const current = await api.GET("/t/{slug}", {
      params: { path: { slug: named } },
      headers: { accept: "application/json" },
    });
    if (!current.data) {
      window.alert(`no collection at /t/${named}`);
      return;
    }
    const spelled = spelling();
    const take = cutoff(spelled);
    if (take === undefined) return;
    const { data, error } = await api.PUT("/t/{slug}/rule", {
      params: { path: { slug: named } },
      body: { expected_rev: current.data.definition_rev, ...asked(spelled, take) },
    });
    if (!data) {
      window.alert(refusal(error, "the rule could not be replaced"));
      return;
    }
    window.location.assign(`/t/${data.slug}`);
  });

  const rail = findElement(document, "[data-rail]", HTMLElement);
  if (!rail) return;
  const thumb = requireElement(rail, "[data-rail-thumb]", HTMLElement);
  const pop = requireElement(rail, "[data-rail-pop]", HTMLElement);
  const popLabel = requireElement(pop, "[data-rail-pop-label]", HTMLElement);
  const popGrid = requireElement(pop, "[data-rail-pop-grid]", HTMLElement);

  type Shape = { page: number; pages: number; currency: string; answer: string; qbase: string };

  const shape = (): Shape | null => {
    const mounted = grid();
    if (!mounted) return null;
    return {
      page: Number(requireData(mounted, "page")),
      pages: Number(requireData(mounted, "pages")),
      currency: requireData(mounted, "currency"),
      answer: requireData(mounted, "answer"),
      qbase: requireData(mounted, "qbase"),
    };
  };

  // The rail is the ORDERED RESULT SET at full height: a fraction of the
  // track is a fraction of the answer, never of scroll height.
  const pageAt = (clientY: number, s: Shape) => {
    const box = rail.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (clientY - box.top) / box.height));
    return Math.min(s.pages, Math.max(1, Math.round(fraction * (s.pages - 1)) + 1));
  };

  // The rail hangs from the header's REAL bottom edge. The header is
  // sticky and wraps with its controls, so a fixed offset left the top of
  // the rail -- the newest pages -- under it, where a hover reached the
  // header and never the rail.
  const bar = findElement(document, "header.bar", HTMLElement);
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

  // A preview must belong to the SAME answer as the grid it floats beside
  // -- the answer identity, not the currency. Currency is the library's
  // data_version, which every commit moves: a running job's bookkeeping
  // moved it once per item, so every hover during a job was a 409 and a
  // whole-page reload. The answer is the sha of the ordering itself; the
  // same answer means the same ordering, and a preview of it is true. A
  // different answer redraws the gallery from the URL -- two orderings are
  // never presented as one.
  const peeked = new Map<string, PeekView>();
  const peek = async (page: number, s: Shape): Promise<PeekView | null> => {
    const key = `${s.answer}:${page}`;
    const held = peeked.get(key);
    if (held) return held;
    const question = new URLSearchParams(s.qbase);
    const { data } = await api.GET("/g/peek", {
      params: {
        query: {
          folder: question.get("folder"),
          album: question.get("album"),
          person: question.get("person"),
          artifact: question.get("artifact"),
          kind: question.get("kind"),
          favorite: question.get("favorite"),
          rating_min: question.get("rating_min") === null ? null : Number(question.get("rating_min")),
          q: question.get("q"),
          sort: question.get("sort"),
          f: question.getAll("f"),
          page,
          count: 9,
        },
      },
    });
    if (!data) return null;
    if (data.answer !== s.answer) {
      window.location.reload();
      return null;
    }
    peeked.set(key, data);
    return data;
  };

  // A page is asked for once the pointer RESTS on it. A sweep down the
  // rail crosses dozens of pages; nine thumbs for each of them queued the
  // page actually wanted behind eighty pictures nobody will see.
  const REST_MS = 60;
  let hoverPage: number | null = null;
  let resting = 0;
  const show = async (page: number, s: Shape) => {
    const told = await peek(page, s);
    if (!told || hoverPage !== page) return;
    popLabel.textContent = `page ${told.page} of ${told.pages} · ${told.first_ordinal}–${told.last_ordinal} of ${told.total}`;
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
    resting = window.setTimeout(() => void show(page, s), REST_MS);
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
  // The shell (overlay.ts) owns open/mount, push-replace policy,
  // Back-on-dismiss, popstate and the generation check. What is MEDIA'S
  // alone lives here: which currency the view is walking, and the arrows
  // -- each a REPLACE, so browsing fifty items is one Back out.
  // The viewer inside the mounted fragment. Each open replaces the
  // overlay's contents, so the previous mount's listeners are released
  // and a fresh one is bound over the new stage -- the fragment IS the
  // state, and a viewer outliving its DOM would be pointing at nothing.
  let viewer: Viewer | null = null;

  const lightbox = addressableOverlay({
    root: "[data-lightbox-root]",
    trigger: "a.cell",
    pathPrefix: "/i/",
    dismiss: () => viewer?.unwind() ?? false,
    mounted: (mounted) => {
      viewer?.release();
      const held = mounted && findElement(mounted, "[data-viewer]", HTMLElement);
      // a step REPLACES the mount, so browsing fifty items is one Back out
      viewer = held ? mountViewer(held, (href) => void lightbox?.open(href, "replace")) : null;
    },
    generation: () => {
      const shown = findElement(document, "[data-lightbox]", HTMLElement);
      const held = shown?.dataset.currency;
      if (held) return held;
      const s = shape();
      return s ? s.currency : "";
    },
    // A 409'd arrow proves the generation moved, not that THIS answer did
    // -- a favorite, a background job's bookkeeping, any commit at all
    // moves data_version. Ask locate for the walked context's (currency,
    // answer): the same answer identity means the mounted walk is still
    // true, so adopt the fresh currency and let the shell retry once. A
    // changed or vanished answer stays a full redraw.
    recover: async () => {
      const shown = findElement(document, "[data-lightbox]", HTMLElement);
      const mounted = shown?.dataset.answer || grid()?.dataset.answer || "";
      const slug = shown?.dataset.slug;
      if (!mounted || !slug) return false;
      const question = new URLSearchParams(window.location.search);
      const { data } = await api.GET("/g/locate/{slug}", {
        params: {
          path: { slug },
          query: {
            folder: question.get("folder"),
            album: question.get("album"),
            person: question.get("person"),
            artifact: question.get("artifact"),
            kind: question.get("kind"),
            favorite: question.get("favorite"),
            rating_min: question.get("rating_min") === null ? null : Number(question.get("rating_min")),
            q: question.get("q"),
            sort: question.get("sort"),
            size: question.get("size") === null ? null : Number(question.get("size")),
            f: question.getAll("f"),
          },
        },
      });
      // `in_answer` discriminates: a NotLocated carries no answer to
      // compare, and the type says so rather than leaving it undefined.
      if (!data?.in_answer || data.answer !== mounted) return false;
      for (const surface of [shown, grid()]) {
        if (surface) {
          surface.dataset.currency = data.currency;
          surface.dataset.answer = data.answer;
        }
      }
      return true;
    },
  });
  if (lightbox) {
    document.addEventListener("click", (event) => {
      const nav = closestFrom(event.target, "[data-nav]", HTMLAnchorElement);
      if (nav && isPlainClick(event, nav)) {
        event.preventDefault();
        lightbox.open(nav.href, "replace");
      }
    });
    // The arrow keys are the VIEWER's now (frontend/src/viewer.ts): walking
    // is what somebody looking at a picture does, and it worked in the
    // overlay and nowhere else while this file owned it.
  }
})();
