// The authored strip: favorite, rating and album membership for the
// media item on screen -- the same markup in the full page and the
// lightbox, because /i/{slug} has one state.
//
// Every write states the DESIRED FINAL FACT (favorite=true, rating=4,
// member=false), so a retry is harmless, and the strip redraws from the
// response's authoritative state, never from its own click.
//
// After a commit the library generation has moved (data_version bumps
// on every commit) while the mounted answer usually has not. The
// coherence check asks locate for the walked context's (currency,
// answer) pair: same answer -> adopt the new currency in place, so the
// next arrow does not 409 over an unchanged answer; different answer or
// no longer in it -> the mounted walk is really stale, redraw whole.
//
// Every request here goes through the generated contract, so the three
// desired-state routes are three named calls rather than one function
// that concatenated a path fragment onto a slug and hoped.
import { type Answered, answered, api } from "./api";
import { closestFrom, everyElement, findElement, requireData, requireElement } from "./dom";
import type { components, paths } from "./generated/api";
import { register } from "./keys";

type AuthoredState = components["schemas"]["AuthoredState"];
type AuthoredAnswer = components["schemas"]["AuthoredAnswer"];
type LocateQuery = NonNullable<paths["/g/locate/{slug}"]["get"]["parameters"]["query"]>;

const draw = (root: HTMLElement, authored: AuthoredState) => {
  requireElement(root, "[data-fav]", HTMLElement).setAttribute("aria-pressed", authored.favorite ? "true" : "false");
  const stars = requireElement(root, "[data-stars]", HTMLElement);
  stars.dataset.rating = String(authored.rating ?? 0);
  for (const star of everyElement(stars, "[data-rate]", HTMLElement)) {
    const n = Number(requireData(star, "rate"));
    if (n > 0) {
      star.setAttribute("aria-pressed", authored.rating !== null && authored.rating >= n ? "true" : "false");
    }
  }
  const albums = requireElement(root, "[data-albums]", HTMLElement);
  albums.replaceChildren(
    ...authored.collections.map((held) => {
      const link = document.createElement("a");
      link.href = `/t/${held.slug}`;
      link.textContent = held.name;
      return link;
    }),
  );
};

// The mounted result-set surfaces this item is being walked over:
// the lightbox fragment and/or the gallery grid behind it.
const mounted = (): HTMLElement[] =>
  [findElement(document, "[data-lightbox]", HTMLElement), findElement(document, "[data-grid]", HTMLElement)].filter(
    (one): one is HTMLElement => one !== null,
  );

/**
 * The mounted question, as locate's own declared parameters.
 *
 * The strip carries the walked question as a query string in `data-qs`.
 * Spelling it out against the contract's parameter type is what makes the
 * repeated `f` survive: an object built from the string wholesale would keep
 * only the last facet, and a two-facet view would locate against a one-facet
 * question.
 */
const asked = (qs: string | undefined): LocateQuery => {
  const question = new URLSearchParams(qs ?? "");
  // null, never undefined: the contract spells an absent parameter
  // `str | None`, and under exactOptionalPropertyTypes a property that is
  // present and undefined is not the same as one that is absent.
  const one = (name: string) => question.get(name);
  const counted = (name: string) => {
    const held = question.get(name);
    return held === null ? null : Number(held);
  };
  return {
    folder: one("folder"),
    album: one("album"),
    person: one("person"),
    artifact: one("artifact"),
    kind: one("kind"),
    favorite: one("favorite"),
    rating_min: counted("rating_min"),
    q: one("q"),
    f: question.getAll("f"),
    sort: one("sort"),
    size: counted("size"),
  };
};

const settle = async (root: HTMLElement) => {
  const surfaces = mounted();
  if (!surfaces.length) return;
  const { data, error } = await api.GET("/g/locate/{slug}", {
    params: { path: { slug: requireData(root, "slug") }, query: asked(root.dataset.qs) },
  });
  if (error || !data) {
    window.location.reload();
    return;
  }
  const held = surfaces[0]?.dataset.answer ?? "";
  if (!data.in_answer || (held && data.answer !== held)) {
    // The walked answer really changed -- the URL owns the state.
    window.location.reload();
    return;
  }
  for (const surface of surfaces) {
    surface.dataset.currency = data.currency;
    surface.dataset.answer = data.answer;
  }
};

/**
 * Redraw from the authoritative answer, then check the mounted walk.
 *
 * A refusal is said out loud. It used to return null and the strip simply
 * did not change, which looks exactly like a click that missed.
 */
const applied = async (root: HTMLElement, told: Answered<AuthoredAnswer>) => {
  if (!told.ok) {
    window.alert(told.refusal);
    return;
  }
  draw(root, told.data.authored);
  await settle(root);
};

const setFavorite = async (root: HTMLElement, value: boolean) => {
  const told = await api.POST("/i/{slug}/favorite", {
    params: { path: { slug: requireData(root, "slug") } },
    body: { value },
  });
  await applied(root, answered(told, "the favorite could not be recorded"));
};

const setRating = async (root: HTMLElement, value: number | null) => {
  const told = await api.POST("/i/{slug}/rating", {
    params: { path: { slug: requireData(root, "slug") } },
    body: { value },
  });
  await applied(root, answered(told, "the rating could not be recorded"));
};

const setMembership = async (root: HTMLElement, collection: string, value: boolean) => {
  const told = await api.POST("/i/{slug}/collections/{collection}", {
    params: { path: { slug: requireData(root, "slug"), collection } },
    body: { value },
  });
  await applied(root, answered(told, "the album membership could not be recorded"));
};

const choices = async (root: HTMLElement) => {
  const box = requireElement(root, "[data-album-choices]", HTMLElement);
  if (!box.hidden) {
    box.hidden = true;
    return;
  }
  const told = answered(
    await api.GET("/i/{slug}/collection-choices", { params: { path: { slug: requireData(root, "slug") } } }),
    "the albums could not be read",
  );
  if (!told.ok) {
    window.alert(told.refusal);
    return;
  }
  const data = told.data;
  box.replaceChildren(
    ...data.map((one) => {
      const row = document.createElement("label");
      const tick = document.createElement("input");
      tick.type = "checkbox";
      tick.checked = one.filed;
      tick.addEventListener("change", () => {
        void setMembership(root, one.slug, tick.checked);
      });
      row.append(tick, ` ${one.name}`);
      return row;
    }),
  );
  if (!data.length) box.textContent = "no albums yet — make one on /albums";
  box.hidden = false;
};

const pressed = (root: HTMLElement) =>
  requireElement(root, "[data-fav]", HTMLElement).getAttribute("aria-pressed") === "true";

document.addEventListener("click", (event) => {
  const root = closestFrom(event.target, "[data-authored]", HTMLElement);
  if (!root) return;
  if (closestFrom(event.target, "[data-fav]", HTMLElement)) {
    void setFavorite(root, !pressed(root));
    return;
  }
  const star = closestFrom(event.target, "[data-rate]", HTMLElement);
  if (star) {
    const n = Number(requireData(star, "rate"));
    void setRating(root, n > 0 ? n : null);
    return;
  }
  if (closestFrom(event.target, "[data-album-picker]", HTMLElement)) void choices(root);
});

/**
 * The strip on the surface being looked at.
 *
 * The lightbox's, when one is mounted: the grid behind it carries its own
 * markup, and a key pressed over an open picture means that picture.
 */
const strip = (): HTMLElement | null =>
  findElement(document, "[data-lightbox] [data-authored]", HTMLElement) ??
  findElement(document, "[data-authored]", HTMLElement);

// Registered, not listened for: these keys and the viewer's ship in the
// same bundle on the same surfaces, and a second claim on one of them is
// refused where it is made rather than firing twice (frontend/src/keys.ts).
// 1-5 stay ratings because every photo tool spells them that way, so the
// viewer's framing moved to Z rather than these moving anywhere.
register([
  {
    key: "f",
    by: "authored: favorite",
    run: () => {
      const root = strip();
      if (root) void setFavorite(root, !pressed(root));
    },
  },
  {
    key: "a",
    by: "authored: albums",
    run: () => {
      const root = strip();
      if (root) void choices(root);
    },
  },
  ...[1, 2, 3, 4, 5].map((stars) => ({
    key: String(stars),
    by: `authored: ${stars} star${stars === 1 ? "" : "s"}`,
    run: () => {
      const root = strip();
      if (root) void setRating(root, stars);
    },
  })),
  {
    key: "0",
    by: "authored: clear rating",
    run: () => {
      const root = strip();
      if (root) void setRating(root, null);
    },
  },
]);
