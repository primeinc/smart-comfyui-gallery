// Selection: ephemeral browser state, anchored to ONE ResultSet answer.
//
//   { answer, selected: Set<entity uuid> }
//
// Selection belongs to an answer: it survives page swaps while the mounted
// grid's data-answer is unchanged (select on page 1, more on page 3, curate
// all of it at once), and clears the moment a different answer mounts -- a
// toolbar must never operate on files the current question no longer shows.
// Nothing here is durable, nothing rides the URL, and no membership or
// ordering is ever computed in the browser.
//
// Every action states ONE desired fact for the whole selection, sent to a
// bulk route that proves the selection against the authoritative projection
// inside its write transaction. Settlement is the same answer-identity
// contract single writes use: an unchanged after-answer adopts the new
// currency and KEEPS the selection for the next operation; a changed one
// clears and redraws.
import { type Answered, answered, api, refusal } from "./api";
import { closestFrom, findElement, requireData, requireElement } from "./dom";
import type { components, operations } from "./generated/api";

type Curated = components["schemas"]["Curated"];

/**
 * The question the toolbar is curating within, from the URL.
 *
 * Spelled out against the contract's own parameter type rather than handed
 * over as a search string: an object built from the string wholesale would
 * keep only the last `f`, and a two-facet view would then prove its selection
 * against a one-facet question.
 */
type Question = NonNullable<operations["GSelectionFavoriteBulkFavorite"]["parameters"]["query"]>;

type PlaceKind = components["schemas"]["BulkPlace"]["kind"];

const asked = (): Question => {
  const question = new URLSearchParams(window.location.search);
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
    sort: one("sort"),
    size: counted("size"),
    f: question.getAll("f"),
  };
};

(() => {
  const bar = findElement(document, "[data-curate]", HTMLElement);
  if (!bar) return;

  const count = requireElement(bar, "[data-curate-count]", HTMLElement);
  const albums = requireElement(bar, "[data-bulk-album]", HTMLSelectElement);
  const grid = () => findElement(document, "[data-grid]", HTMLElement);

  // The answer the selection was made against. Empty until a grid mounts,
  // which is also what makes a first swap adopt rather than clear.
  let answer = "";
  const selected = new Set<string>();

  const draw = () => {
    bar.hidden = selected.size === 0;
    count.textContent = `${selected.size} selected`;
    for (const shell of document.querySelectorAll<HTMLElement>("[data-selection-key]")) {
      const pick = findElement(shell, "[data-pick]", HTMLInputElement);
      if (pick) pick.checked = selected.has(requireData(shell, "selectionKey"));
    }
  };

  const sync = () => {
    const mounted = grid();
    if (!mounted) return;
    const held = requireData(mounted, "answer");
    if (answer !== held) {
      // A different answer mounted: the old selection named members of a
      // question that is no longer on screen.
      answer = held;
      selected.clear();
    }
    draw();
  };

  /** The albums a picture can be filed into, offered once and kept. */
  const shelve = async () => {
    const { data, error } = await api.GET("/albums", { headers: { accept: "application/json" } });
    if (error || !data) return;
    albums.replaceChildren(
      ...data
        .filter((one) => one.kind !== "smart") // rule-derived membership is not filed
        .map((one) => {
          const choice = document.createElement("option");
          choice.value = one.slug;
          choice.textContent = one.name;
          return choice;
        }),
    );
  };

  /**
   * Settle on what the server read back after the write.
   *
   * A refusal is said out loud rather than swallowed: a rating the server
   * will not take used to leave the toolbar looking like a click that missed.
   */
  const settle = (told: Answered<Curated>) => {
    if (!told.ok) {
      window.alert(told.refusal);
      return;
    }
    const mounted = grid();
    if (told.data.after.answer !== answer) {
      // The selected files left (or re-entered) this answer: the URL owns
      // what renders now.
      window.location.reload();
      return;
    }
    // The facts changed; the question did not. Adopt the generation in place
    // and keep the selection mounted for the next operation.
    if (mounted) {
      mounted.dataset.currency = told.data.after.currency;
      mounted.dataset.answer = told.data.after.answer;
    }
    draw();
  };

  /**
   * A 409 means the answer moved underneath the selection and NOTHING was
   * written. The honest move is a whole redraw and a fresh selection, so it
   * is separated from an ordinary refusal, which the person can act on.
   */
  const told = (result: { data?: Curated | undefined; error?: unknown; response: Response }) => {
    if (result.response.status === 409) {
      window.location.reload();
      return;
    }
    settle(answered(result, refusal(result.error, "the selection could not be curated")));
  };

  const items = () => [...selected];

  const favorite = async (value: boolean) => {
    told(
      await api.POST("/g/selection/favorite", { params: { query: asked() }, body: { answer, items: items(), value } }),
    );
  };

  const rate = async (value: number | null) => {
    told(
      await api.POST("/g/selection/rating", { params: { query: asked() }, body: { answer, items: items(), value } }),
    );
  };

  const file = async (collection: string, value: boolean) => {
    told(
      await api.POST("/g/selection/collections/{collection}", {
        params: { path: { collection }, query: asked() },
        body: { answer, items: items(), value },
      }),
    );
  };

  /**
   * `kind` and `within_kind` are sent every time, including when the name is
   * null and nothing is being minted. They carry defaults in Python, but a
   * defaulted field is still `required` in the document Litestar generates,
   * so the contract says to send them and the browser sends them.
   */
  const place = async (name: string | null, kind: PlaceKind) => {
    told(
      await api.POST("/g/selection/place", {
        params: { query: asked() },
        body: { answer, items: items(), name, kind, within_kind: "country" },
      }),
    );
  };

  document.addEventListener("change", (event) => {
    const pick = closestFrom(event.target, "[data-pick]", HTMLInputElement);
    if (!pick) return;
    const shell = pick.closest<HTMLElement>("[data-selection-key]");
    if (!shell) return;
    const key = requireData(shell, "selectionKey");
    if (pick.checked) selected.add(key);
    else selected.delete(key);
    if (selected.size === 1 && !albums.options.length) void shelve();
    draw();
  });

  bar.addEventListener("click", (event) => {
    const flag = closestFrom(event.target, "[data-bulk-favorite]", HTMLElement);
    if (flag) {
      void favorite(requireData(flag, "bulkFavorite") === "1");
      return;
    }
    const stars = closestFrom(event.target, "[data-bulk-rate]", HTMLElement);
    if (stars) {
      const n = Number(requireData(stars, "bulkRate"));
      void rate(n > 0 ? n : null);
      return;
    }
    const filed = closestFrom(event.target, "[data-bulk-file]", HTMLElement);
    if (filed && albums.value) {
      void file(albums.value, requireData(filed, "bulkFile") === "1");
      return;
    }
    const placed = closestFrom(event.target, "[data-bulk-place]", HTMLElement);
    if (placed) {
      const kind = asPlaceKind(requireElement(bar, "[data-bulk-place-kind]", HTMLSelectElement).value);
      if (requireData(placed, "bulkPlace") !== "1") {
        void place(null, kind);
        return;
      }
      const name = requireElement(bar, "[data-bulk-place-name]", HTMLInputElement).value.trim();
      if (!name) return;
      void place(name, kind);
      return;
    }
    if (closestFrom(event.target, "[data-curate-clear]", HTMLElement)) {
      selected.clear();
      draw();
    }
  });

  document.body.addEventListener("htmx:afterSwap", sync);
  sync();
})();

/**
 * The place kinds the contract admits, proven rather than asserted.
 *
 * The `<select>` is rendered from the same vocabulary the schema constrains
 * (sglint SG709 holds the Python Literal against the CHECK), but its `value`
 * is a string at runtime, and the alternative to this check is `as PlaceKind`
 * -- which would let a template typo reach the server as a 400 nobody
 * expected.
 */
function asPlaceKind(held: string): PlaceKind {
  const known = ["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"] as const;
  const found = known.find((one) => one === held);
  if (found === undefined) throw new Error(`the place picker offered ${held}, which is not a place kind`);
  return found;
}
