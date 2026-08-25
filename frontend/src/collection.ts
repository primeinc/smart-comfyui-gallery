// Collection lifecycle, from the browser: the same one write adapter
// machines use, never a second opinion about the rules.
//
// Every control states the DESIRED FINAL STATE -- name is X, parent is Y,
// archived is true -- with the definition revision the page rendered at
// (data-rev). A stale revision returns 409, and this module's whole
// reaction is to reload: the server is authoritative and the browser never
// invents the resulting state. An empty input means CLEAR (sent as null);
// a fact the form does not carry is simply absent and stays unchanged --
// which the contract keeps apart, because pydantic knows which fields a
// request actually named.
import { api, refusal } from "./api";
import { findElement, requireData } from "./dom";
import type { components } from "./generated/api";

type WriteAnswer = components["schemas"]["CollectionWriteAnswer"];
type ListedKind = components["schemas"]["ConvertToListed"]["kind"];

/**
 * Go where the write says the collection now lives, or say why not.
 *
 * A 409 means the definition moved under the open editor: the page is
 * stale in a way no local patch can fix, so it reloads rather than
 * pretending the click landed.
 */
const landed = (result: { data?: WriteAnswer | undefined; error?: unknown; response: Response }) => {
  if (result.data) {
    window.location.assign(`/t/${result.data.slug}`);
    return;
  }
  window.alert(refusal(result.error, "the collection did not accept that"));
  if (result.response.status === 409) window.location.reload();
};

/** A form field's value, or null when it was left empty -- which is CLEAR. */
const said = (form: HTMLFormElement, name: string): string | null => {
  const held = new FormData(form).get(name);
  return typeof held === "string" && held.trim() ? held : null;
};

/**
 * The listed kinds, proven rather than asserted.
 *
 * `data-convert` is markup, so its value is a string at runtime. The
 * alternative is `as ListedKind`, which would let a template typo become a
 * 400 nobody was expecting.
 */
const asListedKind = (held: string): ListedKind => {
  if (held !== "album" && held !== "flag") {
    throw new Error(`data-convert offered ${held}, which is not a listed kind`);
  }
  return held;
};

(() => {
  const creating = findElement(document, "[data-new-collection]", HTMLFormElement);
  creating?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = said(creating, "name");
    if (!name) return;
    const kind = said(creating, "kind");
    const { data, error } = await api.POST("/albums", {
      body: { name, kind: kind === null ? "album" : asListedKind(kind) },
    });
    if (!data) {
      window.alert(refusal(error, "the collection could not be created"));
      return;
    }
    window.location.assign(`/t/${data.slug}`);
  });

  const root = findElement(document, "[data-collection]", HTMLElement);
  if (!root) return;
  const slug = requireData(root, "collection");
  // The revision the page was RENDERED at. Read through requireData so a
  // page that stops carrying it throws here, rather than sending NaN as a
  // concurrency token and being refused for the wrong reason.
  const expected_rev = Number(requireData(root, "rev"));

  const editing = findElement(root, "[data-edit-definition]", HTMLFormElement);
  editing?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = said(editing, "name");
    if (!name) return;
    landed(
      await api.PATCH("/t/{slug}", {
        params: { path: { slug } },
        body: {
          expected_rev,
          name,
          description: said(editing, "description"),
          color: said(editing, "color"),
          parent: said(editing, "parent"),
        },
      }),
    );
  });

  const onClick = (selector: string, ask: (control: HTMLElement) => Promise<void>) => {
    const control = findElement(root, selector, HTMLElement);
    control?.addEventListener("click", () => void ask(control));
  };

  const archived = async (value: boolean) => {
    landed(await api.PATCH("/t/{slug}", { params: { path: { slug } }, body: { expected_rev, archived: value } }));
  };

  onClick("[data-archive]", () => archived(true));
  onClick("[data-restore]", () => archived(false));

  // Becoming smart and becoming listed are different requests now, and the
  // contract is a union over `kind`: a rule-shaped field on a listed
  // conversion, or a discard on a smart one, is refused by name.
  onClick("[data-convert]", async (control) => {
    const wanted = requireData(control, "convert");
    const body =
      wanted === "smart"
        ? ({ kind: "smart", expected_rev } as const)
        : { kind: asListedKind(wanted), expected_rev, discard_rule: false };
    landed(await api.POST("/t/{slug}/convert", { params: { path: { slug } }, body }));
  });

  onClick("[data-discard-rule]", async () => {
    if (!window.confirm("discard this collection's rule and keep it as a plain album?")) return;
    landed(
      await api.POST("/t/{slug}/convert", {
        params: { path: { slug } },
        body: { kind: "album", expected_rev, discard_rule: true },
      }),
    );
  });
})();
