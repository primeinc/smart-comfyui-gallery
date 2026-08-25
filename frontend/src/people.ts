// The person drawer: the second concrete addressable-overlay Adapter.
// Same history contract as the media lightbox -- open PUSHES /p/{slug}
// over the mounted People index, Back/Escape/close leave in one step,
// Forward re-opens what the URL names -- but a person is an entity with a
// collection, not a piece of media, so there are no arrows: the drawer
// shows who they are and hands off to the full profile.
import { api, refusal } from "./api";
import { closestFrom, everyElement, requireData, requireElement } from "./dom";
import { addressableOverlay } from "./overlay";

// days on the cards and in the drawer, formatted for the reader
const pad = (n: number) => String(n).padStart(2, "0");

/**
 * Format every epoch not yet formatted.
 *
 * Only nodes without `data-spelled`: the observer below fires on this very
 * write, and formatting an already-formatted node again would loop forever.
 */
const spellDays = (root: ParentNode) => {
  for (const node of everyElement(root, "time[data-epoch]:not([data-spelled])", HTMLTimeElement)) {
    const d = new Date(Number(requireData(node, "epoch")) * 1000);
    node.textContent = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
    node.dataset.spelled = "";
  }
};

(() => {
  spellDays(document);
  new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });

  // Renaming is the People page's primary action, on the drawer and the
  // full page alike: POST the name, then go live at the new address.
  document.addEventListener("submit", async (event) => {
    const form = closestFrom(event.target, "[data-rename]", HTMLFormElement);
    if (!form) return;
    event.preventDefault();
    // The address the form posts to carries the person; the slug comes
    // from the markup rather than from parsing the action back apart.
    const slug = requireData(form, "rename");
    const name = requireElement(form, '[name="name"]', HTMLInputElement).value;
    const { data, error } = await api.POST("/p/{slug}/name", { params: { path: { slug } }, body: { name } });
    if (!data) {
      window.alert(refusal(error, "that name was refused"));
      return;
    }
    // REPLACE, never assign: the identity's address just moved, and the
    // retired slug must not remain as a history stop -- Back from the
    // renamed profile goes to /people in one step, not through a 301
    // bounce off the old address.
    window.location.replace(`/p/${data.slug}`);
  });

  // The drawer is the person adapter over the AddressableOverlay: the
  // shell (overlay.ts) owns everything an overlay shares, a person is not
  // media, so there is nothing left to add here -- no arrows, no
  // generation evidence.
  //
  // Asked for unconditionally. The full profile at /p/{slug} renders no
  // drawer root, so this returns null there -- the absence is a fact about
  // that page's DOM, where it used to be a fact about whether the template
  // happened to list overlay.js beside this file.
  addressableOverlay({
    root: "[data-drawer-root]",
    trigger: "[data-person]",
    pathPrefix: "/p/",
  });
})();
