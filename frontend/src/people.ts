// The person drawer: the second concrete addressable-overlay Adapter.
// Same history contract as the media lightbox -- open PUSHES /p/{slug}
// over the mounted People index, Back/Escape/close leave in one step,
// Forward re-opens what the URL names -- but a person is an entity with
// a collection, not a piece of media, so there are no arrows: the
// drawer shows who they are and hands off to the full profile.
import { addressableOverlay } from "./overlay";

(() => {
  // days on the cards and in the drawer, spelled for the reader
  const pad = (n) => String(n).padStart(2, "0");
  // only nodes not yet spelled: the observer below fires on this very
  // write, and spelling an already-spelled node again would loop forever
  const spellDays = (root) => {
    for (const node of root.querySelectorAll("time[data-epoch]:not([data-spelled])")) {
      const d = new Date(Number(node.dataset.epoch) * 1000);
      node.textContent = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`;
      node.dataset.spelled = "";
    }
  };
  spellDays(document);
  new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });

  // Renaming is the People page's primary action, on the drawer and the
  // full page alike: POST the name, then go live at the new address.
  document.addEventListener("submit", async (event) => {
    const form = event.target.closest("[data-rename]");
    if (!form) return;
    event.preventDefault();
    const answer = await fetch(form.getAttribute("action"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: form.querySelector('[name="name"]').value }),
    });
    if (!answer.ok) {
      const why = await answer.json().catch(() => ({}));
      window.alert(why.detail || "that name was refused");
      return;
    }
    const told = await answer.json();
    // REPLACE, never assign: the identity's address just moved, and the
    // retired slug must not remain as a history stop -- Back from the
    // renamed profile goes to /people in one step, not through a 301
    // bounce off the old address.
    window.location.replace(`/p/${told.slug}`);
  });

  // The drawer is the person adapter over the AddressableOverlay: the
  // shell (overlay.ts) owns everything an overlay shares, a person is
  // not media, so there is nothing left to add here -- no arrows, no
  // generation evidence. Rename above is this page's own primary
  // action, drawer or not.
  //
  // Asked for unconditionally. The full profile at /p/{slug} renders no
  // drawer root, so this returns null there -- the absence is a fact
  // about that page's DOM, where it used to be a fact about whether the
  // template happened to list overlay.js beside this file.
  addressableOverlay({
    root: "[data-drawer-root]",
    trigger: "[data-person]",
    pathPrefix: "/p/",
  });
})();
