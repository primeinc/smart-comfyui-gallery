// The person drawer: the second concrete addressable-overlay Adapter.
// Same history contract as the media lightbox -- open PUSHES /p/{slug}
// over the mounted People index, Back/Escape/close leave in one step,
// Forward re-opens what the URL names -- but a person is an entity with
// a collection, not a piece of media, so there are no arrows: the
// drawer shows who they are and hands off to the full profile.
(() => {
  "use strict";

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
  // shell (static/overlay.js) owns everything an overlay shares, a
  // person is not media, so there is nothing left to add here -- no
  // arrows, no generation evidence. Rename above is this page's own
  // primary action, drawer or not.
  if (window.sgAddressableOverlay) {
    window.sgAddressableOverlay({
      root: "[data-drawer-root]",
      trigger: "[data-person]",
      pathPrefix: "/p/",
    });
  }
})();
