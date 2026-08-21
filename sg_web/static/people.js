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
    window.location.assign(`/p/${told.slug}`);
  });

  const drawer = document.querySelector("[data-drawer-root]");
  if (!drawer) return;

  const open = async (href, mode) => {
    const answer = await fetch(href, { headers: { "HX-Request": "true" } });
    if (!answer.ok) {
      window.location.assign(href);
      return;
    }
    drawer.innerHTML = await answer.text();
    drawer.hidden = false;
    if (mode === "push") history.pushState({ sgDrawer: true }, "", href);
  };
  const close = () => {
    drawer.hidden = true;
    drawer.replaceChildren();
  };

  document.addEventListener("click", (event) => {
    const card = event.target.closest("[data-person]");
    if (card) {
      event.preventDefault();
      open(card.getAttribute("href"), "push");
      return;
    }
    if (event.target.closest("[data-close]")) {
      event.preventDefault();
      history.back();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (!drawer.hidden && event.key === "Escape") history.back();
  });
  window.addEventListener("popstate", () => {
    if (window.location.pathname.startsWith("/p/")) open(window.location.href, "none");
    else if (!drawer.hidden) close();
  });
})();
