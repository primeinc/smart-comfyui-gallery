// The story page: the same plan under another profile is one POST to
// /stories/renders -- content-addressed, so asking twice is one row --
// and the page moves to the render it names.
(() => {
  "use strict";
  const main = document.querySelector("[data-story-render]");
  if (!main) return;
  const status = document.querySelector("[data-story-status]");
  for (const button of document.querySelectorAll("[data-story-profile-ask]")) {
    button.addEventListener("click", async () => {
      const profile = button.dataset.storyProfileAsk;
      status.textContent = `rendering ${profile}…`;
      const r = await fetch("/stories/renders", {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: JSON.stringify({ plan_id: Number(main.dataset.storyPlan), profile, locale: main.dataset.storyLocale }),
      });
      const told = await r.json().catch(() => ({}));
      if (!r.ok) { status.textContent = told.detail || r.statusText; return; }
      window.location.href = `/stories/renders/${told.id}`;
    });
  }
})();
