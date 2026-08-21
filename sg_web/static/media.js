// The standalone media page: a direct/pasted item has no gallery in its
// history, so Escape goes to the computed return-to-results URL -- never
// a blind history.back() that could leave the site entirely.
(() => {
  "use strict";

  const back = document.querySelector("[data-return]");
  if (!back) return;
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") window.location.assign(back.getAttribute("href"));
  });
})();
