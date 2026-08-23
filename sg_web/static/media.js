// The standalone media page: a direct/pasted item has no gallery in its
// history, so Escape goes to the computed return-to-results URL -- never
// a blind history.back() that could leave the site entirely.
(() => {
  "use strict";
  // the "when" block speaks its clock domain
  const pad = (n) => String(n).padStart(2, "0");
  for (const node of document.querySelectorAll("time[data-epoch]")) {
    const d = new Date(Number(node.dataset.epoch) * 1000);
    const z = node.dataset.domain === "instant" ? "Z" : " wall";
    node.textContent = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}${z}`;
  }

  // a moment's caption is a link into the clip: play from that second
  const video = document.querySelector("video");
  for (const at of document.querySelectorAll("[data-said-seek]")) {
    at.addEventListener("click", () => {
      if (!video) return;
      video.currentTime = Number(at.dataset.saidSeek) / 1000;
      video.play();
    });
  }

  // where it happened: one POST of desired state, then the page re-reads
  const placeForm = document.querySelector("[data-place-form]");
  if (placeForm) {
    const say = async (body) => {
      const answer = await fetch(`/i/${placeForm.dataset.slug}/place`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!answer.ok) {
        window.alert((await answer.json().catch(() => ({}))).detail || "the place could not be recorded");
        return;
      }
      window.location.reload();
    };
    placeForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const name = placeForm.querySelector('[name="name"]').value.trim();
      if (!name) return;
      const within = placeForm.querySelector('[name="within"]').value.trim();
      say({
        name,
        kind: placeForm.querySelector('[name="kind"]').value,
        within: within || null,
        within_kind: placeForm.querySelector('[name="within_kind"]').value,
      });
    });
    placeForm.querySelector("[data-place-clear]")?.addEventListener("click", () => say({ name: null }));
  }

  const back = document.querySelector("[data-return]");
  if (!back) return;
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") window.location.assign(back.getAttribute("href"));
  });
})();
