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

  // a moment's caption is a door into the clip: play from that second
  const video = document.querySelector("video");
  for (const at of document.querySelectorAll("[data-said-seek]")) {
    at.addEventListener("click", () => {
      if (!video) return;
      video.currentTime = Number(at.dataset.saidSeek) / 1000;
      video.play();
    });
  }

  const back = document.querySelector("[data-return]");
  if (!back) return;
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") window.location.assign(back.getAttribute("href"));
  });
})();
