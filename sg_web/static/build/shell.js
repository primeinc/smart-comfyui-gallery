"use strict";
(() => {
  // src/install.ts
  var DISMISSED = "sg-install-dismissed";
  var standalone = () => navigator.standalone === true;
  var installed = () => window.matchMedia("not (display-mode: browser)").matches || standalone();
  var dismissed = () => {
    try {
      return localStorage.getItem(DISMISSED) === "1";
    } catch {
      return false;
    }
  };
  var markDismissed = () => {
    try {
      localStorage.setItem(DISMISSED, "1");
    } catch {
    }
  };
  var isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent) || /macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1;
  function mountInstall() {
    const button = document.querySelector("[data-install]");
    const hint = document.querySelector("[data-install-ios]");
    if (installed() || dismissed()) return;
    let stashed = null;
    window.addEventListener("beforeinstallprompt", (event) => {
      event.preventDefault();
      stashed = event;
      if (button instanceof HTMLElement) button.hidden = false;
    });
    if (button instanceof HTMLElement) {
      button.addEventListener("click", async () => {
        if (!stashed) return;
        stashed.prompt();
        const { outcome } = await stashed.userChoice;
        stashed = null;
        button.hidden = true;
        if (outcome === "dismissed") markDismissed();
      });
    }
    window.addEventListener("appinstalled", () => {
      if (button instanceof HTMLElement) button.hidden = true;
      if (hint instanceof HTMLElement) hint.hidden = true;
    });
    if (isIOS && !standalone() && hint instanceof HTMLElement) {
      hint.hidden = false;
      hint.querySelector("[data-dismiss]")?.addEventListener("click", () => {
        hint.hidden = true;
        markDismissed();
      });
    }
  }
  function mountServiceWorker() {
    if (!("serviceWorker" in navigator)) return;
    window.addEventListener("load", async () => {
      const reg = await navigator.serviceWorker.register("/sw.js", { updateViaCache: "none" });
      reg.addEventListener("updatefound", () => {
        const next = reg.installing;
        if (!next) return;
        next.addEventListener("statechange", () => {
          if (next.state !== "installed" || !navigator.serviceWorker.controller) return;
          const notice = document.querySelector("[data-shell-notice]");
          if (!(notice instanceof HTMLElement)) return;
          notice.textContent = "a new version of the gallery is ready \u2014 ";
          const go = document.createElement("button");
          go.type = "button";
          go.className = "link";
          go.textContent = "reload";
          go.addEventListener("click", () => reg.waiting?.postMessage({ type: "SKIP_WAITING" }));
          notice.append(go);
        });
      });
    });
    let refreshing = false;
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!refreshing) {
        refreshing = true;
        location.reload();
      }
    });
  }

  // src/entries/shell.ts
  mountInstall();
  mountServiceWorker();
})();
//# sourceMappingURL=shell.js.map
