// Installing the gallery as an app -- all three behavior classes, because
// there is no single "install a PWA" API and the biggest field failure is
// code gated on `beforeinstallprompt`, a NON-STANDARD Chromium-only event
// (WICG manifest-incubations; BCD marks it standard_track: false):
//
//   1  Chromium (Chrome/Edge, desktop + Android): stash the prompt event,
//      reveal the shell's install button, prompt on click. Single-use.
//   2  Safari iOS: no prompt event exists at all -- a detection-gated hint
//      says Share -> Add to Home Screen. UA sniffing is acceptable for
//      exactly this decision, because there is no feature to detect.
//   3  Android ambient install: the manifest itself is the affordance;
//      no code here.
//
// Firefox gets nothing on purpose: it does not implement manifest install;
// its Windows "Add to taskbar" is browser chrome, not ours.
//
// Nothing renders when the app is already installed (display-mode media
// queries; `navigator.standalone` is WebKit's spelling) or after the
// person has dismissed an affordance once -- nagging installed users is a
// failure equal to showing nothing.

const DISMISSED = "sg-install-dismissed";

// Chromium's non-standard event, absent from lib.dom.
type InstallPrompt = Event & { prompt(): Promise<void>; userChoice: Promise<{ outcome: string }> };
type MaybeStandalone = Navigator & { standalone?: boolean };

const standalone = (): boolean => (navigator as MaybeStandalone).standalone === true;

export const installed = (): boolean => window.matchMedia("not (display-mode: browser)").matches || standalone();

const dismissed = (): boolean => {
  try {
    return localStorage.getItem(DISMISSED) === "1";
  } catch {
    return false;
  }
};

const markDismissed = (): void => {
  try {
    localStorage.setItem(DISMISSED, "1");
  } catch {
    /* a browser refusing storage just gets asked again next visit */
  }
};

// iPadOS reports a Macintosh UA; touch points are what give it away.
const isIOS =
  /iphone|ipad|ipod/i.test(navigator.userAgent) ||
  (/macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1);

export function mountInstall(): void {
  const button = document.querySelector("[data-install]");
  const hint = document.querySelector("[data-install-ios]");
  if (installed() || dismissed()) return;

  let stashed: InstallPrompt | null = null;
  window.addEventListener("beforeinstallprompt", (event) => {
    event.preventDefault();
    stashed = event as InstallPrompt;
    if (button instanceof HTMLElement) button.hidden = false;
  });

  if (button instanceof HTMLElement) {
    button.addEventListener("click", async () => {
      if (!stashed) return;
      stashed.prompt();
      const { outcome } = await stashed.userChoice;
      stashed = null; // single-use by contract
      button.hidden = true;
      if (outcome === "dismissed") markDismissed();
    });
  }

  window.addEventListener("appinstalled", () => {
    if (button instanceof HTMLElement) button.hidden = true;
    if (hint instanceof HTMLElement) hint.hidden = true;
  });

  // Never weaken this gate: the isIOS half keeps it off Android and
  // desktop, the standalone half keeps it out of the installed app.
  if (isIOS && !standalone() && hint instanceof HTMLElement) {
    hint.hidden = false;
    hint.querySelector("[data-dismiss]")?.addEventListener("click", () => {
      hint.hidden = true;
      markDismissed();
    });
  }
}

// Registration + the safe update flow: a new worker WAITS until the
// person opts in -- an unconditional skipWaiting would swap the version
// under a page whose lazy chunks came from the old one.
export function mountServiceWorker(): void {
  if (!("serviceWorker" in navigator)) return; // http:// on the LAN, or an old browser
  window.addEventListener("load", async () => {
    const reg = await navigator.serviceWorker.register("/sw.js", { updateViaCache: "none" });
    reg.addEventListener("updatefound", () => {
      const next = reg.installing;
      if (!next) return;
      next.addEventListener("statechange", () => {
        // installed + an existing controller = a new version waiting;
        // a first-ever install has no controller and needs no toast.
        if (next.state !== "installed" || !navigator.serviceWorker.controller) return;
        const notice = document.querySelector("[data-shell-notice]");
        if (!(notice instanceof HTMLElement)) return;
        notice.textContent = "a new version of the gallery is ready — ";
        const go = document.createElement("button");
        go.type = "button";
        go.className = "link";
        go.textContent = "reload";
        go.addEventListener("click", () => reg.waiting?.postMessage({ type: "SKIP_WAITING" }));
        notice.append(go);
      });
    });
  });
  // Reload only when a NEW version took over, never when the first one
  // arrives. The same distinction the notice above draws, and for the
  // same reason: `sw.js` calls `clients.claim()` on activate
  // (static/sw.js), so the very first visit fires `controllerchange`
  // with nothing to swap -- the page already holds exactly what the new
  // worker would serve. Reloading there is a whole second page load
  // handed to every first-time visitor.
  //
  // Read BEFORE the listener, because by the time it fires the
  // controller is the new one either way; what tells an update from a
  // first claim is whether there was one to begin with.
  //
  // It also put a navigation under the browser suite. One `goto` was
  // two main-frame navigations with service workers allowed and one
  // with them blocked (measured), landing at whatever moment activation
  // finished -- which is what "Execution context was destroyed, most
  // likely because of a navigation" reads like from a test.
  const wasControlled = Boolean(navigator.serviceWorker.controller);
  let refreshing = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (!wasControlled || refreshing) return;
    refreshing = true;
    location.reload();
  });
}
