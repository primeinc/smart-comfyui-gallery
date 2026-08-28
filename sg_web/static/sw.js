/**
 * The service worker, deliberately small. Every page this application
 * serves is live data over the library's database, so caching documents
 * would hand back yesterday's library as if it were today's -- the
 * honest offline story is one cached fallback page and nothing else.
 * Media and static assets already ride immutable content-addressed HTTP
 * caching (sg_web/app.py ASSET_CACHE, ?v= on /static); a second cache
 * in front of that would only hide it.
 *
 * A fetch handler must exist at all: Chromium's ambient install prompt
 * requires one even though menu-install no longer does (Chrome 108+
 * mobile / 112+ desktop). Navigations go network-first with navigation
 * preload and fall back to the cached offline page -- a real page,
 * never the browser's error screen. Everything else passes through
 * untouched.
 *
 * VERSION busts the precache; activation is gated on the page's
 * "reload for the new version" prompt (sg_web/static/build/app.js)
 * rather than an unconditional skipWaiting, so a mid-session update
 * cannot skew a page against its lazy chunks.
 */
const VERSION = "v1";
const PRECACHE = `precache-${VERSION}`;
const OFFLINE_URL = "/static/offline.html";

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(PRECACHE).then((held) => held.addAll([OFFLINE_URL])));
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      if (self.registration.navigationPreload) {
        await self.registration.navigationPreload.enable();
      }
      const keys = await caches.keys();
      await Promise.all(keys.filter((key) => key !== PRECACHE).map((key) => caches.delete(key)));
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.mode !== "navigate") return;
  event.respondWith(
    (async () => {
      try {
        const preloaded = await event.preloadResponse;
        return preloaded || (await fetch(event.request));
      } catch {
        return await caches.match(OFFLINE_URL);
      }
    })(),
  );
});
