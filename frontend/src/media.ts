// The standalone media page: a direct/pasted item has no gallery in its
// history, so Escape goes to the computed return-to-results URL -- never a
// blind history.back() that could leave the site entirely.
import { api, refusal } from "./api";
import { everyElement, findElement, requireData, requireElement } from "./dom";
import type { components } from "./generated/api";
import { mountViewer } from "./viewer";

type DesiredPlace = components["schemas"]["DesiredPlace"];
type PlaceKind = DesiredPlace["kind"];

const pad = (n: number) => String(n).padStart(2, "0");

/**
 * The place kinds the contract admits, proven rather than asserted.
 *
 * The `<select>` is rendered from the vocabulary the schema constrains
 * (sglint SG709 holds the Python Literal against the CHECK), but its value
 * is a string at runtime, and `as PlaceKind` would let a template typo
 * reach the server as a 400 nobody expected.
 */
const asPlaceKind = (held: string): PlaceKind => {
  const known = ["country", "region", "island", "county", "city", "locality", "neighborhood", "poi"] as const;
  const found = known.find((one) => one === held);
  if (found === undefined) throw new Error(`the place picker offered ${held}, which is not a place kind`);
  return found;
};

(() => {
  // the "when" block speaks its clock domain
  for (const node of everyElement(document, "time[data-epoch]", HTMLTimeElement)) {
    const d = new Date(Number(requireData(node, "epoch")) * 1000);
    const z = node.dataset.domain === "instant" ? "Z" : " wall";
    node.textContent =
      `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ` +
      `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}${z}`;
  }

  // a moment's caption is a link into the clip: play from that second
  const video = findElement(document, "video", HTMLVideoElement);
  for (const at of everyElement(document, "[data-said-seek]", HTMLElement)) {
    at.addEventListener("click", () => {
      if (!video) return;
      video.currentTime = Number(requireData(at, "saidSeek")) / 1000;
      void video.play();
    });
  }

  // where it happened: one POST of desired state, then the page re-reads
  const placeForm = findElement(document, "[data-place-form]", HTMLFormElement);
  if (placeForm) {
    const slug = requireData(placeForm, "slug");
    const value = (name: string) => requireElement(placeForm, `[name="${name}"]`, HTMLInputElement).value.trim();
    const chosen = (name: string) => requireElement(placeForm, `[name="${name}"]`, HTMLSelectElement).value;

    const say = async (body: DesiredPlace) => {
      const { data, error } = await api.POST("/i/{slug}/place", { params: { path: { slug } }, body });
      if (!data) {
        window.alert(refusal(error, "the place could not be recorded"));
        return;
      }
      window.location.reload();
    };

    placeForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const name = value("name");
      if (!name) return;
      const within = value("within");
      void say({
        name,
        kind: asPlaceKind(chosen("kind")),
        within: within || null,
        within_kind: asPlaceKind(chosen("within_kind")),
      });
    });

    // Withdrawing the claim still names the kinds: they carry defaults in
    // Python, but a defaulted field is `required` in the document litestar
    // generates, so the contract asks for them and the browser sends them.
    findElement(placeForm, "[data-place-clear]", HTMLElement)?.addEventListener("click", () => {
      void say({ name: null, kind: "locality", within: null, within_kind: "country" });
    });
  }

  // The viewer, on the page's container. Same module the overlay mounts;
  // what this file owns is only what dismissal MEANS here.
  const mounted = findElement(document, "[data-viewer]", HTMLElement);
  // A page walks by BEING the next page: no gallery is mounted here, so
  // there is nothing to swap a fragment into.
  const viewer = mounted
    ? mountViewer(mounted, (href) => {
        window.location.assign(href);
      })
    : null;

  const back = findElement(document, "[data-return]", HTMLAnchorElement);
  if (!back) return;
  const leave = () => {
    window.location.assign(back.href);
  };

  // A direct or pasted item has no gallery in its history, so leaving goes
  // to the computed return-to-results URL -- never a blind history.back()
  // that could take the browser off the site entirely.
  for (const close of everyElement(document, "[data-close]", HTMLElement)) {
    close.addEventListener("click", leave);
  }
  document.addEventListener("keydown", (event) => {
    // The same ladder the overlay gets: the viewer spends the press on its
    // own state -- a zoomed picture fits, an open inspector closes -- and
    // only a viewer with nothing left to unwind lets Escape mean "leave".
    if (event.key !== "Escape") return;
    if (viewer?.unwind()) return;
    leave();
  });
})();
