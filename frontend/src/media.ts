// The standalone media page: a direct/pasted item has no gallery in its
// history, so Escape goes to the computed return-to-results URL -- never a
// blind history.back() that could leave the site entirely.
import { api, refusal } from "./api";
import { say } from "./ask";
import { everyElement, findElement, requireData, requireElement } from "./dom";
import type { components } from "./generated/api";
import { register } from "./keys";
import { mountViewer } from "./viewer";

type DesiredPlace = components["schemas"]["DesiredPlace"];
type PlaceKind = DesiredPlace["kind"];

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
  // a moment's caption is a link into the clip: play from that second
  const video = findElement(document, "video", HTMLVideoElement);
  for (const at of everyElement(document, "[data-said-seek]", HTMLElement)) {
    at.addEventListener("click", () => {
      if (!video) return;
      video.currentTime = Number(requireData(at, "saidSeek")) / 1000;
      void video.play();
    });
  }

  // A thumb on what a model said, where the sentence is shown.
  //
  // The claim is named by WHAT IT IS -- the kind of annotation and the
  // producer that made it -- never by the annotation's row id: the
  // derived layer is disposable and this judgement has to outlive being
  // rebuilt, so an id would point at something a re-run deletes.
  //
  // Clicking the lit thumb sends `null`, which retracts. "I take that
  // back" is no row, not a third verdict nobody expressed.
  const judged = findElement(document, "[data-viewer]", HTMLElement);
  if (judged) {
    const slug = requireData(judged, "slug");
    for (const box of everyElement(document, "[data-said-judge]", HTMLElement)) {
      const line = box.closest("[data-said-kind]");
      if (!(line instanceof HTMLElement)) continue;
      for (const thumb of everyElement(box, "[data-said-verdict-set]", HTMLElement)) {
        thumb.addEventListener("click", async () => {
          const wanted = requireData(thumb, "saidVerdictSet");
          const held = line.dataset.saidVerdict;
          const { data, error } = await api.POST("/i/{slug}/said/verdict", {
            params: { path: { slug } },
            body: {
              kind: requireData(line, "saidKind") as "caption",
              model_id: requireData(line, "saidModel"),
              model_version: requireData(line, "saidVersion"),
              verdict: held === wanted ? null : (wanted as "right" | "wrong"),
            },
          });
          if (!data) {
            await say(refusal(error, "that verdict was not recorded"));
            return;
          }
          // Drawn from what the SERVER says it now holds, never from
          // what was clicked: the two differ on a retraction, and a
          // control that lies about its own state is worse than one
          // that does nothing.
          if (data.verdict) line.dataset.saidVerdict = data.verdict;
          else delete line.dataset.saidVerdict;
          for (const one of everyElement(box, "[data-said-verdict-set]", HTMLElement)) {
            one.setAttribute("aria-pressed", String(one.dataset.saidVerdictSet === data.verdict));
          }
        });
      }
    }
  }

  // "That is not her", said where the name is shown.
  //
  // A DENIAL, not a retraction, and the difference is the point: a
  // retraction deletes the claim and the next clustering run is free to
  // decide the same thing again, which is how correcting a false merge
  // became a chore somebody repeated after every re-run. This is a
  // record that stops it (db/authored.py `deny_person`).
  //
  // Redrawn from the answer, which is the faces the SERVER now holds --
  // never by removing the chip that was clicked. The two differ the
  // moment anything else has a say, and a browser that drew its own
  // guess would be inventing state.
  const people = findElement(document, "[data-people]", HTMLElement);
  if (judged && people) {
    const slug = requireData(judged, "slug");
    for (const deny of everyElement(people, "[data-person-deny]", HTMLElement)) {
      deny.addEventListener("click", async () => {
        const who = requireData(deny, "personDeny");
        const { data, error } = await api.POST("/i/{slug}/people/{person}/deny", {
          params: { path: { slug, person: who } },
          body: { value: true },
        });
        if (!data) {
          await say(refusal(error, "that was not recorded"));
          return;
        }
        people.replaceChildren();
        for (const [at, one] of data.people.entries()) {
          if (at) people.append(document.createTextNode(" · "));
          const held = document.createElement("span");
          held.className = "person-said";
          held.dataset.personSaid = one.slug;
          const link = document.createElement("a");
          link.href = one.href;
          link.dataset.person = one.slug;
          link.textContent = one.name ?? one.slug;
          held.append(link);
          people.append(held);
        }
        if (data.people.length === 0) {
          const none = document.createElement("span");
          none.className = "muted";
          none.dataset.peopleNone = "";
          none.textContent = "nobody named here now";
          people.append(none);
        }
      });
    }
  }

  // where it happened: one POST of desired state, then the page re-reads
  const placeForm = findElement(document, "[data-place-form]", HTMLFormElement);
  if (placeForm) {
    const slug = requireData(placeForm, "slug");
    const value = (name: string) => requireElement(placeForm, `[name="${name}"]`, HTMLInputElement).value.trim();
    const chosen = (name: string) => requireElement(placeForm, `[name="${name}"]`, HTMLSelectElement).value;

    const record = async (body: DesiredPlace) => {
      const { data, error } = await api.POST("/i/{slug}/place", { params: { path: { slug } }, body });
      if (!data) {
        await say(refusal(error, "the place could not be recorded"));
        return;
      }
      window.location.reload();
    };

    placeForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const name = value("name");
      if (!name) return;
      const within = value("within");
      void record({
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
      void record({ name: null, kind: "locality", within: null, within_kind: "country" });
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
  // The same ladder the overlay gets, through the same registry: the viewer
  // spends the press on its own state -- a zoomed picture fits, an open
  // inspector closes -- and only a viewer with nothing left to unwind lets
  // Escape mean "leave". No overlay is mounted on this page, so this is the
  // one claim on the key here; if that ever stopped being true, keys.ts
  // would say so rather than the page dismissing twice.
  register([
    {
      key: "Escape",
      by: "media page: leave",
      run: () => {
        if (viewer?.unwind()) return;
        leave();
      },
    },
  ]);
})();
