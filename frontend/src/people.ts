// The person drawer: the second concrete addressable-overlay Adapter.
// Same history contract as the media lightbox -- open PUSHES /p/{slug}
// over the mounted People index, Back/Escape/close leave in one step,
// Forward re-opens what the URL names -- but a person is an entity with a
// collection, not a piece of media, so there are no arrows: the drawer
// shows who they are and hands off to the full profile.
import { api, refusal } from "./api";
import { askChoice, say } from "./ask";
import { closestFrom, everyElement, requireData, requireElement } from "./dom";
import { addressableOverlay } from "./overlay";
import { spellDays } from "./spelling";

// days on the cards and in the drawer, spelled for the reader. The
// speller moved to ./spelling when four other surfaces turned out to
// render epochs with nothing to spell them.

/**
 * What stands where a denied picture was.
 *
 * Not the thumbnail greyed out. The name is off the picture NOW
 * (db/derived.py `withdraw_attribution`), so a cell still showing it
 * under this person would contradict what was just said.
 *
 * Undo withdraws the CLAIM and nothing else, which is the honest thing
 * and worth saying plainly: retracting deletes the record that this was
 * wrong, so the next clustering run is free to decide it again -- but it
 * does not put the name back, because no run has said so since. The
 * picture returns to this page when clustering next names them in it.
 */
const denied = (picture: string, who: string): HTMLElement => {
  const held = document.createElement("div");
  held.className = "cell-denied";
  held.dataset.personDenied = picture;

  const what = document.createElement("span");
  what.textContent = "not them";
  held.append(what);

  const undo = document.createElement("button");
  undo.type = "button";
  undo.className = "link";
  undo.textContent = "undo";
  undo.addEventListener("click", async () => {
    undo.disabled = true;
    const { data, error } = await api.POST("/i/{slug}/people/{person}/deny", {
      params: { path: { slug: picture, person: who } },
      body: { value: false },
    });
    if (!data) {
      undo.disabled = false;
      await say(refusal(error, "that was not withdrawn"));
      return;
    }
    held.dataset.personDenied = "";
    held.dataset.personWithdrawn = picture;
    what.textContent = "withdrawn — they are named here again only when clustering next says so";
    undo.replaceWith(wayBack(picture));
  });
  held.append(undo);
  return held;
};

/** The way back to the picture itself, which is where the truth is. */
const wayBack = (picture: string): HTMLElement => {
  const link = document.createElement("a");
  link.href = `/i/${picture}`;
  link.textContent = "open the picture";
  return link;
};

(() => {
  spellDays(document);
  new MutationObserver(() => spellDays(document)).observe(document.body, { childList: true, subtree: true });

  // Renaming is the People page's primary action, on the drawer and the
  // full page alike: POST the name, then go live at the new address.
  document.addEventListener("submit", async (event) => {
    const form = closestFrom(event.target, "[data-rename]", HTMLFormElement);
    if (!form) return;
    event.preventDefault();
    // The address the form posts to carries the person; the slug comes
    // from the markup rather than from parsing the action back apart.
    const slug = requireData(form, "rename");
    const name = requireElement(form, '[name="name"]', HTMLInputElement).value;
    const { data, error } = await api.POST("/p/{slug}/name", { params: { path: { slug } }, body: { name } });
    if (!data) {
      await say(refusal(error, "that name was refused"));
      return;
    }
    // A card in the "who is this?" queue stays where it is. The whole
    // point of naming in place is that twelve people cost twelve names
    // rather than twelve page loads, and navigating to the profile of
    // the one just named is the page load.
    const card = form.closest("[data-unknown]");
    if (card instanceof HTMLElement) {
      card.dataset.named = data.name;
      const named = document.createElement("span");
      named.className = "person-name";
      named.textContent = data.name;
      form.replaceWith(named);
      // The queue says how much is left, so somebody can see it ending.
      const heading = document.querySelector("[data-unknown-faces] .analyze-of");
      const left = document.querySelectorAll("[data-unknown]:not([data-named])").length;
      if (heading) heading.textContent = left === 0 ? "all named" : `${left} unnamed`;
      return;
    }
    // From the profile itself: REPLACE, never assign. The identity's
    // address just moved, and the retired slug must not remain as a
    // history stop -- Back from the renamed profile goes to /people in
    // one step, not through a 301 bounce off the old address.
    window.location.replace(`/p/${data.slug}`);
  });

  // "Not them", said on the page where somebody is actually reviewing
  // who a person is.
  //
  // The claim could already be made -- db/authored.py `deny_person`, and
  // the media inspector's chip has offered it since -- but only one
  // picture at a time, from inside that picture. The wrong face is
  // noticed HERE, looking at a wall of someone's photographs, and this
  // page could only send you somewhere else to say so.
  //
  // Delegated, not bound per cell: the grid is server-rendered today and
  // paged tomorrow, and a listener per cell would miss whatever arrives.
  const grid = document.querySelector("[data-person-pictures]");
  if (grid instanceof HTMLElement) {
    const who = requireData(grid, "personPictures");
    grid.addEventListener("click", async (event) => {
      const button = closestFrom(event.target, "[data-person-not-here]", HTMLButtonElement);
      if (!button) return;
      const shell = closestFrom(button, "[data-person-picture]", HTMLElement);
      if (!shell) return;
      const picture = requireData(shell, "personPicture");
      button.disabled = true;
      const { data, error } = await api.POST("/i/{slug}/people/{person}/deny", {
        params: { path: { slug: picture, person: who } },
        body: { value: true },
      });
      if (!data) {
        button.disabled = false;
        await say(refusal(error, "that was not recorded"));
        return;
      }
      shell.replaceWith(denied(picture, who));
    });
  }

  // "Take their face from THIS one."
  //
  // Delegated at the grid beside the denial, and for the same reason it
  // is here at all: this is where somebody is looking at the pictures,
  // and the avatar being wrong is something you notice while looking at
  // the right one.
  const pictures = document.querySelector("[data-person-pictures]");
  if (pictures instanceof HTMLElement) {
    const whose = requireData(pictures, "personPictures");
    pictures.addEventListener("click", async (event) => {
      const button = closestFrom(event.target, "[data-person-face]", HTMLButtonElement);
      if (!button) return;
      const picture = requireData(button, "personFace");
      const held = await api.POST("/p/{slug}/face", {
        params: { path: { slug: whose } },
        body: { file: picture },
      });
      if (held.error) {
        await say(refusal(held.error, "that face was not chosen"));
        return;
      }
      // The avatar is a cached content address, so the browser will hand
      // back the one it already has. The cache-buster is on the IMG
      // only -- nothing about the person's address changed.
      for (const face of everyElement(document, ".person-face-big", HTMLImageElement)) {
        face.src = `/avatar/${whose}?chosen=${Date.now()}`;
      }
      button.dataset.chosen = "";
    });
  }

  // "These two were always one person."
  //
  // Denying says "not them, in this picture". This says the other thing
  // a durable naming model needs and did not have: a clustering run
  // splits somebody into four, and a threshold cannot fix that without
  // trading away somebody else's correct grouping. Said here it is
  // local, permanent, and re-applied after every future run.
  const folder = document.querySelector("[data-same-as]");
  if (folder instanceof HTMLElement) {
    const keeping = requireData(folder, "sameAs");
    folder.addEventListener("click", async () => {
      const shelf = await api.GET("/people", { headers: { accept: "application/json" } });
      // Not this one: folding somebody into themselves is not a thing
      // to be offered and then refused.
      const others = (shelf.data ?? []).filter((one) => one.slug !== keeping);
      if (others.length === 0) {
        await say("there is nobody else to fold in");
        return;
      }
      // Names, with the address underneath -- the same shape the
      // smart-collection chooser uses, so nobody has to know the
      // internal spelling of a person they named themselves.
      const chosen = await askChoice(
        "who is the same person as this one?",
        others.map((one) => ({
          value: one.slug,
          label: one.name ?? one.slug,
          note: `${one.pictures} ${one.pictures === 1 ? "picture" : "pictures"} · /p/${one.slug}`,
        })),
        { detail: "their pictures, names and corrections come here, and their address redirects here afterwards" },
      );
      if (chosen === null) return;
      const { data, error } = await api.POST("/p/{slug}/same-as", {
        params: { path: { slug: keeping } },
        body: { other: chosen },
      });
      if (!data) {
        await say(refusal(error, "those two were not merged"));
        return;
      }
      // REPLACE: the folded address now redirects here, and leaving it
      // in history would put a 301 between Back and wherever they came
      // from.
      window.location.replace(`/p/${data.slug}`);
    });
  }

  // The drawer is the person adapter over the AddressableOverlay: the
  // shell (overlay.ts) owns everything an overlay shares, a person is not
  // media, so there is nothing left to add here -- no arrows, no
  // generation evidence.
  //
  // Asked for unconditionally. The full profile at /p/{slug} renders no
  // drawer root, so this returns null there -- the absence is a fact about
  // that page's DOM, where it used to be a fact about whether the template
  // happened to list overlay.js beside this file.
  addressableOverlay({
    root: "[data-drawer-root]",
    trigger: "[data-person]",
    pathPrefix: "/p/",
  });
})();
